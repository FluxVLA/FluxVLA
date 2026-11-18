# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mask-driven relative EEF/SO(3) actions shared across embodiments.

``delta_mask`` marks the action dimensions expressed relative to the current
state and ``rotation_mask`` marks the contiguous rotation-vector triples among
them. Linear delta dimensions subtract the state, rotation triples compose
spatial deltas (``Ra Rs^T``; inverse ``Rd Rs``), and unmasked dimensions such
as grippers stay absolute. The conventions match SLA's ``DeltaActions`` and
``AbsoluteActions`` transforms, so SLA-pretrained checkpoints keep their
input/output space on any embodiment.

Embodiment-specific geometry lives in one sibling module per robot (for
example ``robotwin_eef``), created by name via :func:`create_pose_adapter`.
An adapter exposes ``forward`` (raw values -> absolute ``[position, rotation
vector, gripper]`` layout), ``inverse`` (absolute EEF targets -> joint
values as ``(joints, failures, attempts)``), ``delta_mask`` and
``rotation_mask``. A new robot adds such a module plus a branch in
:func:`create_pose_adapter`. Benchmarks whose recorded actions are already
delta EEF values (for example LIBERO) need neither an adapter nor these
helpers.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ['create_pose_adapter', 'make_bool_mask', 'so3_delta']


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Expand run lengths into a mask; positive runs are True, negative False.

    ``make_bool_mask(2, -2, 1)`` returns ``(True, True, False, False, True)``,
    matching OpenPI's helper of the same name.
    """
    mask: list[bool] = []
    for length in dims:
        if length == 0:
            raise ValueError('Mask run lengths must be non-zero')
        mask.extend([length > 0] * abs(length))
    return tuple(mask)


def validate_values(values: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1:] != (dim, ) or not np.isfinite(values).all():
        raise ValueError(f'Expected finite values with last dimension {dim}')
    return values


def _rotation_starts(delta_mask: tuple[bool, ...],
                     rotation_mask: tuple[bool, ...]) -> tuple[int, ...]:
    delta = np.asarray(delta_mask, dtype=bool)
    rotation = np.asarray(rotation_mask, dtype=bool)
    if delta.shape != rotation.shape:
        raise ValueError('delta_mask and rotation_mask lengths differ')
    if np.any(rotation & ~delta):
        raise ValueError('rotation_mask must be a subset of delta_mask')
    starts = []
    index = 0
    while index < len(rotation):
        if not rotation[index]:
            index += 1
            continue
        if not rotation[index:index + 3].all() or index + 3 > len(rotation):
            raise ValueError('rotation_mask requires contiguous groups of 3')
        starts.append(index)
        index += 3
    return tuple(starts)


def so3_delta(actions: np.ndarray,
              state: np.ndarray,
              delta_mask: tuple[bool, ...],
              rotation_mask: tuple[bool, ...],
              inverse: bool = False) -> np.ndarray:
    """Convert absolute actions to deltas w.r.t. ``state``, or back.

    ``state`` must broadcast to ``actions`` (e.g. ``[D]``, ``[B, 1, D]``, or
    ``[N, D]``). Every target in a chunk is referenced to the same
    observation.
    """
    starts = _rotation_starts(delta_mask, rotation_mask)
    dim = len(delta_mask)
    actions = validate_values(actions, dim)
    state = validate_values(state, dim)
    actions, state = np.broadcast_arrays(actions, state)
    linear = np.asarray(delta_mask, bool) & ~np.asarray(rotation_mask, bool)
    result = actions.copy()
    result[..., linear] = (
        actions[..., linear] +
        state[..., linear] if inverse else actions[..., linear] -
        state[..., linear])
    for start in starts:
        rot = slice(start, start + 3)
        shape = actions[..., rot].shape
        ra = Rotation.from_rotvec(actions[..., rot].reshape(-1, 3).copy())
        rs = Rotation.from_rotvec(state[..., rot].reshape(-1, 3).copy())
        relative = ra * rs if inverse else ra * rs.inv()
        result[..., rot] = relative.as_rotvec().reshape(shape)
    return result.astype(np.float32)


def create_pose_adapter(name, urdf_path=None):
    """Instantiate the pose adapter registered under ``name``."""
    if name == 'robotwin-eef':
        from fluxvla.datasets.kinematics.robotwin_eef import get_robotwin_eef
        if urdf_path is None:
            raise ValueError('robotwin-eef requires urdf_path')
        return get_robotwin_eef(urdf_path)
    raise ValueError(f'Unknown pose adapter: {name!r}')
