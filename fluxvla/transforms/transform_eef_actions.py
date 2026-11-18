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
"""Pose-adapter driven EEF training inputs and inference outputs.

Both transforms are embodiment-agnostic: like ``RelativeActions`` with its
delta mask, one registered type covers every robot, selected by the
``pose_adapter`` name (mirroring the ``pose_adapter`` field of statistics
profiles). The adapter owns all robot specifics: ``forward`` for the EEF
layout and masks, ``inverse`` for the execution bridge back to joint
targets. Benchmarks whose recorded actions are already delta EEF values
(for example LIBERO) do not need these transforms.
"""

import warnings

import numpy as np

from fluxvla.datasets.kinematics.eef_so3_actions import (create_pose_adapter,
                                                         so3_delta)
from fluxvla.engines import TRANSFORMS
from .normalize import DenormalizePrivateAction

__all__ = ['RelativeEEFInputs', 'DenormalizeEEFAction']


@TRANSFORMS.register_module()
class RelativeEEFInputs:
    """Absolute source state/targets -> absolute EEF state + delta chunk."""

    def __init__(self, pose_adapter, urdf_path=None, model_path=None):
        self.adapter = create_pose_adapter(pose_adapter, urdf_path)

    def __call__(self, data):
        state = self.adapter.forward(data['states'])
        data = dict(data, states=state)
        if 'actions' in data:
            actions = self.adapter.forward(data['actions'])
            data['actions'] = so3_delta(actions, state[..., None, :],
                                        self.adapter.delta_mask,
                                        self.adapter.rotation_mask)
        return data


@TRANSFORMS.register_module()
class DenormalizeEEFAction(DenormalizePrivateAction):
    """Delta EEF -> absolute EEF -> adapter IK -> joint-space targets.

    ``state`` is the raw joint observation saved by PrivateInferenceDataset,
    not a normalized or EEF state; the adapter maps it to the EEF reference
    pose and seeds its ``inverse`` execution bridge. The caller passes only
    the executed chunk prefix (``RobotwinEvalRunner`` slices it with
    ``eval_chunk_size``). Failed targets hold the preceding solution;
    failures are counted and warnings expose them in worker logs.
    """

    def __init__(self,
                 pose_adapter,
                 urdf_path=None,
                 position_tolerance=0.002,
                 rotation_tolerance=0.02,
                 max_nfev=100,
                 **kwargs):
        super().__init__(**kwargs)
        self.adapter = create_pose_adapter(pose_adapter, urdf_path)
        if max_nfev <= 0:
            raise ValueError('max_nfev must be positive')
        if position_tolerance <= 0 or rotation_tolerance <= 0:
            raise ValueError('IK tolerances must be positive')
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.max_nfev = max_nfev
        self.ik_failures = 0
        self.ik_attempts = 0

    def __call__(self, data):
        actions = np.asarray(super().__call__(data), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        dim = len(self.adapter.delta_mask)
        if actions.ndim != 2 or actions.shape[1] != dim:
            raise ValueError(f'Expected a [T, {dim}] EEF action chunk')
        state = np.asarray(data['state'], dtype=np.float64).reshape(-1)
        targets = so3_delta(
            actions,
            self.adapter.forward(state),
            self.adapter.delta_mask,
            self.adapter.rotation_mask,
            inverse=True)
        result, failures, attempts = self.adapter.inverse(
            targets, state, self.position_tolerance, self.rotation_tolerance,
            self.max_nfev)
        self.ik_failures += failures
        self.ik_attempts += attempts
        if failures:
            warnings.warn(
                f'EEF IK held {failures} arm targets; '
                f'cumulative={self.ik_failures}/{self.ik_attempts}',
                RuntimeWarning)
        return result
