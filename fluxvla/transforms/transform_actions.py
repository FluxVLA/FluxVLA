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

import json
from typing import Dict, List, Optional

import numpy as np

from fluxvla.engines import TRANSFORMS
from fluxvla.transforms.normalize import DenormalizeLiberoAction


def _axisangle_to_matrix(axisangle: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(axisangle, dtype=np.float32)
    if rotvec.shape[-1] != 3:
        raise ValueError(f'axisangle must have last dim 3, got {rotvec.shape}')

    original_shape = rotvec.shape[:-1]
    flat = rotvec.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=-1, keepdims=True)
    axis = np.divide(
        flat,
        theta,
        out=np.zeros_like(flat),
        where=theta > 1e-8,
    )
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    skew = np.zeros((flat.shape[0], 3, 3), dtype=np.float32)
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x

    eye = np.eye(3, dtype=np.float32)[None, :, :]
    sin_theta = np.sin(theta)[:, :, None]
    cos_theta = np.cos(theta)[:, :, None]
    matrix = eye + sin_theta * skew + (1.0 - cos_theta) * (skew @ skew)
    matrix[(theta[:, 0] <= 1e-8)] = eye
    return matrix.reshape(*original_shape, 3, 3)


def _normalize_rotation_matrices(matrices: np.ndarray) -> np.ndarray:
    original_shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3).astype(np.float32, copy=False)
    u, _, vh = np.linalg.svd(flat)
    projected = u @ vh
    reflection = np.linalg.det(projected) < 0
    if np.any(reflection):
        u[reflection, :, -1] *= -1
        projected[reflection] = u[reflection] @ vh[reflection]
    return projected.astype(
        np.float32, copy=False).reshape(*original_shape, 3, 3)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'rotation matrix must end with [3,3], got '
                         f'{matrix.shape}')
    return matrix[..., :, :2].swapaxes(-1, -2).reshape(*matrix.shape[:-2], 6)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] != 6:
        raise ValueError(f'rot6d must have last dim 6, got {rot6d.shape}')
    original_shape = rot6d.shape[:-1]
    flat = rot6d.reshape(-1, 6)
    col0 = flat[:, :3]
    col1 = flat[:, 3:]
    col2 = np.cross(col0, col1, axis=-1)
    matrix = np.stack((col0, col1, col2), axis=-1)
    return _normalize_rotation_matrices(matrix).reshape(*original_shape, 3, 3)


def _matrix_to_axisangle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f'rotation matrix must end with [3,3], got '
                         f'{matrix.shape}')

    original_shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    qw = np.sqrt(
        np.maximum(0.0,
                   1.0 + flat[:, 0, 0] + flat[:, 1, 1] + flat[:, 2, 2])) * 0.5
    qx = np.sqrt(
        np.maximum(0.0,
                   1.0 + flat[:, 0, 0] - flat[:, 1, 1] - flat[:, 2, 2])) * 0.5
    qy = np.sqrt(
        np.maximum(0.0,
                   1.0 - flat[:, 0, 0] + flat[:, 1, 1] - flat[:, 2, 2])) * 0.5
    qz = np.sqrt(
        np.maximum(0.0,
                   1.0 - flat[:, 0, 0] - flat[:, 1, 1] + flat[:, 2, 2])) * 0.5
    qx = np.copysign(qx, flat[:, 2, 1] - flat[:, 1, 2])
    qy = np.copysign(qy, flat[:, 0, 2] - flat[:, 2, 0])
    qz = np.copysign(qz, flat[:, 1, 0] - flat[:, 0, 1])

    quat = np.stack((qx, qy, qz, qw), axis=-1).astype(np.float32)
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.divide(
        quat,
        quat_norm,
        out=np.zeros_like(quat),
        where=quat_norm > 1e-8,
    )
    quat = np.where(quat[:, 3:4] < 0.0, -quat, quat)

    vec = quat[:, :3]
    w = np.clip(quat[:, 3], -1.0, 1.0)
    sin_half = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    axis = np.divide(
        vec,
        sin_half[:, None],
        out=np.zeros_like(vec),
        where=sin_half[:, None] > 1e-8,
    )
    rotvec = axis * angle[:, None]
    small = sin_half <= 1e-8
    rotvec[small] = 2.0 * vec[small]
    return rotvec.astype(np.float32, copy=False).reshape(*original_shape, 3)


def _libero_axisangle_action_to_rot6d(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 7:
        raise ValueError('LIBERO frame-wise action must have shape [..., 7], '
                         f'got {action.shape}.')
    rotation = _matrix_to_rot6d(_axisangle_to_matrix(action[..., 3:6]))
    return np.concatenate((action[..., :3], rotation, action[..., 6:7]),
                          axis=-1)


def _libero_rot6d_action_to_axisangle(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 10:
        raise ValueError('LIBERO rot6d action must have shape [..., 10], '
                         f'got {action.shape}.')
    rotation = _matrix_to_axisangle(_rot6d_to_matrix(action[..., 3:9]))
    return np.concatenate((action[..., :3], rotation, action[..., 9:10]),
                          axis=-1)


@TRANSFORMS.register_module()
class DeltaActions:
    """Convert selected absolute action dimensions to state-relative deltas.

    This is the same generic operation used by OpenPI.  ``state_key`` and
    ``action_key`` keep it usable by both the training pipeline and robot
    post-processing code instead of tying it to one embodiment.
    """

    def __init__(self,
                 mask: List[bool] = None,
                 state_key: str = 'states',
                 action_key: str = 'actions') -> None:
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.state_key = state_key
        self.action_key = action_key

    def __call__(self, data: Dict) -> Dict:
        if self.action_key not in data or self.mask is None:
            return data

        states = np.asarray(data[self.state_key])
        actions = np.asarray(data[self.action_key]).copy()
        dims = self.mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(self.mask, states[..., :dims], 0), axis=-2)
        data[self.action_key] = actions

        return data


@TRANSFORMS.register_module()
class AbsoluteActions(DeltaActions):
    """Invert :class:`DeltaActions` for selected action dimensions."""

    def __call__(self, data: Dict) -> Dict:
        if self.action_key not in data or self.mask is None:
            return data

        states = np.asarray(data[self.state_key])
        actions = np.asarray(data[self.action_key]).copy()
        dims = self.mask.shape[-1]
        actions[..., :dims] += np.expand_dims(
            np.where(self.mask, states[..., :dims], 0), axis=-2)
        data[self.action_key] = actions
        return data


@TRANSFORMS.register_module()
class ProcessLiberoActions(DeltaActions):
    """Backward-compatible name for the generic delta-action transform."""


@TRANSFORMS.register_module()
class JointSignTransform:
    """Apply configured joint-axis signs to state and/or action vectors."""

    def __init__(self,
                 signs: List[float],
                 state_key: Optional[str] = 'states',
                 action_key: Optional[str] = 'actions') -> None:
        signs = np.asarray(signs, dtype=np.float32)
        if signs.ndim != 1 or not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError('signs must be a one-dimensional list of +/-1.')
        self.signs = signs
        self.state_key = state_key
        self.action_key = action_key

    def __call__(self, data: Dict) -> Dict:
        for key in (self.state_key, self.action_key):
            if key is None or key not in data:
                continue
            values = np.asarray(data[key], dtype=np.float32)
            if values.shape[-1] != self.signs.shape[0]:
                raise ValueError(
                    f'JointSignTransform expected {self.signs.shape[0]} '
                    f'dimensions for {key!r}, got {values.shape}.')
            data[key] = values * self.signs
        return data


_ALOHA_DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


def _normalize_range(x, min_value, max_value):
    return (x - min_value) / (max_value - min_value)


def _unnormalize_range(x, min_value, max_value):
    return x * (max_value - min_value) + min_value


def _aloha_gripper_to_angular(value):
    value = _unnormalize_range(value, 0.01844, 0.05800)
    arm_length = 0.036
    horn_radius = 0.022
    argument = (horn_radius**2 + value**2 - arm_length**2) / (2 * horn_radius *
                                                              value)
    value = np.arcsin(np.clip(argument, -1.0, 1.0))
    return _normalize_range(value, 0.5476, 1.6296)


def _aloha_gripper_from_angular(value):
    value = value + 0.5476
    return _normalize_range(value, -0.6213, 1.4910)


def _aloha_gripper_from_angular_inv(value):
    value = _unnormalize_range(value, -0.6213, 1.4910)
    return value - 0.5476


def _aloha_state_gripper_to_pi(state):
    state = np.asarray(state, dtype=np.float32).copy()
    state[..., [6, 13]] = _aloha_gripper_to_angular(state[..., [6, 13]])
    return state


def _aloha_action_gripper_to_pi(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., [6, 13]] = _aloha_gripper_from_angular_inv(action[...,
                                                                  [6, 13]])
    return action


def _aloha_action_gripper_from_pi(action):
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., [6, 13]] = _aloha_gripper_from_angular(action[..., [6, 13]])
    return action


@TRANSFORMS.register_module()
class OpenPIAlohaGripperCoordinates:
    """Convert only ALOHA grippers to OpenPI's internal coordinates.

    Joint signs, delta actions, and normalization deliberately remain in the
    generic ``JointSignTransform``, ``DeltaActions``, and
    ``NormalizeStatesAndActions`` stages.
    """

    def __init__(self,
                 adapt_to_pi: bool = True,
                 gripper_input_range=None,
                 *args,
                 **kwargs):
        self.adapt_to_pi = adapt_to_pi
        if gripper_input_range is not None:
            if (len(gripper_input_range) != 2
                    or gripper_input_range[1] <= gripper_input_range[0]):
                raise ValueError(
                    'gripper_input_range must be an increasing pair.')
            gripper_input_range = tuple(map(float, gripper_input_range))
        self.gripper_input_range = gripper_input_range

    def _standardize_grippers(self, value):
        value = np.asarray(value, dtype=np.float32).copy()
        if self.gripper_input_range is not None:
            low, high = self.gripper_input_range
            value[..., [6, 13]] = (value[..., [6, 13]] - low) / (high - low)
        return value

    def __call__(self, data: Dict) -> Dict:
        states = self._standardize_grippers(data['states'])
        if states.shape[-1] != 14:
            raise ValueError(
                'OpenPIAlohaGripperCoordinates expects 14-dimensional '
                f'states, got {states.shape}.')
        if self.adapt_to_pi:
            states = _aloha_state_gripper_to_pi(states)
        data['states'] = states

        if 'actions' not in data:
            return data

        actions = self._standardize_grippers(data['actions'])
        if actions.shape[-1] != 14:
            raise ValueError(
                'OpenPIAlohaGripperCoordinates expects 14-dimensional '
                f'actions, got {actions.shape}.')
        if self.adapt_to_pi:
            actions = _aloha_action_gripper_to_pi(actions)
        data['actions'] = actions
        return data


@TRANSFORMS.register_module()
class OpenPIAlohaActionPostprocess:
    """Invert PI0.5 ALOHA quantile, delta, and coordinate transforms."""

    def __init__(self,
                 norm_stats: Optional[Dict | str] = None,
                 openpi_norm_stats: Optional[Dict | str] = None,
                 action_dim: int = 14,
                 adapt_to_pi: bool = True,
                 use_delta_joint_actions: bool = True,
                 *args,
                 **kwargs):
        source = openpi_norm_stats or norm_stats
        if isinstance(source, str):
            with open(source, 'r', encoding='utf-8') as f:
                source = json.load(f)
        if isinstance(source, dict) and 'norm_stats' in source:
            source = source['norm_stats']
        if source is None or 'actions' not in source:
            raise ValueError('OpenPI ALOHA action statistics are required.')
        self.action_stats = source['actions']
        self.action_dim = action_dim
        self.adapt_to_pi = adapt_to_pi
        self.use_delta_joint_actions = use_delta_joint_actions
        joint_signs = [
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
        ]
        self.state_joint_signs = JointSignTransform(
            signs=joint_signs, state_key='state', action_key=None)
        self.action_joint_signs = JointSignTransform(
            signs=joint_signs, state_key=None, action_key='action')
        self.absolute_actions = AbsoluteActions(
            mask=_ALOHA_DELTA_MASK, state_key='state', action_key='action')

    def __call__(self, data: Dict) -> np.ndarray:
        actions = np.asarray(data['action'])
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError('Only batch size one is supported for ALOHA '
                                 'robot inference.')
            actions = actions[0]
        actions = actions[..., :self.action_dim]

        low = np.asarray(self.action_stats['q01'], dtype=np.float32)
        high = np.asarray(self.action_stats['q99'], dtype=np.float32)
        actions = 0.5 * (actions + 1.0) * (high - low + 1e-6) + low

        if data.get('state') is None:
            raise ValueError('Current ALOHA state is required to invert '
                             'delta actions.')
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape[-1] != self.action_dim:
            raise ValueError('A 14-dimensional current ALOHA state is '
                             'required to invert delta actions.')
        if self.adapt_to_pi:
            state = self.state_joint_signs({'state': state})['state']
            state = _aloha_state_gripper_to_pi(state)
        if self.use_delta_joint_actions:
            actions = self.absolute_actions({
                'state': state,
                'action': actions,
            })['action']
        if self.adapt_to_pi:
            actions = self.action_joint_signs({'action': actions})['action']
            actions = _aloha_action_gripper_from_pi(actions)
        return actions.astype(np.float32, copy=False)


@TRANSFORMS.register_module()
class LiberoFramewiseActionToRot6D:
    """Convert LIBERO stored 7D frame-wise deltas to official 10D rot6d."""

    def __init__(self, action_key: str = 'actions') -> None:
        self.action_key = action_key

    def __call__(self, data: Dict) -> Dict:
        if self.action_key not in data:
            return data
        data[self.action_key] = _libero_axisangle_action_to_rot6d(
            data[self.action_key])
        return data


@TRANSFORMS.register_module()
class DenormalizeLiberoFramewiseRot6DAction:
    """Denormalize official 10D rot6d LIBERO actions to 7D env commands."""

    def __init__(self, *args, norm_type: str = 'quantile', **kwargs) -> None:
        if norm_type == 'quantile_rot':
            norm_type = 'quantile'
        self.denormalize = DenormalizeLiberoAction(
            *args, norm_type=norm_type, **kwargs)

    def __call__(self, data: Dict) -> np.ndarray:
        return _libero_rot6d_action_to_axisangle(self.denormalize(data))
