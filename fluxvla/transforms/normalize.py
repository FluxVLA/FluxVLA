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
import torch

from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.eval_utils import quat2axisangle
from fluxvla.engines.utils.robot_utils import (invert_gripper_action,
                                               normalize_gripper_action)


@TRANSFORMS.register_module()
class Normalize:
    """Normalize the data using provided statistics.
    This transform normalizes the data by subtracting
    the mean and dividing by the standard deviation.
    Supports different normalization types: 'mean_std',
        'quantile', or 'min_max'.

    Args:
        norm_stats (List): List of normalization statistics,
            where each element is a dictionary  containing
            'mean', 'std', 'q01', 'q99', 'min', and 'max' for each feature.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
    """

    def __init__(self,
                 norm_stats: List,
                 norm_type: str = 'mean_std',
                 strict: bool = False):
        self.norm_stats = norm_stats
        self.norm_type = norm_type
        self.strict = strict

    def __call__(self, data: Dict) -> Dict:
        if self.norm_stats is None:
            return data
        for key, value in data.items():
            if key in self.norm_stats.keys():
                if self.norm_type == 'quantile':
                    data[key] = self._normalize_quantile(
                        value, self.norm_stats[key])
                elif self.norm_type == 'min_max':
                    data[key] = self._normalize_min_max(
                        value, self.norm_stats[key])
                else:  # norm_type == 'mean_std'
                    data[key] = self._normalize(value, self.norm_stats[key])
        return data

    def _normalize(self, x, stats: Dict):
        return (x - torch.tensor(stats['mean'])) / (
            torch.tensor(stats['std']) + 1e-6)

    def _normalize_quantile(self, x, stats: torch.tensor):
        assert stats['q01'] is not None
        assert stats['q99'] is not None
        return (x - torch.tensor(stats['q01'])) / (torch.tensor(
            stats['q99']) - torch.tensor(stats['q01']) + 1e-6) * 2.0 - 1.0

    def _normalize_min_max(self, x, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        return (x - torch.tensor(stats['min'])) / (torch.tensor(
            stats['max']) - torch.tensor(stats['min']) + 1e-6) * 2.0 - 1.0


@TRANSFORMS.register_module()
class DenormalizeLiberoAction:
    """Denormalize the data using provided statistics.
    This transform reverses the normalization done using
    mean/std, quantiles, or min_max.

    Args:
        norm_stats (str or Dict): Normalization statistics,
            which can be a JSON string or a dictionary
            containing 'mean', 'std', 'q01', 'q99', 'min', and 'max' for each
            feature. If a string, it should be a JSON representation
            of the normalization statistics.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
        denorm_action (bool): If True, denormalize the action.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
        normalize_gripper_action (bool): If True, normalize
            the gripper action. This is useful for tasks
            where the gripper action is part of the state
            and needs to be denormalized.
        invert_gripper_action (bool): If True, invert the
            gripper action. This is useful for tasks where
            the gripper action is represented in a way that
            requires inversion (e.g., opening vs. closing).
            This is useful for tasks where the gripper action
            is represented in a way that requires inversion
            (e.g., opening vs. closing).
    """

    def __init__(self,
                 norm_stats: str,
                 action_dim: int = None,
                 norm_type: str = 'mean_std',
                 strict: bool = False,
                 denorm_action: bool = True,
                 normalize_gripper_action: bool = True,
                 invert_gripper_action: bool = True,
                 action_norm_mask: List[bool] = None,
                 clip_normalized_action: bool = False):
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.strict = strict
        self.denorm_action = denorm_action
        self.normalize_gripper_action = normalize_gripper_action
        self.invert_gripper_action = invert_gripper_action
        self.action_norm_mask = action_norm_mask
        self.clip_normalized_action = bool(clip_normalized_action)

    def __call__(self, data: Dict) -> Dict:
        """Denormalize the data using the provided statistics.
        This method denormalizes the action in the data
        if the `denorm_action` flag is set to True.
        It retrieves the normalization statistics based on
        the `task_suite_name` from the data and applies
        the appropriate denormalization method.  # noqa: E501

        Args:
            data (Dict): The data to be denormalized, which should
                contain keys that match the keys in `norm_stats`.
        """
        if self.norm_stats is not None and self.denorm_action:
            norm_stats_key = data.get('norm_stats_key')
            norm_stats = self.norm_stats[norm_stats_key]
            action = data.get('action', None)
            assert action is not None, \
                f'Action is not found in the data: {data.keys()}'
            if self.norm_type == 'quantile':
                action = self._denormalize_quantile(action,
                                                    norm_stats['action'])
            elif self.norm_type == 'min_max':
                action = self._denormalize_min_max(action,
                                                   norm_stats['action'])
            else:  # norm_type == 'mean_std'
                action = self._denormalize(action, norm_stats['action'])
        if self.normalize_gripper_action:
            action = normalize_gripper_action(action, binarize=True)
        if self.invert_gripper_action:
            action = invert_gripper_action(action)

        if self.action_dim is not None:
            action = action[:self.action_dim]
        return action

    def _denormalize(self, normalized_action: np.ndarray, stats: Dict):
        assert 'mean' in stats and stats['mean'] is not None
        assert 'std' in stats and stats['std'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]
        if self.clip_normalized_action:
            normalized_action = np.clip(normalized_action, -1.0, 1.0)

        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(stats['mean'], dtype=bool)
        action = np.where(
            mask,
            normalized_action * np.array(stats['std']) +
            np.array(stats['mean']), normalized_action)
        return action

    def _denormalize_quantile(self, normalized_action: np.ndarray,
                              stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]
        if self.clip_normalized_action:
            normalized_action = np.clip(normalized_action, -1.0, 1.0)
        if self.action_norm_mask is not None:
            mask = np.array(self.action_norm_mask)
        else:
            mask = np.ones_like(stats['q01'], dtype=bool)  # noqa: E501
        action_high = np.array(stats['q99'])
        action_low = np.array(stats['q01'])
        mask = np.array(mask)
        action = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) +
            action_low,  # noqa: E501
            normalized_action,
        )
        return action

    def _denormalize_min_max(self, normalized_action: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        if self.action_dim is not None:
            normalized_action = normalized_action[..., :self.action_dim]
        if self.clip_normalized_action:
            normalized_action = np.clip(normalized_action, -1.0, 1.0)
        if self.action_norm_mask is not None:
            mask = np.array(self.action_norm_mask)
        else:
            mask = np.ones_like(stats['min'], dtype=bool)
        action_high = np.array(stats['max'])
        action_low = np.array(stats['min'])
        mask = np.array(mask)
        action = np.where(
            mask,
            0.5 * (normalized_action + 1) * (action_high - action_low) +
            action_low,
            normalized_action,
        )
        return action


@TRANSFORMS.register_module()
class DenormalizePrivateAction(DenormalizeLiberoAction):
    """Denormalize the data using provided statistics.
    This transform reverses the normalization done using
    mean/std, quantiles, or min_max.

    Args:
        norm_stats (str or Dict): Normalization statistics,
            which can be a JSON string or a dictionary
            containing 'mean', 'std', 'q01', 'q99', 'min', and 'max' for each
            feature. If a string, it should be a JSON representation
            of the normalization statistics.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'mean_std'.
        strict (bool): If True, raise an error if the
            data does not match the expected structure.
        denorm_action (bool): If True, denormalize the action.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
            This is useful for tasks where the action is
            part of the state and needs to be denormalized.
        normalize_gripper_action (bool): If True, normalize
            the gripper action. This is useful for tasks
            where the gripper action is part of the state
            and needs to be denormalized.
        invert_gripper_action (bool): If True, invert the
            gripper action. This is useful for tasks where
            the gripper action is represented in a way that
            requires inversion (e.g., opening vs. closing).
            This is useful for tasks where the gripper action
            is represented in a way that requires inversion
            (e.g., opening vs. closing).
    """

    def __init__(self,
                 norm_stats: str,
                 action_dim: int = None,
                 norm_type: str = 'mean_std',
                 strict: bool = False,
                 denorm_action: bool = True,
                 normalize_gripper_action: bool = True,
                 invert_gripper_action: bool = True,
                 action_norm_mask: List[bool] = None,
                 statistic_name: str = 'private',
                 discrete_action_dims: List[int] = None,
                 discrete_norm_type: str = 'min_max'):
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.action_dim = action_dim
        self.norm_type = norm_type
        self.strict = strict
        self.denorm_action = denorm_action
        self.action_norm_mask = action_norm_mask
        self.statistic_name = statistic_name
        self.discrete_action_dims = (
            list(discrete_action_dims) if discrete_action_dims else None)
        self.discrete_norm_type = discrete_norm_type

    def __call__(self, data: Dict) -> Dict:
        """Denormalize the data using the provided statistics.
        This method denormalizes the action in the data
        if the `denorm_action` flag is set to True.
        It retrieves the normalization statistics based on
        the `task_suite_name` from the data and applies
        the appropriate denormalization method.  # noqa: E501

        Args:
            data (Dict): The data to be denormalized, which should
                contain keys that match the keys in `norm_stats`.
        """
        if self.norm_stats is not None and self.denorm_action:
            norm_stats = self.norm_stats[self.statistic_name]
            action = data.get('action', None)
            assert action is not None, \
                f'Action is not found in the data: {data.keys()}'
            action = np.asarray(action)
            if action.ndim == 3:
                if action.shape[0] != 1:
                    raise ValueError(
                        'Only batch size one is supported for action '
                        f'denormalization, got {action.shape}.')
                action = action[0]
            elif action.ndim == 2 and action.shape[0] == 1:
                action = action[0]
            stats = norm_stats['action']
            cont = self._denormalize_by_type(action, stats, self.norm_type,
                                             self.action_norm_mask)
            if self.discrete_action_dims:
                feat_dim = cont.shape[-1]
                assert all(
                    0 <= d < feat_dim for d in self.discrete_action_dims), (
                        f'discrete_action_dims {self.discrete_action_dims} '
                        f'out of range for action width {feat_dim}')
                disc_mask = np.zeros(feat_dim, dtype=bool)
                disc_mask[list(self.discrete_action_dims)] = True
                disc = self._denormalize_by_type(action, stats,
                                                 self.discrete_norm_type,
                                                 disc_mask.tolist())
                action = np.where(disc_mask, disc, cont)
            else:
                action = cont
        return action

    def _denormalize_by_type(self,
                             action: np.ndarray,
                             stats: Dict,
                             norm_type: str,
                             mask: List[bool] = None) -> np.ndarray:
        saved = self.action_norm_mask
        self.action_norm_mask = mask
        try:
            if norm_type == 'quantile':
                return self._denormalize_quantile(action, stats)
            if norm_type == 'min_max':
                return self._denormalize_min_max(action, stats)
            return self._denormalize(action, stats)
        finally:
            self.action_norm_mask = saved


@TRANSFORMS.register_module()
class DenormalizeDeltaAction(DenormalizePrivateAction):
    """Denormalize selected state-relative dimensions to absolute commands.

    Quantile/mean-std denormalization is applied first.  Dimensions selected
    by ``delta_action_mask`` are then offset by the current raw robot state;
    unselected dimensions, such as grippers, remain absolute.
    """

    def __init__(self, delta_action_mask: List[bool], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_action_mask = np.asarray(delta_action_mask, dtype=bool)
        if self.delta_action_mask.ndim != 1:
            raise ValueError('delta_action_mask must be one-dimensional')

    def __call__(self, data: Dict) -> np.ndarray:
        action = np.asarray(super().__call__(data), dtype=np.float32)
        state = data.get('state')
        if state is None:
            raise ValueError(
                'Current raw robot state is required to restore delta '
                'actions.')
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        if state.ndim != 1:
            raise ValueError(
                f'Current robot state must have shape [D], got {state.shape}.')
        dims = len(self.delta_action_mask)
        if action.shape[-1] < dims or state.shape[-1] < dims:
            raise ValueError(
                f'Delta mask length {dims} exceeds action/state dimensions '
                f'{action.shape[-1]}/{state.shape[-1]}.')
        action[..., :dims] += np.where(self.delta_action_mask, state[:dims],
                                       0.0)
        return action


@TRANSFORMS.register_module()
class NormalizeStatesAndActions:
    """Normalize states and actions in the data.
    This transform normalizes the state and action
    dimensions in the data to match the specified
    action dimension. It pads the state and action
    dimensions to the specified action dimension.

    Args:
        action_dim (int): The dimension to which the state
            and action should be normalized.
        pad_value (float): The value to use for padding.
            Defaults to 0.0.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', 'min_max', or 'none'.
            Defaults to 'mean_std'.
        state_norm_type (str): Optional normalization type for states.
            Defaults to `norm_type`.
        action_norm_type (str): Optional normalization type for actions.
            Defaults to `norm_type`.
        clip_norm (bool): Whether to clip min_max/quantile normalized values
            to [-1, 1]. Defaults to False.
        normalize_states (bool): Whether to normalize states before optional
            padding/truncation. Defaults to True.
        preserve_input_dtype (bool): Keep normalization arithmetic in the
            input array dtype even when ``output_dtype`` is configured.
        output_dtype (str | None): Optional NumPy dtype used for normalization
            arithmetic and outputs. ``None`` preserves the legacy NumPy dtype
            promotion behavior. Defaults to None.
        state_key (str | None): The key in the data dictionary
            that contains the state information.
        action_key (str | None): The key in the data dictionary
            that contains the action information. If None, actions are skipped.
        valid_action_dim (int | None): Number of non-padding action dimensions
            used when constructing a per-dimension action mask.
        mark_all_action_steps_valid (bool): Replace the temporal action mask
            with a mask that enables ``valid_action_dim`` dimensions at every
            action step. Defaults to False.
    """

    def __init__(self,
                 state_key: Optional[str],
                 action_key: Optional[str],
                 action_dim: int = None,
                 state_dim: int = None,
                 norm_type: str = 'mean_std',
                 state_norm_type: str = None,
                 action_norm_type: str = None,
                 pad_value: float = 0.0,
                 action_norm_mask: List[bool] = None,
                 clip_norm: bool = False,
                 normalization_epsilon: float = 1e-6,
                 preserve_input_dtype: bool = False,
                 normalize_states: bool = True,
                 discrete_action_dims: List[int] = None,
                 discrete_state_dims: List[int] = None,
                 discrete_norm_type: str = 'min_max',
                 pad_invalid_action_delta_dims: bool = False,
                 delta_action_dim_mask: List[bool] = None,
                 action_pad_mask_key: str = 'action_masks',
                 valid_action_dim: int = None,
                 mark_all_action_steps_valid: bool = False,
                 output_dtype: Optional[str] = None,
                 *args,
                 **kwargs):
        self.state_key = state_key
        self.action_key = action_key
        self.norm_type = norm_type
        self.state_norm_type = state_norm_type or norm_type
        self.action_norm_type = action_norm_type or norm_type
        self.pad_value = pad_value
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.clip_norm = clip_norm
        self.normalization_epsilon = float(normalization_epsilon)
        self.preserve_input_dtype = bool(preserve_input_dtype)
        self.normalize_states = normalize_states
        self.output_dtype = (None if output_dtype is None else
                             np.dtype(output_dtype))
        if (self.output_dtype is not None
                and not np.issubdtype(self.output_dtype, np.floating)):
            raise ValueError(
                f'output_dtype must be a floating dtype, got {output_dtype!r}')
        if action_norm_mask is not None:
            if (action_dim is not None and len(action_norm_mask) > action_dim):
                raise ValueError(
                    'Action norm mask cannot be wider than the target action '
                    f'dimension {action_dim}, got {len(action_norm_mask)}.')
            self.action_norm_mask = action_norm_mask
        else:
            self.action_norm_mask = None
        self.discrete_action_dims = (
            list(discrete_action_dims) if discrete_action_dims else None)
        self.discrete_state_dims = (
            list(discrete_state_dims) if discrete_state_dims else None)
        self.discrete_norm_type = discrete_norm_type
        self.pad_invalid_action_delta_dims = pad_invalid_action_delta_dims
        self.action_pad_mask_key = action_pad_mask_key
        self.valid_action_dim = (
            int(valid_action_dim) if valid_action_dim is not None else None)
        self.mark_all_action_steps_valid = bool(mark_all_action_steps_valid)
        if self.mark_all_action_steps_valid:
            if self.action_dim is None or self.valid_action_dim is None:
                raise ValueError('`action_dim` and `valid_action_dim` are '
                                 'required when '
                                 '`mark_all_action_steps_valid=True`.')
            if not 0 < self.valid_action_dim <= self.action_dim:
                raise ValueError('`valid_action_dim` must be in '
                                 '(0, action_dim].')
        if delta_action_dim_mask is not None:
            assert len(delta_action_dim_mask) == action_dim, \
                f'Delta action dim mask must be of length {action_dim}'
            self.delta_action_dim_mask = np.asarray(
                delta_action_dim_mask, dtype=bool)
        else:
            self.delta_action_dim_mask = None

    def __call__(self, data: Dict) -> Dict:
        states = np.asarray(data['states'], dtype=np.float32)
        actions = None
        if self.action_key is not None and 'actions' in data:
            actions = np.asarray(data['actions'], dtype=np.float32)
            if (self.action_norm_mask is not None
                    and len(self.action_norm_mask) != actions.shape[-1]):
                raise ValueError(
                    'Action norm mask must match the unpadded action '
                    f'dimension {actions.shape[-1]}, got '
                    f'{len(self.action_norm_mask)}.')
            actions = self._zero_padded_delta_action_dims(data, actions)

        needs_state_stats = (
            self.normalize_states and self.state_norm_type != 'none')
        needs_action_stats = (
            actions is not None and self.action_norm_type != 'none')
        if needs_state_stats or needs_action_stats:
            assert 'stats' in data, "Input data must contain 'stats' key"

        if needs_state_stats:
            state_stats = data['stats'][self.state_key]
            states = self._normalize_mixed(
                states,
                state_stats,
                self.state_norm_type,
                discrete_dims=self.discrete_state_dims)
        if self.output_dtype is not None:
            states = np.asarray(states, dtype=self.output_dtype)
        data['states'] = states

        if actions is not None:
            if needs_action_stats:
                action_stats = data['stats'][self.action_key]
                actions = self._normalize_mixed(
                    actions,
                    action_stats,
                    self.action_norm_type,
                    discrete_dims=self.discrete_action_dims,
                    base_mask=self.action_norm_mask)
            if self.output_dtype is not None:
                actions = np.asarray(actions, dtype=self.output_dtype)
            data['actions'] = actions
        if self.state_dim is not None:
            data['states'] = self._pad_or_truncate_last_dim(
                states, self.state_dim)
        if self.action_dim is not None and actions is not None:
            data['actions'] = self._pad_or_truncate_last_dim(
                actions, self.action_dim)
        if self.output_dtype is not None:
            data['states'] = np.asarray(data['states']).astype(
                self.output_dtype, copy=False)
            if actions is not None:
                data['actions'] = np.asarray(data['actions']).astype(
                    self.output_dtype, copy=False)
        if actions is not None and self.mark_all_action_steps_valid:
            action_masks = np.zeros(
                np.asarray(data['actions']).shape, dtype=np.float32)
            action_masks[..., :self.valid_action_dim] = 1.0
            data[self.action_pad_mask_key] = action_masks
        return data

    def _zero_padded_delta_action_dims(self, data: Dict,
                                       actions: np.ndarray) -> np.ndarray:
        if (not self.pad_invalid_action_delta_dims
                or self.delta_action_dim_mask is None
                or self.action_pad_mask_key not in data):
            return actions
        action_valid = np.asarray(data[self.action_pad_mask_key]).astype(bool)
        if action_valid.ndim != 1:
            action_valid = action_valid.reshape(-1)
        if action_valid.shape[0] != actions.shape[0]:
            raise ValueError(
                f'{self.action_pad_mask_key} length {action_valid.shape[0]} '
                f'does not match actions length {actions.shape[0]}.')
        if self.delta_action_dim_mask.shape[0] != actions.shape[-1]:
            raise ValueError(
                f'Delta action dim mask length '
                f'{self.delta_action_dim_mask.shape[0]} does not match '
                f'action dim {actions.shape[-1]}.')
        invalid_delta = (
            ~action_valid)[:, None] & self.delta_action_dim_mask[None, :]
        if not invalid_delta.any():
            return actions
        actions = actions.copy()
        actions[invalid_delta] = 0.0
        return actions

    def _pad_or_truncate_last_dim(self, values: np.ndarray,
                                  target_dim: int) -> np.ndarray:
        current_dim = values.shape[-1]
        if current_dim >= target_dim:
            return values[..., :target_dim]
        padded_shape = (*values.shape[:-1], target_dim)
        padded = np.full(padded_shape, self.pad_value, dtype=values.dtype)
        padded[..., :current_dim] = values
        return padded

    def _normalize_mixed(self,
                         x,
                         stats: Dict,
                         norm_type: str,
                         discrete_dims: List[int] = None,
                         base_mask: List[bool] = None):
        cont = self._normalize_by_type(x, stats, norm_type, base_mask)
        if not discrete_dims:
            return cont
        feat_dim = x.shape[-1]
        assert all(0 <= d < feat_dim for d in discrete_dims), (
            f'discrete_dims {discrete_dims} out of range for feat_dim '
            f'{feat_dim}')
        disc_mask = np.zeros(feat_dim, dtype=bool)
        disc_mask[list(discrete_dims)] = True
        if base_mask is not None:
            disc_mask = disc_mask & np.asarray(base_mask, dtype=bool)
            if not disc_mask.any():
                return cont
        disc = self._normalize_by_type(x, stats, self.discrete_norm_type,
                                       disc_mask)
        return np.where(disc_mask, disc, cont)

    def _normalize_by_type(self,
                           x,
                           stats: Dict,
                           norm_type: str,
                           norm_mask: List[bool] = None):
        if self.output_dtype is not None and not self.preserve_input_dtype:
            x = np.asarray(x, dtype=self.output_dtype)
        if norm_type == 'none':
            return x
        if norm_type == 'quantile':
            return self._normalize_quantile(x, stats, norm_mask)
        if norm_type == 'min_max':
            return self._normalize_min_max(x, stats, norm_mask)
        return self._normalize(x, stats, norm_mask)

    def _normalize(self, x, stats: Dict, norm_mask: List[bool] = None):
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        mean = self._statistics_array(stats['mean'], x)
        std = self._statistics_array(stats['std'], x)
        epsilon = self._typed_epsilon(x)
        normalized = (x - mean) / (std + epsilon)
        return np.where(norm_mask, normalized, x)

    def _normalize_quantile(self,
                            x,
                            stats: torch.tensor,
                            norm_mask: List[bool] = None):
        assert stats['q01'] is not None
        assert stats['q99'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        low = self._statistics_array(stats['q01'], x)
        high = self._statistics_array(stats['q99'], x)
        epsilon = self._typed_epsilon(x)
        normalized = (x - low) / (high - low + epsilon) * 2.0 - 1.0
        if self.clip_norm:
            normalized = np.clip(normalized, -1, 1)
        return np.where(norm_mask, normalized, x)

    def _normalize_min_max(self, x, stats: Dict, norm_mask: List[bool] = None):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        if norm_mask is None:
            norm_mask = [True] * x.shape[-1]
        low = self._statistics_array(stats['min'], x)
        high = self._statistics_array(stats['max'], x)
        epsilon = self._typed_epsilon(x)
        normalized = (x - low) / (high - low + epsilon) * 2.0 - 1.0
        if self.clip_norm:
            normalized = np.clip(normalized, -1, 1)
        return np.where(norm_mask, normalized, x)

    def _statistics_array(self, values, reference: np.ndarray) -> np.ndarray:
        dtype = (reference.dtype if self.preserve_input_dtype else
                 self.output_dtype)
        return np.asarray(values, dtype=dtype)

    def _typed_epsilon(self, reference: np.ndarray):
        dtype = (reference.dtype if self.preserve_input_dtype else
                 self.output_dtype)
        if dtype is None:
            return self.normalization_epsilon
        return np.asarray(self.normalization_epsilon, dtype=dtype)


@TRANSFORMS.register_module()
class SinCosKeys:
    """Apply sin/cos encoding to vector-valued keys.

    This is useful for source datasets that encode each scalar state dimension
    as ``[sin(x), cos(x)]`` before concatenating the final proprio vector.
    ``expand_axis`` can add a preserved singleton/token dimension after the
    encoding without requiring a model-specific reshape transform.
    """

    def __init__(self,
                 keys: List[str],
                 target_dims=None,
                 interleave: bool = True,
                 backend: str = 'numpy',
                 dtype: str = 'float32',
                 pad_value: float = 0.0,
                 expand_axis: Optional[int] = None) -> None:
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.target_dims = target_dims
        self.interleave = interleave
        self.backend = str(backend).lower()
        if self.backend not in ('numpy', 'torch'):
            raise ValueError("SinCosKeys backend must be 'numpy' or 'torch'.")
        self.dtype = np.dtype(dtype)
        self.pad_value = pad_value
        self.expand_axis = expand_axis

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            if key not in data:
                raise KeyError(f"Key '{key}' not found for SinCosKeys.")
            data[key] = self._encode_value(key, data[key])
        return data

    def _target_dim(self, key: str):
        if self.target_dims is None:
            return None
        if isinstance(self.target_dims, dict):
            return self.target_dims.get(key)
        return int(self.target_dims)

    def _encode_value(self, key: str, value):
        is_tensor = torch.is_tensor(value)
        device = value.device if is_tensor else None
        tensor_dtype = value.dtype if is_tensor else None
        arr = value.detach().cpu().numpy() if is_tensor else np.asarray(value)
        if arr.ndim == 0:
            raise ValueError(f"SinCosKeys expects vector input for '{key}'.")
        arr = arr.astype(np.float32, copy=False)

        if self.backend == 'torch':
            source = torch.from_numpy(arr)
            sin_value = torch.sin(source).numpy()
            cos_value = torch.cos(source).numpy()
        else:
            sin_value = np.sin(arr)
            cos_value = np.cos(arr)
        if self.interleave:
            encoded = np.stack([sin_value, cos_value],
                               axis=-1).reshape(*arr.shape[:-1],
                                                arr.shape[-1] * 2)
        else:
            encoded = np.concatenate([sin_value, cos_value], axis=-1)

        target_dim = self._target_dim(key)
        if target_dim is not None:
            encoded = self._pad_or_truncate_last_dim(encoded, int(target_dim))
        if self.expand_axis is not None:
            encoded = np.expand_dims(encoded, axis=self.expand_axis)

        if is_tensor:
            dtype = tensor_dtype if tensor_dtype.is_floating_point else None
            return torch.from_numpy(encoded).to(device=device, dtype=dtype)
        return encoded.astype(self.dtype, copy=False)

    def _pad_or_truncate_last_dim(self, values: np.ndarray,
                                  target_dim: int) -> np.ndarray:
        current_dim = values.shape[-1]
        if current_dim >= target_dim:
            return values[..., :target_dim]
        padded_shape = (*values.shape[:-1], target_dim)
        padded = np.full(padded_shape, self.pad_value, dtype=values.dtype)
        padded[..., :current_dim] = values
        return padded


@TRANSFORMS.register_module()
class LiberoProprioFromInputs:
    """Build and normalize Libero proprio state from inputs.

    Reads `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`,
    converts quaternion to axis-angle, concatenates into a
    state vector, and normalizes using `norm_stats[task_suite_name +
    '_no_noops']['proprio']`.

    Expects `task_suite_name` to be present in the input dict.

    Args:
        norm_stats (str | Dict): Path to JSON or dict of normalization stats.
        norm_type (str): Type of normalization to use.
            Options: 'mean_std', 'quantile', or 'min_max'.
            Defaults to 'quantile'.
        pos_key (str): Key for end-effector position.
        quat_key (str): Key for end-effector quaternion.
        gripper_key (str): Key for gripper position.
        out_key (str): Output key for normalized state (default 'states').
    """

    def __init__(self,
                 norm_type: str = 'quantile',
                 state_dim: int = None,
                 pos_key: str = 'robot0_eef_pos',
                 quat_key: str = 'robot0_eef_quat',
                 gripper_key: str = 'robot0_gripper_qpos',
                 stat_key: str = 'proprio',
                 out_key: str = 'states',
                 stat_field: str = 'state',
                 stat_subkey: str = 'default',
                 prefix: str = 'global',
                 linear_mode: str = 'min/max',
                 clamp: float = 5.0) -> None:
        self.norm_type = norm_type
        self.state_dim = state_dim
        self.pos_key = pos_key
        self.quat_key = quat_key
        self.gripper_key = gripper_key
        self.out_key = out_key
        self.stat_key = stat_key
        self.stat_field = stat_field
        self.stat_subkey = stat_subkey
        self.prefix = prefix
        self.linear_mode = linear_mode
        self.clamp = float(clamp)

    def __call__(self, data: Dict) -> Dict:
        assert self.pos_key in data and self.quat_key in \
            data and self.gripper_key in data, \
            f'Missing proprio keys in data: {self.pos_key}, {self.quat_key}, {self.gripper_key}'  # noqa: E501
        robot0_eef_pos = np.asarray(data[self.pos_key])
        robot0_eef_quat = np.asarray(data[self.quat_key])
        robot0_gripper_qpos = np.asarray(data[self.gripper_key])

        state = np.concatenate((
            robot0_eef_pos,
            quat2axisangle(robot0_eef_quat),
            robot0_gripper_qpos,
        ))

        if self.norm_type == 'linear':
            raw = data['norm_stats'][self.stat_field][self.stat_subkey]
            lin_stats = _select_prefixed_stats(raw, self.prefix)
            scale, offset = _linear_norm_scale_offset(lin_stats,
                                                      self.linear_mode)
            state_t = torch.as_tensor(state, dtype=torch.float32)
            state = torch.clamp(state_t * scale + offset, -self.clamp,
                                self.clamp).numpy()
        elif self.norm_type == 'none':
            state = state.astype(np.float32, copy=False)
        else:
            stats = data['norm_stats'][self.stat_key]
            if self.norm_type == 'quantile':
                state = self._normalize_quantile(state, stats)
            elif self.norm_type == 'min_max':
                state = self._normalize_min_max(state, stats)
            else:  # norm_type == 'mean_std'
                state = self._normalize(state, stats)

        out = dict(data)
        if self.state_dim is not None:
            out[self.out_key] = np.zeros((self.state_dim), dtype=state.dtype)
            out[self.out_key][:state.shape[0]] = state
        else:
            out[self.out_key] = state
        return out

    def _normalize(self, normalized_states: np.ndarray, stats: Dict):
        assert 'mean' in stats and stats['mean'] is not None
        assert 'std' in stats and stats['std'] is not None
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(stats['mean'], dtype=bool)
        # Keep eval-time mean/std normalization consistent with training:
        # (x - mean) / (std + eps), without clipping.
        states = np.where(
            mask,
            (normalized_states - np.array(stats['mean'])) /
            (np.array(stats['std']) + 1e-6),
            normalized_states,
        )
        return states

    def _normalize_quantile(self, normalized_states: np.ndarray, stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None
        state_high = np.array(stats['q99'])
        state_low = np.array(stats['q01'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states

    def _normalize_min_max(self, normalized_states: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        state_high = np.array(stats['max'])
        state_low = np.array(stats['min'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states


def _select_prefixed_stats(raw: Dict, prefix: Optional[str]) -> Dict:
    """Strip a ``{prefix}_`` prefix from flat ``dataset_stats`` keys.

    Some statistics files store multiple families such as ``global_*`` and
    ``stepwise_*`` per field. This selects one family and drops the prefix.
    With ``prefix=None`` the stats are returned unchanged.
    """
    if not prefix:
        return dict(raw)
    token = prefix + '_'
    return {k[len(token):]: v for k, v in raw.items() if k.startswith(token)}


def _linear_norm_scale_offset(stats: Dict, mode: str):
    """Return ``(scale, offset)`` for ``clamp(x * scale + offset)``."""
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0

    if mode == 'z-score':
        input_mean = torch.as_tensor(stats['mean'], dtype=torch.float32)
        input_std = torch.as_tensor(stats['std'], dtype=torch.float32)
        scale = 1.0 / (input_std + std_reg)
        offset = -input_mean / (input_std + std_reg)
        return scale, offset

    if mode == 'min/max':
        input_min = torch.as_tensor(stats['min'], dtype=torch.float32)
        input_max = torch.as_tensor(stats['max'], dtype=torch.float32)
    elif mode == 'q01/q99':
        input_min = torch.as_tensor(stats['q01'], dtype=torch.float32)
        input_max = torch.as_tensor(stats['q99'], dtype=torch.float32)
    else:
        lo, hi = map(float, mode.split('/'))
        ref = torch.as_tensor(stats['min'], dtype=torch.float32)
        input_min = torch.full_like(ref, lo)
        input_max = torch.full_like(ref, hi)

    input_range = (input_max - input_min).clone()
    ignore_dim = input_range < range_tol
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return scale, offset
