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

from typing import Dict, Optional

import numpy as np
import torch

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class PrepareDiT4DiTInputs:
    """Finalize one sample for the canonical DiT4DiT batch contract.

    Existing transforms remain responsible for prompt tokenization,
    multi-view composition, state encoding, action normalization, and action
    padding. This model-specific finalizer converts the explicit per-sample
    video layout from ``TCHW`` to ``CTHW``, adds the single state-token axis,
    and verifies that state/action fields already match the configured model.

    After collation, the resulting tensors have shapes ``[B, C, T, H, W]``,
    ``[B, 1, state_dim]``, and ``[B, action_horizon, action_dim]``.
    """

    def __init__(
        self,
        image_key: str = 'images',
        input_image_layout: str = 'tchw',
        state_key: str = 'states',
        state_dim: int = 0,
        action_key: str = 'actions',
        action_mask_key: str = 'action_masks',
        action_horizon: int = 0,
        action_dim: int = 0,
        valid_action_dim: Optional[int] = None,
        mark_all_action_steps_valid: bool = False,
        require_actions: bool = True,
    ) -> None:
        if input_image_layout not in ('tchw', 'cthw'):
            raise ValueError(
                "`input_image_layout` must be either 'tchw' or 'cthw'.")
        if state_dim < 0 or action_horizon < 0 or action_dim < 0:
            raise ValueError('DiT4DiT dimensions must be non-negative.')
        if require_actions and (action_horizon <= 0 or action_dim <= 0):
            raise ValueError('Positive `action_horizon` and `action_dim` are '
                             'required when `require_actions=True`.')

        self.image_key = image_key
        self.input_image_layout = input_image_layout
        self.state_key = state_key
        self.state_dim = int(state_dim)
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.valid_action_dim = (
            int(valid_action_dim) if valid_action_dim is not None else None)
        if (self.valid_action_dim is not None
                and not 0 < self.valid_action_dim <= self.action_dim):
            raise ValueError('`valid_action_dim` must be in '
                             '(0, action_dim].')
        self.mark_all_action_steps_valid = bool(mark_all_action_steps_valid)
        if self.mark_all_action_steps_valid and self.valid_action_dim is None:
            raise ValueError('`valid_action_dim` is required when '
                             '`mark_all_action_steps_valid=True`.')
        self.require_actions = bool(require_actions)

    @staticmethod
    def _shape(value) -> tuple[int, ...]:
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            raise TypeError('PrepareDiT4DiTInputs expects numpy arrays or '
                            f'torch tensors, got {type(value)!r}.')
        return tuple(value.shape)

    def _prepare_images(self, images):
        shape = self._shape(images)
        if len(shape) != 4:
            raise ValueError(
                f"'{self.image_key}' must be a per-sample 4D video, got "
                f'{shape}.')

        channel_axis = 1 if self.input_image_layout == 'tchw' else 0
        if shape[channel_axis] != 3:
            raise ValueError(
                f"'{self.image_key}' with layout "
                f'{self.input_image_layout.upper()} must have exactly three '
                f'channels, got {shape}.')

        if self.input_image_layout == 'cthw':
            return images
        if torch.is_tensor(images):
            return images.permute(1, 0, 2, 3).contiguous()
        return np.ascontiguousarray(images.transpose(1, 0, 2, 3))

    def _prepare_states(self, states):
        shape = self._shape(states)
        if len(shape) == 1:
            states = states.unsqueeze(0) if torch.is_tensor(
                states) else np.expand_dims(
                    states, axis=0)
            shape = tuple(states.shape)
        if len(shape) != 2 or shape != (1, self.state_dim):
            raise ValueError(
                f"'{self.state_key}' must have shape [{self.state_dim}] or "
                f'[1, {self.state_dim}], got {shape}.')
        return states

    def _validate_actions(self, inputs: Dict) -> None:
        has_actions = self.action_key in inputs
        if not has_actions:
            if self.require_actions:
                raise KeyError(f"Action key '{self.action_key}' is missing.")
            if self.action_mask_key in inputs:
                raise ValueError(
                    f"'{self.action_mask_key}' is present without "
                    f"'{self.action_key}'.")
            return

        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError('Positive `action_horizon` and `action_dim` are '
                             'required when actions are present.')
        action_shape = self._shape(inputs[self.action_key])
        expected_shape = (self.action_horizon, self.action_dim)
        if action_shape != expected_shape:
            raise ValueError(
                f"'{self.action_key}' must have shape {expected_shape}, got "
                f'{action_shape}. Apply action windowing/padding before '
                'PrepareDiT4DiTInputs.')
        if self.action_mask_key not in inputs:
            raise KeyError(
                f"Action mask key '{self.action_mask_key}' is missing.")
        mask_shape = self._shape(inputs[self.action_mask_key])
        if mask_shape != expected_shape:
            raise ValueError(
                f"'{self.action_mask_key}' must have shape {expected_shape}, "
                f'got {mask_shape}.')

        # The source DiT4DiT loader repeats the last absolute action at an
        # episode boundary and keeps every temporal row valid. Its mask only
        # distinguishes real action dimensions from the padded model width.
        # Keep this opt-in so other pipelines retain their temporal masks.
        if self.mark_all_action_steps_valid:
            action_masks = inputs[self.action_mask_key]
            if torch.is_tensor(action_masks):
                source_mask = torch.zeros_like(action_masks)
            else:
                source_mask = np.zeros_like(action_masks)
            source_mask[..., :self.valid_action_dim] = 1
            inputs[self.action_mask_key] = source_mask

    def __call__(self, inputs: Dict) -> Dict:
        if self.image_key not in inputs:
            raise KeyError(f"Image key '{self.image_key}' is missing.")
        inputs[self.image_key] = self._prepare_images(inputs[self.image_key])

        if self.state_dim > 0:
            if self.state_key not in inputs:
                raise KeyError(f"State key '{self.state_key}' is missing.")
            inputs[self.state_key] = self._prepare_states(
                inputs[self.state_key])
        elif self.state_key in inputs:
            raise ValueError(
                f"'{self.state_key}' is present but `state_dim` is zero.")

        self._validate_actions(inputs)
        return inputs


__all__ = ['PrepareDiT4DiTInputs']
