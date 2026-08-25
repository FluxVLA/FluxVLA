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
#
# Adapted from Mondo-Robotics/DiT4DiT:
# DiT4DiT/model/framework/DiT4DiT.py

from __future__ import annotations
from contextlib import nullcontext
from typing import Callable, Dict, Optional

import torch
from transformers import PretrainedConfig

from fluxvla.engines import VLAS
from fluxvla.engines.utils.fsdp_wrapping import build_module_wrap_policy
from fluxvla.models.backbones.vlms.outputs import (
    VLMBackboneOutput, normalize_vlm_backbone_output)
from .base_vla import BaseVLA
from .continuous_action_utils import (add_auxiliary_losses,
                                      normalize_action_head_output,
                                      repeat_batch_tensor)


@VLAS.register_module()
class DiT4DiTVLA(BaseVLA):
    """FluxVLA wrapper for DiT4DiT-style Cosmos backbone + ActionDiT head.

    The wrapper follows FluxVLA runner contracts:
    ``forward(**batch) -> {"loss": loss, ...}`` for training and
    ``predict_action(**batch) -> Tensor[B, T, action_dim]`` for evaluation.
    Dataset transforms provide canonical ``BCTHW`` videos, one state token,
    exact action tensors/masks, and Cosmos-tokenized language. The wrapper
    validates that contract without inferring layouts or mutating dimensions.
    """

    def __init__(
        self,
        vlm_backbone: Dict = None,
        vla_head: Dict = None,
        repeated_diffusion_steps: int = 1,
        auxiliary_loss_weights: Optional[Dict[str, float]] = None,
        pretrained_name_or_path: str = None,
        name_mapping: Dict = None,
        strict_mapping: bool = False,
        freeze_vlm_backbone: bool = True,
        init_empty_weights: bool = False,
        *args,
        **kwargs,
    ) -> None:
        if init_empty_weights:
            from accelerate import init_empty_weights as empty_weights_context
            build_context = empty_weights_context()
        else:
            build_context = nullcontext()
        with build_context:
            super().__init__(
                vlm_backbone=vlm_backbone,
                vla_head=vla_head,
                pretrained_name_or_path=pretrained_name_or_path,
                name_mapping=name_mapping,
                strict_mapping=strict_mapping,
                freeze_vlm_backbone=freeze_vlm_backbone,
            )
        if self.vlm_backbone is None:
            raise ValueError('DiT4DiTVLA requires `vlm_backbone`.')
        if self.vla_head is None:
            raise ValueError('DiT4DiTVLA requires `vla_head`.')

        self.repeated_diffusion_steps = int(repeated_diffusion_steps)
        self.auxiliary_loss_weights = dict(auxiliary_loss_weights or {})
        self.all_module_keys = ['vlm_backbone', 'vla_head']

    def load_state_dict(self, state_dict, strict: bool = True, assign=False):
        """Materialize architecture-only inference models from checkpoints."""
        has_meta_parameters = any(param.is_meta for param in self.parameters())
        if has_meta_parameters:
            target_state = super().state_dict()
            original_state = state_dict
            state_dict = state_dict.copy()
            if hasattr(original_state, '_metadata'):
                state_dict._metadata = original_state._metadata
            for key, value in state_dict.items():
                target = target_state.get(key)
                if (torch.is_tensor(value) and torch.is_tensor(target)
                        and value.dtype != target.dtype):
                    state_dict[key] = value.to(dtype=target.dtype)
        incompatible = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign or has_meta_parameters,
        )
        if has_meta_parameters:
            tie_weights = getattr(self.vlm_backbone.text_encoder,
                                  'tie_weights', None)
            if callable(tie_weights):
                tie_weights()
        return incompatible

    @property
    def config(self):
        cfg = PretrainedConfig()
        cfg.is_encoder_decoder = False
        return cfg

    def freeze_backbones(self) -> None:
        if hasattr(self.vlm_backbone, 'trainable'):
            self.vlm_backbone.trainable = not self.freeze_vlm_backbone
        super().freeze_backbones()
        if (not self.freeze_vlm_backbone and hasattr(
                self.vlm_backbone, 'freeze_configured_submodules')):
            self.vlm_backbone.freeze_configured_submodules()

    @staticmethod
    def _require_tensor(name: str,
                        value: Optional[torch.Tensor]) -> torch.Tensor:
        if value is None:
            raise ValueError(f'DiT4DiTVLA requires `{name}`.')
        if not torch.is_tensor(value):
            raise TypeError(
                f'`{name}` must be a torch.Tensor, got {type(value)!r}.')
        return value

    def _validate_images(self, images: Optional[torch.Tensor]) -> torch.Tensor:
        images = self._require_tensor('images', images)
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError('DiT4DiTVLA expects canonical images shaped '
                             '[B, 3, T, H, W], got '
                             f'{tuple(images.shape)}.')
        if not images.dtype.is_floating_point:
            raise TypeError('DiT4DiTVLA expects floating-point images from '
                            f'the dataset transforms, got {images.dtype}.')
        return images

    def _validate_language(
        self,
        lang_tokens: Optional[torch.Tensor],
        lang_masks: Optional[torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lang_tokens = self._require_tensor('lang_tokens', lang_tokens)
        lang_masks = self._require_tensor('lang_masks', lang_masks)
        if lang_tokens.ndim != 2 or lang_tokens.shape[0] != batch_size:
            raise ValueError(
                '`lang_tokens` must have shape [B, L] matching '
                f'the image batch, got {tuple(lang_tokens.shape)} '
                f'for B={batch_size}.')
        if lang_masks.shape != lang_tokens.shape:
            raise ValueError('`lang_masks` must match `lang_tokens`, got '
                             f'{tuple(lang_masks.shape)} and '
                             f'{tuple(lang_tokens.shape)}.')
        return lang_tokens, lang_masks

    def _validate_states(
        self,
        states: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        state_dim = int(getattr(self.vla_head, 'state_dim', 0))
        if state_dim <= 0:
            if states is not None:
                raise ValueError('`states` was provided, but the DiT4DiT '
                                 'action head has state_dim=0.')
            return None
        states = self._require_tensor('states', states)
        expected_shape = (batch_size, 1, state_dim)
        if tuple(states.shape) != expected_shape:
            raise ValueError('DiT4DiTVLA expects canonical states shaped '
                             f'{expected_shape}, got {tuple(states.shape)}.')
        return states

    def _validate_actions(
        self,
        actions: Optional[torch.Tensor],
        action_masks: Optional[torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self._require_tensor('actions', actions)
        action_masks = self._require_tensor('action_masks', action_masks)
        expected_shape = (
            batch_size,
            int(self.vla_head.action_horizon),
            int(self.vla_head.action_dim),
        )
        if tuple(actions.shape) != expected_shape:
            raise ValueError('DiT4DiTVLA expects canonical actions shaped '
                             f'{expected_shape}, got {tuple(actions.shape)}.')
        if tuple(action_masks.shape) != expected_shape:
            raise ValueError('DiT4DiTVLA expects canonical action_masks '
                             f'shaped {expected_shape}, got '
                             f'{tuple(action_masks.shape)}.')
        return actions, action_masks

    def _encode_backbone(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> VLMBackboneOutput:
        raw_outputs = self.vlm_backbone(
            images=images,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return normalize_vlm_backbone_output(raw_outputs)

    def _action_head_autocast(self, reference: torch.Tensor):
        if torch.is_tensor(reference) and reference.device.type == 'cuda':
            return torch.autocast('cuda', dtype=torch.float32)
        return nullcontext()

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        del kwargs
        images = self._validate_images(images)
        batch_size = images.shape[0]
        lang_tokens, lang_masks = self._validate_language(
            lang_tokens, lang_masks, batch_size)
        actions, action_masks = self._validate_actions(actions, action_masks,
                                                       batch_size)
        states = self._validate_states(states, batch_size)

        backbone_outputs = self._encode_backbone(
            images=images,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        last_hidden = backbone_outputs.last_hidden_state
        attention_mask = backbone_outputs.attention_mask
        actions = actions.to(
            device=last_hidden.device, dtype=last_hidden.dtype)
        action_masks = action_masks.to(
            device=last_hidden.device, dtype=last_hidden.dtype)
        if states is not None:
            states = states.to(
                device=last_hidden.device, dtype=last_hidden.dtype)

        repeat = max(1, self.repeated_diffusion_steps)
        if repeat > 1:
            last_hidden = last_hidden.repeat(repeat, 1, 1)
            actions = actions.repeat(repeat, 1, 1)
            action_masks = action_masks.repeat(repeat, 1, 1)
            if states is not None:
                states = states.repeat(repeat, 1, 1)
            attention_mask = repeat_batch_tensor(attention_mask, repeat,
                                                 batch_size)

        with self._action_head_autocast(last_hidden):
            head_output = self.vla_head(
                input_features=last_hidden,
                actions=actions,
                action_masks=action_masks,
                states=states,
                attention_mask=attention_mask,
            )
        output = normalize_action_head_output(head_output)
        return add_auxiliary_losses(
            output,
            backbone_outputs.auxiliary_losses,
            self.auxiliary_loss_weights,
        )

    @torch.inference_mode()
    def predict_action(
        self,
        images: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        images = self._validate_images(images)
        batch_size = images.shape[0]
        lang_tokens, lang_masks = self._validate_language(
            lang_tokens, lang_masks, batch_size)
        states = self._validate_states(states, batch_size)

        backbone_outputs = self._encode_backbone(
            images=images,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        last_hidden = backbone_outputs.last_hidden_state
        if states is not None:
            states = states.to(
                device=last_hidden.device, dtype=last_hidden.dtype)
        with self._action_head_autocast(last_hidden):
            actions = self.vla_head.predict_action(
                input_features=last_hidden,
                states=states,
                attention_mask=backbone_outputs.attention_mask,
            )
        if not torch.is_tensor(actions):
            raise TypeError('DiT4DiTActionHead.predict_action must return a '
                            f'torch.Tensor, got {type(actions)!r}.')
        output_dim = (
            getattr(self.vla_head, 'ori_action_dim', None)
            or self.vla_head.action_dim)
        expected_shape = (
            batch_size,
            int(self.vla_head.action_horizon),
            int(output_dim),
        )
        if tuple(actions.shape) != expected_shape:
            raise ValueError('DiT4DiTActionHead returned actions shaped '
                             f'{tuple(actions.shape)}; expected '
                             f'{expected_shape}.')
        return actions

    def get_fsdp_wrapping_policy(self) -> Callable:
        module_classes = {
            self.vlm_backbone.transformer_layer_cls,
            self.vla_head.transformer_layer_cls,
        }
        module_classes.discard(torch.nn.Module)
        if not module_classes:
            raise ValueError(
                'DiT4DiTVLA could not resolve any FSDP wrapper classes.')
        return build_module_wrap_policy(module_classes)

    def get_fsdp_ignored_modules(self) -> list[torch.nn.Module]:
        """Keep configured frozen Cosmos modules resident in BF16.

        The source DeepSpeed ZeRO-2 job does not shard the frozen text encoder
        or VAE. Exposing them to the runner also prevents the generic FSDP
        policy from recursively wrapping their internal blocks.
        """
        getter = getattr(self.vlm_backbone, 'get_fsdp_ignored_modules', None)
        return list(getter()) if callable(getter) else []


__all__ = ['DiT4DiTVLA']
