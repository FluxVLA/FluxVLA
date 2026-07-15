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
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _or_policy
from transformers import PretrainedConfig

from fluxvla.engines import VLAS
from .base_vla import BaseVLA


@VLAS.register_module()
class DiT4DiTVLA(BaseVLA):
    """FluxVLA wrapper for DiT4DiT-style Cosmos backbone + ActionDiT head.

    The wrapper follows FluxVLA runner contracts:
    ``forward(**batch) -> {"loss": loss, ...}`` for training and
    ``predict_action(**batch) -> Tensor[B, T, action_dim]`` for evaluation.
    It primarily consumes Cosmos-tokenized ``lang_tokens`` and ``lang_masks``
    from the transform pipeline. Raw prompt text remains a compatibility path
    for older callers.
    """

    def __init__(
        self,
        vlm_backbone: Dict = None,
        vla_head: Dict = None,
        repeated_diffusion_steps: int = 1,
        image_layout: str = 'auto',
        multiview_strategy: str = 'tile',
        pretrained_name_or_path: str = None,
        name_mapping: Dict = None,
        strict_mapping: bool = False,
        freeze_vlm_backbone: bool = True,
        *args,
        **kwargs,
    ) -> None:
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

        if multiview_strategy not in ('tile', 'first', 'temporal'):
            raise ValueError('`multiview_strategy` must be one of '
                             "'tile', 'first', or 'temporal'.")
        if image_layout not in ('auto', 'bcthw', 'btchw', 'bvchw'):
            raise ValueError(
                "`image_layout` must be one of 'auto', 'bcthw', 'btchw', "
                "or 'bvchw'.")
        self.repeated_diffusion_steps = int(repeated_diffusion_steps)
        self.image_layout = image_layout
        self.multiview_strategy = multiview_strategy
        self.all_module_keys = ['vlm_backbone', 'vla_head']

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
    def _resolve_prompts(
        prompt: Optional[Union[str, Sequence[str]]],
        batch_size: int,
        task_description: Optional[Union[str, Sequence[str]]] = None,
        lang: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> list[str]:
        prompts = prompt
        prompts = prompts if prompts is not None else task_description
        prompts = prompts if prompts is not None else lang
        prompts = prompts if prompts is not None else kwargs.get('instruction')
        prompts = prompts if prompts is not None else kwargs.get(
            'instructions')
        prompts = prompts if prompts is not None else kwargs.get('text')
        if prompts is None:
            raise ValueError(
                'DiT4DiTVLA requires Cosmos `lang_tokens` or a raw prompt.')
        if isinstance(prompts, str):
            prompts = [prompts] * batch_size
        else:
            prompts = list(prompts)
        if len(prompts) != batch_size:
            raise ValueError(
                f'Prompt batch size {len(prompts)} does not match image '
                f'batch size {batch_size}.')
        return [str(item) for item in prompts]

    def _prepare_images(self, images):
        if not torch.is_tensor(images):
            return images
        if images.ndim == 4:
            return images
        if images.ndim != 5:
            raise ValueError(
                f'Expected images with 4D/5D shape, got {tuple(images.shape)}.'
            )

        if self.image_layout == 'bcthw':
            return images[:, :3]
        if self.image_layout == 'btchw':
            return images[:, :, :3]
        if self.image_layout == 'bvchw':
            return self._prepare_multiview_images(images)

        # Already [B, C, T, H, W], e.g. after PrepareVideo.
        if images.shape[1] in (1, 3, 4):
            return images[:, :3]

        # Common FluxVLA multi-view layout: [B, V, C, H, W].
        if images.shape[2] in (1, 3, 4):
            return self._prepare_multiview_images(images)

        raise ValueError('Could not infer channel dimension from images shape '
                         f'{tuple(images.shape)}.')

    def _prepare_multiview_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] not in (1, 3, 4):
            raise ValueError('`image_layout="bvchw"` expects images shaped '
                             '[B, V, C, H, W], '
                             f'got {tuple(images.shape)}.')
        images = images[:, :, :3]
        if self.multiview_strategy == 'temporal':
            return images
        if self.multiview_strategy == 'first':
            return images[:, 0]
        bsz, num_views, channels, height, width = images.shape
        return images.permute(0, 2, 1, 3, 4).reshape(bsz, channels, 1,
                                                     num_views * height, width)

    def _match_last_dim(self, tensor: torch.Tensor, target_dim: int,
                        pad_value: float) -> torch.Tensor:
        if tensor.shape[-1] == target_dim:
            return tensor
        if tensor.shape[-1] < target_dim:
            return F.pad(
                tensor, (0, target_dim - tensor.shape[-1]), value=pad_value)
        return tensor[..., :target_dim]

    def _prepare_actions_and_masks(
        self,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actions is None:
            raise ValueError('DiT4DiTVLA.forward requires `actions`.')

        actions = actions[..., -self.vla_head.action_horizon:, :]
        actions = self._match_last_dim(actions, self.vla_head.action_dim, 0.0)

        if action_masks is None:
            action_masks = torch.ones_like(actions, dtype=torch.bool)
        else:
            action_masks = action_masks[..., -actions.shape[1]:]
            if action_masks.ndim == 2:
                action_masks = action_masks.unsqueeze(-1).expand_as(actions)
            action_masks = self._match_last_dim(action_masks,
                                                self.vla_head.action_dim, 0.0)
        return actions, action_masks.to(dtype=actions.dtype)

    def _prepare_states(
        self,
        states: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if getattr(self.vla_head, 'state_dim', 0) <= 0:
            return None
        if states is None:
            raise ValueError(
                'DiT4DiTVLA was configured with state_dim > 0, but `states` '
                'is None.')
        states = states.to(device=device, dtype=dtype)
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.ndim != 3:
            raise ValueError(f'Expected states shape [B, D] or [B, N, D], got '
                             f'{tuple(states.shape)}.')
        if states.shape[0] != batch_size:
            raise ValueError(
                f'State batch {states.shape[0]} != image batch {batch_size}.')
        target_dim = self.vla_head.state_dim
        return self._match_last_dim(states, target_dim, 0.0)

    def _encode_backbone(
        self,
        images,
        prompts: Optional[Sequence[str]] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, object]:
        backbone_outputs = self.vlm_backbone(
            images=images,
            prompts=prompts,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden_states = getattr(backbone_outputs, 'hidden_states', None)
        if not hidden_states:
            raise RuntimeError(
                'Cosmos backbone did not return `hidden_states`.')
        return hidden_states[-1], backbone_outputs

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
        prompt: Optional[Union[str, Sequence[str]]] = None,
        task_description: Optional[Union[str, Sequence[str]]] = None,
        lang: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> Dict:
        if images is None:
            raise ValueError('DiT4DiTVLA.forward requires `images`.')
        images = self._prepare_images(images)
        batch_size = images.shape[0] if torch.is_tensor(images) else len(
            images)
        prompts = None
        if lang_tokens is None:
            prompts = self._resolve_prompts(
                prompt,
                batch_size,
                task_description=task_description,
                lang=lang,
                **kwargs,
            )

        last_hidden, backbone_outputs = self._encode_backbone(
            images=images,
            prompts=prompts,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        if actions is None:
            raise ValueError('DiT4DiTVLA.forward requires `actions`.')
        actions, action_masks = self._prepare_actions_and_masks(
            actions.to(device=last_hidden.device, dtype=last_hidden.dtype),
            None if action_masks is None else action_masks.to(
                device=last_hidden.device),
        )
        states = self._prepare_states(
            states,
            batch_size=batch_size,
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )

        repeat = max(1, self.repeated_diffusion_steps)
        if repeat > 1:
            last_hidden = last_hidden.repeat(repeat, 1, 1)
            actions = actions.repeat(repeat, 1, 1)
            action_masks = action_masks.repeat(repeat, 1, 1)
            if states is not None:
                states = states.repeat(repeat, 1, 1)

        with self._action_head_autocast(last_hidden):
            action_loss = self.vla_head(
                last_hidden,
                actions=actions,
                action_mask=action_masks,
                state=states,
            )
        loss = action_loss
        output = dict(loss=loss, action_loss=action_loss)
        future_video_loss = getattr(backbone_outputs, 'future_video_loss',
                                    None)
        if future_video_loss is not None:
            output['future_video_loss'] = future_video_loss
            output['loss'] = output['loss'] + future_video_loss
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        images: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, Sequence[str]]] = None,
        task_description: Optional[Union[str, Sequence[str]]] = None,
        lang: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        images = self._prepare_images(images)
        batch_size = images.shape[0] if torch.is_tensor(images) else len(
            images)
        prompts = None
        if lang_tokens is None:
            prompts = self._resolve_prompts(
                prompt,
                batch_size,
                task_description=task_description,
                lang=lang,
                **kwargs,
            )

        last_hidden, _ = self._encode_backbone(
            images=images,
            prompts=prompts,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        states = self._prepare_states(
            states,
            batch_size=batch_size,
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )
        with self._action_head_autocast(last_hidden):
            return self.vla_head.predict_action(last_hidden, state=states)

    def get_fsdp_wrapping_policy(self) -> Callable:
        wrapping_policies = []
        if (self.vlm_backbone is not None
                and hasattr(self.vlm_backbone, 'get_fsdp_wrapping_policy')):
            policy = self.vlm_backbone.get_fsdp_wrapping_policy()
            if policy is not None:
                wrapping_policies.append(policy)
        if self.vla_head is not None and hasattr(self.vla_head,
                                                 'get_fsdp_wrapping_policy'):
            policy = self.vla_head.get_fsdp_wrapping_policy()
            if policy is not None:
                wrapping_policies.append(policy)
        if not wrapping_policies:
            return None
        if len(wrapping_policies) == 1:
            return wrapping_policies[0]
        return partial(_or_policy, policies=wrapping_policies)


__all__ = ['DiT4DiTVLA']
