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

from __future__ import annotations
import copy
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from fluxvla.engines import VLAS, initialize_overwatch
from fluxvla.engines.utils.fsdp_wrapping import build_combined_wrap_policy
from fluxvla.models.projectors.linear_projector import \
    LinearProjector  # noqa: F401
from fluxvla.models.third_party_models.cosmos3.data.vfm import sequence_packing
from fluxvla.models.third_party_models.cosmos3.model.vfm.mot import \
    modeling_utils
from fluxvla.tokenizers.cosmos3_wan22_vae import Cosmos3Wan22VAE
from .base_vla import BaseVLA
from .cosmos3_flowmatching_inference_mixin import Cosmos3InferenceMixin
from .cosmos3_flowmatching_model_mixins import (Cosmos3CodecMixin,
                                                Cosmos3ComponentsMixin)
from .cosmos3_flowmatching_training_mixins import (Cosmos3LossMixin,
                                                   Cosmos3ScheduleMixin,
                                                   Cosmos3SequenceMixin)
from .cosmos3_flowmatching_utils import (_as_action_list, _as_long_list,
                                         _as_text_ids, merge_config,
                                         read_json_if_exists,
                                         resolve_inference_config,
                                         resolve_torch_dtype,
                                         resolve_training_config)

SequencePlan = sequence_packing.SequencePlan
TimestepEmbedder = modeling_utils.TimestepEmbedder
overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class Cosmos3FlowMatching(Cosmos3ComponentsMixin, Cosmos3ScheduleMixin,
                          Cosmos3SequenceMixin, Cosmos3CodecMixin,
                          Cosmos3LossMixin, Cosmos3InferenceMixin, BaseVLA):
    """FluxVLA-native Cosmos3 MoT flow matching model.

    This class intentionally exposes the MoT backbone, modality projectors,
    and losses at the VLA layer.  The vendored Cosmos3 code is only used for
    the compact MoT transformer and sequence packing primitives.
    """

    def __init__(
        self,
        vlm_backbone: Dict,
        vision_latent_dim: int = 16,
        latent_patch_size: int = 1,
        max_action_dim: int = 32,
        vision_in_proj: Optional[Dict | nn.Module] = None,
        vision_out_proj: Optional[Dict | nn.Module] = None,
        action_in_proj: Optional[Dict | nn.Module] = None,
        action_out_proj: Optional[Dict | nn.Module] = None,
        ori_action_dim: Optional[int] = None,
        action_horizon: int = 10,
        rectified_flow_training_config: Optional[Dict] = None,
        rectified_flow_inference_config: Optional[Dict] = None,
        num_embodiment_domains: int = 32,
        timestep_scale: float = 0.001,
        packed_attention_backend: str = 'flash2',
        position_embedding_type: str = 'unified_3d_mrope',
        unified_3d_mrope_reset_spatial_ids: bool = True,
        unified_3d_mrope_temporal_modality_margin: int = 0,
        enable_fps_modulation: bool = False,
        base_fps: float = 24.0,
        vision_vae: Optional[Dict | Cosmos3Wan22VAE] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        freeze_vlm_backbone: bool = False,
        freeze_non_moe_vlm_backbone: bool = False,
        enable_vision_loss: bool = False,
        pretrained_name_or_path: Optional[str] = None,
        name_mapping: Optional[Dict] = None,
        strict_mapping: bool = False,
        reinitialize_action_policy: bool = False,
        norm_stats: Optional[Dict] = None,
        torch_dtype: Optional[str | torch.dtype] = None,
    ) -> None:
        if freeze_vlm_backbone and freeze_non_moe_vlm_backbone:
            raise ValueError(
                'freeze_vlm_backbone=True conflicts with '
                'freeze_non_moe_vlm_backbone=True. Use the latter to keep '
                'only Cosmos3 VLM MoE generation parameters trainable.')

        super().__init__(
            vision_backbone=None,
            llm_backbone=None,
            vlm_backbone=None,
            projector=None,
            vla_head=None,
            freeze_vlm_backbone=freeze_vlm_backbone,
            norm_stats=norm_stats,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping,
        )

        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self.vision_latent_dim = vision_latent_dim
        self.latent_channel = vision_latent_dim
        self.latent_patch_size = latent_patch_size
        self.patch_latent_dim = (
            vision_latent_dim * latent_patch_size * latent_patch_size)
        self.max_action_dim = max_action_dim
        self.action_dim = max_action_dim
        self.ori_action_dim = ori_action_dim or max_action_dim
        self.action_horizon = action_horizon
        self.rectified_flow_training_config = self._resolve_training_config(
            rectified_flow_training_config)
        self.rectified_flow_inference_config = self._resolve_inference_config(
            rectified_flow_inference_config)
        self.num_train_timesteps = int(
            self.rectified_flow_inference_config['num_train_timesteps'])
        self.action_loss_weight = float(
            self.rectified_flow_training_config['action_loss_weight'])
        self.vision_loss_weight = float(
            self.rectified_flow_training_config['vision_loss_weight'])
        self.num_embodiment_domains = int(num_embodiment_domains)
        self.timestep_scale = timestep_scale
        self.packed_attention_backend = packed_attention_backend
        supported_position_embedding_types = {'unified_3d_mrope'}
        if position_embedding_type not in supported_position_embedding_types:
            raise ValueError(
                'FluxVLA-native Cosmos3 currently supports '
                'position_embedding_type="unified_3d_mrope" only; '
                f'got {position_embedding_type!r}.')
        self.position_embedding_type = position_embedding_type
        self.unified_3d_mrope_reset_spatial_ids = (
            unified_3d_mrope_reset_spatial_ids)
        self.unified_3d_mrope_temporal_modality_margin = (
            unified_3d_mrope_temporal_modality_margin)
        self.enable_fps_modulation = enable_fps_modulation
        self.base_fps = base_fps
        self.enable_vision_loss = enable_vision_loss
        self.reinitialize_action_policy = bool(reinitialize_action_policy)
        self.freeze_non_moe_vlm_backbone = bool(freeze_non_moe_vlm_backbone)
        if vision_vae is None:
            raise ValueError('Cosmos3FlowMatching requires '
                             '`vision_vae=dict('
                             'type="Cosmos3Wan22VAE")`.')
        self.vision_vae_config = (
            copy.deepcopy(vision_vae)
            if isinstance(vision_vae, dict) else None)
        self.special_tokens = special_tokens or {
            'eos_token_id': 2,
            'start_of_generation': 3,
            'end_of_generation': 4,
        }

        self.vlm_backbone = self._build_vlm_backbone(vlm_backbone)
        self.hidden_size = self._derive_hidden_size_from_vlm_config()
        hidden_size = self.hidden_size
        self.vision_vae = self._build_vision_vae(vision_vae)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.vision_in_proj = self._build_projector(
            vision_in_proj,
            dict(
                type='LinearProjector',
                in_dim=self.patch_latent_dim,
                out_dim=hidden_size,
            ),
        )
        self.vision_out_proj = self._build_projector(
            vision_out_proj,
            dict(
                type='LinearProjector',
                in_dim=hidden_size,
                out_dim=self.patch_latent_dim,
            ),
        )
        self.action_in_proj = self._build_projector(
            action_in_proj,
            dict(
                type='DomainAwareLinear',
                input_size=max_action_dim,
                output_size=hidden_size,
                num_domains=num_embodiment_domains,
            ),
        )
        self.action_out_proj = self._build_projector(
            action_out_proj,
            dict(
                type='DomainAwareLinear',
                input_size=hidden_size,
                output_size=max_action_dim,
                num_domains=num_embodiment_domains,
            ),
        )
        self.action_modality_embed = nn.Parameter(torch.zeros(hidden_size))
        self._validate_projector_shapes()
        self._init_projection_weights_like_cosmos3()

        if self.torch_dtype is not None:
            self.to(dtype=self.torch_dtype)
            self._keep_rotary_buffers_fp32()

        if freeze_vlm_backbone:
            self.vlm_backbone.requires_grad_(False)

        self.all_module_keys = [
            'vlm_backbone',
            'vision_vae',
            'time_embedder',
            'vision_in_proj',
            'vision_out_proj',
            'action_in_proj',
            'action_out_proj',
            'action_modality_embed',
        ]

    def from_pretrained(self):
        if self.pretrained_name_or_path is None:
            return

        vision_vae = self._modules.pop('vision_vae', None)
        visual = self.vlm_backbone.model._modules.pop('visual', None)
        action_in_proj = action_out_proj = action_modality_embed = None
        if self.reinitialize_action_policy and self.name_mapping:
            action_in_proj = self._modules.pop('action_in_proj')
            action_out_proj = self._modules.pop('action_out_proj')
            action_modality_embed = self._parameters.pop(
                'action_modality_embed')
        skipped_modules = []
        if vision_vae is not None:
            skipped_modules.append('VAE')
        if visual is not None:
            skipped_modules.append('visual tower')
        if skipped_modules:
            overwatch.info(
                f"Temporarily skipping Cosmos3 {', '.join(skipped_modules)} "
                'while loading the transformer checkpoint; those weights are '
                'loaded by their owning modules.')
        if action_in_proj is not None:
            overwatch.info(
                'Keeping the Cosmos3 action policy freshly initialized for '
                'LIBERO post-training.')
        try:
            super().from_pretrained()
        finally:
            if action_in_proj is not None:
                self._modules['action_in_proj'] = action_in_proj
                self._modules['action_out_proj'] = action_out_proj
                self._parameters[
                    'action_modality_embed'] = action_modality_embed
            if visual is not None:
                self.vlm_backbone.model._modules['visual'] = visual
            if vision_vae is not None:
                self._modules['vision_vae'] = vision_vae

    @property
    def config(self):
        return self.vlm_backbone.config

    @staticmethod
    def _resolve_torch_dtype(
            dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
        return resolve_torch_dtype(dtype)

    @staticmethod
    def _merge_config(defaults: Dict, config: Optional[Dict]) -> Dict:
        return merge_config(defaults, config)

    @classmethod
    def _resolve_training_config(cls, config: Optional[Dict]) -> Dict:
        return resolve_training_config(config)

    @classmethod
    def _resolve_inference_config(cls, config: Optional[Dict]) -> Dict:
        return resolve_inference_config(config)

    @staticmethod
    def _read_json_if_exists(path) -> Optional[Dict]:
        return read_json_if_exists(path)

    @staticmethod
    def _is_vlm_moe_generation_parameter(name: str) -> bool:
        return 'moe_gen' in name or 'k_norm_und_for_gen' in name

    def _freeze_non_moe_vlm_backbone(self) -> None:
        if self.vlm_backbone is None:
            return

        trainable_numel = 0
        frozen_numel = 0
        self.vlm_backbone.requires_grad_(False)
        for name, param in self.vlm_backbone.named_parameters():
            if self._is_vlm_moe_generation_parameter(name):
                param.requires_grad_(True)
                trainable_numel += param.numel()
            else:
                frozen_numel += param.numel()

        overwatch.info(
            'Cosmos3 VLM backbone MoE-only tuning enabled: '
            f'trainable_moe_numel={trainable_numel:,}, '
            f'frozen_non_moe_numel={frozen_numel:,}.',
            ctx_level=1,
        )

    def freeze_backbones(self) -> None:
        super().freeze_backbones()
        if self.freeze_non_moe_vlm_backbone:
            self._freeze_non_moe_vlm_backbone()

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_token_ids: Optional[torch.Tensor | List[List[int]]] = None,
        actions: Optional[torch.Tensor | List[torch.Tensor]] = None,
        embodiment_ids: Optional[torch.Tensor | List[int]] = None,
        raw_action_dim: Optional[torch.Tensor | List[int]] = None,
        *,
        sequence_plan: List[SequencePlan],
        conditioning_fps: Optional[torch.Tensor] = None,
        action_fps: Optional[torch.Tensor] = None,
        **unused_batch_fields,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        del unused_batch_fields
        vision_tokens = self._tokenize_vision(images)
        raw_state_action = _as_action_list(
            actions) if actions is not None else None
        batch_size = len(raw_state_action or vision_tokens or [None])
        device = self.device
        vision_resolutions = self._resolve_vision_resolutions(
            images, vision_tokens, batch_size)

        text_ids = _as_text_ids(
            text_token_ids,
            batch_size,
            pad_token_id=self._text_pad_token_id(),
        )
        if len(sequence_plan) != batch_size:
            raise ValueError(
                f'Expected {batch_size} Cosmos3 SequencePlan objects, got '
                f'{len(sequence_plan)}.')
        sequence_plans = sequence_plan
        if raw_state_action is not None:
            if embodiment_ids is None:
                raise ValueError(
                    'Cosmos3FlowMatching action training requires '
                    '`embodiment_ids` from the FluxVLA data pipeline.')
            action_embodiment_ids = _as_long_list(embodiment_ids, batch_size,
                                                  device)
        else:
            action_embodiment_ids = None
        raw_action_dims = (
            _as_long_list(
                raw_action_dim,
                batch_size,
                device,
                default_value=self.ori_action_dim)
            if raw_state_action is not None else None)

        (noised_vision, target_vision, noised_action, target_action,
         timesteps_vision, timesteps_action) = self._sample_noisy_inputs(
             vision_tokens=vision_tokens,
             raw_state_action=raw_state_action,
             sequence_plans=sequence_plans,
             raw_action_dim=raw_action_dims,
             vision_resolutions=vision_resolutions,
         )
        packed_seq = self._pack_generation_data(
            sequence_plans=sequence_plans,
            text_token_ids=text_ids,
            vision_tokens=noised_vision,
            action_tokens=noised_action,
            embodiment_ids=action_embodiment_ids,
            raw_action_dim=raw_action_dims,
            timesteps=timesteps_vision,
            fps_vision=conditioning_fps,
            fps_action=action_fps,
        )
        self._override_action_timesteps(packed_seq, timesteps_action)

        packed_sequence, target_dtype = self._encode_text(packed_seq)
        original_latent_shapes = self._encode_vision(packed_seq,
                                                     packed_sequence,
                                                     target_dtype)
        self._encode_action(packed_seq, packed_sequence, target_dtype)
        last_hidden_state = self._run_backbone(packed_seq, packed_sequence)

        preds_action = self._decode_action(packed_seq, last_hidden_state)
        raw_action_loss = self._compute_action_loss(
            preds_action,
            target_action,
            raw_action_dims,
            packed_seq.action.condition_mask if packed_seq.action else None,
        )
        action_loss = raw_action_loss * self.action_loss_weight

        outputs: Dict[str, torch.Tensor | List[torch.Tensor]] = {
            'loss': action_loss,
            'flow_matching_loss_action': raw_action_loss,
            'preds_action': preds_action,
            'last_hidden_state': last_hidden_state,
        }
        if self.enable_vision_loss:
            preds_vision = self._decode_vision(packed_seq, last_hidden_state,
                                               original_latent_shapes)
            raw_vision_loss = self._compute_vision_loss(
                preds_vision,
                target_vision,
                packed_seq.vision.condition_mask
                if packed_seq.vision else None,
            )
            vision_loss = raw_vision_loss * self.vision_loss_weight
            outputs['loss'] = outputs['loss'] + vision_loss
            outputs['flow_matching_loss_vision'] = raw_vision_loss
            outputs['preds_vision'] = preds_vision
        return outputs

    def get_fsdp_wrapping_policy(self) -> Callable:
        transformer_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=(
                self.vlm_backbone.fsdp_transformer_layer_cls()),
        )

        return build_combined_wrap_policy([transformer_policy])
