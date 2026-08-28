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
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines import (build_head_from_cfg, build_llm_backbone_from_cfg,
                             build_projector_from_cfg,
                             build_vision_backbone_from_cfg,
                             build_vlm_backbone_from_cfg, initialize_overwatch)
from fluxvla.engines.utils.fsdp_wrapping import (build_combined_wrap_policy,
                                                 build_module_wrap_policy)

overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for arbitrary Vision-Language Models ===
class BaseVLA(nn.Module, GenerationMixin, ABC):

    def __init__(self,
                 vision_backbone: str = None,
                 llm_backbone: str = None,
                 vlm_backbone: str = None,
                 projector: str = None,
                 vla_head: str = None,
                 enable_mixed_precision_training: bool = True,
                 freeze_vision_backbone=True,
                 freeze_llm_backbone=True,
                 freeze_vlm_backbone=True,
                 freeze_projector=False,
                 vision_backbone_fp32: bool = False,
                 unfreeze_last_layer: bool = False,
                 ignore_index: int = -100,
                 norm_stats: Dict = None,
                 pretrained_name_or_path: str = None,
                 name_mapping: Dict = None,
                 strict_mapping: bool = False) -> None:
        super().__init__()
        if vision_backbone is not None:
            self.vision_backbone = build_vision_backbone_from_cfg(
                copy.deepcopy(vision_backbone))
        else:
            self.vision_backbone = None
        if llm_backbone is not None:
            self.llm_backbone = build_llm_backbone_from_cfg(llm_backbone)
        else:
            self.llm_backbone = None
        if vlm_backbone is not None:
            self.vlm_backbone = build_vlm_backbone_from_cfg(
                copy.deepcopy(vlm_backbone))
        else:
            self.vlm_backbone = None
        if projector is not None:
            self.projector = build_projector_from_cfg(projector)
        else:
            self.projector = None
        if vla_head is not None:
            self.vla_head = build_head_from_cfg(vla_head)
        else:
            self.vla_head = None

        self.freeze_vision_backbone = freeze_vision_backbone
        self.freeze_llm_backbone = freeze_llm_backbone
        self.freeze_vlm_backbone = freeze_vlm_backbone
        self.freeze_projector = freeze_projector
        self.vision_backbone_requires_grad = not freeze_vision_backbone
        self.vision_backbone_fp32 = vision_backbone_fp32
        self.unfreeze_last_layer = unfreeze_last_layer
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.ignore_index = ignore_index
        self.norm_stats = norm_stats
        self.pretrained_name_or_path = pretrained_name_or_path
        self.name_mapping = name_mapping
        self.strict_mapping = strict_mapping
        # Instance Attributes for a generic VLM
        self.all_module_keys = None

    def _mapped_name_candidates(self, name: str) -> List:
        candidates = []
        replacements = []
        for key, val in self.name_mapping.items():
            if key in name:
                replacements.append((key, val))
                candidates.append((f'{key}->{val}', name.replace(key, val)))

        if len(replacements) > 1:
            mapped_name = name
            desc = []
            for key, val in replacements:
                mapped_name = mapped_name.replace(key, val)
                desc.append(f'{key}->{val}')
            candidates.append((' + '.join(desc), mapped_name))

        deduped = []
        seen = set()
        for desc, mapped_name in candidates:
            if mapped_name not in seen:
                deduped.append((desc, mapped_name))
                seen.add(mapped_name)
        return deduped

    @property
    def device(self) -> torch.device:
        """Borrowed from `transformers.modeling_utils.py` -- checks
        parameter device; assumes model on *ONE* device!"""
        return next(self.parameters()).device

    def freeze_backbones(self) -> None:
        """
        Freeze the designated modules of the model.
        """
        if self.vision_backbone is not None:
            self.vision_backbone.requires_grad_(
                not self.freeze_vision_backbone)
        if self.llm_backbone is not None:
            self.llm_backbone.requires_grad_(not self.freeze_llm_backbone)
        if self.vlm_backbone is not None:
            self.vlm_backbone.requires_grad_(not self.freeze_vlm_backbone)
        if self.projector is not None:
            self.projector.requires_grad_(not self.freeze_projector)

        # Update Trackers
        self.vision_backbone_requires_grad = not self.freeze_vision_backbone

        # Explicitly Log Frozen / Trainable Components
        if self.vision_backbone is not None:
            if self.freeze_vision_backbone:
                overwatch.info(
                    '[Frozen]    🥶 =>> Vision Backbone', ctx_level=1)
            else:
                overwatch.info(
                    '[TRAINABLE] 🔥 =>> Vision Backbone', ctx_level=1)
        if self.llm_backbone is not None:
            if self.freeze_llm_backbone:
                overwatch.info('[Frozen]    🥶 =>> LLM Backbone', ctx_level=1)
            else:
                overwatch.info('[TRAINABLE] 🔥 =>> LLM Backbone', ctx_level=1)
        if self.vlm_backbone is not None:
            if self.freeze_vlm_backbone:
                overwatch.info('[Frozen]    🥶 =>> VLM Backbone', ctx_level=1)
            else:
                overwatch.info('[TRAINABLE] 🔥 =>> VLM Backbone', ctx_level=1)
        if self.projector is not None:
            if self.freeze_projector:
                overwatch.info('[Frozen]    🥶 =>> Projector', ctx_level=1)
            else:
                overwatch.info('[TRAINABLE] 🔥 =>> Projector', ctx_level=1)

        # Some VLM backbones need finer-grained tuning than the generic
        # freeze_vlm_backbone switch, e.g. GR00T N1.5 freezes Eagle LLM but
        # tunes Eagle vision tower for RoboCasa.
        if self.vlm_backbone is not None and hasattr(self.vlm_backbone,
                                                     'apply_trainable_policy'):
            self.vlm_backbone.apply_trainable_policy()

        if self.vision_backbone_fp32:
            self.vision_backbone.dtype = torch.float32

        if self.unfreeze_last_layer:
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

        overwatch.debug('##################################################')
        overwatch.debug('#####      Trainable Network Parameters:     #####')
        overwatch.debug('##################################################')
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    def forward_model(self,
                      input_ids: Optional[torch.LongTensor] = None,
                      attention_mask: Optional[torch.Tensor] = None,
                      pixel_values: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      inputs_embeds: Optional[torch.FloatTensor] = None,
                      past_key_values: Optional[List[
                          torch.FloatTensor]] = None,
                      use_cache: Optional[bool] = None,
                      output_attentions: Optional[bool] = None,
                      output_hidden_states: Optional[bool] = None,
                      return_dict: Optional[bool] = None,
                      multimodal_indices: Optional[torch.LongTensor] = None,
                      return_fused_labels: bool = False,
                      *args,
                      **kwargs) -> CausalLMOutputWithPast:
        """
        Fuse separate vision and language backbones before the action head.

        Supports:
            - Autoregressive decoding with cached past key values
            - Fully multimodal batches (image + text)
            - Unimodal fallback (text only)

        Args:
            input_ids (LongTensor): Input token IDs [B, T].
            attention_mask (Tensor): Mask for input tokens [B, T].
            pixel_values (FloatTensor): Image tensor or dict for vision model.
            labels (LongTensor): Language modeling target tokens [B, T].
            inputs_embeds (FloatTensor): Optional precomputed input embeddings.
            past_key_values (List[FloatTensor]): LLM cache for fast decoding.
            use_cache (bool): Whether to return cache for next step.
            output_attentions (bool): Whether to return attention maps.
            output_hidden_states (bool): Whether to return hidden states.
            return_dict (bool): Whether to return a CausalLMOutputWithPast.
            multimodal_indices (LongTensor): Indices of samples using image +
                text.

        Returns:
            CausalLMOutputWithPast: Outputs including logits and optional
                cache.
        """
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to
            # `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if return_fused_labels:
                return output, None, None
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError('Invalid `forward()` call!')

        # Handle Multimodal Indices is None --> pretend like the batch is fully
        # multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(
                len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple
        # unimodal forward
        elif len(multimodal_indices) == 0:
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if return_fused_labels:
                return output, attention_mask, labels
            return output

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({
                    k: pixel_values[k][multimodal_indices]
                    for k in pixel_values
                })
            else:
                patch_features = self.vision_backbone(
                    pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>>
        # num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # === Step 1: Get Input Embeddings from LLM ===
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # === Step 2: Build Multimodal Embeddings & Attention Mask ===
        multimodal_embeddings = torch.cat([
            input_embeddings[multimodal_indices, :1, :],
            projected_patch_embeddings, input_embeddings[multimodal_indices,
                                                         1:, :]
        ],
                                          dim=1)

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat([
                attention_mask[multimodal_indices, :1],
                projected_patch_attention_mask,
                attention_mask[multimodal_indices, 1:]
            ],
                                                  dim=1)

        # === Step 3: Build Multimodal Labels (Ignore patch embeddings) ===
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1]),
                self.ignore_index,
                dtype=labels.dtype,
                device=labels.device)
            multimodal_labels = torch.cat([
                labels[multimodal_indices, :1], projected_patch_labels,
                labels[multimodal_indices, 1:]
            ],
                                          dim=1)

        # === Step 4: Handle Unimodal Cases ===
        unimodal_indices = torch.tensor([
            idx
            for idx in range(len(input_ids)) if idx not in multimodal_indices
        ],
                                        dtype=torch.long,
                                        device=multimodal_indices.device)

        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels
        else:
            patch_len = projected_patch_embeddings.shape[1]
            embed_dim = input_embeddings.shape[2]

            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), patch_len, embed_dim),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device)
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), patch_len),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device)
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), patch_len),
                self.ignore_index,
                dtype=labels.dtype,
                device=labels.device)

            unimodal_embeddings = torch.cat(
                [input_embeddings[unimodal_indices], unimodal_embeddings_pad],
                dim=1)

            unimodal_attention_mask = torch.cat(
                [attention_mask[unimodal_indices], unimodal_attention_pad],
                dim=1)

            unimodal_labels = torch.cat(
                [labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # === Step 5: Merge Multimodal and Unimodal ===
            fused_embeddings = torch.vstack(
                [multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack(
                [multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # === Step 6: Final LLM Forward Pass ===
        output = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        if return_fused_labels:
            return output, fused_attention_mask, fused_labels
        return output, fused_attention_mask

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        ...

    # === GenerationMixin Expected Properties & Methods (DO NOT MODIFY) ===
    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.llm_backbone.llm.config

    # => Beam Search Utility
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm_backbone.llm._reorder_cache(past_key_values, beam_idx)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Returns the FSDP wrapping policy for the model.

        Returns:
            Callable: The wrapping policy for FSDP.
        """
        fsdp_policy_list = list()
        if hasattr(self, 'vision_backbone') and hasattr(
                self.vision_backbone, 'get_fsdp_wrapping_policy'):
            # Get Vision Backbone FSDP Wrapping Policy
            # =>> just a module wrapping policy around `self.vision_backbone`
            vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            fsdp_policy_list.append(vision_fsdp_wrapping_policy)
        if hasattr(self, 'llm_backbone') and hasattr(
                self.llm_backbone, 'get_fsdp_wrapping_policy'):
            # Get LLM Backbone FSDP Wrapping Policy
            # =>> just a module wrapping policy around `self.llm_backbone`
            llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            fsdp_policy_list.append(llm_fsdp_wrapping_policy)
        if hasattr(self, 'vlm_backbone') and hasattr(
                self.vlm_backbone, 'get_fsdp_wrapping_policy'):
            # Get VLM Backbone FSDP Wrapping Policy
            # =>> just a module wrapping policy around `self.vlm_backbone`
            vlm_fsdp_wrapping_policy = self.vlm_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            fsdp_policy_list.append(vlm_fsdp_wrapping_policy)
        if hasattr(self, 'vla_head') and hasattr(self.vla_head,
                                                 'get_fsdp_wrapping_policy'):
            fsdp_policy_list.append(self.vla_head.get_fsdp_wrapping_policy())
        from fluxvla.engines import PROJECTORS

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy
        # around `self.projector`
        projector_fsdp_wrapping_policy = build_module_wrap_policy(
            set(PROJECTORS._module_dict.values()))
        fsdp_policy_list.append(projector_fsdp_wrapping_policy)
        # Return union (_or_) over constituent policies
        # => Note: there is *not* a fall-through policy; any module that isn't
        # covered by the above constituents will automatically be folded into
        # the root VLM FSDP instance.
        return build_combined_wrap_policy(fsdp_policy_list)

    def from_pretrained(self):
        # Load weights based on file format
        if self.pretrained_name_or_path is None:
            return
        if self.pretrained_name_or_path.endswith(
                '.safetensors') or os.path.isdir(self.pretrained_name_or_path):
            # Handle safetensors format
            if self.pretrained_name_or_path.endswith('.safetensors'):
                pretrained_weights = load_file(
                    self.pretrained_name_or_path, device='cpu')
            else:
                # Load from directory containing safetensors files
                pretrained_weights = dict()
                for file in os.listdir(self.pretrained_name_or_path):
                    if file.endswith('.safetensors'):
                        file_path = os.path.join(self.pretrained_name_or_path,
                                                 file)
                        pretrained_weights.update(
                            load_file(file_path, device='cpu'))
        elif self.pretrained_name_or_path.endswith(
                '.pt') or self.pretrained_name_or_path.endswith('.pth'):
            # Handle pt/pth format using torch.load
            checkpoint = torch.load(
                self.pretrained_name_or_path, map_location='cpu')
            # Handle both dict format {'model': state_dict}
            # and direct state_dict
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                pretrained_weights = checkpoint['model']
            else:
                pretrained_weights = checkpoint
        else:
            raise ValueError(f'Unsupported checkpoint format: '
                             f'{self.pretrained_name_or_path}')

        # Load weights with name_mapping handling
        if not self.name_mapping:
            self.load_state_dict(
                pretrained_weights, strict=self.strict_mapping)
        else:
            for name, param in self.named_parameters():
                if self.name_mapping is None:
                    if self.strict_mapping and name not in pretrained_weights:
                        raise ValueError(
                            f"Parameter '{name}' not found in pretrained weights."  # noqa: E501, E713
                        )
                    if name in pretrained_weights and param.size(
                    ) == pretrained_weights[name].size():
                        # Copy the parameter withname the correct dtype
                        with torch.no_grad():
                            param.copy_(pretrained_weights[name].to(
                                param.dtype))
                    else:
                        overwatch.info(
                            f"Parameter '{name}' not found in pretrained weights, skipping."  # noqa: E501, E713
                        )
                else:
                    matched = False
                    matched_name = None
                    candidates = self._mapped_name_candidates(name)
                    for _, mapped_name in candidates:
                        if mapped_name not in pretrained_weights:
                            continue
                        if matched:
                            raise ValueError(
                                f"Parameter '{name}' matched multiple "
                                f"pretrained weights: '{matched_name}' and "
                                f"'{mapped_name}'.")
                        with torch.no_grad():
                            if param.size(
                            ) == pretrained_weights[mapped_name].size():
                                param.copy_(pretrained_weights[mapped_name].to(
                                    param.dtype))
                            else:
                                overwatch.info(
                                    f"[*] Size mismatch for '{name}': "
                                    f'model={list(param.size())} vs '
                                    f"ckpt '{mapped_name}'="
                                    f'{list(pretrained_weights[mapped_name].size())}'  # noqa: E501
                                )
                                continue
                        matched = True
                        matched_name = mapped_name
                    if not matched:
                        if self.strict_mapping:
                            raise ValueError(
                                f"Parameter '{name}' not found in pretrained weights with mapping."  # noqa: E501, E713
                            )
                        else:
                            # Debug: show what mapping was attempted
                            attempted = [
                                f"{desc}: '{mn}' "
                                f'(in_ckpt={mn in pretrained_weights})'
                                for desc, mn in candidates
                            ]
                            overwatch.info(
                                f"[*] Parameter '{name}' not found in "  # noqa: E713, E501
                                f'pretrained weights, skipping. '
                                f'Attempted: {attempted}')
