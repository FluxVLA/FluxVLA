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
"""Thin wrapper for VLM SFT: only vlm_backbone, forward returns HF
CausalLMOutputWithPast (with loss).
"""

import os
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines import VLAS, build_vlm_backbone_from_cfg


@VLAS.register_module()
class VLMForSFT(nn.Module):
    """VLM-only model for SFT: wraps a VLM backbone and forwards HF-style
    batches (input_ids, attention_mask, pixel_values, image_grid_thw, labels)
    to produce loss. Checkpoints can be saved in HuggingFace format so that
    the official transformers code can load and run inference.

    VLM is split into VisionEncoder, Projection (e.g. merger), and Backbone
    (LLM). You can freeze each part via freeze_vision_encoder,
    freeze_projection, and freeze_backbone (e.g. stage1 alignment: only
    projection trainable; stage2 SFT: only vision encoder frozen).

    Args:
        vlm_backbone (Dict): Config for the VLM backbone (e.g. Qwen3VL).
        freeze_vision_encoder (bool): If True, freeze vision encoder.
            Default False.
        freeze_projection (bool): If True, freeze projection/merger.
            Default False.
        freeze_backbone (bool): If True, freeze LLM backbone. Default False.
    """

    def __init__(
        self,
        vlm_backbone: Dict,
        freeze_vision_encoder: bool = False,
        freeze_projection: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.vlm_backbone = build_vlm_backbone_from_cfg(vlm_backbone)
        self._trainable_module_keys = ['vlm_backbone']
        self._all_module_keys = ['vlm_backbone']
        self._freeze_vision_encoder = freeze_vision_encoder
        self._freeze_projection = freeze_projection
        self._freeze_backbone = freeze_backbone

    @property
    def all_module_keys(self):
        return self._all_module_keys

    @property
    def trainable_module_keys(self):
        return self._trainable_module_keys

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        """Forward for SFT. Expects batch from VLMSFTCollator (input_ids,
        attention_mask, labels, optional pixel_values, image_grid_thw).
        Returns CausalLMOutputWithPast with loss.
        """
        return self.vlm_backbone(**kwargs)

    def get_fsdp_wrapping_policy(self) -> Callable:
        return self.vlm_backbone.get_fsdp_wrapping_policy()

    def freeze_backbones(self) -> None:
        """Apply freeze flags to VisionEncoder, Projection, Backbone inside
        the VLM backbone (if supported)."""
        if hasattr(self.vlm_backbone, 'apply_vlm_freeze'):
            self.vlm_backbone.apply_vlm_freeze(
                freeze_vision_encoder=self._freeze_vision_encoder,
                freeze_projection=self._freeze_projection,
                freeze_backbone=self._freeze_backbone,
            )

    def from_pretrained(self) -> None:
        """No-op (backbone already loaded from path in config)."""
        pass

    def clip_grad_norm_(self, max_norm: float) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def get_hf_model(self):
        """Return the inner HuggingFace model for save_pretrained."""
        return self.vlm_backbone.vlm

    def save_pretrained_hf(
        self,
        save_directory: str,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Save the inner HuggingFace VLM so transformers can load it.

        Writes the same layout as Transformers Trainer / model.save_pretrained:
        - config.json
        - model.safetensors (preferred) or pytorch_model.bin
        - Processor files (tokenizer, preprocessor_config, chat_template, etc.)
          when vlm_path is set and backbone has a known processor type.

        If state_dict is provided (e.g. from FSDP), keys under
        'vlm_backbone.vlm.' or '_fsdp_wrapped_module.vlm_backbone.vlm.' are
        extracted, prefix stripped, and saved.
        """
        hf_model = self.vlm_backbone.vlm
        prefixes = ('vlm_backbone.vlm.',
                    '_fsdp_wrapped_module.vlm_backbone.vlm.')
        src = state_dict if state_dict is not None else self.state_dict()
        hf_state = {}
        for k, v in src.items():
            for prefix in prefixes:
                if k.startswith(prefix):
                    hf_state[k[len(prefix):]] = v
                    break
        os.makedirs(save_directory, exist_ok=True)
        # Config (same as HF save_pretrained)
        hf_model.config.save_pretrained(save_directory)
        # Weights: prefer safetensors so AutoModel.from_pretrained works
        model_safetensors = os.path.join(save_directory, 'model.safetensors')
        try:
            from safetensors.torch import save_file
            save_file(hf_state, model_safetensors)
        except Exception:
            torch.save(hf_state,
                       os.path.join(save_directory, 'pytorch_model.bin'))
        # Processor (tokenizer + image/video processor + chat_template, etc.)
        self._save_processor_pretrained(save_directory)

    def _save_processor_pretrained(self, save_directory: str) -> None:
        """Save processor (tokenizer, preprocessor_config, chat_template)."""
        processor_path = getattr(self.vlm_backbone, 'vlm_path', None)
        if not processor_path or not os.path.isdir(processor_path):
            return
        backbone_id = getattr(self.vlm_backbone, 'vlm_backbone_id', '') or ''
        if backbone_id.startswith('qwen3_'):
            try:
                from transformers import Qwen3VLProcessor
                processor = Qwen3VLProcessor.from_pretrained(
                    processor_path, trust_remote_code=True)
                processor.save_pretrained(save_directory)
            except Exception:  # noqa: S110
                pass
        if backbone_id.startswith('qwen3_5_'):
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(
                    processor_path, trust_remote_code=True)
                processor.save_pretrained(save_directory)
            except Exception:  # noqa: S110
                pass
