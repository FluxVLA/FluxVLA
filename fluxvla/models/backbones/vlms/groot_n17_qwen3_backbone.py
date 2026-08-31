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
"""FluxVLA-native GR00T N1.7 Qwen3 backbone."""

from __future__ import annotations
import logging
from collections.abc import Mapping
from typing import Any, Dict, Optional

import torch

from fluxvla.engines.utils import VLM_BACKBONES
from .outputs import VLMBackboneOutput
from .qwen3_vl import Qwen3VL

logger = logging.getLogger(__name__)


@VLM_BACKBONES.register_module()
class GrootN17Qwen3Backbone(Qwen3VL):
    """Native equivalent of official GR00T N1.7 ``Qwen3Backbone``.

    This class intentionally mirrors the official backbone output contract so
    it can be compared and later swapped into ``GrootN17VLA`` without changing
    the action head.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        qwen3_runtime: str = 'compat_457',
    ) -> None:
        del reproject_vision, projector_dim

        self.qwen3_runtime = qwen3_runtime
        if qwen3_runtime:
            compat = __import__(
                'fluxvla.models.compat.qwen3vl_457_compat',
                fromlist=['apply_qwen3vl_runtime'])
            self.qwen3_runtime_summary = compat.apply_qwen3vl_runtime(
                qwen3_runtime,
                patch_gr00t_backbone=False,
            )
        else:
            self.qwen3_runtime_summary = None

        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                attention_implementation = 'flash_attention_2'
            except ImportError:
                logger.warning(
                    'flash_attn is not installed. Falling back to sdpa '
                    'attention.')
                attention_implementation = 'sdpa'
        else:
            attention_implementation = 'sdpa'

        # The GR00T checkpoint owns all backbone weights. Build only the Qwen
        # architecture here; GrootN17VLA materializes these meta parameters
        # directly from the ``backbone.*`` tensors in that checkpoint.
        with torch.device('meta'):
            super().__init__(
                vlm_backbone_id='qwen3_2b_vl_pt',
                vlm_config=dict(model_config),
                vlm_path=None,
                use_projection=False,
                attn_implementation=attention_implementation,
                torch_dtype='bf16',
            )
        self.vlm.eval()

        if select_layer > 0:
            while len(self.vlm.model.language_model.layers) > select_layer:
                self.vlm.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.load_bf16 = load_bf16

    def finalize_checkpoint_load(self) -> None:
        """Materialize buffers after assigning checkpoint weights."""
        device = next(self.parameters()).device
        visual_rotary = self.vlm.model.visual.rotary_pos_emb
        with torch.device(device):
            fresh_visual_rotary = type(visual_rotary)(visual_rotary.dim,
                                                      visual_rotary.theta)
        visual_rotary.register_buffer(
            'inv_freq', fresh_visual_rotary.inv_freq, persistent=False)

        text_rotary = self.vlm.model.language_model.rotary_emb
        with torch.device(device):
            fresh_text_rotary = type(text_rotary)(
                self.vlm.config.text_config, device=device)
        text_rotary.register_buffer(
            'inv_freq', fresh_text_rotary.inv_freq, persistent=False)
        text_rotary.register_buffer(
            'original_inv_freq',
            fresh_text_rotary.original_inv_freq,
            persistent=False)
        text_rotary.attention_scaling = fresh_text_rotary.attention_scaling

        if self.load_bf16:
            self.vlm.to(dtype=torch.bfloat16)

    @staticmethod
    def _trim_common_left_padding(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover the official collator's dynamic left-padded sequence.

        FluxVLA tokenizes each sample before ``DictCollator``, so training
        configs use a fixed maximum length to make samples stackable. The
        official N1.7 collator instead pads each batch only to its longest
        sequence. Removing columns that are padding for every sample makes
        the two model inputs identical while retaining the generic collator.
        """
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            return input_ids, attention_mask
        valid_columns = attention_mask.to(dtype=torch.bool).any(dim=0)
        if not torch.any(valid_columns):
            raise ValueError(
                'Qwen3-VL attention mask contains no valid token.')
        first_valid_column = int(
            torch.nonzero(valid_columns, as_tuple=False)[0].item())
        if first_valid_column == 0:
            return input_ids, attention_mask
        return (input_ids[:, first_valid_column:],
                attention_mask[:, first_valid_column:])

    def forward(
        self,
        vl_input: Optional[Mapping[str, torch.Tensor]] = None,
        images: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> VLMBackboneOutput:
        del img_masks, kwargs
        if vl_input is None:
            vl_input = {
                'lang_tokens': lang_tokens,
                'lang_masks': lang_masks,
                'images': images,
                'image_grid_thw': image_grid_thw,
            }
        hf_input = {
            'input_ids': vl_input['lang_tokens'],
            'attention_mask': vl_input['lang_masks'],
            'pixel_values': vl_input['images'],
            'image_grid_thw': vl_input['image_grid_thw'],
        }
        hf_input['input_ids'], hf_input['attention_mask'] = (
            self._trim_common_left_padding(hf_input['input_ids'],
                                           hf_input['attention_mask']))
        # DictCollator stacks sample-level packed vision tensors. Hugging Face
        # Qwen3-VL expects the same tensors packed across the whole batch.
        if hf_input['pixel_values'].ndim == 3:
            hf_input['pixel_values'] = hf_input['pixel_values'].flatten(0, 1)
        if hf_input['image_grid_thw'].ndim == 3:
            hf_input['image_grid_thw'] = hf_input['image_grid_thw'].flatten(
                0, 1)
        outputs = self.vlm(**hf_input, output_hidden_states=True)
        backbone_features = outputs.hidden_states[-1]
        image_mask = hf_input['input_ids'] == self.vlm.config.image_token_id
        attention_mask = hf_input['attention_mask'] == 1
        return VLMBackboneOutput(
            last_hidden_state=backbone_features,
            attention_mask=attention_mask,
            auxiliary_outputs={'image_mask': image_mask},
        )

    def enable_gradient_checkpointing(self) -> None:
        """Enable HuggingFace gradient checkpointing on the inner Qwen3-VL."""
        if not any(param.requires_grad for param in self.parameters()):
            logger.info('Skipping Qwen3-VL gradient checkpointing because the '
                        'backbone is frozen.')
            return
        if not hasattr(self.vlm, 'gradient_checkpointing_enable'):
            return
        gradient_checkpointing_kwargs = {'use_reentrant': False}
        try:
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            self.vlm.gradient_checkpointing_enable()
