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
"""Qwen3-VL compatibility shims for GR00T N1.7 on transformers 5.3.

The official GR00T N1.7 checkpoint was validated with transformers 4.57.3.
FluxVLA currently carries transformers 5.3.0, where Qwen3-VL's public surface
and attention dispatch have changed. This module is the single place where we
make the 5.3 runtime look like the 4.57 runtime for the official
``gr00t.model.modules.qwen3_backbone`` code path.

This first layer is intentionally conservative: it restores the interfaces
the official Qwen3Backbone directly consumes and records which patches were
applied.
Numerical compatibility is verified separately by the tensor diff tools.
"""

from __future__ import annotations
import importlib
from typing import Any, Dict

import torch

_Compat = Dict[str, Any]


def _flash_attention_forward_457(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Transformers 4.57-style flash_attention_2 wrapper."""
    flash_utils = importlib.import_module(
        'transformers.modeling_flash_attention_utils')
    flash_integration = importlib.import_module(
        'transformers.integrations.flash_attention')
    logger = getattr(flash_integration, 'logger')
    use_top_left_mask = getattr(flash_integration, '_use_top_left_mask')
    flash_forward = getattr(flash_utils, '_flash_attention_forward')

    if kwargs.get('output_attentions',
                  False) or kwargs.get('head_mask') is not None:
        logger.warning_once(
            '`flash_attention_2` does not support `output_attentions=True` '
            'or `head_mask`. Please set your attention to `eager` if you '
            'want any of these features.')

    seq_len = query.shape[2]
    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            'FlashAttention does not support inputs with dim=0. Please check '
            'your input shapes or use SDPA instead.')

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, '_pre_quantization_dtype'):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(
                layer for layer in module.modules()
                if isinstance(layer, torch.nn.Linear)).weight.dtype

    is_causal = kwargs.pop('is_causal', None)
    if is_causal is None:
        is_causal = module.is_causal

    attn_output = flash_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=use_top_left_mask,
        target_dtype=target_dtype,
        attn_implementation=module.config._attn_implementation,
        layer_idx=module.layer_idx if hasattr(module, 'layer_idx') else None,
        **kwargs,
    )
    return attn_output, None


def _patch_attention_interface() -> Dict[str, Any]:
    modeling_utils = importlib.import_module('transformers.modeling_utils')
    attention_fns = getattr(modeling_utils, 'ALL_ATTENTION_FUNCTIONS')
    previous = attention_fns['flash_attention_2']
    attention_fns.register('flash_attention_2', _flash_attention_forward_457)
    # Transformers 5.3 aliases flash_attention_3 to the same wrapper on this
    # machine. Patch it only when it points to the same object as FA2.
    patched_fa3 = False
    try:
        if attention_fns['flash_attention_3'] is previous:
            attention_fns.register('flash_attention_3',
                                   _flash_attention_forward_457)
            patched_fa3 = True
    except Exception:
        patched_fa3 = False
    return {
        'previous_flash_attention_2':
        f'{previous.__module__}.{previous.__name__}',
        'new_flash_attention_2': f'{_flash_attention_forward_457.__module__}.'
        f'{_flash_attention_forward_457.__name__}',
        'patched_flash_attention_3': patched_fa3,
    }


def _get_rope_index_457(
    self,
    input_ids: torch.LongTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transformers 4.57 Qwen3-VL multimodal RoPE index implementation."""
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(
            video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None
                                  or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for batch_idx, current_input_ids in enumerate(total_input_ids):
            current_input_ids = current_input_ids[attention_mask[batch_idx] ==
                                                  1]
            vision_start_indices = torch.argwhere(
                current_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = current_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = current_input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st
                st_idx = (
                    llm_pos_ids_list[-1].max() +
                    1 if len(llm_pos_ids_list) > 0 else 0)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                    -1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len +
                    st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() +
                    1 if len(llm_pos_ids_list) > 0 else 0)
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., batch_idx,
                         attention_mask[batch_idx] == 1] = llm_positions.to(
                             position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 -
                                         len(total_input_ids[batch_idx]))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
            attention_mask.device)
        max_position_ids = position_ids.max(
            0, keepdim=False)[0].max(
                -1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = torch.arange(
            input_ids.shape[1],
            device=input_ids.device).view(1, 1,
                                          -1).expand(3, input_ids.shape[0], -1)
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
    return position_ids, mrope_position_deltas


def _compute_3d_position_ids_457(
    self,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: torch.Tensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
) -> torch.Tensor | None:
    del mm_token_type_ids
    if input_ids is None:
        return None

    past_key_values_length = (0 if past_key_values is None else
                              past_key_values.get_seq_length())
    can_compute_mrope = (
        image_grid_thw is not None or video_grid_thw is not None)
    if can_compute_mrope and (self.rope_deltas is None
                              or past_key_values_length == 0):
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        self.rope_deltas = rope_deltas
        return position_ids

    if self.rope_deltas is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids.view(1, batch_size, -1).repeat(
                3, 1, 1).to(inputs_embeds.device)
        else:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                device=inputs_embeds.device,
            )
            position_ids = position_ids.view(1, 1,
                                             -1).expand(3, batch_size, -1)
        delta = self.rope_deltas.repeat_interleave(
            batch_size // self.rope_deltas.shape[0], dim=0)
        return position_ids + delta.to(device=inputs_embeds.device)

    position_ids, rope_deltas = self.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    self.rope_deltas = rope_deltas
    return position_ids


def _patch_rope_position_logic(modeling_module) -> Dict[str, Any]:
    """Patch 5.3 Qwen3-VL position id generation to the 4.57 path."""
    qwen_cls = getattr(modeling_module, 'Qwen3VLForConditionalGeneration')
    qwen_model_cls = getattr(modeling_module, 'Qwen3VLModel')
    text_model_cls = getattr(modeling_module, 'Qwen3VLTextModel')
    previous_cond_forward = qwen_cls.forward
    previous_get_rope = qwen_model_cls.get_rope_index
    previous_compute = getattr(qwen_model_cls, 'compute_3d_position_ids', None)
    previous_text_forward = text_model_cls.forward

    def text_forward_457_bridge(self, *args, **kwargs):
        position_ids = kwargs.get('position_ids')
        visual_pos_masks = kwargs.get('visual_pos_masks')
        deepstack_visual_embeds = kwargs.get('deepstack_visual_embeds')
        last_decoder_output = None

        def capture_last_layer(_module, _inputs, _kwargs, output):
            nonlocal last_decoder_output
            last_decoder_output = output

        last_layer_handle = None
        if len(getattr(self, 'layers', [])) > 0:
            last_layer_handle = self.layers[-1].register_forward_hook(
                capture_last_layer, with_kwargs=True)
        if (isinstance(position_ids, torch.Tensor) and position_ids.ndim == 3
                and position_ids.shape[0] == 3):
            kwargs['position_ids'] = torch.cat(
                [position_ids[:1], position_ids], dim=0)
        try:
            outputs = previous_text_forward(self, *args, **kwargs)
        finally:
            if last_layer_handle is not None:
                last_layer_handle.remove()
        self._fluxvla_last_decoder_output_457 = last_decoder_output
        hidden_states = getattr(outputs, 'hidden_states', None)
        if hidden_states is not None and deepstack_visual_embeds is not None:
            patched_hidden_states = list(hidden_states)
            for layer_idx, visual_embeds in enumerate(deepstack_visual_embeds):
                hidden_idx = layer_idx + 1
                if hidden_idx >= len(patched_hidden_states):
                    break
                patched_hidden_states[hidden_idx] = self._deepstack_process(
                    patched_hidden_states[hidden_idx],
                    visual_pos_masks,
                    visual_embeds,
                )
            outputs.hidden_states = tuple(patched_hidden_states)
        return outputs

    def conditional_forward_457_bridge(self, *args, **kwargs):
        outputs = previous_cond_forward(self, *args, **kwargs)
        hidden_states = getattr(outputs, 'hidden_states', None)
        last_decoder_output = getattr(
            self.model.language_model,
            '_fluxvla_last_decoder_output_457',
            None,
        )
        if hidden_states is not None and last_decoder_output is not None:
            patched_hidden_states = list(hidden_states)
            patched_hidden_states[-1] = last_decoder_output
            outputs.hidden_states = tuple(patched_hidden_states)
        return outputs

    qwen_cls.forward = conditional_forward_457_bridge
    qwen_model_cls.get_rope_index = _get_rope_index_457
    qwen_model_cls.compute_3d_position_ids = _compute_3d_position_ids_457
    text_model_cls.forward = text_forward_457_bridge
    return {
        'qwen3vl_model_get_rope_index':
        f'{previous_get_rope.__module__}.{previous_get_rope.__name__}',
        'qwen3vl_model_compute_3d_position_ids':
        (f'{previous_compute.__module__}.{previous_compute.__name__}'
         if previous_compute is not None else None),
        'qwen3vl_text_forward_bridge':
        True,
        'qwen3vl_conditional_forward_bridge':
        f'{previous_cond_forward.__module__}.{previous_cond_forward.__name__}',
    }


def _patch_public_backbone_properties(qwen_cls: type) -> Dict[str, Any]:
    """Restore 4.57 top-level aliases consumed by official Qwen3Backbone."""
    patched = {
        'language_model': False,
        'visual': False,
    }
    if not hasattr(qwen_cls, 'language_model'):
        qwen_cls.language_model = property(
            lambda self: self.model.language_model)
        patched['language_model'] = True
    if not hasattr(qwen_cls, 'visual'):
        qwen_cls.visual = property(lambda self: self.model.visual)
        patched['visual'] = True
    return patched


def _patch_transformers_qwen3vl_class() -> tuple[type, Dict[str, Any]]:
    module = importlib.import_module(
        'transformers.models.qwen3_vl.modeling_qwen3_vl')
    qwen_cls = getattr(module, 'Qwen3VLForConditionalGeneration')
    patched = _patch_public_backbone_properties(qwen_cls)
    patched['rope_position_logic'] = _patch_rope_position_logic(module)
    return qwen_cls, patched


def apply_qwen3vl_457_compat(patch_gr00t_backbone: bool = True) -> _Compat:
    """Apply the current Qwen3-VL 4.57 compatibility layer.

    Returns a structured summary so probes and debug JSON can show exactly
    which compat hooks were active for a run.
    """
    qwen_cls, class_patches = _patch_transformers_qwen3vl_class()
    summary: Dict[str, Any] = {
        'runtime':
        'compat_457',
        'transformers_qwen3vl_class':
        (f'{qwen_cls.__module__}.{qwen_cls.__name__}'),
        'class_patches':
        class_patches,
        'attention_patches':
        _patch_attention_interface(),
        'gr00t_qwen3_backbone_patched':
        False,
    }
    if not patch_gr00t_backbone:
        summary['gr00t_qwen3_backbone_skipped'] = True
        return summary

    try:
        qwen_backbone = importlib.import_module(
            'gr00t.model.modules.qwen3_backbone')
    except Exception as exc:  # pragma: no cover - probe path
        summary['gr00t_qwen3_backbone_error'] = (
            f'{type(exc).__name__}: {str(exc).splitlines()[0]}')
        return summary

    # qwen3_backbone imports Qwen3VLForConditionalGeneration at module import
    # time, so patching the transformers class alone is not enough after the
    # official module has already been imported.
    setattr(qwen_backbone, 'Qwen3VLForConditionalGeneration', qwen_cls)
    setattr(qwen_backbone, '_QWEN3VL_AVAILABLE', True)
    summary['gr00t_qwen3_backbone_patched'] = True
    return summary


def apply_qwen3vl_runtime(
    runtime: str,
    patch_gr00t_backbone: bool = True,
) -> Dict[str, Any]:
    """Apply a named Qwen3-VL runtime shim."""
    runtime = str(runtime).lower()
    if runtime in ('hf_53', 'none'):
        return {'runtime': runtime, 'applied': False}
    if runtime == 'compat_457':
        summary = apply_qwen3vl_457_compat(
            patch_gr00t_backbone=patch_gr00t_backbone)
        summary['applied'] = True
        return summary
    raise ValueError(f'Unsupported qwen3_runtime: {runtime!r}')
