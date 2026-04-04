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
"""Modular weight loading for VLM: load vision encoder and LLM backbone from
different checkpoints via config (vision_encoder_path, llm_backbone_path).

This module provides a generic interface and per-backbone implementations so
that different VLMs can mix vision/LLM from different models. Other VLMs
(e.g. PaliGemma, Qwen2.5-VL) can add their own get_*_modular_config and
load_*_modular_weights in the future.
"""

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Keys to copy from LLM config to VLM text_config when merging (Qwen3 family)
QWEN_LLM_TO_TEXT_CONFIG_KEYS = [
    'hidden_size',
    'intermediate_size',
    'num_hidden_layers',
    'num_attention_heads',
    'num_key_value_heads',
    'head_dim',
    'hidden_act',
    'rms_norm_eps',
    'vocab_size',
    'tie_word_embeddings',
    'attention_bias',
    'attention_dropout',
]


def get_qwen3vl_modular_config(
    vlm_path: str,
    vision_encoder_path: Optional[str] = None,
    llm_backbone_path: Optional[str] = None,
    vlm_backbone_id: str = 'qwen3_2b_vl_pt',
) -> Dict[str, Any]:
    """Build merged Qwen3VL config for modular loading.

    - Vision structure and out_hidden_size come from the vision template
      (vision_encoder_path or vlm_path). If llm_backbone_path is set, text
      config is taken from the LLM and vision_config.out_hidden_size is set
      to the LLM hidden_size so image embeddings match.

    Args:
        vlm_path: Main VLM path (used as vision template when
            vision_encoder_path is None).
        vision_encoder_path: If set, use this path's config for vision.
        llm_backbone_path: If set, use this path's config for text and set
            vision out_hidden_size to LLM hidden_size.
        vlm_backbone_id: Key in VLM_BACKBONE_CONFIGS for config class.

    Returns:
        Config dict suitable for VLMBackbone (vlm_cfg(**config)).
    """
    from transformers import AutoConfig

    from .configs import VLM_BACKBONE_CONFIGS

    # Resolve vision template path
    template_path = vision_encoder_path if vision_encoder_path else vlm_path
    if not template_path:
        raise ValueError(
            'Modular loading requires vlm_path or vision_encoder_path.')

    if vlm_backbone_id not in VLM_BACKBONE_CONFIGS:
        raise KeyError(
            f'vlm_backbone_id={vlm_backbone_id!r} not in VLM_BACKBONE_CONFIGS')
    vlm_cfg_cls = VLM_BACKBONE_CONFIGS[vlm_backbone_id]['config']
    qwen3_vl_cfg = vlm_cfg_cls.from_pretrained(template_path)

    if llm_backbone_path is None:
        # Only vision swapped or default: use template config as-is
        return qwen3_vl_cfg.to_dict()

    # Merge: use LLM config for text, keep VL vision and rope_scaling
    qwen3_llm_cfg = AutoConfig.from_pretrained(llm_backbone_path)
    keep_rope_scaling = getattr(qwen3_vl_cfg.text_config, 'rope_scaling', None)

    for k in QWEN_LLM_TO_TEXT_CONFIG_KEYS:
        if hasattr(qwen3_llm_cfg, k) and hasattr(qwen3_vl_cfg.text_config, k):
            setattr(qwen3_vl_cfg.text_config, k, getattr(qwen3_llm_cfg, k))

    if keep_rope_scaling is not None:
        qwen3_vl_cfg.text_config.rope_scaling = keep_rope_scaling

    qwen3_vl_cfg.vision_config.out_hidden_size = (
        qwen3_vl_cfg.text_config.hidden_size)

    return qwen3_vl_cfg.to_dict()


def _copy_linear_out_features(
    old_linear: torch.nn.Linear,
    new_linear: torch.nn.Linear,
) -> None:
    """Copy first new_out rows from old to new (same in_features)."""
    assert old_linear.in_features == new_linear.in_features
    new_out = new_linear.out_features
    with torch.no_grad():
        new_linear.weight.copy_(old_linear.weight[:new_out, :])
        if (old_linear.bias is not None and new_linear.bias is not None):
            new_linear.bias.copy_(old_linear.bias[:new_out])


def _copy_merger_last_layer_from_template(
    visual: torch.nn.Module,
    orig_visual: torch.nn.Module,
) -> None:
    """Copy only the last linear (linear_fc2) of merger from template to
    current model with slicing when out_features differ. Other merger layers
    are assumed already loaded in step 1.
    """
    if hasattr(visual, 'merger') and hasattr(orig_visual, 'merger'):
        if (hasattr(visual.merger, 'linear_fc2')
                and hasattr(orig_visual.merger, 'linear_fc2')):
            _copy_linear_out_features(
                orig_visual.merger.linear_fc2,
                visual.merger.linear_fc2,
            )
    if (hasattr(visual, 'deepstack_merger_list')
            and hasattr(orig_visual, 'deepstack_merger_list')):
        for old_m, new_m in zip(orig_visual.deepstack_merger_list,
                                visual.deepstack_merger_list):
            if (hasattr(old_m, 'linear_fc2') and hasattr(new_m, 'linear_fc2')):
                _copy_linear_out_features(old_m.linear_fc2, new_m.linear_fc2)


def load_qwen3vl_modular_weights(
    model: torch.nn.Module,
    vlm_path: str,
    vision_encoder_path: Optional[str] = None,
    llm_backbone_path: Optional[str] = None,
    vlm_backbone_id: str = 'qwen3_2b_vl_pt',
    attn_implementation: str = 'flash_attention_2',
    low_cpu_mem_usage: bool = False,
) -> None:
    """Load vision and/or LLM weights from separate checkpoints into a Qwen3VL
    model that was built with merged config (e.g. 2B vision + 0.6B LLM).

    - Vision: loaded from template in step 1 (load_state_dict strict=False).
      Vision merger: when merger output dim matches backbone, all merger
      weights are loaded as-is in step 1; when it does not match (e.g. 2B
      vision + 0.6B LLM), only the last layer (linear_fc2) has a shape
      mismatch and is skipped in step 1, then in step 3 we copy that layer
      from the template with slicing (first new_out rows); other merger
      layers are loaded normally in step 1.
    - LLM: if llm_backbone_path is set, load LLM state dict with key mapping
      model.* -> model.language_model.*, lm_head.* -> lm_head.*.

    Args:
        model: The Qwen3VLForConditionalGeneration instance (already created
            with merged config).
        vlm_path: Main path (processor / default template).
        vision_encoder_path: Path for vision encoder weights.
        llm_backbone_path: Path for LLM weights (e.g. Qwen3-0.6B).
        vlm_backbone_id: Key in VLM_BACKBONE_CONFIGS.
        attn_implementation: Passed when loading template VL.
        low_cpu_mem_usage: Passed when loading template VL.
    """
    from transformers import AutoModelForCausalLM

    from .configs import VLM_BACKBONE_CONFIGS

    if vlm_backbone_id not in VLM_BACKBONE_CONFIGS:
        raise KeyError(
            f'vlm_backbone_id={vlm_backbone_id!r} not in VLM_BACKBONE_CONFIGS')
    vlm_cls = VLM_BACKBONE_CONFIGS[vlm_backbone_id]['model_cls']

    template_path = vision_encoder_path if vision_encoder_path else vlm_path

    # 1) Load full VL from template so we get vision weights (and later merger)
    logger.info(
        'Modular load: loading template VLM from %s for vision weights',
        template_path)
    model_from_template = vlm_cls.from_pretrained(
        template_path,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=low_cpu_mem_usage,
        ignore_mismatched_sizes=True,
    )
    # Load template state into our model. PyTorch load_state_dict(strict=False)
    # still raises on shape mismatch, so we only pass keys that match model
    # shape (vision encoder + non-output merger layers); mismatch keys (merger
    # linear_fc2, language_model, lm_head) are skipped and filled in steps 2–3.
    load_sd = model_from_template.state_dict()
    model_sd = model.state_dict()
    load_sd_filtered = {
        k: v
        for k, v in load_sd.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    skipped = len(load_sd) - len(load_sd_filtered)
    missing, unexpected = model.load_state_dict(load_sd_filtered, strict=False)
    logger.info(
        'Modular load: template state_dict applied (shape-matched), '
        'loaded=%d skipped_shape_mismatch=%d missing=%d unexpected=%d',
        len(load_sd_filtered), skipped, len(missing), len(unexpected))
    del model_from_template
    torch.cuda.empty_cache()

    # 2) If LLM from another path, load and map keys
    if llm_backbone_path:
        logger.info('Modular load: loading LLM from %s', llm_backbone_path)
        llm = AutoModelForCausalLM.from_pretrained(
            llm_backbone_path,
            torch_dtype='auto',
            device_map={'': 'cpu'},
        )
        llm_sd = llm.state_dict()
        mapped = {}
        for k, v in llm_sd.items():
            if k.startswith('model.'):
                mapped['model.language_model.' + k[len('model.'):]] = v
            elif k.startswith('lm_head.'):
                mapped[k] = v
        missing_llm, unexpected_llm = model.load_state_dict(
            mapped, strict=False)
        logger.info(
            'Modular load: LLM state_dict applied, missing=%d unexpected=%d',
            len(missing_llm), len(unexpected_llm))
        del llm
        torch.cuda.empty_cache()

    # 3) Vision merger: when merger output dim matches backbone, all merger
    # weights were already loaded in step 1. When we changed out_hidden_size
    # (llm_backbone_path set), only the last layer (linear_fc2) has different
    # out_features and was skipped in step 1; we re-load the template and
    # copy that layer with slicing; other merger layers remain as loaded.
    if llm_backbone_path:
        visual = getattr(model.model, 'visual', None)
        if visual is not None:
            logger.info(
                'Modular load: re-filling vision merger last layer from '
                'template (sliced).')
            orig_vl = vlm_cls.from_pretrained(
                template_path,
                torch_dtype='auto',
                device_map={'': 'cpu'},
            )
            orig_visual = getattr(orig_vl.model, 'visual', None)
            if orig_visual is not None:
                _copy_merger_last_layer_from_template(visual, orig_visual)
            del orig_vl
            torch.cuda.empty_cache()

    # 4) Tie weights
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
