# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""FluxVLA-native GR00T N1.7 Qwen3 backbone."""

from __future__ import annotations

from functools import partial
import logging
from typing import Any, Callable, Dict, Optional, Type

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.modeling_qwen3_vl import \
    Qwen3VLTextDecoderLayer

from fluxvla.engines.utils import VLM_BACKBONES


logger = logging.getLogger(__name__)


@VLM_BACKBONES.register_module()
class GrootN17Qwen3Backbone(torch.nn.Module):
    """Native equivalent of official GR00T N1.7 ``Qwen3Backbone``.

    This class intentionally mirrors the official backbone output contract so
    it can be compared and later swapped into ``GrootN17VLA`` without changing
    the action head.
    """

    def __init__(
        self,
        model_name: str = 'nvidia/Cosmos-Reason2-2B',
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: Optional[Dict[str, Any]] = None,
        qwen3_runtime: str = 'compat_457',
    ) -> None:
        super().__init__()
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

        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                extra_kwargs['attn_implementation'] = 'flash_attention_2'
            except ImportError:
                logger.warning(
                    'flash_attn is not installed. Falling back to sdpa attention.')
                extra_kwargs['attn_implementation'] = 'sdpa'
        if load_bf16:
            extra_kwargs['torch_dtype'] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **(transformers_loading_kwargs or {}),
        ).eval()

        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)
                    logger.debug('Casting trainable parameter %s to fp32', name)

    @property
    def transformer_layer_cls(self) -> Type[torch.nn.Module]:
        return Qwen3VLTextDecoderLayer

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool,
                                 tune_top_llm_layers: int) -> None:
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for param in self.parameters():
            param.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)
        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        if not any(param.requires_grad for param in self.parameters()):
            logger.warning('No backbone trainable parameters found.')

    def set_frozen_modules_to_eval_mode(self) -> None:
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = [
            'input_ids',
            'attention_mask',
            'pixel_values',
            'image_grid_thw',
        ]
        vl_input = {key: vl_input[key] for key in keys_to_use}
        outputs = self.model(**vl_input, output_hidden_states=True)
        backbone_features = outputs.hidden_states[-1]
        image_mask = vl_input['input_ids'] == self.model.config.image_token_id
        attention_mask = vl_input['attention_mask'] == 1
        return BatchFeature(
            data={
                'backbone_features': backbone_features,
                'backbone_attention_mask': attention_mask,
                'image_mask': image_mask,
            })

    def enable_gradient_checkpointing(self) -> None:
        """Enable HuggingFace gradient checkpointing on the inner Qwen3-VL."""
        if not any(param.requires_grad for param in self.parameters()):
            logger.info(
                'Skipping Qwen3-VL gradient checkpointing because the '
                'backbone is frozen.')
            return
        if not hasattr(self.model, 'gradient_checkpointing_enable'):
            return
        gradient_checkpointing_kwargs = {'use_reentrant': False}
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            self.model.gradient_checkpointing_enable()

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for Qwen3-VL text decoder layers."""
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self.transformer_layer_cls},
        )
