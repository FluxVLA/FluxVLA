# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Construct a native PI0.5 graph from the explicit deployment schema.

This module owns construction only. Prefix, suffix, cached denoising and Euler
inference are the same computation mixin used by the training adapter. No
registry, training import, pretrained download or checkpoint remapping is used.
"""

from __future__ import annotations
from dataclasses import asdict

import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from ...components.condition_gemma import ConditionGemmaModel
from ...components.pi_policy import PI05Computation
from ...components.projector import LinearProjector
from ...components.siglip import SigLIPViTBackbone
from ...config import (DeploymentConfig, GemmaConfig, LinearConfig,
                       PrecisionConfig, _inference, _model, _precision)
from ...errors import ConfigError, ContractError


def _gemma(config: GemmaConfig, precision: PrecisionConfig,
           fp32_layers: tuple[int, ...]) -> ConditionGemmaModel:
    """Map every schema Gemma field; non-schema HF fields are fixed here."""
    model = ConditionGemmaModel(
        attention_bias=config.attention_bias,
        adarms_cond_dim=config.adarms_cond_dim,
        attention_dropout=config.attention_dropout,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        head_dim=config.head_dim,
        hidden_act=config.hidden_activation,
        hidden_activation=config.hidden_activation,
        hidden_size=config.hidden_size,
        initializer_range=0.02,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        model_type='gemma',
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        num_key_value_heads=config.num_key_value_heads,
        pad_token_id=config.pad_token_id,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        torch_dtype=precision.parameter_dtype,
        transformers_version='4.48.1',
        use_adarms=config.use_adarms,
        use_cache=config.use_cache,
        vocab_size=config.vocab_size,
        adarms_fp32=precision.adarms_fp32,
        attention_math_fp32=precision.attention_math_fp32,
        fp32_layers=fp32_layers,
    )
    # Shared inference assigns this same value. Set it before the first call
    # so inference does not change the initially prepared HF configuration.
    model.config._attn_implementation = precision.attention_backend
    return model


def _projector(config: LinearConfig) -> LinearProjector:
    # Schema 1 requires type='linear', bias=True: the shared projector's exact
    # native structure, including its .projector child and checkpoint keys.
    return LinearProjector(in_dim=config.in_dim, out_dim=config.out_dim)


def _validate_config(config: DeploymentConfig) -> None:
    if not isinstance(config, DeploymentConfig):
        raise ConfigError('build_pi05_model requires a DeploymentConfig.')
    if type(config.schema_version) is not int or config.schema_version != 1:
        raise ConfigError('schema_version: PI0.5 requires schema version 1.')
    if config.recipe != 'pi05-v1':
        raise ConfigError('recipe: PI0.5 requires pi05-v1.')
    # Dataclasses can be constructed or replaced outside load_config. Reuse
    # the schema validators rather than silently accepting unsupported flags
    # or maintaining a second architecture/precision validation implementation.
    model = _model(asdict(config.model))
    _inference(asdict(config.inference))
    precision = asdict(config.precision)
    for branch in ('backbone', 'expert'):
        precision[f'{branch}_fp32_layers'] = list(
            precision[f'{branch}_fp32_layers'])
    _precision(precision, model)


class PI05Model(PI05Computation, nn.Module):
    """Native six-input PI0.5 model with FP32 master parameters.

    Construction consumes the caller's CPU RNG normally, in the original
    BaseVLA/PI0 submodule order. It never seeds the RNG or loads weights.
    ``forward`` requires explicit noise and does not modify it. Autocast and
    no-grad execution belong to the reference runner or an explicit caller.
    """

    def __init__(self,
                 config: DeploymentConfig,
                 *,
                 _initialization_device: str = 'cpu') -> None:
        _validate_config(config)
        if _initialization_device not in ('cpu', 'meta'):
            raise ContractError(
                'PI0.5 initialization supports only cpu or meta.')
        if torch.get_default_dtype() != torch.float32:
            raise ContractError(
                'PI0.5 construction requires torch default dtype float32 '
                'for native FP32 initialization; refusing an implicit cast.')
        super().__init__()
        architecture = config.model
        precision = config.precision
        vision = architecture.vision

        # Preserve the original BaseVLA -> PI0 constructor RNG consumption and
        # registration order. None fields must not be replaced with wrappers.
        # The private meta path checks artifact-producer structure without
        # allocating a second model; it is never a reference inference model.
        with torch.device(_initialization_device):
            self.vision_backbone = SigLIPViTBackbone(
                vision=SiglipVisionModel(
                    SiglipVisionConfig(
                        hidden_size=vision.hidden_size,
                        intermediate_size=vision.intermediate_size,
                        num_hidden_layers=vision.num_hidden_layers,
                        num_attention_heads=vision.num_attention_heads,
                        num_channels=vision.num_channels,
                        image_size=vision.image_size,
                        patch_size=vision.patch_size,
                        hidden_act=vision.hidden_activation,
                        layer_norm_eps=vision.layer_norm_eps,
                        attention_dropout=vision.attention_dropout,
                        vision_use_head=vision.vision_use_head,
                    )),
                openpi_stem_fp32=precision.vision_stem_fp32,
                preserve_fp32_residuals=precision.vision_residual_fp32,
            )
            self.llm_backbone = _gemma(architecture.backbone, precision,
                                       precision.backbone_fp32_layers)
            self.vlm_backbone = None
            self.projector = _projector(architecture.projector)
            self.vla_head = None
            self.state_proj = None
            self.action_in_proj = _projector(architecture.action_in_proj)
            self.action_out_proj = _projector(architecture.action_out_proj)
            self.time_mlp_in = _projector(architecture.time_mlp_in)
            self.time_mlp_out = _projector(architecture.time_mlp_out)
            self.action_time_mlp_in = None
            self.action_time_mlp_out = None
            # Keep the unused expert token embedding: it is part of the full
            # native checkpoint and participates in seeded initialization.
            self.llm_expert = _gemma(architecture.expert, precision,
                                     precision.expert_fp32_layers)

        self.proj_width = architecture.proj_width
        self.n_action_steps = architecture.action_horizon
        self.max_action_dim = architecture.action_dim
        self.ori_action_dim = None
        self.num_steps = config.inference.num_steps
        self.enable_mixed_precision_training = (
            precision.autocast_dtype == 'bfloat16')
        self.openpi_fp32_flow = precision.flow_fp32
        self.preserve_fp32_residuals = precision.residual_fp32
        self.attention_implementation = precision.attention_backend
        self.attention_interface = self.get_attention_interface()
        # Native unconditional Gemma RMSNorm allocates BF16 zeros even with
        # torch_dtype='float32'. Normalize only after native construction to
        # preserve seeded initialization while enforcing schema FP32 masters.
        # Checkpoint loading still requires FP32 and never casts artifact data.
        self.float()
        self.eval()

    def forward(self, images: torch.Tensor, img_masks: torch.Tensor,
                lang_tokens: torch.Tensor, lang_masks: torch.Tensor,
                states: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.predict_action(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            states=states,
            noise=noise.clone(),
            prev_actions=None,
            prefix_len=0,
            rtc_config=None,
        )


def build_pi05_model(config: DeploymentConfig) -> PI05Model:
    """Build the real CPU graph in eval mode, without reseeding or loading."""
    return PI05Model(config)
