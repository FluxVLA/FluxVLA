# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Nemotron-3 Dense VL text components used by Cosmos3-Edge."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class Nemotron3DenseVLTextConfig(PretrainedConfig):
    """Nemotron-H config after pairing 56 hybrid blocks into 28 MoT layers."""

    model_type = 'nemotron_3_dense_vl_text'

    def __init__(
        self,
        vocab_size: int = 131072,
        tie_word_embeddings: bool = False,
        hidden_size: int = 2048,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        head_dim: int = 128,
        num_key_value_heads: int = 8,
        mlp_hidden_act: str = 'relu2',
        attention_bias: bool = False,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        use_cache: bool = True,
        num_logits_to_keep: int = 1,
        pad_token_id: int = 11,
        bos_token_id: int = 1,
        eos_token_id: int = 11,
        sliding_window: int | None = None,
        max_position_embeddings: int = 131072,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        enable_rope: bool = True,
        rope_scaling: dict | None = None,
        rope_theta: float = 100_000_000.0,
        enable_mrope: bool = True,
        mrope_section: list[int] | None = None,
        use_und_k_norm_for_gen: bool = True,
        torch_dtype: str = 'bfloat16',
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.enable_rope = enable_rope
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.enable_mrope = enable_mrope
        self.mrope_section = ([24, 20, 20]
                              if mrope_section is None else mrope_section)
        self.use_und_k_norm_for_gen = bool(use_und_k_norm_for_gen)
        self.torch_dtype = torch_dtype
        self._attn_implementation = kwargs.pop('_attn_implementation', 'eager')
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def rms_norm_eps(self) -> float:
        return self.layer_norm_epsilon


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class Nemotron3DenseVLRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * normalized).to(input_dtype)


class Nemotron3DenseVLMLP(nn.Module):

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(relu2(self.up_proj(x)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Nemotron partial RoPE; Edge rotates the full attention head."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return (torch.cat((q_embed, q_pass),
                      dim=-1), torch.cat((k_embed, k_pass), dim=-1))


class MultiModalRotaryEmbedding(nn.Module):
    """Nemotron multimodal RoPE used by the Edge packed transformer."""

    def __init__(self, config: Nemotron3DenseVLTextConfig) -> None:
        super().__init__()
        self.mrope_section = config.mrope_section
        if sum(self.mrope_section) != config.head_dim // 2:
            raise ValueError(
                'Nemotron mrope_section must sum to head_dim / 2, got '
                f'{self.mrope_section} for head_dim={config.head_dim}.')
        inv_freq = 1.0 / (
            config.rope_theta**(torch.arange(
                0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer(
            'original_inv_freq', inv_freq.clone(), persistent=False)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            freqs_t[...,
                    slice(offset, length, 3)] = freqs[dim, ...,
                                                      slice(offset, length, 3)]
        return freqs_t

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(
                3, position_ids.shape[0], -1)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(
                'Nemotron mRoPE expects [N], [B,N], or [3,B,N] position '
                f'ids, got {tuple(position_ids.shape)}.')

        inv_freq = self.inv_freq[None, None, :,
                                 None].float().expand(3, position_ids.shape[1],
                                                      -1, 1)
        pos = position_ids[:, :, None, :].float()
        device_type = x.device.type if x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ pos).transpose(2, 3)
            freqs = self._apply_interleaved_mrope(freqs)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


__all__ = [
    'MultiModalRotaryEmbedding',
    'Nemotron3DenseVLMLP',
    'Nemotron3DenseVLRMSNorm',
    'Nemotron3DenseVLTextConfig',
    'apply_rotary_pos_emb_partial',
]
