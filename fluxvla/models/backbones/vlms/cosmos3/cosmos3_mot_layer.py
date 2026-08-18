# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# flake8: noqa

from typing import Any, Callable, Type

import torch
from torch import nn
# Qwen3-VL imports
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS, Qwen3VLTextMLP, Qwen3VLTextRMSNorm)
from transformers.models.qwen3_vl.modeling_qwen3_vl import \
    apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb
from transformers.models.qwen3_vl.modeling_qwen3_vl import \
    eager_attention_forward

from fluxvla.models.third_party_models.cosmos3.data.vfm.sequence_packing import (
    FactoredSequencePack, from_und_gen_splits, get_gen_seq, get_und_seq)
from .cosmos3_attention import SplitInfo, two_way_attention

# Torch optimization settings
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096


class Cosmos3TextAttention(nn.Module):
    """
    Dual-pathway packed attention for MoT architectures.
    Implements understanding and generation pathways with separate projections.

    This FluxVLA variant supports the dense Qwen3-VL layers used by Nano/Super
    and the dense Nemotron layers used by Edge.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        rms_norm_cls: Type[nn.Module] = Qwen3VLTextRMSNorm,
        apply_rotary_pos_emb: Callable = qwen3_vl_apply_rotary_pos_emb,
    ):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(
            config, 'layer_types') else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, 'head_dim',
            config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.apply_rotary_pos_emb = apply_rotary_pos_emb

        eps = config.rms_norm_eps

        # Understanding pathway projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias)

        # Understanding pathway QK norm
        if qk_norm_for_text:
            self.q_norm = rms_norm_cls(self.head_dim, eps=eps)
            self.k_norm = rms_norm_cls(self.head_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Generation pathway QK norm
        if qk_norm_for_diffusion:
            self.q_norm_moe_gen = rms_norm_cls(self.head_dim, eps=eps)
            self.k_norm_moe_gen = rms_norm_cls(self.head_dim, eps=eps)
        else:
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        # Edge applies a separate RMSNorm to understanding keys when they are
        # consumed by generation queries. This cannot be folded into k_norm:
        # reasoner queries must continue to see the unmodified key path.
        self.use_und_k_norm_for_gen = bool(
            getattr(config, 'use_und_k_norm_for_gen', False))
        if self.use_und_k_norm_for_gen:
            self.k_norm_und_for_gen = rms_norm_cls(self.head_dim, eps=eps)
        else:
            self.k_norm_und_for_gen = nn.Identity()

        # Generation pathway linear projections
        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias)
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias)
        self.o_proj_moe_gen = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias)

        self.attention_backend = getattr(config, 'packed_attention_backend',
                                         'flash2')

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run standard Qwen3-VL text attention through the reasoner path."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(
            1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {
                'sin': sin,
                'cos': cos,
                'cache_position': cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward_packed(
        self,
        pack: FactoredSequencePack,
        attention_mask: SplitInfo,
        packed_position_embeddings: tuple[FactoredSequencePack,
                                          FactoredSequencePack],
    ) -> FactoredSequencePack:
        """Run dense two-way MoT attention over packed tokens.

        Args:
            pack: Packed sequence with und/gen tokens
            attention_mask: Two-way packed attention metadata.
            packed_position_embeddings: RoPE embeddings (cos, sin)
        """

        q_und_in = self.q_proj(get_und_seq(pack))  # [N_und,num_heads*head_dim]
        q_gen_in = self.q_proj_moe_gen(
            get_gen_seq(pack))  # [N_gen,num_heads*head_dim]

        k_und_in = self.k_proj(
            get_und_seq(pack))  # [N_und,num_kv_heads*head_dim]
        k_gen_in = self.k_proj_moe_gen(
            get_gen_seq(pack))  # [N_gen,num_kv_heads*head_dim]

        v_und_in = self.v_proj(
            get_und_seq(pack))  # [N_und,num_kv_heads*head_dim]
        v_gen_in = self.v_proj_moe_gen(
            get_gen_seq(pack))  # [N_gen,num_kv_heads*head_dim]

        q_und = q_und_in.view(-1, self.num_attention_heads,
                              self.head_dim)  # [N_und,num_heads,head_dim]
        k_und_raw = k_und_in.view(
            -1, self.num_key_value_heads,
            self.head_dim)  # [N_und,num_kv_heads,head_dim]
        v_und = v_und_in.view(-1, self.num_key_value_heads,
                              self.head_dim)  # [N_und,num_kv_heads,head_dim]

        q_gen = q_gen_in.view(-1, self.num_attention_heads,
                              self.head_dim)  # [N_gen,num_heads,head_dim]
        k_gen = k_gen_in.view(-1, self.num_key_value_heads,
                              self.head_dim)  # [N_gen,num_kv_heads,head_dim]
        v_gen = v_gen_in.view(-1, self.num_key_value_heads,
                              self.head_dim)  # [N_gen,num_kv_heads,head_dim]

        q_und = self.q_norm(q_und)  # [N_und,num_heads,head_dim]
        k_und = self.k_norm(k_und_raw)  # [N_und,num_kv_heads,head_dim]
        k_und_for_gen = self.k_norm_und_for_gen(
            k_und_raw)  # [N_und,num_kv_heads,head_dim]

        q_gen = self.q_norm_moe_gen(q_gen)  # [N_gen,num_heads,head_dim]
        k_gen = self.k_norm_moe_gen(k_gen)  # [N_gen,num_kv_heads,head_dim]

        packed_cos = packed_position_embeddings[0]
        packed_sin = packed_position_embeddings[1]

        q_und_, k_und_ = self.apply_rotary_pos_emb(
            q_und,
            k_und,
            get_und_seq(packed_cos),
            get_und_seq(packed_sin),
            unsqueeze_dim=1,
        )  # q_und_: [N_und,num_heads,head_dim], k_und_: [N_und,num_kv_heads,head_dim]
        q_gen_, k_gen_ = self.apply_rotary_pos_emb(
            q_gen,
            k_gen,
            get_gen_seq(packed_cos),
            get_gen_seq(packed_sin),
            unsqueeze_dim=1,
        )  # q_gen_: [N_gen,num_heads,head_dim], k_gen_: [N_gen,num_kv_heads,head_dim]
        packed_key_states_normalized_ = None
        if self.use_und_k_norm_for_gen:
            _, k_und_for_gen_ = self.apply_rotary_pos_emb(
                q_und,
                k_und_for_gen,
                get_und_seq(packed_cos),
                get_und_seq(packed_sin),
                unsqueeze_dim=1,
            )
            packed_key_states_normalized_ = from_und_gen_splits(
                k_und_for_gen_, k_gen_, pack)

        packed_query_states_ = from_und_gen_splits(
            q_und_, q_gen_, pack)  # [N_und+N_gen,num_heads,head_dim]
        packed_key_states_ = from_und_gen_splits(
            k_und_, k_gen_, pack)  # [N_und+N_gen,num_kv_heads,head_dim]
        packed_value_states_ = from_und_gen_splits(
            v_und, v_gen, pack)  # [N_und+N_gen,num_kv_heads,head_dim]

        if not isinstance(attention_mask, SplitInfo):
            raise TypeError(
                'Cosmos3 MoT only supports SplitInfo attention metadata, '
                f'got {type(attention_mask)}.')

        packed_attn_output = two_way_attention(
            packed_query_states_,
            packed_key_states_,
            packed_value_states_,
            backend=self.attention_backend,
            packed_key_states_normalized=packed_key_states_normalized_,
        )

        # Apply projections directly to get final results
        und_seq = self.o_proj(
            get_und_seq(packed_attn_output))  # [N_und,hidden_size]
        gen_seq = self.o_proj_moe_gen(
            get_gen_seq(packed_attn_output))  # [N_gen,hidden_size]
        return from_und_gen_splits(und_seq, gen_seq,
                                   pack)  # [N_und+N_gen,hidden_size]


def _run_unpadded_mlp(
    mlp: nn.Module,
    hidden_states: torch.Tensor,
    active_len: int,
) -> torch.Tensor:
    """Run MLP on real tokens and leave padded tokens untouched."""
    active_out = mlp(hidden_states[:active_len])
    if active_len == hidden_states.shape[0]:
        return active_out
    return torch.cat([active_out, hidden_states[active_len:]], dim=0)


class Cosmos3TextDecoderLayer(nn.Module):
    """
    Unified MoT (Mixture of Transformers) decoder layer.
    Features dual-pathway attention for understanding vs generation.

    Sub-layer types are injectable so Nano/Super retain Qwen3-VL modules while
    Edge supplies Nemotron RMSNorm, ReLU2 MLP, and partial RoPE.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        mlp_cls: Type[nn.Module] = Qwen3VLTextMLP,
        rms_norm_cls: Type[nn.Module] = Qwen3VLTextRMSNorm,
        apply_rotary_pos_emb: Callable = qwen3_vl_apply_rotary_pos_emb,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Cosmos3TextAttention(
            config,
            layer_idx=layer_idx,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            rms_norm_cls=rms_norm_cls,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
        )

        self.mlp = mlp_cls(config)
        self.mlp_moe_gen = mlp_cls(config)

        self.input_layernorm = rms_norm_cls(
            config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = rms_norm_cls(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rms_norm_cls(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = rms_norm_cls(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        packed_sequence: FactoredSequencePack | None = None,
        packed_position_embeddings: tuple[FactoredSequencePack,
                                          FactoredSequencePack] | None = None,
        packed_attention_mask: SplitInfo | None = None,
        **kwargs,
    ) -> torch.Tensor | FactoredSequencePack:
        """Standard Qwen3-VL text decoder path using reasoner weights only."""
        if packed_sequence is not None:
            if packed_position_embeddings is None:
                raise ValueError(
                    'packed_position_embeddings is required for packed '
                    'Cosmos3 forward.')
            return self.forward_packed(
                packed_sequence,
                packed_attention_mask,
                packed_position_embeddings,
            )
        if hidden_states is None or position_embeddings is None:
            raise ValueError(
                'hidden_states and position_embeddings are required for '
                'Cosmos3 text forward.')

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_packed(
        self,
        pack: FactoredSequencePack,
        attention_mask,
        packed_position_embeddings: tuple[FactoredSequencePack,
                                          FactoredSequencePack],
    ) -> FactoredSequencePack:
        """Forward pass with MoT routing over packed und/gen tokens."""
        pack_norm_out = from_und_gen_splits(
            self.input_layernorm(get_und_seq(pack)),  # [N_und,hidden_size]
            self.input_layernorm_moe_gen(
                get_gen_seq(pack)),  # [N_gen,hidden_size]
            pack,
        )  # [N_und+N_gen,hidden_size]

        pack_attn_out = self.self_attn.forward_packed(
            pack_norm_out,
            attention_mask,
            packed_position_embeddings,
        )
        residual_und = get_und_seq(pack) + get_und_seq(pack_attn_out)
        residual_gen = get_gen_seq(pack) + get_gen_seq(pack_attn_out)

        ln_out_und = self.post_attention_layernorm(residual_und)
        ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)

        und_len = pack_attn_out['_num_causal_tokens']
        gen_len = pack_attn_out['_num_full_tokens']
        mlp_out_und = _run_unpadded_mlp(self.mlp, ln_out_und, und_len)
        mlp_out_gen = _run_unpadded_mlp(self.mlp_moe_gen, ln_out_gen, gen_len)

        mlp_out_und_seq = residual_und + mlp_out_und
        mlp_out_gen_seq = residual_gen + mlp_out_gen
        return from_und_gen_splits(mlp_out_und_seq, mlp_out_gen_seq, pack)
