"""PI0/PI0.5 transformer layer implementations using Megatron-Core parallel layers.

Replaces vanilla nn.Linear with Megatron ColumnParallelLinear / RowParallelLinear
for tensor-parallel (TP) training.

Architecture overview
---------------------
Each PI0ModelLayerTP wraps ONE layer from the 18-layer shared transformer:

  PI0ModelLayerTP
  ├── vlm_input_norm     : norm for VLM prefix hidden states
  ├── exp_input_norm     : norm for Expert suffix hidden states
  ├── PI0AttentionTP     : joint cross-branch attention (TP-sharded by heads)
  ├── vlm_post_attn_norm : post-attention norm for VLM
  ├── exp_post_attn_norm : post-attention norm for Expert
  ├── vlm_mlp            : PI0MLPBranchTP for VLM
  └── exp_mlp            : PI0MLPBranchTP for Expert

Joint attention (PI0AttentionTP)
--------------------------------
Both VLM and Expert share the same num_heads=8 and head_dim=256.
Q,K,V are projected independently from each branch's hidden states,
then concatenated along the SEQUENCE dimension before attention.
This allows full cross-branch information flow.

TP sharding (heads):
  - Q projections (ColumnParallel): each rank holds num_heads/TP heads.
  - K/V projections: only 1 KV head (GQA). Since 1 < TP, K and V are NOT
    split — each rank replicates the single KV head. This is correct because
    flash_attn handles the GQA case (num_kv_heads < num_q_heads) by expanding
    KV. We use gather_output=True on K/V proj so each TP rank gets the full
    1-head KV output, matching what flash_attn expects for GQA.
  - O projections (RowParallel): each rank receives attn output for its
    local Q heads (num_heads/TP * head_dim columns) and reduces across TP.

MLP TP sharding (standard):
  - gate/up projections: ColumnParallel.
  - down projection:     RowParallel (AllReduce across TP).
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from megatron.core import parallel_state

try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False

from .embedders import AdaRMSNorm, GatedRMSNorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace_layer_backward(
    layer_idx: int,
    name: str,
    tensor: Optional[Tensor],
    *,
    save_value: bool = False,
) -> None:
    """Trace one selected layer's value and gradient during fixed-replay diagnosis."""
    trace_dir = os.environ.get("PI05_BACKWARD_ALIGNMENT_TRACE_DIR")
    target_layer = int(os.environ.get("PI05_BACKWARD_ALIGNMENT_LAYER", "17"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if (
        not trace_dir
        or rank != 0
        or layer_idx != target_layer
        or tensor is None
        or not tensor.requires_grad
    ):
        return

    from pathlib import Path

    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    if save_value:
        torch.save(
            tensor.detach().cpu(),
            path / f"native_layer_{layer_idx:02d}_{name}_value.pt",
        )

    def _save(gradient: Tensor) -> Tensor:
        torch.save(
            gradient.detach().cpu(), path / f"native_layer_{layer_idx:02d}_{name}.pt"
        )
        return gradient

    tensor.register_hook(_save)


def _gated_residual(residual: Tensor, x: Tensor, gate: Optional[Tensor]) -> Tensor:
    """Gemma-style gated residual: residual + gate * x (or residual + x if gate is None)."""
    if gate is None:
        return residual + x
    return residual + gate * x


class _ActivationDTypeBridge(torch.autograd.Function):
    """Cast activations for PI0 consumers while preserving MCore grad dtype."""

    @staticmethod
    def forward(ctx, value: Tensor, dtype: torch.dtype) -> Tensor:
        ctx.source_dtype = value.dtype
        return value.to(dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.to(dtype=ctx.source_dtype), None


def _linear_full_sequence(
    layer: nn.Module,
    hidden_states: Tensor,
) -> Tensor:
    """Run an MCore linear on PI0's batch-first tensor and restore full sequence.

    MCore sequence parallelism shards the first dimension of a ``[S, B, H]``
    activation. PI0.5 keeps its branch tensors as ``[B, S, H]`` because the
    joint-attention mask and flow loss are batch-first. The adapter makes the
    layout conversion explicit and gathers the sequence result before any
    operation that needs both branches or the complete mask. Backward still
    uses MCore's reduce-scatter/gather autograd path, so this is a correctness
    contract rather than a silent TP-only fallback.
    """
    B, L, _ = hidden_states.shape
    sequence_parallel = bool(getattr(layer, "sequence_parallel", False))
    accuracy_contract = os.environ.get("PI05_MCORE_TP_ACCURACY_CONTRACT") == "1"
    if accuracy_contract and not sequence_parallel:
        tp_group = getattr(layer, "tp_group", None)
        if isinstance(layer, RowParallelLinear):
            full_input = gather_from_tensor_model_parallel_region(
                hidden_states,
                group=tp_group,
            )
            full_weight = gather_from_tensor_model_parallel_region(
                layer.weight,
                group=tp_group,
            )
            return F.linear(full_input, full_weight, layer.bias)

        full_weight = gather_from_tensor_model_parallel_region(
            layer.weight.transpose(0, 1).contiguous(),
            group=tp_group,
        ).transpose(0, 1).contiguous()
        full_bias = None
        if layer.bias is not None:
            full_bias = gather_from_tensor_model_parallel_region(
                layer.bias.unsqueeze(0),
                group=tp_group,
            ).squeeze(0)
        output = F.linear(hidden_states, full_weight, full_bias)
        if not layer.gather_output:
            output = scatter_to_tensor_model_parallel_region(output, group=tp_group)
        return output

    if sequence_parallel:
        tp_size = max(1, parallel_state.get_tensor_model_parallel_world_size())
        padded_length = (L + tp_size - 1) // tp_size * tp_size
        if padded_length != L:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length - L))
        sequence_first = hidden_states.transpose(0, 1).contiguous()
        if accuracy_contract:
            tp_group = getattr(layer, "tp_group", None)
            if isinstance(layer, RowParallelLinear):
                full_input = gather_from_tensor_model_parallel_region(
                    sequence_first,
                    group=tp_group,
                )
                full_weight = gather_from_tensor_model_parallel_region(
                    layer.weight,
                    group=tp_group,
                )
                output = F.linear(full_input, full_weight, layer.bias)
                output = scatter_to_sequence_parallel_region(output, group=tp_group)
            else:
                full_weight = gather_from_tensor_model_parallel_region(
                    layer.weight.transpose(0, 1).contiguous(),
                    group=tp_group,
                ).transpose(0, 1).contiguous()
                full_bias = None
                if layer.bias is not None:
                    full_bias = gather_from_tensor_model_parallel_region(
                        layer.bias.unsqueeze(0),
                        group=tp_group,
                    ).squeeze(0)
                output = F.linear(sequence_first, full_weight, full_bias)
                if not layer.gather_output:
                    output = scatter_to_tensor_model_parallel_region(
                        output,
                        group=tp_group,
                    )
            if isinstance(layer, RowParallelLinear):
                output = gather_from_sequence_parallel_region(
                    output,
                    tensor_parallel_output_grad=False,
                    group=tp_group,
                )
            return output.transpose(0, 1).contiguous()[:, :L]
        # MCore's SP custom backward requires its saved input, runtime weight,
        # and dgrad reduce-scatter buffer to have one dtype. Keep FP32 model
        # parameters for optimizer alignment, but derive a BF16 runtime weight
        # when PI0 is under BF16 compute. Autograd propagates its gradient
        # through the cast back to the FP32 parameter.
        activation_compute_dtype = (
            torch.bfloat16
            if getattr(getattr(layer, "config", None), "bf16", False)
            else sequence_first.dtype
        )
        if isinstance(layer, RowParallelLinear):
            # Accuracy contract: accumulate TP partials and reduce-scatter in
            # FP32, then round once after the complete RowParallel result.
            # This avoids one BF16 rounding per TP partial, the first-bad
            # tensor observed against the non-TP FSDP reference.
            linear_compute_dtype = layer.weight.dtype
            layer_input = sequence_first.to(dtype=linear_compute_dtype)
            runtime_weight = layer.weight
        else:
            linear_compute_dtype = activation_compute_dtype
            sequence_first = sequence_first.to(dtype=linear_compute_dtype)
            runtime_weight = layer.weight.to(dtype=linear_compute_dtype)
            layer_input = scatter_to_sequence_parallel_region(
                sequence_first,
                group=getattr(layer, "tp_group", None),
            )
        if isinstance(layer, RowParallelLinear):
            # RowParallelLinear has no runtime-weight argument. Reproduce its
            # native forward with the same MCore linear implementation and SP
            # mapping, substituting only the dtype-compatible derived weight.
            output_parallel = layer._forward_impl(
                input=layer_input,
                weight=runtime_weight,
                bias=None,
                gradient_accumulation_fusion=layer.gradient_accumulation_fusion,
                allreduce_dgrad=False,
                sequence_parallel=False,
                tp_group=None,
                grad_output_buffer=None,
            )
            output = reduce_scatter_to_sequence_parallel_region(
                output_parallel,
                group=getattr(layer, "tp_group", None),
            )
            if layer.bias is not None and not layer.skip_bias_add:
                output = output + layer.bias.to(dtype=output.dtype)
        else:
            output, _ = layer(layer_input, weight=runtime_weight)
        # ColumnParallelLinear returns the full sequence and sharded hidden
        # output. RowParallelLinear performs sequence reduce-scatter, so only
        # that path needs an explicit gather before batch-first consumers.
        if isinstance(layer, RowParallelLinear) and layer.sequence_parallel:
            output = gather_from_sequence_parallel_region(
                output,
                # PI0 restores a full batch-first activation and executes the
                # following norm/residual block redundantly on every TP rank.
                # Its backward must therefore only split the duplicate
                # gradient. Reducing here would multiply it by TP size once
                # per transformer layer.
                tensor_parallel_output_grad=False,
                group=getattr(layer, "tp_group", None),
            )
        output = output.transpose(0, 1).contiguous()[:, :L]
        if isinstance(layer, RowParallelLinear):
            output = output.to(dtype=activation_compute_dtype)
        return _ActivationDTypeBridge.apply(output, hidden_states.dtype)
    output, _ = layer(hidden_states.reshape(B * L, -1))
    return output.reshape(B, L, -1)


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embedding to q and k.

    q, k: [B, num_heads_local, L, head_dim]
    cos, sin: [B, L, head_dim] (precomputed from position_ids, float32)
    """
    # HF Gemma computes the frequencies in FP32, then returns cos/sin in the
    # query dtype. Keep the rotation itself in that dtype as well. Promoting q/k
    # here changes the BF16 rounding contract before the first VLM layer.
    cos = cos[:, None, :, :]   # [B, 1, L, D]
    sin = sin[:, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def _precompute_rope(position_ids: Tensor, head_dim: int,
                     rope_theta: float = 10000.0
                     ) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute RoPE cos/sin from position_ids.

    position_ids: [B, L]
    Returns cos, sin: [B, L, head_dim], and the FP32 frequency vector.
    """
    B, L = position_ids.shape
    device = position_ids.device
    half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, half, dtype=torch.float32, device=device) / half
    ))
    # [B, L, half]
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)   # [B, L, head_dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin, inv_freq


# ---------------------------------------------------------------------------
# PI0AttentionTP: joint cross-branch attention with TP
# ---------------------------------------------------------------------------

class PI0AttentionTP(nn.Module):
    """Tensor-parallel joint attention for PI0/PI0.5.

    Both VLM and Expert branches contribute Q,K,V. The sequences are
    concatenated along the sequence dimension before attention, enabling
    full cross-branch information flow.

    Tensor Parallel sharding
    ------------------------
    Q projections use ColumnParallelLinear (gather_output=False):
      each TP rank holds num_heads/TP query heads.

    K/V projections use ColumnParallelLinear with gather_output=True:
      GQA has only 1 KV head, which cannot be split across TP ranks.
      gather_output=True replicates the single KV head to all ranks.

    O projections use RowParallelLinear (input_is_parallel=True):
      receives (num_heads/TP * head_dim) columns from the TP rank's
      local attention output; AllReduce across TP ranks.

    Parameters
    ----------
    config : PI0TransformerConfig
    rope_theta : float  RoPE base frequency (default 10000 matching Gemma)
    """

    def __init__(self, config, rope_theta: float = 10000.0):
        super().__init__()
        self.config = config
        self.rope_theta = rope_theta

        H = config.num_attention_heads     # 8  (Q heads per branch)
        D = config.head_dim                # 256
        D_vlm = config.hidden_size         # 2048
        D_exp = config.action_expert_hidden_size  # 1024
        KV = config.vlm_num_kv_heads       # 1

        # ---- VLM branch projections ----
        # Q: ColumnParallel, each rank holds H/TP heads
        self.vlm_q = ColumnParallelLinear(
            D_vlm, H * D, config=config, bias=False, gather_output=False,
            init_method=nn.init.xavier_normal_,
        )
        # K,V: gather_output=True — replicate the single KV head to all TP ranks
        self.vlm_k = ColumnParallelLinear(
            D_vlm, KV * D, config=config, bias=False, gather_output=True,
            init_method=nn.init.xavier_normal_,
        )
        self.vlm_v = ColumnParallelLinear(
            D_vlm, KV * D, config=config, bias=False, gather_output=True,
            init_method=nn.init.xavier_normal_,
        )
        # O: RowParallel — input_is_parallel=True (receives TP-sharded output)
        self.vlm_o = RowParallelLinear(
            H * D, D_vlm, config=config, bias=False, input_is_parallel=True,
            skip_bias_add=False, init_method=nn.init.xavier_normal_,
        )

        # ---- Expert branch projections ----
        # Same head_dim=256 and num_heads=8 as VLM, different input dim=1024
        self.exp_q = ColumnParallelLinear(
            D_exp, H * D, config=config, bias=False, gather_output=False,
            init_method=nn.init.xavier_normal_,
        )
        self.exp_k = ColumnParallelLinear(
            D_exp, KV * D, config=config, bias=False, gather_output=True,
            init_method=nn.init.xavier_normal_,
        )
        self.exp_v = ColumnParallelLinear(
            D_exp, KV * D, config=config, bias=False, gather_output=True,
            init_method=nn.init.xavier_normal_,
        )
        # O: maps 2048 (H*D) → 1024 (D_exp); RowParallel input is H*D/TP per rank
        self.exp_o = RowParallelLinear(
            H * D, D_exp, config=config, bias=False, input_is_parallel=True,
            skip_bias_add=False, init_method=nn.init.xavier_normal_,
        )

        self.head_dim = D
        self.num_q_heads = H
        self.num_kv_heads = KV
        # Local Q heads per TP rank (computed lazily to avoid distributed init issues)
        self._local_q_heads: Optional[int] = None
        self.scale = D ** -0.5

    @property
    def local_q_heads(self) -> int:
        if self._local_q_heads is None:
            try:
                tp = parallel_state.get_tensor_model_parallel_world_size()
            except Exception:
                tp = 1
            self._local_q_heads = self.num_q_heads // tp
        return self._local_q_heads

    def _proj_qkv(
        self,
        hs: Tensor,
        q_proj: ColumnParallelLinear,
        k_proj: ColumnParallelLinear,
        v_proj: ColumnParallelLinear,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Project hidden states to Q, K, V and reshape to (L, num_heads, head_dim)."""
        B, L, _ = hs.shape
        D = self.head_dim
        local_h = self.local_q_heads
        kv_h = self.num_kv_heads

        q_full = _linear_full_sequence(q_proj, hs)
        k_full = _linear_full_sequence(k_proj, hs)
        v_full = _linear_full_sequence(v_proj, hs)
        q_flat = q_full.reshape(B * L, -1)             # [BL, local_h*D]
        k_flat = k_full.reshape(B * L, -1)             # [BL, kv_h*D]
        v_flat = v_full.reshape(B * L, -1)             # [BL, kv_h*D]

        # K/V have one GQA head and are gathered in full on every TP rank. Each
        # rank owns different Q heads, so its attention backward contributes a
        # different gradient to the same full K/V tensors. ColumnParallelLinear's
        # gather backward only splits that local gradient; first sum the complete
        # gradient across TP so every K/V weight shard receives all Q-head terms.
        accuracy_contract = os.environ.get("PI05_MCORE_TP_ACCURACY_CONTRACT") == "1"
        if not accuracy_contract:
            k_flat = copy_to_tensor_model_parallel_region(k_flat)
            v_flat = copy_to_tensor_model_parallel_region(v_flat)

        q = q_flat.reshape(B, L, local_h, D)
        k = k_flat.reshape(B, L, kv_h, D)
        v = v_flat.reshape(B, L, kv_h, D)
        return q, k, v

    def forward(
        self,
        vlm_hs: Optional[Tensor],  # [B, L_prefix, D_vlm]
        exp_hs: Optional[Tensor],  # [B, L_suffix, D_exp]
        attention_mask: Tensor,    # [B, 1, L_total, L_total]  additive
        position_ids: Tensor,      # [B, L_total]  long
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Joint attention forward.

        Returns updated (vlm_hs, exp_hs) with same shapes as inputs.
        """
        B = (vlm_hs if vlm_hs is not None else exp_hs).shape[0]
        D = self.head_dim
        trace_dir = os.environ.get("PI05_ALIGNMENT_TRACE_DIR")
        trace_path = None
        if trace_dir and int(os.environ.get("RANK", "0")) == 0:
            from pathlib import Path
            trace_path = Path(trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)

        # ---------- Project Q, K, V from both branches ----------
        parts_q, parts_k, parts_v = [], [], []
        vlm_L = 0
        exp_L = 0

        if vlm_hs is not None and vlm_hs.shape[1] > 0:
            q_vlm, k_vlm, v_vlm = self._proj_qkv(vlm_hs, self.vlm_q, self.vlm_k, self.vlm_v)
            parts_q.append(q_vlm)
            parts_k.append(k_vlm)
            parts_v.append(v_vlm)
            vlm_L = vlm_hs.shape[1]

        if exp_hs is not None and exp_hs.shape[1] > 0:
            q_exp, k_exp, v_exp = self._proj_qkv(exp_hs, self.exp_q, self.exp_k, self.exp_v)
            layer_idx = getattr(self, "alignment_layer_idx", -1)
            _trace_layer_backward(layer_idx, "suffix_q_proj", q_exp, save_value=True)
            _trace_layer_backward(layer_idx, "suffix_k_proj", k_exp, save_value=True)
            _trace_layer_backward(layer_idx, "suffix_v_proj", v_exp, save_value=True)
            parts_q.append(q_exp)
            parts_k.append(k_exp)
            parts_v.append(v_exp)
            exp_L = exp_hs.shape[1]

        trace_first = (trace_path is not None
                       and getattr(self, "alignment_layer_idx", -1) == 0
                       and not getattr(self, "_alignment_trace_done", False))
        if trace_first:
            payload = {"attention_mask": attention_mask[:1].detach().cpu(),
                       "position_ids": position_ids[:1].detach().cpu()}
            labels = ("prefix_q", "prefix_k", "prefix_v", "suffix_q", "suffix_k", "suffix_v")
            values = (*((q_vlm, k_vlm, v_vlm) if vlm_L else (None, None, None)),
                      *((q_exp, k_exp, v_exp) if exp_L else (None, None, None)))
            payload.update({label: value[:1].detach().cpu()
                            for label, value in zip(labels, values) if value is not None})
            torch.save(payload, trace_path / "pmp_layer_00_qkv.pt")

        if not parts_q:
            return vlm_hs, exp_hs

        # Concat along sequence dim: [B, L_total, local_h, D]
        q = torch.cat(parts_q, dim=1)
        k = torch.cat(parts_k, dim=1)
        v = torch.cat(parts_v, dim=1)

        L_total = q.shape[1]
        local_h = self.local_q_heads
        kv_h = self.num_kv_heads

        accuracy_contract = os.environ.get("PI05_MCORE_TP_ACCURACY_CONTRACT") == "1"
        if accuracy_contract:
            # Recreate the non-TP attention GEMM shape. Sharding attention by
            # heads changes the BF16 backward reduction order even when its
            # forward result is equal. The output scatter keeps O projection
            # parameters sharded while its autograd inverse gives every rank
            # the complete attention-output gradient.
            q = gather_from_tensor_model_parallel_region(
                q.reshape(B, L_total, local_h * D),
                group=getattr(self.vlm_q, "tp_group", None),
            ).reshape(B, L_total, self.num_q_heads, D)
            local_h = self.num_q_heads

        # ---------- Apply RoPE ----------
        cos, sin, inv_freq = _precompute_rope(
            position_ids, D, self.rope_theta)
        # Match GemmaRotaryEmbedding.forward(): frequency construction is FP32,
        # while the returned lookup tensors carry the Q/K dtype.
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        # RoPE operates on [B, H, L, D] → reshape temporarily
        q = q.permute(0, 2, 1, 3)   # [B, local_h, L, D]
        k = k.permute(0, 2, 1, 3)   # [B, kv_h, L, D]
        q, k = _apply_rope(q, k, cos, sin)
        layer_idx = getattr(self, "alignment_layer_idx", -1)
        _trace_layer_backward(layer_idx, "q_rope", q, save_value=True)
        _trace_layer_backward(layer_idx, "k_rope", k, save_value=True)
        _trace_layer_backward(
            layer_idx,
            "v_attention_input",
            v,
            save_value=True,
        )
        if trace_first:
            torch.save({"q": q[:1].detach().cpu(), "k": k[:1].detach().cpu(),
                        "cos": cos[:1].detach().cpu(), "sin": sin[:1].detach().cpu(),
                        "inv_freq": inv_freq.detach().cpu(),
                        "inv_freq_dtype": str(inv_freq.dtype),
                        "rope_theta": self.rope_theta,
                        "head_dim": D},
                       trace_path / "pmp_layer_00_rope.pt")
        q = q.permute(0, 2, 1, 3)   # [B, L, local_h, D]
        k = k.permute(0, 2, 1, 3)   # [B, L, kv_h, D]

        # ---------- Attention ----------
        # Use manual matmul (same algorithm as vlash's Attention class) to ensure
        # bit-identical forward results across frameworks.  Contiguous tensors are
        # required so cuBLAS picks the same GEMM kernel as vlash's contiguous path.
        q_t = q.permute(0, 2, 1, 3).contiguous()  # [B, local_h, L, D]
        k_t = k.permute(0, 2, 1, 3).contiguous()  # [B, kv_h,    L, D]
        v_t = v.permute(0, 2, 1, 3).contiguous()  # [B, kv_h,    L, D]
        _trace_layer_backward(
            layer_idx, "q_attention_input", q_t, save_value=True
        )
        # attn_scores: [B, local_h, L, L]
        from .subtask import joint_attention

        attn_out = joint_attention(
            q_t, k_t, v_t, attention_mask, self.scale, vlm_L,
            self.config.knowledge_insulation,
            self.config.attention_implementation,
        )  # [B, local_h, L, D]
        _trace_layer_backward(
            layer_idx,
            "attention_context",
            attn_out,
            save_value=True,
        )
        if trace_first:
            torch.save({"attention_output": attn_out.transpose(1, 2).contiguous()[:1].detach().cpu()},
                       trace_path / "pmp_layer_00_attention.pt")

        # Reshape to [B, L, local_h * D]
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()  # [B, L, local_h, D]
        attn_out = attn_out.reshape(B, L_total, local_h * D)   # [B, L, local_h*D]
        if accuracy_contract:
            attn_out = scatter_to_tensor_model_parallel_region(
                attn_out,
                group=getattr(self.vlm_o, "tp_group", None),
            )

        # ---------- Output projections (RowParallel → AllReduce) ----------
        vlm_out = exp_out = None
        pos = 0

        if vlm_hs is not None and vlm_L > 0:
            slice_vlm = attn_out[:, pos: pos + vlm_L, :]  # [B, L_vlm, local_h*D]
            vlm_out = _linear_full_sequence(self.vlm_o, slice_vlm)
            if trace_first:
                torch.save({"output": vlm_out[:1].detach().cpu()},
                           trace_path / "pmp_layer_00_prefix_o.pt")
            pos += vlm_L

        if exp_hs is not None and exp_L > 0:
            slice_exp = attn_out[:, pos: pos + exp_L, :]
            _trace_layer_backward(
                layer_idx, "suffix_o_input", slice_exp, save_value=True
            )
            exp_out = _linear_full_sequence(self.exp_o, slice_exp)
            if trace_first:
                torch.save({"output": exp_out[:1].detach().cpu()},
                           trace_path / "pmp_layer_00_suffix_o.pt")

        if trace_first:
            self._alignment_trace_done = True

        return vlm_out, exp_out


# ---------------------------------------------------------------------------
# PI0MLPBranchTP: one-branch MLP with TP
# ---------------------------------------------------------------------------

class PI0MLPBranchTP(nn.Module):
    """Single-branch SwiGLU MLP with Megatron TP.

    gate_proj and up_proj: ColumnParallelLinear (each rank holds intermediate/TP cols).
    down_proj:             RowParallelLinear (AllReduce across TP ranks).
    """

    def __init__(self, in_dim: int, intermediate: int, config):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            in_dim, intermediate, config=config, bias=False, gather_output=False,
            init_method=nn.init.xavier_normal_,
        )
        self.up_proj = ColumnParallelLinear(
            in_dim, intermediate, config=config, bias=False, gather_output=False,
            init_method=nn.init.xavier_normal_,
        )
        self.down_proj = RowParallelLinear(
            intermediate, in_dim, config=config, bias=False, input_is_parallel=True,
            skip_bias_add=False, init_method=nn.init.xavier_normal_,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = _linear_full_sequence(self.gate_proj, x)
        up = _linear_full_sequence(self.up_proj, x)
        hidden = F.gelu(gate, approximate="tanh") * up
        return _linear_full_sequence(self.down_proj, hidden)


# ---------------------------------------------------------------------------
# PI0ModelLayerTP: one PI0 transformer layer (VLM + Expert, TP-aware)
# ---------------------------------------------------------------------------

class PI0ModelLayerTP(nn.Module):
    """A single PI0 transformer layer with tensor-parallel support.

    Contains:
      - Separate input LayerNorms for VLM and Expert branches.
        In PI0 mode:   GatedRMSNorm (no conditioning, gate=None).
        In PI0.5 mode: AdaRMSNorm  (state+time conditioning via adaRMS).
      - PI0AttentionTP: joint cross-branch attention.
      - Separate post-attention LayerNorms.
      - PI0MLPBranchTP: independent MLP for each branch.

    Gated residual (Gemma convention):
      output = residual + gate * delta    (gate = None → standard residual)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        D_vlm = config.hidden_size                # 2048
        D_exp = config.action_expert_hidden_size  # 1024

        def _make_norm_vlm():
            return AdaRMSNorm(D_vlm, 0)  # VLM branch has no adaRMS conditioning

        def _make_norm_exp():
            # PI0.5: use adaRMS conditioning with cond_dim = action_expert_hidden_size (1024)
            return AdaRMSNorm(D_exp, config.adarms_cond_dim if config.use_adarms else 0)

        self.vlm_input_norm = _make_norm_vlm()
        self.exp_input_norm = _make_norm_exp()
        self.attn = PI0AttentionTP(config)
        self.vlm_post_attn_norm = _make_norm_vlm()
        self.exp_post_attn_norm = _make_norm_exp()
        self.vlm_mlp = PI0MLPBranchTP(D_vlm, config.vlm_ffn_hidden_size, config)
        self.exp_mlp = PI0MLPBranchTP(D_exp, config.action_expert_ffn_hidden_size, config)

    def forward(
        self,
        vlm_hs: Optional[Tensor],      # [B, L_prefix, D_vlm] or None
        exp_hs: Optional[Tensor],      # [B, L_suffix, D_exp] or None
        attention_mask: Tensor,        # [B, 1, L_total, L_total] additive mask
        position_ids: Tensor,          # [B, L_total] long
        vlm_cond: Optional[Tensor],    # [B, D_cond] adaRMS cond for VLM branch (always None for PI0)
        exp_cond: Optional[Tensor],    # [B, D_cond] adaRMS cond for Expert branch
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """One transformer layer forward pass."""
        layer_idx = getattr(self, "alignment_layer_idx", -1)

        # ---- Pre-attention LayerNorm ----
        vlm_res = vlm_hs
        exp_res = exp_hs

        if vlm_hs is not None:
            vlm_hs, vlm_gate_pre = self.vlm_input_norm(vlm_hs, vlm_cond)
        if exp_hs is not None:
            exp_hs, exp_gate_pre = self.exp_input_norm(exp_hs, exp_cond)
            _trace_layer_backward(layer_idx, "input_norm_output", exp_hs)
            _trace_layer_backward(layer_idx, "input_norm_gate", exp_gate_pre, save_value=True)
        else:
            exp_gate_pre = None

        trace_dir = os.environ.get("PI05_ALIGNMENT_TRACE_DIR")
        trace_first = trace_dir and int(os.environ.get("RANK", "0")) == 0 \
            and getattr(self, "alignment_layer_idx", -1) == 0 \
            and not getattr(self, "_alignment_trace_done", False)
        if trace_first:
            from pathlib import Path
            path = Path(trace_dir)
            path.mkdir(parents=True, exist_ok=True)
            exp_variance = exp_res.float().pow(2).mean(-1, keepdim=True)
            exp_rms = exp_res * torch.rsqrt(exp_variance + self.exp_input_norm.eps)
            exp_modulation = self.exp_input_norm.cond_proj(exp_cond)
            torch.save(
                {
                    "prefix_norm": vlm_hs[:1].detach().cpu(),
                    "suffix_input": exp_res[:1].detach().cpu(),
                    "suffix_cond": exp_cond[:1].detach().cpu(),
                    "suffix_rms": exp_rms[:1].detach().cpu(),
                    "suffix_modulation": exp_modulation[:1].detach().cpu(),
                    "suffix_norm": exp_hs[:1].detach().cpu(),
                    "suffix_gate": exp_gate_pre[:1].detach().cpu(),
                },
                path / "pmp_layer_00_norm.pt",
            )

        # ---- Joint attention ----
        vlm_attn_out, exp_attn_out = self.attn(
            vlm_hs, exp_hs, attention_mask, position_ids
        )
        _trace_layer_backward(layer_idx, "attention_output", exp_attn_out)

        # ---- Gated residual after attention ----
        if vlm_hs is not None and vlm_attn_out is not None:
            vlm_hs = _gated_residual(vlm_res, vlm_attn_out, vlm_gate_pre)
        else:
            vlm_hs = vlm_res
        if exp_hs is not None and exp_attn_out is not None:
            exp_hs = _gated_residual(exp_res, exp_attn_out, exp_gate_pre)
            _trace_layer_backward(layer_idx, "post_attention_residual", exp_hs)
        else:
            exp_hs = exp_res

        # ---- Post-attention LayerNorm + MLP ----
        if vlm_hs is not None:
            # Match FluxVLA's explicit after_first_residual clone. Besides
            # preserving the forward value, this fixes the BF16 backward
            # branch accumulation order at the residual/AdaRMS fork.
            vlm_res = vlm_hs.clone()
            vlm_hs, vlm_gate_post = self.vlm_post_attn_norm(vlm_hs, vlm_cond)
            vlm_mlp_out = self.vlm_mlp(vlm_hs)
            if trace_first:
                torch.save({"output": vlm_mlp_out[:1].detach().cpu()},
                           path / "pmp_layer_00_prefix_mlp.pt")
            vlm_hs = _gated_residual(vlm_res, vlm_mlp_out, vlm_gate_post)

        if exp_hs is not None:
            exp_res = exp_hs.clone()
            exp_hs, exp_gate_post = self.exp_post_attn_norm(exp_hs, exp_cond)
            _trace_layer_backward(layer_idx, "post_attention_norm_output", exp_hs)
            exp_mlp_out = self.exp_mlp(exp_hs)
            _trace_layer_backward(layer_idx, "mlp_output", exp_mlp_out)
            if trace_first:
                torch.save({"output": exp_mlp_out[:1].detach().cpu()},
                           path / "pmp_layer_00_suffix_mlp.pt")
            exp_hs = _gated_residual(exp_res, exp_mlp_out, exp_gate_post)

        if trace_first:
            self._alignment_trace_done = True
        return vlm_hs, exp_hs
