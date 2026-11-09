"""Eager routed-MoE fallback for the FluxVLA-native LingBot model."""

from __future__ import annotations

import os

import torch


def robby_moe_forward(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor | None = None,
    selected_experts: torch.Tensor | None = None,
    gate_weight: torch.Tensor | None = None,
    up_weight: torch.Tensor | None = None,
    down_weight: torch.Tensor | None = None,
    workspace: dict[str, torch.Tensor] | None = None,
    **kwargs,
):
    """Differentiable fallback with the same positional contract as official LingBot.

    ``Qwen2FusedExperts`` also calls this helper through the older keyword
    names.  Normalize those names here so both call sites execute the same
    math and cannot silently bind a weight tensor to ``top_k``.
    """
    gate_weight = gate_weight if gate_weight is not None else kwargs.pop("fc1_1_weight", None)
    up_weight = up_weight if up_weight is not None else kwargs.pop("fc1_2_weight", None)
    down_weight = down_weight if down_weight is not None else kwargs.pop("fc2_weight", None)
    if routing_weights is None or selected_experts is None:
        raise ValueError("routing_weights and selected_experts are required")
    router_weights = routing_weights
    expert_indices = selected_experts
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    # Keep one result per (token, top-k slot).  The official grouped-MoE
    # path applies the route weight before the down projection and gathers
    # slots in their original top-k order.  Accumulating directly into a
    # token buffer with index_add_ changes both the BF16 rounding point and
    # the reduction order (expert order instead of slot order).
    num_tokens, top_k = expert_indices.shape
    fp32_accum = os.environ.get("FLUXVLA_MOE_FP32_ACCUM", "0") == "1"
    route_outputs = torch.zeros(
        (num_tokens, top_k, flat.shape[-1]), dtype=flat.dtype, device=flat.device
    )
    if gate_weight is not None:
        for expert_id in range(gate_weight.shape[0]):
            token_ids, slot_ids = (expert_indices == expert_id).nonzero(as_tuple=True)
            if token_ids.numel():
                x = flat[token_ids]
                if fp32_accum:
                    x_compute = x.float()
                    gate = x_compute @ gate_weight[expert_id].float().transpose(-1, -2)
                    up = x_compute @ up_weight[expert_id].float().transpose(-1, -2)
                else:
                    gate = x @ gate_weight[expert_id].transpose(-1, -2)
                    up = x @ up_weight[expert_id].transpose(-1, -2)
                # Match official FusedMoeExpertFunction: route weighting is
                # part of the down-projection input, not its output.
                intermediate = torch.nn.functional.silu(gate) * up
                intermediate = intermediate * router_weights[token_ids, slot_ids, None]
                if fp32_accum:
                    y = intermediate @ down_weight[expert_id].float().transpose(-1, -2)
                else:
                    y = intermediate @ down_weight[expert_id].transpose(-1, -2)
                route_outputs[token_ids, slot_ids] = y.to(flat.dtype)
    # Official moe_gather accumulates the top-k slots in FP32 and only then
    # stores the result in the model dtype.  Reducing a BF16 buffer directly
    # changes the rounding whenever multiple routes contribute to a token.
    return route_outputs.float().sum(dim=1).to(flat.dtype).reshape_as(hidden_states)
