# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import os
from pathlib import Path

import torch

from .group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
from .group_gemm.kernel.moe import expert_histogram, moe_gather, moe_scatter


_backward_trace_call_index = 0


def _trace_backward_outputs(
    grad_output: torch.Tensor,
    grad_gate_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
) -> None:
    """Save opt-in fused-MoE backward outputs without changing autograd."""
    root_value = os.environ.get("FLUXVLA_LINGBOT_FUSED_MOE_BACKWARD_TRACE_DIR")
    if not root_value:
        return

    global _backward_trace_call_index
    _backward_trace_call_index += 1
    calls_per_step = int(os.environ.get("LINGBOT_FUSED_MOE_CALLS_PER_STEP", "36"))
    target_step = int(os.environ.get("LINGBOT_FUSED_MOE_BACKWARD_TRACE_STEP", "1"))
    step = (_backward_trace_call_index - 1) // calls_per_step + 1
    if step != target_step:
        return

    call_in_step = (_backward_trace_call_index - 1) % calls_per_step + 1
    predicted_layer = calls_per_step - call_in_step
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)

    def cpu_float(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu().clone()

    torch.save(
        {
            "backend": "fluxvla",
            "step": step,
            "rank": rank,
            "call_in_step": call_in_step,
            "predicted_layer": predicted_layer,
            "grad_output": cpu_float(grad_output),
            "grad_gate_weight": cpu_float(grad_gate_weight),
            "grad_hidden_states": cpu_float(grad_hidden_states),
        },
        root
        / (
            f"flux_step{step:06d}_rank{rank}_call{call_in_step:03d}_"
            f"layer{predicted_layer:03d}.pt"
        ),
    )


class FusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        # MOE Step 3: dispatch input tokens to the experts
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        # MOE Step 3-1: compute the token num for each expert
        # splits shape (num_experts)
        splits = expert_histogram(expert_index, num_experts)

        # MOE Step 3-2: compute the each token's index in result
        # scatter_index shape (batch_size * sequence_len, topk)
        # TODO(wenyawei): opt it
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)

        # MOE Step 3-3: compute the result, select tokens by scatter_index, and put them together
        # scatter_output shape (batch_size * sequence_len * topk, hidden_size)
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # MOE Step 4: compute linear layer 1-1
        # Not consistent.
        cumsum_t = torch.cumsum(splits, dim=0)
        fc1_1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 6: compute linear layer 1-2
        # fc1_2_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_2_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 5: compute the actication of linear layer 1-1
        # TODO(wenyawei): act function
        # fc1_1_activation shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7: compute final result of linear layer 1
        fc1_activation = fc1_1_activation * fc1_2_output

        # MOE Step 8: compute the the weighted linear layer 1 result
        # MOE Step 8-1: compute scattered_gate_weight, shape is (batch_size * sequence_len * topk)
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        # MOE Step 8-2: multiply activate with scattered_gate_weight
        # fc1_weighted_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # MOE Step 9: compute linear layer 2
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 10: gather the final token result by averate the the topk token results
        expert_output = moe_gather(fc2_output, scatter_index)

        # reshape the output with input shape
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # MOE Step 10
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # MOE Step 9
        # grad_fc1_weighted_output = torch.empty_like(fc1_weighted_output)

        # dgrad
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 8
        # MOE Step 8-2
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        # MOE Step 8-1
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # recompute during backward
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        # MOE Step 6
        # grad_scatter_output_2 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=scatter_output,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 5
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # MOE Step 4
        # grad_scatter_output_1 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=scatter_output,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 3
        # MOE Step 3-3
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)

        # MOE Step 3-2: no grad
        # MOE Step 3-1: no grad

        # reshape the result with input shape
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        _trace_backward_outputs(
            grad_output,
            grad_gate_weight,
            grad_hidden_states,
        )

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )



def fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
):
    """Run LingBot's differentiable grouped-MoE path for Native FSDP.

    Native FSDP owns complete expert tensors. MCore expert parallelism uses its
    own dispatcher and never enters this function.
    """
    del module
    compute_dtype = torch.bfloat16
    hidden_states = hidden_states.to(compute_dtype)
    routing_weights = routing_weights.to(compute_dtype)
    fc1_1_weight = fc1_1_weight.to(compute_dtype)
    fc1_2_weight = fc1_2_weight.to(compute_dtype)
    fc2_weight = fc2_weight.to(compute_dtype)

    from torch.distributed._tensor import DTensor

    if isinstance(hidden_states, DTensor):
        hidden_states = hidden_states.to_local()
    if isinstance(fc1_1_weight, DTensor):
        fc1_1_weight = fc1_1_weight.to_local()
    if isinstance(fc1_2_weight, DTensor):
        fc1_2_weight = fc1_2_weight.to_local()
    if isinstance(fc2_weight, DTensor):
        fc2_weight = fc2_weight.to_local()

    return FusedMoeExpertFunction.apply(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    )


__all__ = ["FusedMoeExpertFunction", "fused_moe_forward"]
