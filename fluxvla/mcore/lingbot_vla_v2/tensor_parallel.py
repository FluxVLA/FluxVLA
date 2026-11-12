"""LingBot-specific module selection for the shared MCore TP primitives."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from ..tensor_parallel import MCoreTPLinear, SequenceParallelMLP


def _fsdp2_ring_average(gathered: list[torch.Tensor]) -> torch.Tensor:
    """Reproduce FSDP2's per-parameter, owner-dependent ring reduction."""
    from torch.distributed.algorithms._comm_hooks.default_hooks import DefaultState

    world_size = len(gathered)
    if world_size == 0:
        raise ValueError("LingBot EP reference wgrad requires at least one rank")
    shape = gathered[0].shape
    if len(shape) < 2:
        raise ValueError("LingBot EP reference wgrad expects a parameter batch")
    if any(gradient.shape != shape for gradient in gathered):
        raise ValueError("LingBot EP reference wgrad tensors must have equal size")
    parameter_shape = shape[1:]
    padded_dim0 = math.ceil(parameter_shape[0] / world_size) * world_size
    padded_shape = (padded_dim0, *parameter_shape[1:])
    parameter_numel = math.prod(padded_shape)

    predivide = DefaultState._get_gradient_predivide_factor(world_size)
    postdivide = world_size / predivide
    shard_numel = parameter_numel // world_size
    reduced_parameters = []
    for parameter_index in range(shape[0]):
        flats = []
        for gradient in gathered:
            value = gradient[parameter_index]
            if value.shape[0] == padded_dim0:
                padded = value.contiguous()
            else:
                padded = torch.zeros(
                    padded_shape,
                    dtype=value.dtype,
                    device=value.device,
                )
                padded[: value.shape[0]].copy_(value)
            flats.append(padded.view(-1))
        reduced_shards = []
        for owner in range(world_size):
            start = owner * shard_numel
            order = [
                (owner + offset) % world_size
                for offset in range(1, world_size + 1)
            ]
            shard = flats[order[0]][start : start + shard_numel].clone()
            if predivide > 1.0:
                shard.div_(predivide)
            for source in order[1:]:
                value = flats[source][start : start + shard_numel]
                shard.add_(value / predivide if predivide > 1.0 else value)
            if postdivide > 1.0:
                shard.div_(postdivide)
            reduced_shards.append(shard)
        reduced_parameters.append(
            torch.cat(reduced_shards).view(padded_shape)[: parameter_shape[0]]
        )
    return torch.stack(reduced_parameters, dim=0)


class _NativeGradientExpertAllGather(torch.autograd.Function):
    """Gather EP weights and reproduce FSDP2's full-DP wgrad reduction tree."""

    @staticmethod
    def forward(
        ctx,
        local_weights: torch.Tensor,
        ep_group,
        dense_dp_group,
        ep_rank: int,
    ) -> torch.Tensor:
        gathered = [
            torch.empty_like(local_weights)
            for _ in range(dist.get_world_size(ep_group))
        ]
        dist.all_gather(gathered, local_weights, group=ep_group)
        ctx.ep_rank = ep_rank
        ctx.local_count = local_weights.shape[0]
        ctx.dense_dp_group = dense_dp_group
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_weights: torch.Tensor):
        dense_dp_size = dist.get_world_size(ctx.dense_dp_group)
        flat = grad_weights.contiguous().view(-1)
        if flat.numel() % dense_dp_size:
            raise RuntimeError(
                "LingBot EP reference wgrad cannot be evenly reduce-scattered "
                f"over dense DP size {dense_dp_size}"
            )
        gathered = [torch.empty_like(grad_weights) for _ in range(dense_dp_size)]
        dist.all_gather(gathered, grad_weights.contiguous(), group=ctx.dense_dp_group)
        reduced = _fsdp2_ring_average(gathered)
        start = ctx.ep_rank * ctx.local_count
        local = reduced[start : start + ctx.local_count].contiguous()
        return local, None, None, None


class LocalExpertTokenMoE(nn.Module):
    """EP expert sharding adapter for LingBot's token-MoE block.

    Router and shared expert remain replicated. Routed expert weights are
    partitioned by contiguous global expert id. The implementation gathers
    token activations across the EP group, evaluates only locally-owned
    experts, and uses autograd-aware collectives to return routed outputs.
    This is deliberately correctness-first; a fused all-to-all dispatcher can
    replace the gather/reduce path without changing the model contract.
    """

    def __init__(self, block: nn.Module, ep_size: int, ep_rank: int) -> None:
        super().__init__()
        experts = block.experts
        if not isinstance(experts, nn.ModuleList):
            raise TypeError("LocalExpertTokenMoE requires eager ModuleList experts")
        if len(experts) % ep_size:
            raise ValueError("num_experts must be divisible by EP size")
        self.num_experts = int(block.num_experts)
        self.top_k = int(block.top_k)
        self.norm_topk_prob = bool(block.norm_topk_prob)
        self.routed_scaling_factor = block.routed_scaling_factor
        self._router_activation = block._router_activation
        self._use_shared_expert_gate = block._use_shared_expert_gate
        self.gate = block.gate
        self.shared_expert = block.shared_expert
        if self._use_shared_expert_gate:
            self.shared_expert_gate = block.shared_expert_gate
        for name in ("e_score_correction_bias", "tokens_per_expert", "last_tokens_per_expert", "avg_topk_sigmoid_score"):
            if hasattr(block, name):
                self.register_buffer(name, getattr(block, name), persistent=name == "e_score_correction_bias")
        local_count = len(experts) // ep_size
        self.expert_start = ep_rank * local_count
        self.expert_end = self.expert_start + local_count
        self.experts = nn.ModuleList(list(experts)[self.expert_start : self.expert_end])
        for parameter in self.experts.parameters():
            # Routed expert ownership is disjoint across EP ranks. Prevent
            # Megatron DDP from reducing these gradients as if they were
            # replicated parameters.
            parameter.allreduce = False
            # The differentiable full-DP weight gather already produces the
            # FSDP2-reduced average. Keep DistOpt sharding/parameter sync, but
            # suppress the otherwise redundant expert-DP grad collective.
            parameter._fluxvla_grad_already_reduced = True
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        from megatron.core import parallel_state
        self.ep_group = parallel_state.get_expert_model_parallel_group()
        self.dense_dp_group = parallel_state.get_data_parallel_group()
        self._trace_tag = ""
        self._trace_calls = 0

    def _gather_expert_weights(self, projection: str) -> torch.Tensor:
        """Materialize global expert weights with an EP-aware autograd path.

        Every source rank forms its expert wgrad with the same full-token eager
        operators as Native. The differentiable gather reduces those source
        contributions back to the local expert owners; MCore's expert-DP
        reduction then combines the owner replicas.
        """
        local = torch.stack(
            [getattr(expert, projection).weight for expert in self.experts],
            dim=0,
        ).contiguous()
        gathered = _NativeGradientExpertAllGather.apply(
            local,
            self.ep_group,
            self.dense_dp_group,
            self.ep_rank,
        )
        trace_root = os.environ.get("FLUXVLA_LINGBOT_EP_WGRAD_TRACE_DIR")
        trace_call = int(os.environ.get("FLUXVLA_LINGBOT_EP_WGRAD_TRACE_CALL", "1"))
        trace_name = f"{self._trace_tag}.{projection}".strip(".")
        trace_pattern = os.environ.get(
            "FLUXVLA_LINGBOT_EP_WGRAD_TRACE_PATTERN",
            r"^$",
        )
        if (
            trace_root
            and self._trace_calls == trace_call
            and re.search(trace_pattern, trace_name)
        ):
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

            def save_pre_collective_gradient(gradient: torch.Tensor) -> torch.Tensor:
                root = Path(trace_root)
                root.mkdir(parents=True, exist_ok=True)
                safe_name = trace_name.replace(".", "_")
                torch.save(
                    {
                        "backend": "mcore",
                        "phase": "pre_expert_wgrad_collective",
                        "rank": rank,
                        "name": trace_name,
                        "grad": gradient.detach().float().cpu(),
                    },
                    root / f"mcore_rank_{rank:02d}_{safe_name}.pt",
                )
                return gradient

            gathered.register_hook(save_pre_collective_gradient)
        return gathered

    def forward(self, hidden_states: torch.Tensor):
        self._trace_calls += 1
        batch, length, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        with torch.amp.autocast(flat.device.type, enabled=False):
            logits = torch.nn.functional.linear(flat.float(), self.gate.weight.float())
        scores = logits.sigmoid() if self._router_activation == "sigmoid" else torch.softmax(logits, dim=-1)
        choice_scores = scores + self.e_score_correction_bias.unsqueeze(0)
        _, selected = torch.topk(choice_scores, self.top_k, dim=-1)
        weights = scores.gather(1, selected)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor != 1.0:
            weights = weights * self.routed_scaling_factor
        weights = weights.to(flat.dtype)

        trace_root = os.environ.get("FLUXVLA_LINGBOT_EP_TRACE_DIR")
        trace_enabled = bool(trace_root) and self._trace_calls == int(
            os.environ.get("FLUXVLA_LINGBOT_EP_TRACE_CALL", "1")
        )
        ep_trace = {
            "tag": self._trace_tag,
            "ep_rank": self.ep_rank,
            "expert_start": self.expert_start,
            "expert_end": self.expert_end,
            "hidden": flat.detach().float().cpu() if trace_enabled else None,
            "router_logits": logits.detach().float().cpu() if trace_enabled else None,
            "selected_experts": selected.detach().cpu() if trace_enabled else None,
            "routing_weights": weights.detach().float().cpu() if trace_enabled else None,
        }

        if self.ep_size > 1:
            # Native eager MoE evaluates every expert on every token and then
            # applies routing with one_hot + einsum. Preserve those GEMM and
            # reduction operators while sharding expert parameters: replicate
            # each source rank's tokens to the EP owners, evaluate only local
            # experts, return their dense outputs, and concatenate them in
            # global expert order on the source rank.
            from megatron.core.tensor_parallel import all_to_all

            tokens_per_source = flat.shape[0]
            local_count = len(self.experts)
            token_splits = [tokens_per_source] * self.ep_size
            send_hidden = (
                flat.detach().unsqueeze(0)
                .expand(self.ep_size, -1, -1)
                .contiguous()
                .reshape(self.ep_size * tokens_per_source, hidden)
            )
            recv_hidden = all_to_all(
                self.ep_group,
                send_hidden,
                token_splits,
                token_splits,
            )
            source_outputs = []
            for source_rank in range(self.ep_size):
                start = source_rank * tokens_per_source
                source_hidden = recv_hidden[start : start + tokens_per_source]
                source_outputs.append(
                    torch.stack(
                        [expert(source_hidden) for expert in self.experts],
                        dim=0,
                    )
                )
            send_outputs = torch.stack(source_outputs, dim=0).reshape(
                self.ep_size * local_count * tokens_per_source,
                hidden,
            )
            expert_output_splits = [local_count * tokens_per_source] * self.ep_size
            returned_outputs = all_to_all(
                self.ep_group,
                send_outputs,
                expert_output_splits,
                expert_output_splits,
            ).reshape(
                self.num_experts,
                tokens_per_source,
                hidden,
            )
            expert_mask = torch.nn.functional.one_hot(
                selected,
                num_classes=self.num_experts,
            ).float()
            dense_weights = (
                expert_mask * weights.unsqueeze(-1).float()
            ).sum(dim=1).to(flat.dtype)
            routed = torch.einsum(
                "etd,te->td",
                returned_outputs,
                dense_weights.detach(),
            )

            # The all-to-all path above is retained as a dispatcher reference,
            # but its collective layout can change the BF16 accumulation order
            # relative to Native eager. Recompute the Native operator directly
            # from gathered weights. This is the correctness path: it keeps
            # local expert parameter ownership while making forward *and*
            # backward use the same stack + one_hot + einsum sequence as the
            # replicated Native implementation. A fused owner-local dispatcher
            # can be enabled later only after an operator-equivalence proof.
            gate_weights = self._gather_expert_weights("gate_proj")
            up_weights = self._gather_expert_weights("up_proj")
            down_weights = self._gather_expert_weights("down_proj")
            if trace_enabled:
                ep_trace["local_gate_proj"] = torch.stack(
                    [expert.gate_proj.weight for expert in self.experts]
                ).detach().float().cpu()
                ep_trace["local_up_proj"] = torch.stack(
                    [expert.up_proj.weight for expert in self.experts]
                ).detach().float().cpu()
                ep_trace["local_down_proj"] = torch.stack(
                    [expert.down_proj.weight for expert in self.experts]
                ).detach().float().cpu()
                ep_trace["global_gate_proj"] = gate_weights.detach().float().cpu()
                ep_trace["global_up_proj"] = up_weights.detach().float().cpu()
                ep_trace["global_down_proj"] = down_weights.detach().float().cpu()
            act_fn = self.experts[0].act_fn
            reference_outputs = torch.stack(
                [
                    torch.nn.functional.linear(
                        act_fn(
                            torch.nn.functional.linear(flat, gate_weights[index])
                        )
                        * torch.nn.functional.linear(flat, up_weights[index]),
                        down_weights[index],
                    )
                    for index in range(self.num_experts)
                ],
                dim=0,
            )
            reference_routed = torch.einsum(
                "etd,te->td",
                reference_outputs,
                dense_weights,
            )
            routed = reference_routed
        else:
            expert_outputs = torch.stack(
                [expert(flat) for expert in self.experts],
                dim=0,
            )
            expert_mask = torch.nn.functional.one_hot(
                selected,
                num_classes=self.num_experts,
            ).float()
            dense_weights = (
                expert_mask * weights.unsqueeze(-1).float()
            ).sum(dim=1).to(flat.dtype)
            routed = torch.einsum("etd,te->td", expert_outputs, dense_weights)
        shared = self.shared_expert(flat)
        if self._use_shared_expert_gate:
            shared = torch.sigmoid(self.shared_expert_gate(flat)) * shared
        if trace_enabled:
            ep_trace["routed"] = routed.detach().float().cpu()
            ep_trace["shared"] = shared.detach().float().cpu()
            root = Path(trace_root)
            root.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            safe_tag = self._trace_tag.replace(".", "_") or "moe"
            torch.save(ep_trace, root / f"mcore_{safe_tag}_rank{rank}.pt")
        return (routed + shared).reshape(batch, length, hidden), logits


def shard_lingbot_experts(model: nn.Module) -> int:
    """Replace eager LingBot MoE blocks with contiguous local expert shards."""
    from megatron.core import parallel_state

    ep_size = parallel_state.get_expert_model_parallel_world_size()
    if ep_size <= 1:
        return 0
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    replaced = 0
    for module_name, parent in model.named_modules():
        for name, child in list(parent.named_children()):
            if hasattr(child, "experts") and isinstance(child.experts, nn.ModuleList):
                replacement = LocalExpertTokenMoE(child, ep_size, ep_rank)
                replacement._trace_tag = f"{module_name}.{name}".strip(".")
                setattr(parent, name, replacement)
                replaced += 1
    return replaced


def _replace_fused_experts(
    mlp: nn.Module,
    tp_rank: int,
    tp_size: int,
    sequence_parallel: bool,
) -> bool:
    experts = getattr(mlp, "experts", None)
    if experts is None or isinstance(experts, nn.ModuleList):
        return False
    if not all(
        isinstance(getattr(experts, name, None), torch.Tensor)
        for name in ("gate_proj", "up_proj", "down_proj")
    ):
        return False
    if sequence_parallel:
        raise NotImplementedError(
            "LingBot SP for native fused MoE requires a token-dispatch-aware "
            "grouped-GEMM adapter; use eager experts for SP."
        )
    # Native LingBot's EPGroupGemm consumes a contiguous local expert block,
    # but its local expert metadata is not exposed through the HF adapter yet.
    # Refuse EP rather than silently producing incorrect grouped-GEMM shapes.
    from megatron.core import parallel_state

    ep_size = parallel_state.get_expert_model_parallel_world_size()
    if ep_size > 1:
        raise NotImplementedError(
            "LingBot EP for native fused MoE requires local expert metadata and "
            "a verified EPGroupGemm checkpoint contract; use eager experts."
        )
    return False


def _replace(
    parent: nn.Module,
    name: str,
    config,
    *,
    column: bool,
    sequence_parallel: bool = False,
    column_gather_output: bool = True,
    row_input_is_parallel: bool = False,
    row_full_weight_reference: bool = False,
    column_full_weight_reference: bool = False,
) -> bool:
    source = getattr(parent, name, None)
    if not isinstance(source, nn.Linear):
        return False
    setattr(
        parent,
        name,
        MCoreTPLinear(
            source,
            config,
            column=column,
            # The native LingBot MoE block flattens [B, T, H] to [B*T, H]
            # before calling each expert.  SequenceParallelMLP owns the token
            # scatter/gather at that boundary; letting MCore linear apply its
            # own sequence mapping would shard the token dimension twice.
            sequence_parallel=False,
            column_gather_output=column_gather_output,
            row_input_is_parallel=row_input_is_parallel,
            use_native_linear=True,
            row_full_weight_reference=row_full_weight_reference,
            column_full_weight_reference=column_full_weight_reference,
        ),
    )
    return True


def convert_lingbot_dense_modules(
    model: nn.Module,
    config,
    *,
    sequence_parallel: bool = False,
    convert_attention: bool = True,
    convert_mlp: bool = True,
    convert_routed_experts: bool = True,
    convert_shared_expert: bool = True,
    convert_mlp_column: bool = True,
    convert_mlp_row: bool = True,
    parity_reference: bool = False,
) -> dict[str, int]:
    """Shard dense Qwen projections after the HF checkpoint is loaded."""
    from megatron.core import parallel_state

    if parallel_state.get_tensor_model_parallel_world_size() <= 1:
        return {
            "attention": 0,
            "mlp": 0,
            "heads": 0,
            "sp_wrapped": 0,
            "sp_skipped_routed_moe": 0,
        }
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    counts = {
        "attention": 0,
        "mlp": 0,
        "heads": 0,
        "sp_wrapped": 0,
        "sp_skipped_routed_moe": 0,
    }
    roots = (
        # Qwen3-VL nests its text decoder below ``language_model``. Keep the
        # direct ``model.layers`` variant for compatible HF checkpoints, but
        # do not let the production backbone silently remain replicated.
        "model.qwenvl_with_expert.qwenvl.model.language_model.layers.",
        "model.qwenvl_with_expert.qwenvl.model.layers.",
        "model.qwenvl_with_expert.qwen_expert.model.layers.",
    )
    wrapped_mlps: set[int] = set()
    for module_name, module in list(model.named_modules()):
        if not any(module_name.startswith(root) for root in roots):
            continue
        for name in ("q_proj", "k_proj", "v_proj"):
            if convert_attention and _replace(
                module,
                name,
                config,
                column=True,
                column_full_weight_reference=parity_reference,
            ):
                counts["attention"] += 1
        if convert_attention and _replace(
            module,
            "o_proj",
            config,
            column=False,
            row_full_weight_reference=parity_reference,
        ):
            counts["attention"] += 1
        mlp = getattr(module, "mlp", None)
        if mlp is not None and not convert_mlp:
            # Attention-only TP is useful for frozen-backbone/action-head
            # runs: it avoids the second set of routed/shared MLP collectives
            # while retaining the memory-saving QKV/O sharding path.
            continue
        if mlp is not None:
            # Qwen's routed MoE stores expert MLPs in a ModuleList (or in a
            # fused 3-D parameter container).  The eager ModuleList path can
            # use the shared TP linear adapter without changing router
            # semantics.  Fused 3-D experts use the grouped-GEMM adapter below.
            fused_converted = _replace_fused_experts(
                mlp,
                tp_rank,
                tp_size,
                sequence_parallel,
            )
            if fused_converted:
                counts["mlp"] += 3
            expert_modules = getattr(mlp, "experts", None)
            if convert_routed_experts and isinstance(expert_modules, nn.ModuleList):
                for expert in expert_modules:
                    if convert_mlp_column:
                        for name in ("gate_proj", "up_proj"):
                            if _replace(
                                expert,
                                name,
                                config,
                                column=True,
                                sequence_parallel=False,
                                column_gather_output=(
                                    parity_reference or not convert_mlp_row
                                ),
                                column_full_weight_reference=parity_reference,
                            ):
                                counts["mlp"] += 1
                    if convert_mlp_row:
                        if _replace(
                            expert,
                            "down_proj",
                            config,
                            column=False,
                            sequence_parallel=False,
                            row_input_is_parallel=(
                                convert_mlp_column and not parity_reference
                            ),
                            row_full_weight_reference=parity_reference,
                        ):
                            counts["mlp"] += 1
            shared_expert = getattr(mlp, "shared_expert", None)
            if convert_shared_expert and shared_expert is not None:
                if convert_mlp_column:
                    for name in ("gate_proj", "up_proj"):
                        if _replace(
                            shared_expert,
                            name,
                            config,
                            column=True,
                            sequence_parallel=sequence_parallel,
                            column_gather_output=(
                                parity_reference or not convert_mlp_row
                            ),
                            column_full_weight_reference=parity_reference,
                        ):
                            counts["mlp"] += 1
                if convert_mlp_row:
                    if _replace(
                        shared_expert,
                        "down_proj",
                        config,
                        column=False,
                        sequence_parallel=False,
                        row_input_is_parallel=(
                            convert_mlp_column and not parity_reference
                        ),
                        row_full_weight_reference=parity_reference,
                    ):
                        counts["mlp"] += 1
            # LingBot's routed MoE block flattens tokens and performs its own
            # router/expert dispatch.  Wrapping this boundary in the generic
            # sequence scatter/gather adapter is invalid: the TP column/row
            # projections then see different token sets on each rank while
            # still executing ordinary hidden-dimension collectives.  Keep
            # MoE token ownership native until a dispatcher-aware SP adapter
            # is available; dense MLPs can use the shared boundary safely.
            has_routed_experts = expert_modules is not None
            if sequence_parallel and not has_routed_experts and id(mlp) not in wrapped_mlps:
                setattr(module, "mlp", SequenceParallelMLP(mlp))
                wrapped_mlps.add(id(mlp))
                counts["sp_wrapped"] += 1
            elif sequence_parallel and has_routed_experts:
                counts["sp_skipped_routed_moe"] += 1
    # LingBot's action_fp32 path calls F.linear with these modules' weight and
    # bias directly.  Wrapping them would bypass MCoreTPLinear.forward and feed
    # a local shard to an operation that expects the full matrix.  Keep action
    # and state projections replicated until the native block contract itself
    # is migrated to MCore tensor-parallel layers.
    return counts


__all__ = ["convert_lingbot_dense_modules"]
