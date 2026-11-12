"""Shared MCore Column/Row parallel linear adapters."""

from __future__ import annotations

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def _no_init(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def mcore_tp_config(config, dtype: torch.dtype):
    """Create a CPU-initialized config for a checkpoint-backed TP layer."""
    result = copy.copy(config)
    result.params_dtype = dtype
    result.use_cpu_initialization = True
    result.perform_initialization = False
    result.gradient_accumulation_fusion = False
    result.sequence_parallel = False
    return result


class _GatherRowWeightForReference(torch.autograd.Function):
    """Gather a RowParallel weight and return the local wgrad shard."""

    @staticmethod
    def forward(ctx, local_weight: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        shards = [torch.empty_like(local_weight) for _ in range(world_size)]
        dist.all_gather(shards, local_weight.contiguous(), group=group)
        ctx.rank = rank
        ctx.local_width = local_weight.shape[1]
        return torch.cat(shards, dim=1)

    @staticmethod
    def backward(ctx, grad_weight: torch.Tensor):
        start = ctx.rank * ctx.local_width
        return grad_weight[:, start : start + ctx.local_width].contiguous(), None


class _GatherColumnTensorForReference(torch.autograd.Function):
    """Gather a ColumnParallel tensor and return the local gradient shard."""

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        shards = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(shards, local_tensor.contiguous(), group=group)
        ctx.rank = rank
        ctx.local_length = local_tensor.shape[0]
        return torch.cat(shards, dim=0)

    @staticmethod
    def backward(ctx, grad_tensor: torch.Tensor):
        start = ctx.rank * ctx.local_length
        return grad_tensor[start : start + ctx.local_length].contiguous(), None


class MCoreTPLinear(nn.Module):
    """Wrap an MCore Column/RowParallelLinear with an ordinary Linear API.

    ``column=True`` shards output features and gathers the output.  ``column=False``
    shards input features and reduces the output.  This contract lets native HF
    blocks keep their existing tensor shapes while model-specific adapters decide
    which projections are safe to shard.
    """

    def __init__(
        self,
        source: nn.Linear,
        config,
        *,
        column: bool,
        sequence_parallel: bool = False,
        column_gather_output: bool = True,
        row_input_is_parallel: bool = False,
        use_native_linear: bool = False,
        row_full_weight_reference: bool = False,
        column_full_weight_reference: bool = False,
    ) -> None:
        super().__init__()
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
            set_tensor_model_parallel_attributes,
        )
        from megatron.core import parallel_state

        self.in_features = source.in_features
        self.out_features = source.out_features
        self.column = column
        self.sequence_parallel = bool(sequence_parallel)
        self.column_gather_output = bool(column_gather_output)
        self.row_input_is_parallel = bool(row_input_is_parallel)
        self.use_native_linear = bool(use_native_linear)
        self.row_full_weight_reference = bool(row_full_weight_reference)
        self.column_full_weight_reference = bool(column_full_weight_reference)
        if self.column and self.row_full_weight_reference:
            raise ValueError(
                "row_full_weight_reference is valid only for RowParallelLinear"
            )
        if not self.column and self.column_full_weight_reference:
            raise ValueError(
                "column_full_weight_reference is valid only for ColumnParallelLinear"
            )
        self._fluxvla_sp_meta = None
        mconfig = mcore_tp_config(config, source.weight.dtype)
        mconfig.sequence_parallel = self.sequence_parallel
        if column:
            self.linear = ColumnParallelLinear(
                self.in_features,
                self.out_features,
                config=mconfig,
                init_method=_no_init,
                bias=source.bias is not None,
                gather_output=(
                    self.column_gather_output and not self.sequence_parallel
                ),
                disable_grad_reduce=self.sequence_parallel,
            )
        else:
            self.linear = RowParallelLinear(
                self.in_features,
                self.out_features,
                config=mconfig,
                init_method=_no_init,
                bias=source.bias is not None,
                input_is_parallel=(
                    self.row_input_is_parallel or self.sequence_parallel
                ),
                skip_bias_add=False,
            )
        # MCore's CPU-init path only installs TP metadata when it also runs
        # the initializer. Checkpoint-backed adapters intentionally disable
        # initialization, so mark the already-sharded weight explicitly.
        # Optimizer grad-norm and duplicate filtering rely on these fields.
        set_tensor_model_parallel_attributes(
            self.linear.weight,
            is_parallel=True,
            dim=0 if column else 1,
            stride=1,
        )
        rank = parallel_state.get_tensor_model_parallel_rank()
        with torch.no_grad():
            if column:
                local = self.linear.weight.shape[0]
                self.linear.weight.copy_(source.weight.detach()[rank * local : (rank + 1) * local].cpu())
                if source.bias is not None:
                    self.linear.bias.copy_(source.bias.detach()[rank * local : (rank + 1) * local].cpu())
            else:
                local = self.linear.weight.shape[1]
                self.linear.weight.copy_(source.weight.detach()[:, rank * local : (rank + 1) * local].cpu())
                if source.bias is not None:
                    self.linear.bias.copy_(source.bias.detach().cpu())

        # Model-specific distributed adapters may attach ownership metadata to
        # the source parameter before this wrapper replaces the native Linear.
        # Preserve it on the MCore parameter; otherwise EP local-expert
        # gradients can be reduced a second time by DistributedOptimizer.
        for attribute in ("allreduce", "_fluxvla_grad_already_reduced"):
            if hasattr(source.weight, attribute):
                setattr(
                    self.linear.weight,
                    attribute,
                    getattr(source.weight, attribute),
                )
            if source.bias is not None and hasattr(source.bias, attribute):
                setattr(
                    self.linear.bias,
                    attribute,
                    getattr(source.bias, attribute),
                )

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_first = self.sequence_parallel and hidden_states.ndim >= 3
        flattened_batch = False
        if self.sequence_parallel and hidden_states.ndim == 2:
            meta = getattr(self, "_fluxvla_sp_meta", None)
            if meta is not None:
                batch_size, local_length = meta
                hidden_states = hidden_states.reshape(batch_size, local_length, -1)
                hidden_states = hidden_states.transpose(0, 1).contiguous()
                flattened_batch = True
        if batch_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        if self.column_full_weight_reference:
            full_weight = _GatherColumnTensorForReference.apply(
                self.linear.weight,
                self.linear.tp_group,
            )
            full_bias = (
                _GatherColumnTensorForReference.apply(
                    self.linear.bias,
                    self.linear.tp_group,
                )
                if self.linear.bias is not None
                else None
            )
            output = F.linear(hidden_states, full_weight, full_bias)
        elif self.row_full_weight_reference:
            full_weight = _GatherRowWeightForReference.apply(
                self.linear.weight,
                self.linear.tp_group,
            )
            output = F.linear(hidden_states, full_weight, self.linear.bias)
        elif self.use_native_linear and self.column:
            from megatron.core.tensor_parallel.mappings import (
                copy_to_tensor_model_parallel_region,
            )

            local_input = copy_to_tensor_model_parallel_region(
                hidden_states,
                group=self.linear.tp_group,
            )
            output = F.linear(
                local_input,
                self.linear.weight,
                self.linear.bias,
            )
            if self.linear.gather_output:
                from megatron.core.tensor_parallel.mappings import (
                    gather_from_tensor_model_parallel_region,
                )

                output = gather_from_tensor_model_parallel_region(
                    output,
                    group=self.linear.tp_group,
                )
        elif self.use_native_linear:
            from megatron.core.tensor_parallel.mappings import (
                reduce_from_tensor_model_parallel_region,
                scatter_to_tensor_model_parallel_region,
            )

            local_input = hidden_states
            if not self.linear.input_is_parallel:
                local_input = scatter_to_tensor_model_parallel_region(
                    hidden_states,
                    group=self.linear.tp_group,
                )
            local_output = F.linear(local_input, self.linear.weight, None)
            output = reduce_from_tensor_model_parallel_region(
                local_output,
                group=self.linear.tp_group,
            )
            if self.linear.bias is not None:
                output = output + self.linear.bias
        else:
            output = self.linear(hidden_states)
            output = output[0] if isinstance(output, tuple) else output
        if batch_first:
            output = output.transpose(0, 1).contiguous()
        if flattened_batch:
            output = output.transpose(0, 1).contiguous().reshape(-1, output.shape[-1])
        # MCore mapping Functions may return a view.  Ordinary nn.Linear
        # callers are allowed to apply in-place residual updates to their
        # output, so restore that API contract at the adapter boundary.
        return output.clone()


__all__ = ["MCoreTPLinear", "mcore_tp_config"]
