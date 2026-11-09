"""Activation sequence-parallel boundaries shared by Native MCore models."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class SequenceParallelMLP(nn.Module):
    """Run an already TP-sharded MLP on a sequence shard."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            seen: set[int] = set()
            for parameter in self.module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if getattr(parameter, "tensor_model_parallel", False):
                    continue

                def allreduce_replicated_wgrad(gradient, *, group=tp_group):
                    # The wrapper partitions tokens, but an ordinary nn.Linear
                    # still owns a replicated weight. Sum its partial wgrad
                    # across the TP ranks before DP reduction/optimizer step.
                    dist.all_reduce(gradient, group=group)
                    return gradient

                parameter.register_hook(allreduce_replicated_wgrad)

    def forward(self, hidden_states, *args, **kwargs):
        from megatron.core.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        batch_first = hidden_states.ndim >= 3
        batch_size = hidden_states.shape[0] if batch_first else None
        if batch_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        original_length = hidden_states.shape[0]
        from megatron.core import parallel_state

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        pad_length = (-original_length) % tp_size
        if pad_length:
            # MCore's sequence mapping intentionally rejects uneven shards.
            # Padding here keeps the collective contract explicit and trimming
            # after gather restores the exact native token layout.
            if hidden_states.ndim >= 3:
                hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad_length))
            else:
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
        hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        local_length = hidden_states.shape[0]
        if batch_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        sp_meta = (batch_size, local_length) if batch_first else None
        for child in self.module.modules():
            if hasattr(child, "_fluxvla_sp_meta"):
                child._fluxvla_sp_meta = sp_meta
        output = self.module(hidden_states, *args, **kwargs)
        if isinstance(output, tuple):
            value, *rest = output
        else:
            value, rest = output, []
        def gather_value(item):
            if not isinstance(item, torch.Tensor) or item.ndim == 0:
                return item
            flattened_batch = batch_first and item.shape[0] == batch_size * local_length
            if flattened_batch:
                item = item.reshape(batch_size, local_length, *item.shape[1:])
                item = item.transpose(0, 1).contiguous()
            elif batch_first and item.ndim >= 3:
                item = item.transpose(0, 1).contiguous()
            # The wrapper restores a full hidden tensor before returning to
            # the native block, so downstream computation is duplicated on
            # every TP rank. Its backward must split the replicated output
            # gradient, not reduce-scatter and multiply it by TP size.
            item = gather_from_sequence_parallel_region(
                item,
                tensor_parallel_output_grad=False,
            )
            if pad_length:
                item = item[:original_length]
            if flattened_batch:
                # Native MoE returns router logits as [B*T, E].  Restore
                # that flattened contract after the sequence collective;
                # leaving it 3-D makes the auxiliary balance loss construct a
                # 3-D scatter source and fails before the main loss is formed.
                gathered_length = item.shape[0]
                item = item.transpose(0, 1).contiguous().reshape(
                    batch_size * gathered_length, *item.shape[2:]
                )
            elif batch_first and item.ndim >= 3:
                item = item.transpose(0, 1).contiguous()
            return item

        value = gather_value(value)
        rest = [gather_value(item) for item in rest]
        return (value, *rest) if rest else value


__all__ = ["SequenceParallelMLP"]
