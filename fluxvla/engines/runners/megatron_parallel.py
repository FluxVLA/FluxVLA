"""Small Megatron-compatible parallel wrappers used by the FluxVLA runner.

The wrappers deliberately use ordinary PyTorch collectives so they can be
attached to the existing PI0.5 model without modifying Megatron-LM.  A column
partition gathers its output, preserving the original module contract; this
keeps the bridge compatible with HuggingFace/timm modules while reducing the
linear parameter and optimizer footprint by the TP degree.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


class ColumnParallelLinear(nn.Module):
    """Output-partitioned linear layer with a full-output compatibility API."""

    def __init__(self, source: nn.Linear, group: dist.ProcessGroup):
        super().__init__()
        world = dist.get_world_size(group)
        if source.out_features % world:
            raise ValueError(
                f'Linear output {source.out_features} is not divisible by '
                f'TP={world}')
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.tp_world_size = world
        self.tp_rank = dist.get_rank(group)
        self.group = group
        start = self.tp_rank * (source.out_features // world)
        end = start + source.out_features // world
        self.weight = nn.Parameter(source.weight.detach()[start:end].clone(),
                                   requires_grad=source.weight.requires_grad)
        self.bias = (nn.Parameter(source.bias.detach()[start:end].clone(),
                                   requires_grad=source.bias.requires_grad)
                     if source.bias is not None else None)
        self.weight._tp_sharded = True
        if self.bias is not None:
            self.bias._tp_sharded = True

    def forward(self, x):
        local = torch.nn.functional.linear(x, self.weight, self.bias)
        chunks = dist_nn.all_gather(local, group=self.group)
        return torch.cat(chunks, dim=-1)


def _replace_linear_modules(module: nn.Module, group: dist.ProcessGroup,
                            minimum_out_features: int = 0) -> int:
    """Replace eligible nn.Linear children and return the replacement count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if child.out_features >= minimum_out_features:
                setattr(module, name, ColumnParallelLinear(child, group))
                count += 1
        else:
            count += _replace_linear_modules(child, group, minimum_out_features)
    return count


def apply_tensor_parallel(module: nn.Module, group: Optional[dist.ProcessGroup],
                          minimum_out_features: int = 128) -> int:
    """Apply output TP to a model after its full checkpoint has been loaded."""
    if group is None or dist.get_world_size(group) <= 1:
        return 0
    count = _replace_linear_modules(module, group, minimum_out_features)
    # Parameters which remain replicated (embeddings, norms, biases in modules
    # below the threshold) must still see the same TP-averaged gradient when
    # each TP rank owns a different data shard.
    world = dist.get_world_size(group)
    if world > 1:
        for parameter in module.parameters():
            if getattr(parameter, '_tp_sharded', False):
                continue

            def _average_grad(grad, _group=group, _world=world):
                dist.all_reduce(grad, group=_group)
                return grad / _world

            parameter.register_hook(_average_grad)
    return count


class ShardedOptimizer:
    """Bucketized ZeRO-1 optimizer compatible with Megatron DP groups.

    Gradients are synchronized by DDP first.  Each rank updates only its stable
    parameter subset, then packs each owner's updated parameters into bounded
    flat buffers and broadcasts one buffer per bucket.  This mirrors the
    bucket-level synchronization contract of Megatron's distributed optimizer
    without modifying the read-only Megatron-LM checkout.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 group: Optional[dist.ProcessGroup],
                 bucket_cap_mb: float = 25.0):
        self.optimizer = optimizer
        self.group = group
        self.rank = dist.get_rank(group) if group is not None else 0
        self.world_size = dist.get_world_size(group) if group is not None else 1
        if bucket_cap_mb <= 0:
            raise ValueError('bucket_cap_mb must be positive')
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)
        self._all_params = [p for g in optimizer.param_groups for p in g['params']]
        owned = [p for i, p in enumerate(self._all_params)
                 if i % self.world_size == self.rank]
        owned_ids = {id(p) for p in owned}
        self._owned = owned
        for group_cfg in optimizer.param_groups:
            group_cfg['params'] = [p for p in group_cfg['params'] if id(p) in owned_ids]
        self.param_groups = optimizer.param_groups
        self._buckets = self._build_buckets()

    def _build_buckets(self):
        """Build deterministic owner buckets grouped by dtype and device."""
        buckets = []
        grouped = {}
        for index, parameter in enumerate(self._all_params):
            owner = index % self.world_size
            key = (owner, parameter.device, parameter.dtype)
            grouped.setdefault(key, []).append(parameter)
        # Keep owner/key order deterministic, while avoiding one bucket per
        # parameter when the original parameter list interleaves DP owners.
        for key, parameters in grouped.items():
            current = None
            current_bytes = 0
            for parameter in parameters:
                parameter_bytes = parameter.numel() * parameter.element_size()
                if (current is None
                        or current_bytes + parameter_bytes > self.bucket_cap_bytes):
                    current = {'key': key, 'params': [], 'numel': 0}
                    buckets.append(current)
                    current_bytes = 0
                current['params'].append(parameter)
                current['numel'] += parameter.numel()
                current_bytes += parameter_bytes
        return buckets

    def zero_grad(self, set_to_none: bool = True):
        for p in self._all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        result = self.optimizer.step(closure=closure)
        if self.group is not None and self.world_size > 1:
            for bucket in self._buckets:
                owner, device, dtype = bucket['key']
                flat = torch.empty(bucket['numel'], device=device, dtype=dtype)
                offset = 0
                for parameter in bucket['params']:
                    size = parameter.numel()
                    if owner == self.rank:
                        flat[offset:offset + size].copy_(parameter.data.view(-1))
                    offset += size
                # ``owner`` is a rank local to the DP group.  TP/DP topologies
                # commonly create non-contiguous global ranks (for example
                # groups [0, 2] and [1, 3]); passing the local rank as ``src``
                # therefore targets a rank outside the process group.
                group_src = (dist.get_global_rank(self.group, owner)
                             if hasattr(dist, 'get_global_rank') else owner)
                dist.broadcast(flat, src=group_src, group=self.group)
                offset = 0
                for parameter in bucket['params']:
                    size = parameter.numel()
                    parameter.data.view(-1).copy_(flat[offset:offset + size])
                    offset += size
        return result

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)
