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
    """ZeRO-1 style optimizer wrapper sharding optimizer state over a DP group.

    Gradients are synchronized by DDP first.  Each rank updates only its stable
    parameter subset, then broadcasts updated parameters to the DP group.  The
    wrapper exposes the standard optimizer API used by FluxVLA schedulers and
    checkpoint code.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 group: Optional[dist.ProcessGroup]):
        self.optimizer = optimizer
        self.group = group
        self.rank = dist.get_rank(group) if group is not None else 0
        self.world_size = dist.get_world_size(group) if group is not None else 1
        self._all_params = [p for g in optimizer.param_groups for p in g['params']]
        owned = [p for i, p in enumerate(self._all_params)
                 if i % self.world_size == self.rank]
        owned_ids = {id(p) for p in owned}
        self._owned = owned
        for group_cfg in optimizer.param_groups:
            group_cfg['params'] = [p for p in group_cfg['params'] if id(p) in owned_ids]
        self.param_groups = optimizer.param_groups

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
            for index, parameter in enumerate(self._all_params):
                owner = index % self.world_size
                dist.broadcast(parameter.data, src=owner, group=self.group)
        return result

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)
