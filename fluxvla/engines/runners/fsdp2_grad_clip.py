"""FSDP2 gradient norm and clipping compatible with DTensor gradients.

The official LingBot FSDP2 trainer computes the norm from local DTensor
shards, reduces the p-th-power sum over the FSDP group, and then applies one
global coefficient.  ``torch.nn.utils.clip_grad_norm_`` does not guarantee
that contract for mixed replicated/sharded DTensor parameters.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)


def clip_grad_norm_fsdp2(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    process_group=None,
    norm_type: float = 2.0,
    foreach: bool = True,
) -> torch.Tensor:
    """Return the global norm and clip FSDP2 gradients in-place."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0, device=torch.cuda.current_device())
    group = process_group if process_group is not None else dist.group.WORLD

    if math.isinf(norm_type):
        local = _local_max(params)
        dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
        total = local
    else:
        p = float(norm_type)
        local = _local_pth_sum(params, p)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
        total = local ** (1.0 / p)

    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total, foreach=foreach)
    return total


def _local_pth_sum(params, p: float) -> torch.Tensor:
    grads = [
        g.to_local().detach().float() if isinstance(g := param.grad, DTensor)
        else g.detach().float()
        for param in params
    ]
    result = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    grouped = _group_tensors_by_device_and_dtype([grads])
    with torch.no_grad():
        for (device, _), ([device_grads], _) in grouped.items():
            if _has_foreach_support(device_grads, device) or _device_has_foreach_support(device):
                norms = torch._foreach_norm(device_grads, p)
                result += torch.stack(norms).pow_(p).sum().to(result)
            else:
                for grad in device_grads:
                    result += grad.norm(p=p).pow(p).to(result)
    return result


def _local_max(params) -> torch.Tensor:
    result = None
    for param in params:
        grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
        value = grad.detach().float().abs().max()
        result = value if result is None else torch.maximum(result, value)
    return result if result is not None else torch.tensor(0.0, device=torch.cuda.current_device())
