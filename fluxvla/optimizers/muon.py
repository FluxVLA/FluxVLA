"""FluxVLA-native Muon/AdamW optimizer composition.

The implementation follows the LingBot training contract without importing
the LingBot repository: matrix-like weights use Newton-Schulz Muon updates,
while vectors and embeddings use AdamW.  DTensor parameters keep optimizer
state local to their shard and reconstruct only the gradient needed by the
Newton-Schulz step.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.optim import AdamW, Optimizer


def _full_tensor(value: Tensor) -> Tensor:
    return value.full_tensor() if isinstance(value, DTensor) else value


def _local_tensor(value: Tensor) -> Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _newton_schulz(value: Tensor, steps: int = 5) -> Tensor:
    original_dtype = value.dtype
    matrix = value.to(torch.bfloat16)
    transposed = matrix.shape[-2] > matrix.shape[-1]
    if transposed:
        matrix = matrix.mT
    matrix = matrix / matrix.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    for _ in range(steps):
        gram = matrix @ matrix.mT
        matrix = 3.4445 * matrix - 4.7750 * (gram @ matrix) + 2.0315 * (gram @ gram @ matrix)
    return (matrix.mT if transposed else matrix).to(original_dtype)


def _shard_slice(parameter: DTensor) -> tuple[int, int, int]:
    for mesh_dim, placement in enumerate(parameter.placements):
        if isinstance(placement, Shard):
            size = parameter.device_mesh.size(mesh_dim)
            rank = parameter.device_mesh.get_local_rank(mesh_dim)
            dim = placement.dim
            total = parameter.shape[dim]
            base, remainder = divmod(total, size)
            start = rank * base + min(rank, remainder)
            length = base + int(rank < remainder)
            return dim, start, length
    raise ValueError("Muon expected a sharded DTensor")


class DistributedMuon(Optimizer):
    """Muon optimizer with local FSDP2 state and DTensor-aware updates."""

    def __init__(self, params: Iterable[Tensor], *, lr: float, weight_decay: float = 0.0,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 adjust_lr_fn: str = "match_rms_adamw") -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)

    @staticmethod
    def _adjusted_lr(lr: float, shape: Sequence[int], mode: str | None) -> float:
        if mode is None:
            return lr
        fan_out, fan_in = shape[-2:]
        if mode == "original":
            return lr * math.sqrt(max(1.0, fan_out / fan_in))
        if mode == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
        raise ValueError(f"Unsupported Muon learning-rate adjustment: {mode}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = list(group["params"])
            # FSDP2 creates many same-shaped DTensor weights.  Batch their
            # gather and Newton-Schulz work; one collective per shape avoids
            # the minutes-long per-parameter path and bounds peak memory.
            buckets: dict[tuple, list[Tensor]] = {}
            local_params: list[Tensor] = []
            for parameter in params:
                if isinstance(parameter, DTensor) and any(
                        isinstance(p, Shard) for p in parameter.placements):
                    key = (tuple(parameter.shape), str(parameter.dtype),
                           _shard_slice(parameter)[0])
                    buckets.setdefault(key, []).append(parameter)
                else:
                    local_params.append(parameter)

            for key in sorted(buckets, key=str):
                self._step_dtensor_bucket(buckets[key], group)
            for parameter in local_params:
                self._step_local(parameter, group)
        return loss

    def _step_local(self, parameter: Tensor, group: dict) -> None:
        if parameter.grad is None:
            return
        local_parameter = _local_tensor(parameter)
        local_grad = _local_tensor(parameter.grad)
        state = self.state[parameter]
        momentum_buffer = state.setdefault("momentum_buffer", torch.zeros_like(local_parameter))
        momentum_buffer.lerp_(local_grad, 1.0 - group["momentum"])
        update = (local_grad.lerp(momentum_buffer, group["momentum"])
                  if group["nesterov"] else momentum_buffer.clone())
        orthogonal = _newton_schulz(update, int(group["ns_steps"]))
        adjusted_lr = self._adjusted_lr(float(group["lr"]), parameter.shape,
                                         group["adjust_lr_fn"])
        if group["weight_decay"]:
            local_parameter.mul_(1.0 - float(group["lr"]) * group["weight_decay"])
        local_parameter.add_(orthogonal.to(local_parameter.dtype), alpha=-adjusted_lr)

    def _step_dtensor_bucket(self, params: list[DTensor], group: dict) -> None:
        reference = params[0]
        dim, _, _ = _shard_slice(reference)
        mesh_dim = next(i for i, p in enumerate(reference.placements) if isinstance(p, Shard))
        process_group = reference.device_mesh.get_group(mesh_dim)
        world_size = reference.device_mesh.size(mesh_dim)
        global_size = int(reference.shape[dim])
        updates, has_grad = [], []
        for parameter in params:
            local_parameter = parameter.to_local()
            present = parameter.grad is not None
            local_grad = _local_tensor(parameter.grad) if present else torch.zeros_like(local_parameter)
            state = self.state[parameter]
            momentum_buffer = state.setdefault("momentum_buffer", torch.zeros_like(local_parameter))
            momentum_buffer.lerp_(local_grad, 1.0 - group["momentum"])
            update = (local_grad.lerp(momentum_buffer, group["momentum"])
                      if group["nesterov"] else momentum_buffer.clone())
            updates.append(update)
            has_grad.append(present)
        stacked = torch.stack(updates, dim=0)
        local_size = stacked.shape[dim + 1]
        max_size = (global_size + world_size - 1) // world_size
        if local_size < max_size:
            pad = [0] * (2 * stacked.ndim)
            pad[2 * (stacked.ndim - dim - 2) + 1] = max_size - local_size
            stacked = torch.nn.functional.pad(stacked, pad)
        gathered = [torch.empty_like(stacked) for _ in range(world_size)]
        dist.all_gather(gathered, stacked.contiguous(), group=process_group)
        chunks = []
        remainder = global_size % world_size
        for rank, chunk in enumerate(gathered):
            size = global_size // world_size + int(rank < remainder)
            chunks.append(chunk.narrow(dim + 1, 0, size))
        full = torch.cat(chunks, dim=dim + 1)
        ortho = _newton_schulz(full, int(group["ns_steps"]))
        rank = reference.device_mesh.get_local_rank(mesh_dim)
        start = rank * (global_size // world_size) + min(rank, remainder)
        local_ortho = ortho.narrow(dim + 1, start, local_size)
        adjusted_lr = self._adjusted_lr(float(group["lr"]), reference.shape,
                                         group["adjust_lr_fn"])
        for i, parameter in enumerate(params):
            if not has_grad[i]:
                continue
            local_parameter = parameter.to_local()
            if group["weight_decay"]:
                local_parameter.mul_(1.0 - float(group["lr"]) * group["weight_decay"])
            local_parameter.add_(local_ortho[i].to(local_parameter.dtype), alpha=-adjusted_lr)


class CombinedOptimizer(Optimizer):
    """Expose Muon and AdamW as one optimizer for schedulers/checkpointing."""

    def __init__(self, optimizers: Sequence[Optimizer]) -> None:
        if not optimizers:
            raise ValueError("CombinedOptimizer requires at least one optimizer")
        self.optimizers = list(optimizers)
        # LambdaLR and cosine schedulers access optimizer.defaults['lr'] and
        # mutate the shared parameter-group dictionaries in place.
        self.defaults = dict(self.optimizers[0].defaults)
        self._step_pre_hooks = {}

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    @param_groups.setter
    def param_groups(self, value):
        # Optimizer schedulers assign this attribute during construction.  The
        # inner optimizers own the actual groups, so preserve their identity.
        del value

    def register_step_pre_hook(self, hook):
        # Keep the public Optimizer contract used by gradient clipping and
        # checkpoint wrappers.  Registering on the first optimizer is enough
        # because hooks observe the combined step boundary.
        return self.optimizers[0].register_step_pre_hook(hook)

    @property
    def state(self):
        merged = {}
        for optimizer in self.optimizers:
            merged.update(optimizer.state)
        return merged

    def step(self, closure=None):
        result = None
        for optimizer in self.optimizers:
            result = optimizer.step(closure if optimizer is self.optimizers[0] else None)
        return result

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        for optimizer, payload in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(payload)


def build_muon_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> CombinedOptimizer:
    """Split model parameters according to the official Muon contract."""
    expert_pattern = re.compile(r"(?:^|\.)layers\.\d+\.mlp\.experts\.")
    scale = math.sqrt(float(cfg.get("token_num_experts", 32)) / float(cfg.get("token_top_k", 4)))
    embedding_modules = {
        module_name for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Embedding)
    }

    def force_adamw(name: str) -> bool:
        if any(name.startswith(f"{module}.") for module in embedding_modules):
            return True
        lowered = name.lower()
        return any(pattern in lowered
                   for pattern in ("embed_tokens", "embedding", "lm_head", "output_layer"))

    muon_groups: dict[float, list[Tensor]] = {float(cfg["lr"]): []}
    adamw_groups: dict[float, list[Tensor]] = {float(cfg["lr"]): []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = float(cfg["lr"]) * (scale if expert_pattern.search(name) else 1.0)
        target = (muon_groups
                  if parameter.ndim in (2, 3) and not force_adamw(name)
                  else adamw_groups)
        target.setdefault(lr, []).append(parameter)
    muon = DistributedMuon(
        [{"params": params, "lr": lr} for lr, params in muon_groups.items() if params],
        lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("muon_momentum", 0.95)),
        nesterov=bool(cfg.get("muon_nesterov", True)),
        ns_steps=int(cfg.get("muon_ns_steps", 5)),
        adjust_lr_fn=cfg.get("muon_adjust_lr_fn", "match_rms_adamw"),
    )
    optimizers: list[Optimizer] = [muon]
    adamw_params = [{"params": params, "lr": lr} for lr, params in adamw_groups.items() if params]
    if adamw_params:
        optimizers.append(AdamW(adamw_params, lr=float(cfg["lr"]),
                                betas=tuple(cfg.get("betas", (0.9, 0.95))),
                                eps=float(cfg.get("eps", 1e-8)),
                                weight_decay=float(cfg.get("weight_decay", 0.0)),
                                foreach=bool(cfg.get("foreach", True)),
                                fused=bool(cfg.get("fused", False))))
    return CombinedOptimizer(optimizers)
