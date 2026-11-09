"""Deterministic per-sample flow-matching targets for parity experiments.

The normal LingBot training path samples noise and time in the model.  This
module is an opt-in contract for cross-backend comparisons: targets depend only
on the sample index and seed, never on rank, batch order, or device RNG state.
"""

from __future__ import annotations

import torch


_MAX_SEED = 2**63 - 1
_INDEX_MULTIPLIER = 1_000_003


def sample_flow_by_index(
    sample_indices: torch.Tensor,
    action_shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate legacy LingBot flow targets independently for each sample.

    ``action_shape`` is ``(batch, horizon, width)``.  CPU generators are used
    intentionally so the result is independent of CUDA RNG implementation and
    distributed rank.  The beta sampler matches the official legacy formula.
    """
    if len(action_shape) != 3:
        raise ValueError(f"action_shape must be [B, H, D], got {action_shape}")
    indices = torch.as_tensor(sample_indices, dtype=torch.long).reshape(-1)
    if indices.numel() != action_shape[0]:
        raise ValueError(
            f"sample_indices length {indices.numel()} does not match batch {action_shape[0]}"
        )
    noises = []
    times = []
    for index in indices.tolist():
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(seed) + int(index) * _INDEX_MULTIPLIER) % _MAX_SEED)
        noises.append(torch.randn(action_shape[1:], generator=generator, dtype=torch.float32))
        gamma1 = torch.rand((), generator=generator, dtype=torch.float32).pow(1.0 / 1.5)
        gamma2 = torch.rand((), generator=generator, dtype=torch.float32).pow(1.0 / 1.0)
        times.append((gamma1 / (gamma1 + gamma2)) * 0.999 + 0.001)
    noise = torch.stack(noises).to(device=device, dtype=dtype)
    time = torch.stack(times).to(device=device, dtype=dtype)
    return noise, time


__all__ = ["sample_flow_by_index"]
