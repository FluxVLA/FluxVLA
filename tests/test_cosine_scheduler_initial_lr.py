from types import SimpleNamespace

import pytest
import torch

from fluxvla.optimizers.lr_scheduler_policies import (
    LinearWarmupCosineDecayLRScheduler,
)


def _build_policy(warmup_ratio):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    runner = SimpleNamespace(num_training_steps=100)
    policy = LinearWarmupCosineDecayLRScheduler(
        warmup_ratio=warmup_ratio,
        min_lr=5e-5,
    )
    policy.scheduler = policy.build_scheduler(runner, optimizer)
    return optimizer


def test_zero_warmup_starts_at_peak_lr():
    optimizer = _build_policy(warmup_ratio=0.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_positive_warmup_starts_at_zero_lr():
    optimizer = _build_policy(warmup_ratio=0.1)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
