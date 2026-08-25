import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from fluxvla.engines import build_lr_scheduler_from_cfg
from fluxvla.optimizers.lr_scheduler_policies import (
    BaseLRSchedulerPolicy, LinearWarmupCosineDecayLRScheduler,
    LinearWarmupCosineDecayMinLRScheduler)


class _NoSchedulerPolicy(BaseLRSchedulerPolicy):

    def build_scheduler(self, runner, optimizer):
        return None


def test_scheduler_build_keeps_main_weight_decay_override_api():
    runner = SimpleNamespace(
        vla=nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2)),
        optimizer_cfg={
            'type': 'AdamW',
            'lr': 1e-3,
        })

    optimizer, policy = _NoSchedulerPolicy().build(runner, weight_decay=0.25)

    assert policy.optimizer is optimizer
    assert sorted(group['weight_decay']
                  for group in optimizer.param_groups) == [0.0, 0.25]


def _optimizer(lr=1e-3):
    return AdamW([torch.nn.Parameter(torch.ones(()))], lr=lr)


def _runner(num_training_steps=20, lr=1e-3, vla=None, **optimizer_cfg):
    return SimpleNamespace(
        num_training_steps=num_training_steps,
        vla=vla,
        optimizer_cfg={
            'type': 'AdamW',
            'lr': lr,
            **optimizer_cfg,
        })


def _cosine_factor(step, warmup_steps, total_steps, min_lr_rate=0.0):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cosine * (1.0 - min_lr_rate) + min_lr_rate


def _openpi_cosine_factor(step, warmup_steps, decay_steps, min_lr_rate):
    if step < warmup_steps:
        initial = 1.0 / (warmup_steps + 1)
        return initial + (1.0 - initial) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine


def test_linear_warmup_cosine_lr_curve():
    runner = _runner()
    optimizer = _optimizer()
    policy = LinearWarmupCosineDecayLRScheduler(warmup_ratio=0.2)

    scheduler = policy.build_scheduler(runner, optimizer)

    assert optimizer.param_groups[0]['lr'] == 0.0
    for step in (0, 1, 3, 4, 5, 10, 19, 20):
        assert scheduler.lr_lambdas[0](step) == pytest.approx(
            _cosine_factor(step, 4, 20))


def test_linear_warmup_cosine_supports_steps_and_min_lr():
    runner = _runner()
    optimizer = _optimizer()
    policy = LinearWarmupCosineDecayLRScheduler(warmup_steps=4, min_lr=1e-5)

    scheduler = policy.build_scheduler(runner, optimizer)

    assert optimizer.param_groups[0]['lr'] == 0.0
    for step in (0, 1, 3, 4, 5, 10, 19, 20):
        assert scheduler.lr_lambdas[0](step) == pytest.approx(
            _cosine_factor(step, 4, 20, min_lr_rate=0.01))


def test_cosine_warmup_ratio_and_steps_are_mutually_exclusive():
    with pytest.raises(ValueError, match='Use only one'):
        LinearWarmupCosineDecayLRScheduler(warmup_ratio=0.1, warmup_steps=10)


def test_linear_warmup_cosine_preserves_openpi_curve():
    runner = _runner(num_training_steps=20)
    optimizer = _optimizer()
    policy = LinearWarmupCosineDecayLRScheduler(
        schedule_style='openpi', warmup_steps=4, decay_steps=12, min_lr=1e-4)

    scheduler = policy.build_scheduler(runner, optimizer)

    assert optimizer.param_groups[0]['lr'] == pytest.approx(2e-4)
    for step in (0, 1, 3, 4, 5, 11, 12, 13, 20):
        assert scheduler.lr_lambdas[0](step) == pytest.approx(
            _openpi_cosine_factor(step, 4, 12, min_lr_rate=0.1))


def test_fastwam_cosine_rejects_standard_min_lr_fields():
    with pytest.raises(TypeError, match='Unexpected LR scheduler config'):
        LinearWarmupCosineDecayMinLRScheduler(min_lr_rate=0.1)


@pytest.mark.parametrize(('scheduler_type', 'expected_type', 'kwargs'), [
    ('linear-warmup+cosine-decay', LinearWarmupCosineDecayLRScheduler, {}),
    ('LinearWarmupCosineDecayLRScheduler', LinearWarmupCosineDecayLRScheduler,
     {}),
    ('linear-warmup+cosine-decay-min-lr',
     LinearWarmupCosineDecayMinLRScheduler, {}),
    ('LinearWarmupCosineDecayMinLRScheduler',
     LinearWarmupCosineDecayMinLRScheduler, {}),
])
def test_cosine_scheduler_registry_aliases_remain_available(
        scheduler_type, expected_type, kwargs):
    policy = build_lr_scheduler_from_cfg({'type': scheduler_type, **kwargs})

    assert type(policy) is expected_type


def test_fastwam_cosine_keeps_optimizer_groups_and_lr_curve():
    model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    runner = _runner(
        num_training_steps=16, lr=1e-3, vla=model, weight_decay=0.2)
    policy = LinearWarmupCosineDecayMinLRScheduler(
        warmup_ratio=0.25,
        min_lr_ratio=0.1,
        betas=(0.8, 0.88),
        weight_decay_style='decay_no_decay')

    optimizer, policy = policy.build(runner)

    assert isinstance(optimizer, AdamW)
    assert sorted(group['weight_decay']
                  for group in optimizer.param_groups) == [0.0, 0.2]
    assert all(group['betas'] == (0.8, 0.88)
               for group in optimizer.param_groups)

    reference_parameter = nn.Parameter(torch.ones(()))
    reference_optimizer = AdamW([reference_parameter],
                                lr=1e-3,
                                betas=(0.8, 0.88))
    warmup = LinearLR(
        reference_optimizer, start_factor=0.25, end_factor=1.0, total_iters=4)
    cosine = CosineAnnealingLR(reference_optimizer, T_max=12, eta_min=1e-4)
    reference_scheduler = SequentialLR(
        reference_optimizer, schedulers=[warmup, cosine], milestones=[4])

    assert policy.get_last_lr() == pytest.approx(
        [reference_scheduler.get_last_lr()[0]] * len(optimizer.param_groups))
    for _ in range(16):
        optimizer.step()
        reference_optimizer.step()
        policy.step(runner)
        reference_scheduler.step()
        assert policy.get_last_lr() == pytest.approx(
            [reference_scheduler.get_last_lr()[0]] *
            len(optimizer.param_groups))
