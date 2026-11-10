# Copyright 2026 Limx Dynamics
"""Exercise actual PI05 recipe schedulers through the training runner setup."""

import math
from pathlib import Path

import pytest
import torch
from mmengine import Config

from fluxvla.engines.runners.fsdp_train_runner import FSDPTrainRunner
from fluxvla.optimizers.lr_scheduler_policies import \
    LinearWarmupCosineDecayLRScheduler

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / 'configs/pi05').rglob('*.py'))


@pytest.mark.parametrize(
    'path', CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_pi05_config_optimizer_and_scheduler_setup(path):
    """A tiny model is sufficient to catch stale registry names/config fields.

    Use the real BaseTrainRunner setup and optimizer, not just a syntax check
    or a separately constructed scheduler with hand-written arguments.
    """
    cfg = Config.fromfile(path)
    if 'runner' not in cfg:
        pytest.skip('Inference-only config has no training scheduler')
    runner = FSDPTrainRunner.__new__(FSDPTrainRunner)
    runner.vla = torch.nn.Linear(2, 2)
    runner.optimizer_cfg = dict(cfg.runner.optimizer)
    runner.lr_scheduler_cfg = dict(cfg.runner.lr_scheduler)
    runner.max_steps = cfg.runner.get('max_steps')
    runner.max_epochs = cfg.runner.get('max_epochs')
    runner.global_batch_size = 256
    runner._setup_optimizer_and_scheduler(n_train_examples=1859602)

    assert all(math.isfinite(lr) for lr in runner.lr_scheduler.get_last_lr())
    if 'robodojo' in path.name:
        assert isinstance(runner.lr_scheduler,
                          LinearWarmupCosineDecayLRScheduler)
        assert runner.lr_scheduler.schedule_style == 'openpi'
        multiplier = runner.lr_scheduler.scheduler.lr_lambdas[0]
        assert multiplier(0) == pytest.approx(1 / 1001)
        assert multiplier(1000) == pytest.approx(1)
        assert multiplier(15500) == pytest.approx(0.55)
        assert multiplier(30000) == pytest.approx(0.1)
        assert multiplier(60000) == pytest.approx(0.1)

    loss = runner.vla(torch.ones(1, 2)).square().mean()
    loss.backward()
    runner.optimizer.step()
    runner.lr_scheduler.step(runner)
    assert all(math.isfinite(lr) for lr in runner.lr_scheduler.get_last_lr())
