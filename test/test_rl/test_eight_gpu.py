"""Eight-GPU placement, sampling size, and cooperative budget contracts."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.rlinf_registry import register
from fluxvla.rl.train import prepare_environment, validate_frontend_cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    'kind', ['ppo_fluxvla_pi05', 'eval_fluxvla_pi05', 'preflight'])
def test_eight_gpu_config(kind, tiny_assets, monkeypatch):
    import rlinf.config as backend
    prepare_environment()
    register()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(
            config_name=(
                'benchmarks/robotwin/pi05/ppo_8gpu' if kind == 'preflight' else
                f'benchmarks/robotwin/pi05/{kind.split("_")[0]}_8gpu'),
            overrides=[f'actor.model.model_path={tiny_assets[2]}'] + ([
                'runner.max_steps=1', 'runner.max_epochs=1',
                'algorithm.update_epoch=1'
            ] if kind == 'preflight' else []))
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg, evaluation=bool(cfg.runner.only_eval))
    monkeypatch.setattr(backend, 'Cluster', lambda *a, **k: None)
    monkeypatch.setattr(
        backend, 'HybridComponentPlacement',
        lambda *a, **k: SimpleNamespace(get_world_size=lambda name: 4
                                        if name == 'actor' else 2))
    backend.validate_embodied_cfg(cfg)
    assert cfg.cluster.component_placement == {
        'actor': '0-3',
        'rollout': '4-5',
        'env': '6-7'
    }
    assert cfg.env.train.enable_offload and cfg.env.eval.enable_offload
    assert cfg.actor.global_batch_size == 32 * 4 * (200 // 50) == 512
    assert cfg.actor.micro_batch_size == 4
    assert cfg.actor.model.fluxvla.rollout_micro_batch_size == 4
    assert cfg.actor.optim.lr == 1e-6
    assert cfg.experiment.budget_reserve_seconds == 600
    assert not cfg.actor.fsdp_config.enable_gradient_accumulation
    if kind.startswith('eval'):
        assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 150


def test_budget_stop_validates_unscheduled_last_checkpoint(
        tmp_path, monkeypatch):
    from rlinf.runners.embodied_runner import EmbodiedRunner

    from fluxvla.rl.benchmarks.robotwin.runner import (BudgetStop,
                                                       RoboTwinRunner)
    path = tmp_path / 'budget.json'
    path.write_text(json.dumps({'training_deadline': time.time() + 200}))
    monkeypatch.setenv('FLUX_RL_BUDGET_FILE', str(path))
    monkeypatch.setattr(EmbodiedRunner, '_maybe_eval_and_checkpoint',
                        lambda self, step: {})
    runner = object.__new__(RoboTwinRunner)
    runner.cfg = OmegaConf.create({
        'experiment': {
            'split': 'validation'
        },
        'runner': {
            'save_interval': 5,
            'logger': {
                'log_path': str(tmp_path),
                'experiment_name': 'trial'
            }
        }
    })
    runner.artifact_dir.mkdir()
    runner.global_step = 2
    runner.update_rollout_weights = lambda: None
    runner.evaluate = lambda: {'success_once': 0.75}
    calls = []
    runner._save_checkpoint = lambda: calls.append('save')
    runner.metric_logger = SimpleNamespace(log=lambda **kwargs: None)
    runner._maybe_eval_and_checkpoint(1)
    monkeypatch.setattr(EmbodiedRunner, '_log_step_metrics',
                        lambda self: calls.append('log'))
    with pytest.raises(BudgetStop):
        runner._log_step_metrics()
    record = json.loads(
        (runner.artifact_dir / 'checkpoint_selection.json').read_text())
    assert record['best'] == {
        'step': 2,
        'split': 'validation',
        'success_rate': 0.75
    }
    assert calls == ['save', 'log']


def test_actual_accumulated_forward_is_audited(monkeypatch):
    import torch
    from rlinf.workers.actor.embodied_fsdp_actor_worker import \
        EmbodiedFSDPActor

    from fluxvla.rl.workers.actor import AuditedEmbodiedActor

    class Policy(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, **kwargs):
            return {'logprobs': self.weight.expand(2, 1, 1)}

    actor = object.__new__(AuditedEmbodiedActor)
    actor.model = Policy()
    actor.cfg = OmegaConf.create({'experiment': {}})
    actor.optimizer_steps = actor._rollout_optimizer_start = 0
    actor._loss_input_max_drift = 0.0
    monkeypatch.setattr(torch.distributed, 'all_reduce',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(
        EmbodiedFSDPActor, 'train_micro_batch',
        lambda self, *args, **kwargs: self.model(forward_inputs={}))
    sample = {'prev_logprobs': torch.ones(2, 1, 1)}
    actor.train_micro_batch(sample, {}, is_last=False)
    with torch.no_grad():
        actor.model.weight.fill_(2)
    with pytest.raises(FloatingPointError, match='Actual pre-update'):
        actor.train_micro_batch(sample, {}, is_last=False)
    assert not actor.model._forward_hooks
    actor.optimizer_steps = 1
    # Intentional policy changes after a real optimizer update must be allowed.
    actor.train_micro_batch(sample, {}, is_last=True)
