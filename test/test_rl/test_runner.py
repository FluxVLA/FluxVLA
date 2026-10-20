"""Unified RLinf runner audit contracts."""

import json
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


def test_budget_stop_validates_unscheduled_last_checkpoint(
        tmp_path, monkeypatch):
    from rlinf.runners.embodied_runner import EmbodiedRunner

    from fluxvla.engines.runners.rlinf import pilot_runner
    path = tmp_path / 'budget.json'
    path.write_text(json.dumps({'training_deadline': time.time() + 200}))
    monkeypatch.setenv('FLUX_RL_BUDGET_FILE', str(path))
    monkeypatch.setattr(EmbodiedRunner, '_maybe_eval_and_checkpoint',
                        lambda self, step: {})
    runner = object.__new__(pilot_runner.RoboTwinPilotRunner)
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
    with pytest.raises(pilot_runner.PilotBudgetStop):
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

    from fluxvla.engines.runners.rlinf.actor_worker import AuditedEmbodiedActor

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
