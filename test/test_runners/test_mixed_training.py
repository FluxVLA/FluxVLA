# Copyright 2026 Limx Dynamics
"""Mixed task scheduling, normalized gradients and resumable data cursors."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from mmengine import Config

from fluxvla.collators.vlm_sft_collator import VLMSFTCollator
from fluxvla.engines.runners.mixed_training import (MixedTraining,
                                                    TaskDataStream)


def configuration():
    return Config(
        dict(
            train_dataloader=dict(dataset=dict(type='unused-in-unit-test')),
            runner=dict(
                collator=dict(
                    type='VLMSFTCollator',
                    pad_token_id=0,
                    image_layout='views')),
            vlm_dataloader=dict(
                dataset=dict(type='unused-in-unit-test'),
                collator=dict(
                    type='VLMSFTCollator',
                    pad_token_id=0,
                    image_layout='views'),
                per_device_batch_size=1,
                per_device_num_workers=0,
                grad_accumulation_steps=2)))


class ToyPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.7))
        self.action = torch.nn.Parameter(torch.tensor(0.3))

    def inactive_training_parameters(self, task, *, vision_used=True):
        return [self.action] if task == 'vlm' else []


def runner_for(model):
    commits = []
    return SimpleNamespace(
        vla=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        metric=SimpleNamespace(
            global_step=0, commit=lambda **kw: commits.append(kw)),
        _unwrap_vla=lambda: model)


@pytest.mark.parametrize('task', ['vla', 'vlm'])
def test_accumulation_uses_target_count_not_mean_of_means(task):
    model = ToyPolicy()
    runner = runner_for(model)
    mixed = MixedTraining(configuration(), loss_weights={task: 0.5})
    mixed.active_task = task
    targets = (torch.tensor([1.0]), torch.tensor([2.0, 3.0, 4.0]))
    for index, target in enumerate(targets):
        losses = (model.shared - target).square()
        key = 'num_tokens' if task == 'vlm' else 'num_action_targets'
        output = dict(
            loss=losses.mean(),
            loss_sum=losses.sum(),
            num_correct=torch.tensor(0),
            num_vision_forwards=torch.tensor(0),
            **{key: torch.tensor(target.numel())})
        mixed.backward(runner, output, should_step=index == 1)
    expected = 0.5 * 2 * (0.7 - torch.tensor([1., 2., 3., 4.])).mean()
    torch.testing.assert_close(model.shared.grad, expected)
    assert mixed.step_metrics[f'train/{task}/targets'] == 4
    assert mixed.totals is None


def test_vlm_step_does_not_decay_or_apply_momentum_to_action_parameters():
    model = ToyPolicy()
    runner = runner_for(model)
    # Establish nonzero momentum before switching to a language-only step.
    (model.shared.square() + model.action.square()).backward()
    runner.optimizer.step()
    runner.optimizer.zero_grad(set_to_none=True)
    before = model.action.detach().clone()
    mixed = MixedTraining(configuration())
    mixed.active_task = 'vlm'
    # Emulate an unused action slice materialized as zero by FSDP.
    model.action.grad = torch.zeros_like(model.action)
    loss = model.shared.square()
    mixed.backward(
        runner,
        dict(
            loss=loss,
            loss_sum=loss,
            num_tokens=torch.tensor(1),
            num_vision_forwards=torch.tensor(0),
            num_correct=torch.tensor(0)),
        should_step=True)
    assert model.action.grad is None
    runner.optimizer.step()
    torch.testing.assert_close(model.action, before, atol=0, rtol=0)


def test_no_supervised_targets_fails_before_optimizer_step():
    model = ToyPolicy()
    runner = runner_for(model)
    mixed = MixedTraining(configuration())
    mixed.active_task = 'vlm'
    with pytest.raises(ValueError, match='no valid targets'):
        mixed.backward(
            runner,
            dict(
                loss=model.shared * 0,
                loss_sum=model.shared * 0,
                num_tokens=torch.tensor(0),
                num_vision_forwards=torch.tensor(0),
                num_correct=torch.tensor(0)),
            should_step=True)
    assert model.shared.grad is None


@pytest.mark.parametrize('image_counts', [(0, 0), (1, 0), (0, 1)])
def test_mixed_vision_activity_accumulates_all_microbatches(
        monkeypatch, image_counts):
    model = ToyPolicy()
    runner = runner_for(model)
    mixed = MixedTraining(configuration())
    mixed.active_task = 'vlm'
    decisions = []

    def inactive(task, *, vision_used):
        decisions.append(vision_used)
        return []

    monkeypatch.setattr(model, 'inactive_training_parameters', inactive)
    for index, count in enumerate(image_counts):
        loss = model.shared.square()
        mixed.backward(
            runner,
            dict(
                loss=loss,
                loss_sum=loss,
                num_tokens=torch.tensor(1),
                num_correct=torch.tensor(0),
                num_vision_forwards=torch.tensor(count)),
            should_step=index == 1)
    assert decisions == [any(image_counts)]


def test_missing_vision_activity_fails_before_backward():
    model = ToyPolicy()
    mixed = MixedTraining(configuration())
    mixed.active_task = 'vlm'
    loss = model.shared.square()
    with pytest.raises(ValueError, match='num_vision_forwards'):
        mixed.backward(
            runner_for(model),
            dict(loss=loss, loss_sum=loss, num_tokens=torch.tensor(1)),
            should_step=True)
    assert model.shared.grad is None and mixed.totals is None


def test_task_stream_replays_cursor_without_skipping_or_duplicating_batches():
    kwargs = dict(
        dataset=list(range(11)),
        collator=list,
        batch_size=2,
        num_workers=0,
        seed=42,
        rank=1,
        world_size=2)
    stream = TaskDataStream(**kwargs)
    stream.next()
    state = stream.state_dict()
    expected = [stream.next() for _ in range(5)]
    restored = TaskDataStream(**kwargs)
    restored.load_state_dict(state)
    assert [restored.next() for _ in range(5)] == expected
    with pytest.raises(ValueError, match='count changed'):
        restored.load_state_dict(dict(state, length=state['length'] + 1))


def test_schedule_separate_streams_and_checkpoint_contract():
    model = ToyPolicy()
    runner = runner_for(model)
    runner.grad_accumulation_steps = 1
    runner.per_device_batch_size = 1
    runner.per_device_num_workers = 0
    runner.collator = list
    runner.seed = 42
    runner.dataset_sharding_strategy = 'round_robin'
    runner.optimizer_cfg = dict(type='AdamW', lr=0.01)
    runner.lr_scheduler_cfg = dict(type='constant')
    runner._save_vlm_assets = lambda dataset: None
    runner._grad_sync_context = lambda **kwargs: nullcontext()
    mixed = MixedTraining(configuration())
    language = [
        dict(
            input_ids=torch.tensor([2, i, 1]),
            labels=torch.tensor([-100, i, 1])) for i in range(3, 8)
    ]
    assert isinstance(mixed.vlm_collator, VLMSFTCollator)
    assert mixed.vlm_collator.image_layout == 'views'
    assert mixed.vlm_collator.pad_token_id == 0
    mixed.setup(runner, list(range(10)), vlm_dataset=language)
    observed = []

    def training_step(batch, should_step):
        observed.append(mixed.active_task)
        assert isinstance(batch, dict) == (mixed.active_task == 'vlm')
        return torch.tensor(1.0)

    runner._training_step = training_step
    for step in range(4):
        runner.metric.global_step = step
        mixed.step(runner)
    assert observed == ['vla', 'vla', 'vla', 'vlm', 'vlm']
    assert mixed.task_steps == dict(vla=3, vlm=1)
    state = mixed.checkpoint_state()
    assert state['streams']['vla']['offset'] == 3
    assert state['streams']['vlm']['offset'] == 2
    mixed.load_state_dict(state, global_step=4)
    bad = copy.deepcopy(state)
    bad['contract']['world_size'] = 8
    with pytest.raises(ValueError, match='same task schedule'):
        mixed.load_state_dict(bad, global_step=4)
    with pytest.raises(ValueError, match='counters'):
        mixed.load_state_dict(state, global_step=3)
    for version in (None, 1):
        old = copy.deepcopy(state)
        if version is None:
            old['contract'].pop('version')
        else:
            old['contract']['version'] = version
        with pytest.raises(ValueError, match='contract version changed'):
            mixed.load_state_dict(old, global_step=4)
    mixed.close(runner)
    assert not mixed.streams


@pytest.mark.parametrize('schedule', [(), ('vlm', ), ('vla', 'other')])
def test_invalid_mixed_schedules_rejected(schedule):
    with pytest.raises(ValueError, match='task_schedule'):
        MixedTraining(configuration(), task_schedule=schedule)
