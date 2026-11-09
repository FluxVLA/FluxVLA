# Copyright 2026 Limx Dynamics
"""New VLM runner behavior must not change established VLA recipes."""

import math
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner


def _runner(task):
    # Exercise the real runner methods without initializing CUDA/DDP or
    # loading a production-size policy in its constructor.
    runner = DDPTrainRunner.__new__(DDPTrainRunner)
    runner.train_task = task
    runner._mixed_training = None
    runner.keep_lora_trainable_params_fp32 = False
    runner.enable_mixed_precision_training = True
    runner.mixed_precision_dtype = torch.bfloat16
    return runner


@pytest.mark.parametrize('task', ['vla', 'vlm', 'mixed'])
@pytest.mark.parametrize('keep_fp32', [False, True])
@pytest.mark.parametrize('mixed_precision', [False, True])
def test_ddp_fp32_promotion_is_scoped_to_language_tasks(
        task, keep_fp32, mixed_precision):
    runner = _runner(task)
    runner.keep_params_fp32 = keep_fp32
    runner.enable_mixed_precision_training = mixed_precision
    model = torch.nn.Module()
    model.fp32 = torch.nn.Parameter(torch.tensor([1.0001]))
    model.bf16 = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))
    model.frozen = torch.nn.Parameter(
        model.bf16.detach().clone(), requires_grad=False)
    model.register_buffer('scale', torch.tensor(0.25, dtype=torch.bfloat16))
    model.register_buffer('count', torch.tensor(3))
    runner.vla = model
    before = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    parameters = dict(model.named_parameters())

    runner._move_vla_to_device(torch.device('cpu'))

    for name, tensor in model.state_dict().items():
        expected = before[name]
        if expected.is_floating_point():
            if keep_fp32 and task in ('vlm', 'mixed'):
                expected = expected.float()
            elif mixed_precision and not keep_fp32:
                expected = expected.bfloat16()
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    # The pre-device-move LoRA recipe can already have bound an optimizer.
    assert all(parameter is parameters[name]
               for name, parameter in model.named_parameters())
    assert not model.frozen.requires_grad


@pytest.mark.parametrize('task', ['vla', 'vlm', 'mixed'])
@pytest.mark.parametrize('keep_adapters_fp32', [False, True])
def test_ddp_lora_precision_option_keeps_its_existing_semantics(
        task, keep_adapters_fp32):
    runner = _runner(task)
    runner.keep_params_fp32 = False
    runner.keep_lora_trainable_params_fp32 = keep_adapters_fp32
    runner.vla = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(2, 2)),
        LoraConfig(r=1, lora_alpha=1, target_modules=['0']))

    runner._move_vla_to_device(torch.device('cpu'))

    assert any(parameter.requires_grad
               for parameter in runner.vla.parameters())
    for parameter in runner.vla.parameters():
        expected = (
            torch.float32 if keep_adapters_fp32 and parameter.requires_grad
            else torch.bfloat16)
        assert parameter.dtype == expected


class _EpochDataset(Dataset):

    def __init__(self, length):
        self.length = length
        self.epochs = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class _EpochSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas=1):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=0, shuffle=False)
        self.epochs = []

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epochs.append(epoch)


@pytest.mark.parametrize('task', ['vla', 'vlm'])
@pytest.mark.parametrize('steps_per_epoch', [None, 2])
def test_iterator_restart_preserves_legacy_vla_sampler_epochs(
        task, steps_per_epoch):
    runner = _runner(task)
    runner.current_epoch = 4
    runner.steps_per_epoch = steps_per_epoch
    dataset = _EpochDataset(3)
    sampler = _EpochSampler(dataset)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    iterator = None
    observed = []
    for _ in range(4):
        batch, iterator = runner._next_batch(loader, iterator, sampler)
        observed.append(batch.item())

    expected_epoch = 4 if task == 'vlm' and steps_per_epoch else 5
    assert observed == [0, 1, 2, 0]
    assert runner.current_epoch == expected_epoch
    assert sampler.epochs == dataset.epochs == [4, expected_epoch]


@pytest.mark.parametrize('training_mode', ['epoch_based', 'step_based'])
@pytest.mark.parametrize('length,batch_size,accumulation,world_size', [
    (5, 1, 1, 1),
    (6, 1, 2, 1),
    (5, 1, 2, 1),
    (3, 1, 8, 1),
    (16, 4, 2, 2),
    (17, 4, 2, 2),
    (33, 4, 2, 2),
    (3, 4, 2, 2),
])
def test_vlm_loop_counts_each_epoch_once_even_when_accumulation_wraps(
        training_mode, length, batch_size, accumulation, world_size, tmp_path):
    runner = _runner('vlm')
    runner.current_epoch = 0
    runner.training_mode = training_mode
    runner.max_epochs = 3
    runner.grad_accumulation_steps = accumulation
    runner.global_batch_size = batch_size * world_size * accumulation
    dataset = _EpochDataset(length)
    runner.steps_per_epoch = runner._get_steps_per_epoch(dataset)
    assert runner.steps_per_epoch == math.ceil(length /
                                               runner.global_batch_size)
    runner.max_steps = runner.max_epochs * runner.steps_per_epoch
    runner.save_epoch_interval = 1
    runner.save_iter_interval = runner.steps_per_epoch
    runner._loss_accumulator = []
    runner.evaluator = None
    runner.vla = torch.nn.Linear(1, 1)
    runner.metric = SimpleNamespace(
        global_step=0,
        epoch=0,
        get_status=lambda: '',
        push=lambda: '',
        run_dir=str(tmp_path))
    step_epochs, micro_steps, checkpoints = [], [], []

    def commit(**kwargs):
        for key, value in kwargs.items():
            setattr(runner.metric, key, value)
        step_epochs.append(kwargs['epoch'])

    def training_step(batch, should_step):
        micro_steps.append((runner.current_epoch, should_step))
        return batch.float().mean()

    runner.metric.commit = commit
    runner._training_step = training_step
    runner._get_log_lr = lambda: 0.001
    runner._save_and_sync = lambda: checkpoints.append(runner.metric.
                                                       global_step)
    sampler = _EpochSampler(dataset, num_replicas=world_size)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    loop = getattr(runner, f'_run_{training_mode}')
    loop(loader, sampler, dataset)

    assert runner.current_epoch == runner.max_epochs
    assert runner.metric.global_step == runner.max_steps
    assert step_epochs == [
        epoch for epoch in range(3) for _ in range(runner.steps_per_epoch)
    ]
    assert len(micro_steps) == runner.max_steps * accumulation
    assert sum(should_step
               for _, should_step in micro_steps) == runner.max_steps
    assert checkpoints == [
        runner.steps_per_epoch * epoch for epoch in (1, 2, 3)
    ]
    assert sampler.epochs == dataset.epochs
    assert sorted(set(sampler.epochs)) == [0, 1, 2]


@pytest.mark.parametrize('length', [17, 665298])
@pytest.mark.parametrize('epochs', [1, 3])
@pytest.mark.parametrize('world_size', [1, 2])
def test_vlm_epoch_scheduler_uses_full_dataset_budget(length, epochs,
                                                      world_size):
    runner = _runner('vlm')
    runner.max_epochs = epochs
    runner.max_steps = None
    runner.global_batch_size = 4 * world_size * 2
    runner.vla = torch.nn.Linear(1, 1)
    runner.optimizer_cfg = dict(type='AdamW', lr=2e-5, weight_decay=0.01)
    runner.lr_scheduler_cfg = dict(
        type='linear-warmup+cosine-decay', warmup_ratio=0.03)

    runner._setup_optimizer_and_scheduler(length)

    expected_steps = math.ceil(length / runner.global_batch_size) * epochs
    assert runner.num_training_steps == expected_steps
    assert runner.num_training_steps == (
        runner._get_steps_per_epoch(_EpochDataset(length)) * epochs)
    multiplier = runner.lr_scheduler.scheduler.lr_lambdas[0]
    assert multiplier(expected_steps) == pytest.approx(0)
    assert multiplier(int(expected_steps * 0.03)) == pytest.approx(1)
