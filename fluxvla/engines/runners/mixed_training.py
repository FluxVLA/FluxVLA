# Copyright 2026 Limx Dynamics
"""Synchronized VLA/VLM steps with separate, resumable data streams."""

import copy
import hashlib
import json
import math
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from fluxvla.engines.utils import (build_collator_from_cfg,
                                   build_dataset_from_cfg)
from fluxvla.engines.utils.torch_utils import worker_init_function


def _capture_rng():
    numpy_state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1],
                                  np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


class TaskDataStream:
    """One task's loader; epoch/offset refer to consumed local batches."""

    def __init__(self,
                 dataset,
                 collator,
                 batch_size,
                 num_workers,
                 seed,
                 rank,
                 world_size,
                 sharding_strategy='round_robin'):
        if batch_size < 1 or num_workers < 0 or len(dataset) < 1:
            raise ValueError('Mixed training needs nonempty task datasets '
                             'and positive batch sizes.')
        self.seed, self.rank = seed, rank
        self.epoch, self.offset = 0, 0
        self.iterator = None
        iterable = isinstance(dataset, IterableDataset)
        if iterable:
            if len(dataset) < world_size * max(1, num_workers):
                raise ValueError('Mixed iterable data needs at least one '
                                 'sample per rank and worker.')
            configure = getattr(dataset, 'configure_sharding', None)
            if not callable(configure):
                raise ValueError('Mixed iterable datasets must implement '
                                 'distributed configure_sharding().')
            configure(strategy=sharding_strategy, batch_size=batch_size)
        self.sampler = None if iterable else DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False)
        self.generator = torch.Generator()
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collator,
            num_workers=num_workers,
            worker_init_fn=worker_init_function,
            generator=self.generator,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers else None,
            persistent_workers=bool(num_workers))
        # Repeating robot datasets report their global size and shard in
        # __iter__; DataLoader cannot infer their per-rank epoch length.
        self.length = (
            math.ceil(len(dataset) / world_size /
                      batch_size) if iterable else len(self.loader))
        if self.length < 1:
            raise ValueError('Mixed task dataloader has no batches.')

    def next(self):
        if self.offset == self.length:
            self.epoch += 1
            self.offset = 0
            self.iterator = None
        if self.iterator is None:
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            set_epoch = getattr(self.loader.dataset, 'set_epoch', None)
            if callable(set_epoch):
                set_epoch(self.epoch)
            self.generator.manual_seed(self.seed + self.epoch + self.rank)
            # Replaying skipped batches must not advance the training RNG.
            state = _capture_rng()
            try:
                self.iterator = iter(self.loader)
                for _ in range(self.offset):
                    next(self.iterator)
            finally:
                _restore_rng(state)
        try:
            batch = next(self.iterator)
        except StopIteration as exc:
            raise RuntimeError('Task dataset ended before its advertised '
                               'length; check distributed sharding.') from exc
        self.offset += 1
        return batch

    def state_dict(self):
        return dict(epoch=self.epoch, offset=self.offset, length=self.length)

    def load_state_dict(self, state):
        if state['length'] != self.length:
            raise ValueError(
                'Mixed task dataset/batch count changed on resume.')
        if not 0 <= state['offset'] <= self.length or state['epoch'] < 0:
            raise ValueError('Invalid mixed task data cursor.')
        self.epoch, self.offset = state['epoch'], state['offset']
        self.iterator = None


class MixedTraining:
    """One task per update; VLA-only schedules allow controlled ablations."""

    def __init__(self,
                 cfg,
                 task_schedule=('vla', 'vla', 'vla', 'vlm'),
                 loss_weights=None,
                 sync_each_microbatch=True):
        self.sync_each_microbatch = bool(sync_each_microbatch)
        self.schedule = tuple(task_schedule)
        if ('vla' not in self.schedule
                or not set(self.schedule) <= {'vla', 'vlm'}):
            raise ValueError('task_schedule must contain vla and optionally '
                             'vlm, with no other tasks.')
        self.loss_weights = dict(vla=1.0, vlm=1.0)
        self.loss_weights.update(loss_weights or {})
        if (set(self.loss_weights) != {'vla', 'vlm'}
                or any(not math.isfinite(value) or value <= 0
                       for value in self.loss_weights.values())):
            raise ValueError('Mixed loss_weights must be finite and positive.')
        uses_vlm = 'vlm' in self.schedule
        if uses_vlm and 'vlm_dataloader' not in cfg:
            raise ValueError('Mixed training requires cfg.vlm_dataloader.')
        self.vlm_cfg = copy.deepcopy(cfg.vlm_dataloader) if uses_vlm else {}
        data_config = dict(
            vla=cfg.train_dataloader.get('dataset'),
            vlm=self.vlm_cfg.get('dataset'),
            vla_collator=cfg.runner.get('collator'),
            vlm_collator=self.vlm_cfg.get('collator'))
        self.data_signature = hashlib.sha256(
            json.dumps(data_config, sort_keys=True,
                       default=str).encode()).hexdigest()
        if uses_vlm and 'collator' not in self.vlm_cfg:
            raise ValueError('vlm_dataloader must define its own collator.')
        self.vlm_collator = (
            build_collator_from_cfg(self.vlm_cfg['collator'])
            if uses_vlm else None)
        self.active_task = None
        self.streams = {}
        self.accumulation = {}
        self.totals = None
        self.task_steps = dict(vla=0, vlm=0)
        self.step_metrics = {}
        self.contract = None

    def setup(self, runner, vla_dataset, vlm_dataset=None):
        self.accumulation = dict(vla=runner.grad_accumulation_steps)
        batch_sizes = dict(vla=runner.per_device_batch_size)
        workers = dict(vla=runner.per_device_num_workers)
        sources = [('vla', vla_dataset, runner.collator)]
        if 'vlm' in self.schedule:
            vlm_dataset = (
                build_dataset_from_cfg(self.vlm_cfg['dataset'])
                if vlm_dataset is None else vlm_dataset)
            runner._save_vlm_assets(vlm_dataset)
            self.accumulation['vlm'] = self.vlm_cfg.get(
                'grad_accumulation_steps', runner.grad_accumulation_steps)
            batch_sizes['vlm'] = self.vlm_cfg['per_device_batch_size']
            workers['vlm'] = self.vlm_cfg.get('per_device_num_workers', 0)
            sources.append(('vlm', vlm_dataset, self.vlm_collator))
        if any(
                type(n) is not int or n < 1
                for n in self.accumulation.values()):
            raise ValueError(
                'Task accumulation steps must be positive integers.')
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for task, dataset, collator in sources:
            self.streams[task] = TaskDataStream(
                dataset, collator, batch_sizes[task], workers[task],
                (runner.seed or 0) + (task == 'vlm'), rank, world_size,
                runner.dataset_sharding_strategy
                if task == 'vla' else 'round_robin')
        self.contract = dict(
            sync_each_microbatch=self.sync_each_microbatch,
            data_signature=self.data_signature,
            trainable_parameters=tuple(
                name
                for name, parameter in runner._unwrap_vla().named_parameters()
                if parameter.requires_grad),
            optimizer=json.loads(
                json.dumps(runner.optimizer_cfg, default=str)),
            scheduler=json.loads(
                json.dumps(runner.lr_scheduler_cfg, default=str)),
            schedule=self.schedule,
            loss_weights=self.loss_weights,
            accumulation=self.accumulation,
            batch_sizes=batch_sizes,
            workers=workers,
            stream_lengths={
                task: s.length
                for task, s in self.streams.items()
            },
            seed=runner.seed,
            world_size=world_size)
        if dist.is_initialized():
            contracts = [None] * world_size
            dist.all_gather_object(contracts, self.contract)
            if any(item != self.contract for item in contracts):
                raise ValueError(
                    'Mixed training configuration differs by rank.')

    def run(self, runner, vla_dataset, eval_dataset):
        from fluxvla.evaluators.vlm_evaluator import VLMEvaluator
        if isinstance(runner.evaluator, VLMEvaluator) and not self.vlm_cfg:
            raise ValueError('VLA-only ablations do not have a VLM collator; '
                             'use a separate VLM evaluation config.')
        if isinstance(runner.evaluator, VLMEvaluator) and eval_dataset is None:
            raise ValueError('Mixed VLM validation requires an explicit '
                             'val_dataloader.dataset with VLM preprocessing.')
        try:
            self.setup(runner, vla_dataset)
            runner.steps_per_epoch = None
            runner.resume()
            runner._initialize_ema_parameters()
            runner.vla.train()
            runner.optimizer.zero_grad(set_to_none=True)
            return runner._run_step_based(
                None, None,
                vla_dataset if eval_dataset is None else eval_dataset)
        finally:
            self.close(runner)

    def step(self, runner):
        self.active_task = self.schedule[runner.metric.global_step %
                                         len(self.schedule)]
        task = self.active_task
        for micro_step in range(self.accumulation[task]):
            batch = self.streams[task].next()
            should_step = micro_step + 1 == self.accumulation[task]
            # Root FSDP no_sync retains a full-model gradient, which can
            # exceed memory once Adam moments have been initialized.
            with runner._grad_sync_context(
                    should_sync=should_step or self.sync_each_microbatch):
                loss = runner._training_step(batch, should_step=should_step)
        self.task_steps[task] += 1
        runner.current_epoch = self.streams['vla'].epoch
        return float(loss.detach())

    def backward(self, runner, output, should_step):
        task = self.active_task
        count_key = 'num_tokens' if task == 'vlm' else 'num_action_targets'
        if 'loss_sum' not in output or count_key not in output:
            raise ValueError(f'{task} must return loss_sum and {count_key} '
                             'for globally normalized mixed training.')
        values = [
            output['loss_sum'].detach().double(),
            output[count_key].detach().double()
        ]
        if task == 'vlm':
            values.append(output['num_correct'].detach().double())
        values = torch.stack(values)
        self.totals = values if self.totals is None else self.totals + values
        output['loss_sum'].backward()
        if not should_step:
            return output['loss'].detach()
        totals, self.totals = self.totals, None
        world_size = 1
        if dist.is_initialized():
            dist.all_reduce(totals)
            world_size = dist.get_world_size()
        if totals[1] <= 0 or not torch.isfinite(totals).all():
            runner.optimizer.zero_grad(set_to_none=True)
            raise ValueError(
                f'{task} step has no valid targets or finite loss.')
        scale = world_size * self.loss_weights[task] / totals[1]
        for parameter in runner.vla.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(parameter.grad.dtype))
        # FSDP may materialize zeros for unused slices of a flat parameter.
        # None prevents Adam momentum/decay from changing inactive heads.
        for parameter in runner._unwrap_vla().inactive_training_parameters(
                task):
            parameter.grad = None
        raw_loss = totals[0] / totals[1]
        loss = raw_loss.to(output['loss'].dtype) * self.loss_weights[task]
        runner.metric.commit(loss=loss)
        metrics = {
            f'train/{task}/loss': raw_loss.item(),
            f'train/{task}/weighted_loss': loss.item(),
            f'train/{task}/targets': totals[1].item(),
            f'train/{task}/steps': self.task_steps[task] + 1
        }
        if task == 'vlm':
            metrics['train/vlm/token_accuracy'] = (totals[2] /
                                                   totals[1]).item()
        self.step_metrics = metrics
        return loss

    def checkpoint_state(self):
        if self.totals is not None:
            raise RuntimeError(
                'Save mixed checkpoints at optimizer boundaries.')
        state = dict(
            contract=self.contract,
            task_steps=dict(self.task_steps),
            streams={
                task: stream.state_dict()
                for task, stream in self.streams.items()
            })
        rng = _capture_rng()
        states = [rng]
        if dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, rng)
        state['rng'] = states
        return state

    def load_state_dict(self, state, global_step):
        if state is None or state['contract'] != self.contract:
            raise ValueError('Mixed resume requires the same task schedule, '
                             'data, optimizer, trainable parameters, batch '
                             'sizes, losses, seed and world size. '
                             'Use pretrained weights for a new recipe.')
        cycles, remainder = divmod(global_step, len(self.schedule))
        expected = {
            task: cycles * self.schedule.count(task) +
            self.schedule[:remainder].count(task)
            for task in self.task_steps
        }
        if state['task_steps'] != expected:
            raise ValueError('Mixed task counters do not match global_step.')
        for task, stream in self.streams.items():
            stream.load_state_dict(state['streams'][task])
            consumed = stream.epoch * stream.length + stream.offset
            if consumed != expected[task] * self.accumulation[task]:
                raise ValueError('Mixed task data cursor does not match '
                                 'its completed optimizer steps.')
        self.task_steps = dict(state['task_steps'])
        rank = dist.get_rank() if dist.is_initialized() else 0
        _restore_rng(state['rng'][rank])

    def close(self, runner):
        for stream in self.streams.values():
            runner._shutdown_dataloader(stream.loader)
            stream.iterator = None
        self.streams.clear()
