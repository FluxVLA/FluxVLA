# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Dict, Optional

from fluxvla.engines.utils.builder import build_optimizer_from_cfg
from fluxvla.engines.utils.root import LR_SCHEDULERS
from .schedulers import (get_constant_schedule,
                         get_cosine_schedule_with_warmup,
                         get_step_based_schedule)


class BaseLRSchedulerPolicy:
    """Build and advance optimizer-specific learning rate schedules."""

    def __init__(self, **kwargs) -> None:
        if kwargs:
            fields = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected LR scheduler config field(s) for '
                            f'{self.__class__.__name__}: {fields}')
        self.optimizer = None
        self.scheduler = None

    @staticmethod
    def _canonicalize_param_name(name: str) -> str:
        canonical_name = name
        while canonical_name.startswith('module.'):
            canonical_name = canonical_name.removeprefix('module.')
        while canonical_name.startswith('_fsdp_wrapped_module.'):
            canonical_name = canonical_name.removeprefix(
                '_fsdp_wrapped_module.')
        return canonical_name.replace('._fsdp_wrapped_module.', '.')

    def _get_param_lr(self, runner, name: str) -> float:
        optimizer_cfg = runner.optimizer_cfg
        paramwise_lr = optimizer_cfg.get('paramwise_learning_rate', {})
        if not paramwise_lr:
            return float(optimizer_cfg['lr'])

        canonical_name = self._canonicalize_param_name(name)
        matched_lr = float(optimizer_cfg['lr'])
        matched_len = -1
        for prefix, lr in paramwise_lr.items():
            if (canonical_name.startswith(prefix)
                    and len(prefix) > matched_len):
                matched_lr = float(lr)
                matched_len = len(prefix)
        return matched_lr

    def build_param_groups(self, runner, weight_decay=None):
        optimizer_cfg = runner.optimizer_cfg
        paramwise_lr = optimizer_cfg.get('paramwise_learning_rate', {})
        if weight_decay is None:
            weight_decay = optimizer_cfg.get('weight_decay')
        if not paramwise_lr and weight_decay is None:
            return [
                param for param in runner.vla.parameters()
                if param.requires_grad
            ]
        if not paramwise_lr:
            decay, no_decay = [], []
            for name, param in runner.vla.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith('.bias'):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [{
                'params': decay,
                'weight_decay': weight_decay
            }, {
                'params': no_decay,
                'weight_decay': 0.0
            }]

        groups = {}
        for name, param in runner.vla.named_parameters():
            if not param.requires_grad:
                continue
            lr = self._get_param_lr(runner, name)
            decay = 0.0
            if (weight_decay is not None and param.ndim > 1
                    and not name.endswith('.bias')):
                decay = float(weight_decay)
            key = (lr, decay)
            if key not in groups:
                group = {'params': [], 'lr': lr}
                if weight_decay is not None:
                    group['weight_decay'] = decay
                groups[key] = group
            groups[key]['params'].append(param)
        return list(groups.values())

    @staticmethod
    def _optimizer_build_cfg(runner) -> Dict:
        optimizer_cfg = runner.optimizer_cfg
        optimizer_kwargs = dict(optimizer_cfg)
        optimizer_kwargs.pop('paramwise_learning_rate', None)
        optimizer_kwargs.pop('weight_decay', None)
        return optimizer_kwargs

    def build_optimizer(self, runner, weight_decay=None):
        groups = self.build_param_groups(runner, weight_decay)
        return build_optimizer_from_cfg(
            self._optimizer_build_cfg(runner), default_args={'params': groups})

    def build_scheduler(self, runner, optimizer):
        raise NotImplementedError

    def build(self, runner, weight_decay=None):
        runner.optimizer = self.build_optimizer(runner, weight_decay)
        self.optimizer = runner.optimizer
        self.scheduler = self.build_scheduler(runner, runner.optimizer)
        return runner.optimizer, self

    def prepare_step(self, runner) -> None:
        pass

    def step(self, runner) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def get_last_lr(self):
        if self.scheduler is None:
            return []
        return self.scheduler.get_last_lr()

    def state_dict(self):
        if self.scheduler is None:
            return {}
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict) -> None:
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict)

    def bind_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer
        if self.scheduler is not None and hasattr(self.scheduler, 'optimizer'):
            self.scheduler.optimizer = optimizer


@LR_SCHEDULERS.register_module(name=['constant', 'ConstantLRScheduler'])
class ConstantLRScheduler(BaseLRSchedulerPolicy):

    def build_scheduler(self, runner, optimizer):
        return get_constant_schedule(optimizer)


@LR_SCHEDULERS.register_module(
    name=['linear-warmup+cosine-decay', 'LinearWarmupCosineDecayLRScheduler'])
class LinearWarmupCosineDecayLRScheduler(BaseLRSchedulerPolicy):

    def __init__(self, warmup_ratio: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.warmup_ratio = warmup_ratio

    def build_scheduler(self, runner, optimizer):
        num_warmup_steps = int(runner.num_training_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps,
                                                    runner.num_training_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0
        return scheduler


@LR_SCHEDULERS.register_module(name=['step-based', 'StepBasedLRScheduler'])
class StepBasedLRScheduler(BaseLRSchedulerPolicy):

    def __init__(self,
                 lr_schedule: Optional[Dict[float, float]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.lr_schedule = lr_schedule

    def build_scheduler(self, runner, optimizer):
        if self.lr_schedule is None:
            raise ValueError('lr_schedule must be provided when using '
                             'step-based scheduler')
        return get_step_based_schedule(optimizer, runner.num_training_steps,
                                       self.lr_schedule)


@LR_SCHEDULERS.register_module(name=[
    'groupwise-freeze-warmup-cosine', 'GroupwiseFreezeWarmupCosineLRScheduler'
])
class GroupwiseFreezeWarmupCosineLRScheduler(BaseLRSchedulerPolicy):

    def __init__(self,
                 freeze_steps: int = 0,
                 warmup_steps: int = 0,
                 lr_coef: float = 1.0,
                 use_cosine_decay: bool = False,
                 min_lr_ratio: float = 0.1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.freeze_steps = freeze_steps
        self.warmup_steps = warmup_steps
        self.lr_coef = lr_coef
        self.use_cosine_decay = use_cosine_decay
        self.min_lr_ratio = min_lr_ratio

    def build_param_groups(self, runner, weight_decay=None):
        optimizer_cfg = runner.optimizer_cfg
        if weight_decay is None:
            weight_decay = optimizer_cfg.get('weight_decay')
        strategy = getattr(runner.vla, 'get_lr_param_group_strategy', None)
        if callable(strategy):
            param_groups = strategy(
                learning_rate=optimizer_cfg['lr'],
                lr_coef=self.lr_coef,
                weight_decay=weight_decay,
                canonicalize_param_name=self._canonicalize_param_name,
            )
            if param_groups is not None:
                return param_groups
        raise ValueError(
            'Groupwise LR schedule requires the model to implement '
            '`get_lr_param_group_strategy(...)`.')

    @staticmethod
    def _optimizer_build_cfg(runner) -> Dict:
        optimizer_kwargs = BaseLRSchedulerPolicy._optimizer_build_cfg(runner)
        optimizer_kwargs.pop('lr', None)
        return optimizer_kwargs

    def build_scheduler(self, runner, optimizer):
        return None

    def build(self, runner, weight_decay=None):
        super().build(runner, weight_decay)
        self.prepare_step(runner)
        return runner.optimizer, self

    def _groupwise_lr_scale(self, runner, step: int) -> float:
        strategy = getattr(runner.vla, 'get_lr_groupwise_scale', None)
        if callable(strategy):
            scale = strategy(
                step=step,
                freeze_steps=self.freeze_steps,
                warmup_steps=self.warmup_steps,
                use_cosine_decay=self.use_cosine_decay,
                min_lr_ratio=self.min_lr_ratio,
                num_training_steps=runner.num_training_steps,
                max_steps=runner.max_steps,
            )
            if scale is not None:
                return scale

        if not self.use_cosine_decay:
            return 1.0

        progress = max(0, step - self.freeze_steps)
        if progress < self.warmup_steps:
            return progress / max(1, self.warmup_steps)

        remain = max(
            1, runner.num_training_steps -
            (self.freeze_steps + self.warmup_steps))
        cosine_progress = min(1.0, (progress - self.warmup_steps) / remain)
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_ratio

    def prepare_step(self, runner) -> None:
        lr = runner.optimizer_cfg['lr']
        base = {
            'vlm': lr * self.lr_coef,
            'transformer_core': lr,
            'soft_prompts': lr * self.lr_coef,
            'action_heads': lr,
        }
        step = runner.metric.global_step
        for group in runner.optimizer.param_groups:
            name = group.get('name', '')
            if name not in base:
                continue
            if step < self.freeze_steps:
                group['lr'] = 0.0 if name in (
                    'vlm', 'transformer_core') else base[name]
            else:
                group['lr'] = base[name] * self._groupwise_lr_scale(
                    runner, step)

    def step(self, runner) -> None:
        pass

    def get_last_lr(self):
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        if self.optimizer is None:
            return []
        for group in self.optimizer.param_groups:
            if group.get('name') == 'action_heads':
                return [group['lr']]
        return [self.optimizer.param_groups[0]['lr']]

    def get_log_lr(self, runner):
        for group in runner.optimizer.param_groups:
            if group.get('name') == 'action_heads':
                return group['lr']
        return runner.optimizer.param_groups[0]['lr']
