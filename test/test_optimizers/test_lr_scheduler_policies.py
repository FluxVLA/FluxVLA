from types import SimpleNamespace

import torch.nn as nn

from fluxvla.optimizers.lr_scheduler_policies import BaseLRSchedulerPolicy


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
