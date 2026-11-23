"""Run with torchrun --standalone --nproc-per-node=2 in the RL environment."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from fluxvla.engines.runners.rlinf.smolvla_actor import SmolVLAActor


def main():
    dist.init_process_group('gloo')
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    cfg = OmegaConf.create({
        'runner': {'task_type': 'embodied'},
        'algorithm': {'adv_type': 'gae', 'reward_type': 'chunk_level',
                      'gamma': 1., 'gae_lambda': 1., 'normalize_advantages': True},
        'env': {'train': {'auto_reset': False, 'ignore_terminations': False}},
    })
    # Rank 0 succeeds, then loses the success condition within its final
    # chunk. Rank 1 times out. Both ranks have two valid chunk decisions.
    rewards = torch.zeros(2, 1, 3)
    dones = torch.zeros(3, 1, 3, dtype=torch.bool)
    dones[-1, 0, -1] = True
    if rank == 0:
        dones[-1, 0, 0] = True
        rewards[-1, 0, :2] = torch.tensor([1., -1.])
    batch = dict(rewards=rewards, dones=dones,
                 prev_values=torch.zeros(3, 1, 1),
                 loss_mask=torch.ones(2, 1, 1, dtype=torch.bool))
    actor = SimpleNamespace(cfg=cfg, device='cpu', rollout_batch=batch)
    # Omit scheduler profiling and metrics, exercise the actor target path.
    import fluxvla.engines.runners.rlinf.smolvla_actor as module
    with patch.object(module, 'compute_rollout_metrics', return_value={}):
        for normalize in (True, False):
            cfg.algorithm.normalize_advantages = normalize
            SmolVLAActor.compute_advantages_and_returns.__wrapped__(actor)
            raw = torch.full_like(batch['returns'], float(rank == 0))
            torch.testing.assert_close(batch['returns'], raw)
            expected = raw
            if normalize:
                std = torch.tensor([1., 1., 0., 0.]).std()
                expected = (raw - .5) / (std + 1e-5)
            torch.testing.assert_close(batch['advantages'], expected)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
