"""Build SmolVLA PPO targets from the complete distributed rollout."""

import torch
import torch.distributed as dist
from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.scheduler import Worker
from rlinf.utils.metric_utils import compute_loss_mask, compute_rollout_metrics
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


def normalize_rollout_advantages(advantages, mask, device):
    """Use the same masked sample mean and std on every actor rank."""
    valid = advantages if mask is None else advantages[mask.expand_as(
        advantages)]
    valid = valid.to(dtype=torch.float64)
    stats = torch.stack((valid.new_tensor(valid.numel()), valid.sum(),
                         valid.square().sum())).to(device)
    dist.all_reduce(stats)
    count, total, squares = stats.to(advantages.device).unbind()
    mean = total / count.clamp_min(1)
    variance = (squares - total * mean).clamp_min(0) / (count - 1).clamp_min(1)
    normalized = (advantages - mean) / (variance.sqrt() + 1e-5)
    if mask is not None:
        normalized = normalized.masked_fill(~mask, 0)
    return normalized.to(advantages.dtype)


class SmolVLAActor(EmbodiedFSDPActor):

    @Worker.timer('actor/compute_adv')
    def compute_advantages_and_returns(self):
        batch = self.rollout_batch
        algo = self.cfg.algorithm
        rewards = batch['rewards']
        if (algo.reward_type == 'chunk_level'
                and not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations):
            # LIBERO executes the rest of a chunk after a success. Mask
            # those steps before summing relative rewards for chunk GAE.
            step_mask, _ = compute_loss_mask(batch['dones'])
            rewards = rewards.masked_fill(~step_mask, 0)
        mask = batch.get('loss_mask')
        targets = calculate_adv_and_returns(
            task_type=self.cfg.runner.task_type,
            adv_type=algo.adv_type,
            reward_type=algo.reward_type,
            gamma=algo.gamma,
            gae_lambda=algo.gae_lambda,
            rewards=rewards,
            dones=batch['dones'],
            values=batch['prev_values'],
            loss_mask=mask,
            loss_mask_sum=batch.get('loss_mask_sum'),
            normalize_advantages=False)
        if algo.normalize_advantages:
            targets['advantages'] = normalize_rollout_advantages(
                targets['advantages'], mask, self.device)
        batch.update(targets)
        return compute_rollout_metrics(dict(batch, rewards=rewards))
