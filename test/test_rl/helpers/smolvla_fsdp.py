"""Two-GPU synthetic PPO/FSDP2 probe; no simulator or benchmark claim.

Run via the opt-in test_smol_gpu_fsdp test, which supplies a tiny real model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor


def fingerprints(model, frozen_only=False):
    result = {}
    for name, parameter in model.named_parameters():
        if frozen_only and parameter.requires_grad:
            continue
        data = parameter.detach()
        if isinstance(data, DTensor):
            data = data.to_local()
        result[name] = hashlib.sha256(
            data.cpu().contiguous().numpy().tobytes()).hexdigest()
    return result


def main():
    import rlinf.algorithms  # noqa: F401
    from rlinf.algorithms.registry import policy_loss
    from rlinf.hybrid_engines.fsdp.strategy.fsdp2 import FSDP2Strategy
    from rlinf.utils.utils import warmup_optimizer_state

    from fluxvla.rl.models.builder import build_smolvla_policy
    from fluxvla.rl.train import prepare_environment

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    dist.init_process_group('nccl')
    if dist.get_world_size() != 2:
        raise ValueError('This probe requires two GPU ranks')
    root = prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(config_name='benchmarks/libero/smolvla/ppo')
    cfg.actor.model = OmegaConf.load(args.model_config)
    cfg.actor.model.fluxvla.compute_dtype = 'bf16'
    OmegaConf.resolve(cfg)
    model = build_smolvla_policy(cfg.actor.model).cuda()
    obs = dict(
        main_images=torch.randint(0, 256, (2, 20, 24, 3), dtype=torch.uint8),
        wrist_images=torch.randint(0, 256, (2, 20, 24, 3), dtype=torch.uint8),
        states=torch.randn(2, 8),
        task_descriptions=['pick the red cube', 'open the drawer'])
    _, old = model.predict_action_batch(obs, mode='train')
    strategy = FSDP2Strategy(
        cfg.actor, world_size=2, dp_group=dist.group.WORLD)
    model = strategy.wrap_model(model, init_device_mesh('cuda', (2, )))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    warmup_optimizer_state(optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    before = fingerprints(model)
    frozen = fingerprints(model, frozen_only=True)
    model.train()
    drift = None
    for step in range(2):
        output = model(
            forward_inputs=old['forward_inputs'], compute_entropy=True)
        if step == 0:
            drift = float((output['logprobs'] -
                           old['prev_logprobs']).detach().abs().max())
            torch.testing.assert_close(
                output['logprobs'], old['prev_logprobs'], rtol=1e-5, atol=1e-5)
        advantages = torch.tensor([[1.], [-0.5]], device='cuda')
        loss, _ = policy_loss(
            loss_type='actor_critic',
            task_type='embodied',
            reward_type='chunk_level',
            logprob_type='chunk_level',
            single_action_dim=7,
            logprobs=output['logprobs'],
            old_logprobs=old['prev_logprobs'],
            values=output['values'],
            prev_values=old['prev_values'],
            advantages=advantages,
            returns=old['prev_values'] + advantages,
            loss_mask=torch.ones(2, 1, dtype=torch.bool, device='cuda'),
            clip_ratio_low=.2,
            clip_ratio_high=.2,
            value_clip=.2,
            huber_delta=10.)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [
            p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            for p in model.parameters() if p.grad is not None
        ]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        strategy.clip_grad_norm_(model)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    after = fingerprints(model)
    assert frozen == fingerprints(model, frozen_only=True)
    changed = [name for name in before if before[name] != after[name]]
    assert any(name.startswith('llm_expert.') for name in changed)
    assert any(name.startswith('value_head.') for name in changed)
    checkpoint = args.output / 'checkpoint'
    strategy.save_checkpoint(model, optimizer, scheduler, str(checkpoint))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    strategy.load_checkpoint(model, optimizer, scheduler, str(checkpoint))
    assert fingerprints(model) == after
    report = dict(
        rank=rank,
        world_size=2,
        passed=True,
        preupdate_logprob_max_abs_drift=drift,
        frozen_unchanged=True,
        expert_and_value_updated=True,
        checkpoint_restored=True,
        optimizer_steps=2,
        peak_gpu_bytes=torch.cuda.max_memory_allocated())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'rank{rank}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
