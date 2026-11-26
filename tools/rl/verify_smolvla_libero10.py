"""Verify a trained SmolVLA checkpoint on a real LIBERO observation.

Run in the FluxVLA environment, with an idle CUDA device selected through
CUDA_VISIBLE_DEVICES. This checks loading, PPO replay and native export
parity; it does not estimate the policy's task success rate.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from mmengine import Config

from fluxvla.engines import build_vla_from_cfg
from fluxvla.engines.runners.rlinf_train_runner import RLinfTrainRunner
from fluxvla.models.vlas.smolvla_rl_builder import build_smolvla_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = RLinfTrainRunner.prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(
            config_name='default',
            overrides=['model=smolvla', 'simulator=smolvla_libero_10_ppo'])
    policy = build_smolvla_policy(cfg.actor.model).cuda().eval()
    checkpoint = args.checkpoint
    if checkpoint.is_dir():
        checkpoint = checkpoint / 'actor/model_state_dict/full_weights.pt'
    trained = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expert_changed, frozen_changed = 0, 0
    for name, initial in policy.state_dict().items():
        if name.startswith('value_head.'):
            continue
        different = not torch.equal(initial.cpu(), trained[name])
        if name.startswith(policy.rl_trainable_modules):
            expert_changed += int(different)
        if name.startswith(policy.rl_frozen_modules):
            frozen_changed += int(different)
    assert expert_changed > 0, 'PPO did not change any action expert weights'
    assert frozen_changed == 0, 'PPO changed frozen VLM/state weights'
    policy.load_state_dict(trained, strict=True)
    del trained

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.libero.utils import (get_libero_image,
                                         get_libero_wrist_image,
                                         quat2axisangle)

    suite = benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path('bddl_files')) / task.problem_folder /
            task.bddl_file),
        camera_heights=256,
        camera_widths=256)
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        state = np.concatenate([
            obs['robot0_eef_pos'],
            quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']
        ]).astype(np.float32)
        env_obs = dict(
            states=torch.from_numpy(state[None]),
            main_images=torch.from_numpy(get_libero_image(obs)[None].copy()),
            wrist_images=torch.from_numpy(
                get_libero_wrist_image(obs)[None].copy()),
            task_descriptions=[task.language])
        torch.manual_seed(42)
        # Each rollout rank can have one environment while the actor
        # replays shuffled transitions in a larger microbatch.
        micro = int(policy.rollout_micro_batch_size
                    or cfg.actor.micro_batch_size)
        rollouts = [
            policy.predict_action(env_obs, mode='train')[1]
            for _ in range(micro)
        ]
        inputs = {
            key: torch.cat([r['forward_inputs'][key] for r in rollouts])
            for key in rollouts[0]['forward_inputs']
        }
        old = torch.cat([r['prev_logprobs'] for r in rollouts])
        old_values = torch.cat([r['prev_values'][:, 0] for r in rollouts])
        policy.train()
        replay = policy(forward_inputs=inputs)
        new = replay['logprobs'].detach()
        ratio = (new.sum((1, 2)) - old.sum((1, 2))).exp()
        logprob_error = (new - old).abs().max().item()
        value_error = (replay['values'].detach() -
                       old_values).abs().max().item()
        torch.testing.assert_close(
            ratio, torch.ones_like(ratio), atol=1e-4, rtol=0)
        assert logprob_error < 1e-5, logprob_error
        assert value_error < 1e-5, value_error
        del replay
        policy.eval()
        noise = torch.randn(
            1, policy.chunk_size, policy.max_action_dim, device='cuda')
        rl_actions, _ = policy.predict_action(
            env_obs, mode='eval', noise=noise)

        model_cfg = copy.deepcopy(
            Config.fromfile(cfg.actor.model.fluxvla.config_path).model)
        model_cfg.pretrained_name_or_path = None
        model_cfg.name_mapping = None
        model_cfg.strict_mapping = True
        model_cfg.num_steps = int(cfg.actor.model.num_steps)
        native = build_vla_from_cfg(model_cfg).float().cuda().eval()
        if args.export.suffix == '.safetensors':
            from safetensors.torch import load_file
            exported = load_file(str(args.export))
        else:
            exported = torch.load(
                args.export, map_location='cpu', weights_only=True)
        assert not any(name.startswith('value_head.') for name in exported)
        native.load_state_dict(exported, strict=True)
        del exported
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            model_actions = native.predict_action(
                **policy._prepare_obs(env_obs), noise=noise.clone())
        native_actions = policy.observation_adapter.env_actions(
            model_actions, policy.action_chunk, policy.action_env_dim)
        export_error = (rl_actions.cpu() - native_actions).abs().max().item()
        assert export_error < 1e-5, export_error
        assert torch.isfinite(rl_actions).all()
        result = dict(
            status='passed',
            checkpoint=str(checkpoint),
            task=task.language,
            action_shape=list(rl_actions.shape),
            changed_expert_tensors=expert_changed,
            changed_frozen_tensors=frozen_changed,
            pre_update_chunk_ratio=ratio.tolist(),
            max_logprob_replay_error=logprob_error,
            max_value_replay_error=value_error,
            max_export_action_error=export_error)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
