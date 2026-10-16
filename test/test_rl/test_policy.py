import asyncio
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from mmengine import Config
from safetensors.torch import save_file

from fluxvla.engines import build_vla_from_cfg
from fluxvla.rl.models.builder import build_pi05_policy, load_sft_weights
from fluxvla.rl.models.flow.sampler import (gaussian_entropy, gaussian_logprob,
                                            transition)


def test_eval_matches_flux_and_preserves_noise(policy, env_obs, tiny_assets):
    original = build_vla_from_cfg(tiny_assets[0].model).eval()
    original.load_state_dict(
        {
            key: value
            for key, value in policy.state_dict().items()
            if not key.startswith('value_head.')
        },
        strict=True)
    policy.compute_dtype = torch.bfloat16
    noise = torch.randn(3, 3, 32)
    before = noise.clone()
    obs = policy._prepare_obs(env_obs)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        expected = original.predict_action(**obs, noise=noise.clone())
    actions, result = policy.predict_action_batch(
        env_obs, mode='eval', noise=noise)
    torch.testing.assert_close(noise, before, rtol=0, atol=0)
    torch.testing.assert_close(
        result['forward_inputs']['model_action'].reshape_as(expected),
        expected,
        rtol=0,
        atol=0)
    torch.testing.assert_close(
        actions, policy.observation_adapter.env_actions(expected, 2, 7))
    assert actions.shape == (3, 2, 7)
    assert result['prev_values'].shape == (3, 1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_logprobs_recompute_and_rng_replay(policy, env_obs, dtype):
    policy.compute_dtype = dtype
    noise = torch.randn(3, 3, 32)
    _, first = policy.predict_action_batch(
        env_obs,
        mode='train',
        noise=noise,
        rng=torch.Generator().manual_seed(19))
    _, second = policy.predict_action_batch(
        env_obs,
        mode='train',
        noise=noise,
        rng=torch.Generator().manual_seed(19))
    inputs = first['forward_inputs']
    assert inputs['chains'].shape == (3, 4, 3, 32)
    assert inputs['chains'].dtype == torch.float32
    snapshot = inputs['chains'].clone()
    policy.train()
    new = policy(forward_inputs=inputs, compute_entropy=True)
    torch.testing.assert_close(
        new['logprobs'], first['prev_logprobs'], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        new['values'], first['prev_values'][:, 0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        second['forward_inputs']['chains'], snapshot, rtol=0, atol=0)
    torch.testing.assert_close(inputs['chains'], snapshot, rtol=0, atol=0)
    for value in new.values():
        assert torch.isfinite(value).all()
        assert value.dtype == torch.float32
    assert not policy.llm_backbone.training
    assert policy.llm_expert.training


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_real_rlinf_ppo_updates_expert_and_critic(policy, env_obs, dtype):
    import rlinf.algorithms  # noqa: F401
    from rlinf.algorithms.registry import policy_loss

    policy.compute_dtype = dtype
    _, collected = policy.predict_action_batch(env_obs, mode='train')
    before = {
        name: value.detach().clone()
        for name, value in policy.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3)
    policy.train()
    output = policy(forward_inputs=collected['forward_inputs'])
    advantages = torch.tensor([[1.], [-0.5], [0.7]])
    loss, metrics = policy_loss(
        loss_type='actor_critic',
        task_type='embodied',
        reward_type='chunk_level',
        logprob_type='chunk_level',
        single_action_dim=7,
        logprobs=output['logprobs'],
        old_logprobs=collected['prev_logprobs'],
        values=output['values'],
        prev_values=collected['prev_values'],
        advantages=advantages,
        returns=collected['prev_values'] + advantages,
        loss_mask=torch.ones(3, 1, dtype=torch.bool),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        huber_delta=10.)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {
        name: p.grad
        for name, p in policy.named_parameters() if p.grad is not None
    }
    assert gradients and all(
        torch.isfinite(g).all() for g in gradients.values())
    assert any(
        name.startswith('llm_expert.') and grad.abs().sum() > 0
        for name, grad in gradients.items())
    assert any(
        name.startswith('value_head.') and grad.abs().sum() > 0
        for name, grad in gradients.items())
    optimizer.step()
    changed = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        elif not torch.equal(parameter, before[name]):
            changed.append(name)
    assert any(name.startswith('llm_expert.') for name in changed)
    assert any(name.startswith('value_head.') for name in changed)
    assert metrics


@pytest.mark.parametrize('fmt',
                         ['native_pt', 'nested_pt', 'safetensors', 'mapped'])
def test_strict_checkpoint_formats(tiny_assets, tmp_path, fmt):
    cfg, _, _, _ = tiny_assets
    source = build_vla_from_cfg(cfg.model)
    target = build_vla_from_cfg(cfg.model)
    state = source.state_dict()
    if fmt == 'mapped':
        production = Path(__file__).resolve(
        ).parents[2] / 'configs/pi05/pi05_paligemma_libero_10_full_finetune.py'
        target.name_mapping = Config.fromfile(
            str(production)).model.name_mapping
        # Use the real Flux OpenPI name mapping, including overlapping keys.
        state = {
            target._mapped_name_candidates(key)[0][1]: value
            for key, value in state.items()
        }
    path = tmp_path / ('weights.safetensors'
                       if fmt == 'safetensors' else 'weights.pt')
    if fmt == 'safetensors':
        save_file(state, str(path))
    else:
        torch.save({'model': state} if fmt == 'nested_pt' else state, path)
    load_sft_weights(target, path)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(
            target.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['missing', 'shape'])
def test_checkpoint_errors_are_fatal(tiny_assets, tmp_path, fault):
    cfg, _, _, _ = tiny_assets
    model = build_vla_from_cfg(cfg.model)
    state = model.state_dict()
    name = next(iter(state))
    if fault == 'missing':
        state.pop(name)
    else:
        state[name] = torch.empty(1)
    path = tmp_path / 'bad.pt'
    torch.save(state, path)
    with pytest.raises(ValueError, match='checkpoint'):
        load_sft_weights(model, path)


def test_policy_state_restore_and_weight_copy(policy, model_cfg, env_obs,
                                              tmp_path):
    with torch.no_grad():
        policy.action_out_proj.projector.weight.add_(0.03)
        next(policy.value_head.parameters()).add_(0.01)
    path = tmp_path / 'rl_policy.pt'
    torch.save(policy.state_dict(), path)
    rollout = build_pi05_policy(model_cfg)
    rollout.load_state_dict(torch.load(path, weights_only=True), strict=True)
    noise = torch.randn(3, 3, 32)
    for model in (policy, rollout):
        model.eval()
    left = policy.predict_action_batch(env_obs, mode='eval', noise=noise)[1]
    right = rollout.predict_action_batch(env_obs, mode='eval', noise=noise)[1]
    torch.testing.assert_close(left['forward_inputs']['model_action'],
                               right['forward_inputs']['model_action'])
    torch.testing.assert_close(left['prev_values'], right['prev_values'])
    assert all(not key.startswith('model.') for key in rollout.state_dict())


def test_transition_matches_rlinf_reference():
    import rlinf

    path = Path(rlinf.__file__
                ).parent / 'models/embodiment/openpi_rlinf/utils/rl_sampler.py'
    spec = importlib.util.spec_from_file_location('rlinf_reference_sampler',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x, velocity = torch.randn(3, 3, 32), torch.randn(3, 3, 32)
    indices = torch.tensor([0, 1, 2])
    for method in ('flow_ode', 'flow_sde'):
        actual = transition(
            x,
            velocity,
            indices,
            num_steps=3,
            noise_level=0.5,
            stochastic=method == 'flow_sde')
        expected = module.sample_mean_var(
            x,
            velocity,
            indices,
            num_steps=3,
            noise_level=0.5,
            noise_method=method)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
    mean, std = actual
    expected_distribution = torch.distributions.Normal(mean, std)
    torch.testing.assert_close(
        gaussian_logprob(x, mean, std), expected_distribution.log_prob(x))
    torch.testing.assert_close(
        gaussian_entropy(std), expected_distribution.entropy())
    assert torch.equal(gaussian_entropy(torch.zeros(1)), torch.zeros(1))


def test_downcast_chain_is_rejected(policy, env_obs):
    _, result = policy.predict_action_batch(env_obs, mode='train')
    inputs = copy.copy(result['forward_inputs'])
    inputs['chains'] = inputs['chains'].bfloat16()
    with pytest.raises(ValueError, match='FP32'):
        policy(forward_inputs=inputs)


def test_value_uses_only_valid_prefix_tokens(policy, env_obs):
    obs = policy._prepare_obs(env_obs)
    hidden, mask, _ = policy._prefix(obs)
    mask[:, -2:] = False
    reference = policy._value(hidden, mask)
    hidden[:, -2:] = 10000
    torch.testing.assert_close(
        policy._value(hidden, mask), reference, rtol=0, atol=0)
    _, collected = policy.predict_action_batch(env_obs, mode='train')
    torch.testing.assert_close(
        policy.get_values(env_obs), collected['prev_values'])


def test_forward_value_only_needs_no_chain_or_rng(policy, env_obs):
    obs = policy._prepare_obs(env_obs)
    state = torch.random.get_rng_state().clone()
    result = policy(
        forward_inputs=obs,
        compute_logprobs=False,
        compute_values=True,
        compute_entropy=False)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(result['values'],
                               policy.get_values(env_obs)[:, 0])
    assert result['logprobs'].count_nonzero() == 0


def test_real_bucket_sync_on_cpu(policy, model_cfg, env_obs):
    from rlinf.hybrid_engines.weight_syncer.bucket_syncer import \
        BucketWeightSyncer
    from rlinf.utils.utils import collect_param_names_need_sync

    rollout = build_pi05_policy(model_cfg)
    with torch.no_grad():
        policy.action_out_proj.projector.weight.add_(0.03)
        next(policy.value_head.parameters()).add_(0.01)
    syncer = BucketWeightSyncer(
        bucket_size=4096,
        bucket_dtype=None,
        bucket_device='cpu',
        load_instant=False)
    names = collect_param_names_need_sync(policy)
    payloads = []

    async def send(bucket):
        payloads.append(
            {name: value.clone()
             for name, value in bucket.items()})

    async def receive():
        return payloads.pop(0)

    async def copy_weights():
        await syncer.init_sender(policy.state_dict(), names, send)
        await syncer.sync(policy.state_dict(), send, version=7)
        assert len(payloads) > 1
        return await syncer.apply(rollout, receive)

    assert asyncio.run(copy_weights()) == 7
    assert not payloads
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(
            rollout.state_dict()[name], value, rtol=0, atol=0)
    _, collected = policy.predict_action_batch(env_obs, mode='train')
    output = rollout(forward_inputs=collected['forward_inputs'])
    torch.testing.assert_close(output['logprobs'], collected['prev_logprobs'])
    torch.testing.assert_close(output['values'], collected['prev_values'][:,
                                                                          0])
