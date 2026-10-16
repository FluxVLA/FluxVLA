import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import save_file

from fluxvla.engines import build_vla_from_cfg
from fluxvla.rl.benchmarks.robotwin.adapter import (DELTA_MASK, JOINT_FLIP,
                                                    RoboTwinObservationAdapter,
                                                    decode_state)
from fluxvla.rl.benchmarks.robotwin.protocol import (choose_best,
                                                     partition_seeds)
from fluxvla.rl.models.builder import (build_pi05_policy, load_sft_weights,
                                       read_checkpoint)
from fluxvla.rl.train import prepare_environment, validate_frontend_cfg
from fluxvla.rl.utils.logging import require_wandb

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def robotwin_stats():
    return {
        'norm_stats': {
            key: {
                'q01': [-1.] * 14,
                'q99': [1.] * 14
            }
            for key in ('state', 'actions')
        }
    }


@pytest.fixture
def robotwin_obs():
    rng = np.random.default_rng(55)
    state = rng.uniform(0.1, 0.9, (3, 14)).astype(np.float32)
    return {
        'main_images':
        torch.from_numpy(
            rng.integers(256, size=(3, 20, 24, 3), dtype=np.uint8)),
        'wrist_images':
        torch.from_numpy(
            rng.integers(256, size=(3, 2, 20, 24, 3), dtype=np.uint8)),
        'states':
        torch.from_numpy(state),
        'task_descriptions': ['Adjust_Bottle\nleft'] * 3
    }


@pytest.fixture
def robotwin_adapter(tiny_assets, robotwin_stats):
    return RoboTwinObservationAdapter(
        norm_stats=robotwin_stats,
        tokenizer_path=tiny_assets[3],
        image_size=16,
        max_token_len=200)


@pytest.fixture
def robotwin_policy(model_cfg, robotwin_stats, tmp_path):
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps(robotwin_stats))
    cfg = copy.deepcopy(model_cfg)
    cfg.action_dim = 14
    cfg.fluxvla.update({
        'observation_adapter': 'robotwin',
        'norm_stats_path': str(stats),
        'image_size': 16,
        'max_token_len': 200,
        'noise_level': 0.3
    })
    return build_pi05_policy(cfg)


def test_robotwin_three_views_native_state_prompt(robotwin_adapter,
                                                  robotwin_obs):
    obs = robotwin_adapter(robotwin_obs)
    assert obs['images'].shape == (3, 9, 16, 16)
    assert obs['img_masks'].all() and obs['img_masks'].shape == (3, 3)
    assert obs['states'].shape == (3, 32)
    assert not obs['states'][:, 14:].any()
    normalized = robotwin_adapter.normalize_state(
        robotwin_obs['states'].numpy())
    bins = np.digitize(normalized[0], np.linspace(-1, 1, 257)[:-1]) - 1
    state_text = ' '.join(map(str, bins))
    text = f'Task: Adjust Bottle left, State: {state_text};\nAction: '
    expected = robotwin_adapter.tokenizer._tokenizer.encode(text, add_bos=True)
    assert obs['lang_tokens'][0, :len(expected)].tolist() == expected
    assert obs['lang_masks'][0].sum() == len(expected)


def test_robotwin_contextual_delta_restore_and_grippers(
        robotwin_adapter, robotwin_obs):
    chain = torch.zeros(3, 3, 32)
    before = chain.clone()
    actions = robotwin_adapter.env_actions(
        chain, 2, 14, env_obs=robotwin_obs).numpy()
    state = robotwin_obs['states'].numpy()
    np.testing.assert_allclose(
        actions[:, 0, DELTA_MASK], state[:, DELTA_MASK], atol=1e-6)
    # Public Aloha output has a different scale from state decoding. Do not
    # incorrectly "invert" the runtime linear gripper transform here.
    expected_gripper = (0.0000005 + 0.5476 + 0.6213) / (1.4910 + 0.6213)
    np.testing.assert_allclose(
        actions[..., [6, 13]], expected_gripper, atol=1e-6)
    torch.testing.assert_close(chain, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match='current observation'):
        robotwin_adapter.env_actions(chain, 2, 14)
    decoded = decode_state(state)
    np.testing.assert_allclose(decoded[:, DELTA_MASK],
                               (state * JOINT_FLIP)[:, DELTA_MASK])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_robotwin_probability_recompute(robotwin_policy, robotwin_obs, dtype):
    import rlinf.algorithms  # noqa: F401
    from rlinf.algorithms.registry import policy_loss

    model = robotwin_policy
    model.compute_dtype = dtype
    actions, rollout = model.predict_action_batch(robotwin_obs, mode='train')
    assert actions.shape == (3, 2, 14)
    snapshot = rollout['forward_inputs']['chains'].clone()
    actor = model(
        forward_inputs=rollout['forward_inputs'], compute_entropy=True)
    torch.testing.assert_close(actor['logprobs'], rollout['prev_logprobs'])
    assert torch.isfinite(actor['entropy']).all()
    assert snapshot.dtype == torch.float32
    torch.testing.assert_close(
        snapshot, rollout['forward_inputs']['chains'], rtol=0, atol=0)
    advantage = torch.tensor([[1.], [-.5], [.7]])
    loss, _ = policy_loss(
        loss_type='actor_critic',
        task_type='embodied',
        reward_type='chunk_level',
        logprob_type='chunk_level',
        single_action_dim=14,
        logprobs=actor['logprobs'],
        old_logprobs=rollout['prev_logprobs'],
        values=actor['values'],
        prev_values=rollout['prev_values'],
        advantages=advantage,
        returns=rollout['prev_values'] + advantage,
        loss_mask=torch.ones(3, 1, dtype=torch.bool),
        clip_ratio_low=.2,
        clip_ratio_high=.2,
        value_clip=.2,
        huber_delta=10.)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_out_proj.projector.weight.grad.isfinite().all()
    assert all(parameter.grad is None
               for parameter in model.vision_backbone.parameters())


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_openpi_precision_branch_recompute(robotwin_policy, robotwin_obs,
                                           dtype):
    model = robotwin_policy
    model.openpi_fp32_flow = True
    model.vision_backbone.openpi_stem_fp32 = True
    model.configure_rl(
        model.observation_adapter,
        action_chunk=2,
        action_dim=14,
        noise_level=.3,
        compute_dtype=dtype)
    _, rollout = model.predict_action_batch(robotwin_obs, mode='train')
    output = model(forward_inputs=rollout['forward_inputs'])
    torch.testing.assert_close(output['logprobs'], rollout['prev_logprobs'])
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_training_rollout_microbatch_preserves_alignment(
        robotwin_policy, robotwin_obs):
    model = robotwin_policy
    model.rollout_micro_batch_size = 2
    # RLinf EnvOutput includes optional observation channels with None values.
    robotwin_obs = dict(robotwin_obs, extra_view=None)
    actions, result = model.predict_action_batch(robotwin_obs, mode='train')
    assert len(actions) == 3
    for start in (0, 2):
        inputs = {
            key: value[start:start + 2]
            for key, value in result['forward_inputs'].items()
        }
        output = model(forward_inputs=inputs)
        torch.testing.assert_close(output['logprobs'],
                                   result['prev_logprobs'][start:start + 2])
    rng_before = torch.random.get_rng_state().clone()
    bootstrap = model.get_values(robotwin_obs)
    assert bootstrap.shape == (3, 1)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(bootstrap, result['prev_values'])


def test_robotwin_ode_same_flux_model(robotwin_policy, robotwin_obs):
    from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching
    model = robotwin_policy
    obs = model._prepare_obs(robotwin_obs)
    noise = torch.randn(3, 3, 32)
    expected = PI05FlowMatching.predict_action(
        model, **obs, noise=noise.clone())
    _, result = model.predict_action_batch(
        robotwin_obs, mode='eval', noise=noise)
    torch.testing.assert_close(result['forward_inputs']['model_action'],
                               expected.reshape(3, -1))


def test_sharded_checkpoint_strict(tiny_assets, tmp_path):
    model = build_vla_from_cfg(tiny_assets[0].model)
    state = model.state_dict()
    names = list(state)
    first, second = names[::2], names[1::2]
    save_file({key: state[key].clone()
               for key in first}, str(tmp_path / 'part1.safetensors'))
    save_file({key: state[key].clone()
               for key in second}, str(tmp_path / 'part2.safetensors'))
    index = {
        'weight_map':
        {key: f'part{1 if key in first else 2}.safetensors'
         for key in names}
    }
    path = tmp_path / 'model.safetensors.index.json'
    path.write_text(json.dumps(index))
    clone = build_vla_from_cfg(tiny_assets[0].model)
    load_sft_weights(clone, tmp_path)
    for key, value in state.items():
        torch.testing.assert_close(clone.state_dict()[key], value)
    index['weight_map'][names[0]] = '../outside.safetensors'
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='unsafe'):
        read_checkpoint(path)


def test_openpi_tied_alias_and_unused_expert(tiny_assets, tmp_path):
    model = build_vla_from_cfg(tiny_assets[0].model)
    mapping = {
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model'
    }
    model.name_mapping = mapping
    state = {}
    for key, tensor in model.state_dict().items():
        candidates = model._mapped_name_candidates(key)
        state[candidates[0][1] if candidates else key] = tensor.clone()
    embed_key = ('paligemma_with_expert.paligemma.model.'
                 'language_model.embed_tokens.weight')
    head_key = 'paligemma_with_expert.paligemma.lm_head.weight'
    state[embed_key] = model.llm_backbone.embed_tokens.weight.detach().clone()
    state[head_key] = state[embed_key].clone()
    state.pop('paligemma_with_expert.gemma_expert.model.embed_tokens.weight')
    expert_head_key = 'paligemma_with_expert.gemma_expert.lm_head.weight'
    state[expert_head_key] = (
        model.llm_expert.embed_tokens.weight.detach().clone())
    state['value_head.out.weight'] = torch.randn(
        1, 123)  # Reinitialized, not loaded.
    path = tmp_path / 'mapped.pt'
    torch.save(state, path)
    load_sft_weights(model, path)
    state[head_key][0, 0] += 1
    torch.save(state, path)
    with pytest.raises(ValueError, match='aliases disagree'):
        load_sft_weights(model, path)


def test_builder_preserves_fp32_norm_checkpoint(model_cfg, tmp_path):
    checkpoint = torch.load(model_cfg.model_path, weights_only=True)
    key = 'llm_backbone.layers.0.input_layernorm.weight'
    expected = torch.linspace(.0012345, .9987654,
                              checkpoint['model'][key].numel())
    assert not torch.equal(expected, expected.bfloat16().float())
    checkpoint['model'][key] = expected
    path = tmp_path / 'fp32_norm.pt'
    torch.save(checkpoint, path)
    cfg = copy.deepcopy(model_cfg)
    cfg.model_path = str(path)
    model = build_pi05_policy(cfg)
    torch.testing.assert_close(
        model.state_dict()[key], expected, rtol=0, atol=0)


def test_openpi_layer_norm_keeps_fp32_masters():
    from fluxvla.rl.models.pi05.precision import OpenPILayerNorm
    module = OpenPILayerNorm(16).float()
    data = torch.randn(2, 3, 16).bfloat16()
    reference = torch.nn.functional.layer_norm(data, (16, ),
                                               module.weight.bfloat16(),
                                               module.bias.bfloat16(),
                                               module.eps)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = module(data)
    assert actual.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in module.parameters())
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.float().sum().backward()
    assert module.weight.grad.isfinite().all()


@pytest.mark.parametrize('count,envs,rounds', [(150, 10, 15), (32, 8, 4),
                                               (968, 32, None)])
def test_seed_partition_coverage(count, envs, rounds):
    parts = [
        partition_seeds(range(count), seed=1234, rank=rank, world_size=2)
        for rank in range(2)
    ]
    all_seeds = torch.cat(parts).tolist()
    assert sorted(all_seeds) == list(range(count))
    if rounds:
        visited = []
        for part in parts:
            for turn in range(rounds):
                visited.extend(part[turn * (envs // 2):(turn + 1) *
                                    (envs // 2)].tolist())
        assert sorted(visited) == list(range(count))


def test_checkpoint_selection_no_test_leakage():
    records = [{
        'split': 'validation',
        'success_rate': score,
        'step': step
    } for step, score in [(20, 0.8), (10, 0.8), (30, 0.7)]]
    assert choose_best(records)['step'] == 10
    with pytest.raises(ValueError, match='validation'):
        choose_best([{'split': 'test', 'success_rate': 1., 'step': 40}])


def test_wandb_fail_closed(monkeypatch):
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='missing'):
        require_wandb()


@pytest.mark.parametrize('evaluation', [False, True])
def test_robotwin_configs(evaluation, tiny_assets, monkeypatch):
    prepare_environment()
    suffix = 'eval' if evaluation else 'ppo'
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(
            config_name=f'benchmarks/robotwin/pi05/{suffix}',
            overrides=[f'actor.model.model_path={tiny_assets[2]}'])
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg, evaluation=evaluation)
    from types import SimpleNamespace

    import rlinf.config as backend

    from fluxvla.rl.rlinf_registry import register
    register()
    monkeypatch.setattr(backend, 'Cluster', lambda *a, **k: None)
    monkeypatch.setattr(
        backend, 'HybridComponentPlacement',
        lambda *a, **k: SimpleNamespace(get_world_size=lambda name: 2))
    backend.validate_embodied_cfg(cfg)
    if evaluation:
        assert cfg.rollout.model.model_type == 'fluxvla_pi05'
        assert cfg.rollout.model.num_action_chunks == 50
    assert cfg.cluster.component_placement == {
        'actor': '0-1',
        'env,rollout': '2-3'
    }
    assert cfg.actor.global_batch_size == 64
    assert cfg.env.train.total_num_envs == 32
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == (
        150 if evaluation else 32)
