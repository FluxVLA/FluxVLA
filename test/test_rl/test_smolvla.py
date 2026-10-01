"""Real tiny SmolVLM/Llama experts, real Flux transforms and RLinf PPO."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from mmengine import Config
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from fluxvla.engines import build_vla_from_cfg
from fluxvla.rl.models.builder import build_smolvla_policy, load_sft_weights
from fluxvla.rl.rlinf_registry import register
from fluxvla.rl.train import prepare_environment, validate_frontend_cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='session')
def smol_assets(tmp_path_factory, tiny_assets):
    root = tmp_path_factory.mktemp('smolvla_rl')
    cfg = Config.fromfile(
        str(ROOT / 'configs/smolvla/smolvla_libero_10_finetune.py'))
    model = cfg.model
    model.pretrained_name_or_path = None
    model.chunk_size = 3
    model.num_steps = 3
    model.enable_mixed_precision_training = False
    vision = model.vlm_backbone.vision_config
    vision.update(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=4,
        intermediate_size=32)
    text = model.vlm_backbone.text_config
    text.update(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=64,
        vocab_size=64,
        pad_token_id=0)
    model.vlm_backbone.update(
        scale_factor=2, num_vlm_layers=2, torch_dtype='float32')
    model.llm_expert.update(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        vocab_size=64,
        vlm_kv_dim=8,
        torch_dtype='float32')
    model.state_proj.out_dim = 32
    model.action_in_proj.out_dim = 16
    model.action_out_proj.in_dim = 16
    model.action_time_mlp_in.update(in_dim=32, out_dim=16)
    model.action_time_mlp_out.update(in_dim=16, out_dim=16)
    tokenizer_path = root / 'tokenizer'
    words = [
        '[PAD]', '[UNK]', 'pick', 'the', 'red', 'cube', 'place', 'blue', 'cup',
        'open', 'drawer'
    ]
    tokenizer = Tokenizer(
        WordLevel(
            vocab={word: i
                   for i, word in enumerate(words)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='[UNK]',
        pad_token='[PAD]').save_pretrained(tokenizer_path)
    cfg.eval.dataset.transforms[1].input_sizes = [[3, 16, 16]] * 2
    cfg.eval.dataset.transforms[2].max_len = 12
    cfg.eval.dataset.transforms[2].tokenizer.model_path = str(tokenizer_path)
    stats = copy.deepcopy(
        tiny_assets[0].train_dataloader.dataset.dataset_statistics)
    cfg.train_dataloader.dataset.dataset_statistics = stats
    stats_path = root / 'stats.json'
    stats_path.write_text(json.dumps(stats))
    config_path = root / 'tiny_smolvla.py'
    cfg.dump(str(config_path))
    torch.manual_seed(123)
    original = build_vla_from_cfg(model).float()
    weights = root / 'sft.safetensors'
    save_file(original.state_dict(), str(weights))
    return cfg, config_path, weights, tokenizer_path, stats_path


@pytest.fixture
def smol_cfg(smol_assets):
    _, config, weights, tokenizer, stats = smol_assets
    return OmegaConf.create(
        dict(
            model_type='fluxvla_smolvla',
            model_path=str(weights),
            precision='fp32',
            load_to_device=False,
            is_lora=False,
            num_action_chunks=2,
            action_dim=7,
            num_steps=3,
            joint_logprob=False,
            add_value_head=True,
            fluxvla=dict(
                config_path=str(config),
                tokenizer_path=str(tokenizer),
                norm_stats_path=str(stats),
                compute_dtype='fp32',
                noise_level=0.5)))


@pytest.fixture
def smol_policy(smol_cfg):
    return build_smolvla_policy(smol_cfg)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_smol_eval_and_prefix_match_native(smol_policy, smol_assets, env_obs,
                                           dtype):
    original = build_vla_from_cfg(smol_assets[0].model).float().eval()
    load_sft_weights(original, smol_assets[2])
    policy = smol_policy
    policy.compute_dtype = dtype
    obs = policy._prepare_obs(env_obs)
    noise = torch.randn(3, 3, 32)
    saved = noise.clone()
    from fluxvla.engines.utils.model_utils import make_att_2d_masks
    with torch.no_grad(), policy._network_context():
        expected = original.predict_action(**obs, noise=noise.clone())
        prefix, mask, attention = original.embed_prefix(
            obs['images'], obs['img_masks'], obs['lang_tokens'],
            obs['lang_masks'], obs['states'])
        _, cache = original.forward_model(
            attention_mask=make_att_2d_masks(mask, attention),
            position_ids=mask.cumsum(1) - 1,
            inputs_embeds=[prefix, None],
            use_cache=True)
        velocity = original._denoise_step(noise, mask, cache, torch.ones(3))
    _, actual_mask, actual_cache = policy._prefix(obs)
    actual_velocity = policy._velocity(obs, actual_mask, actual_cache, noise,
                                       torch.ones(3))
    torch.testing.assert_close(
        actual_velocity, velocity.float(), rtol=0, atol=0)
    for left, right in zip(cache, actual_cache):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    actions, result = policy.predict_action_batch(env_obs, noise=noise)
    torch.testing.assert_close(noise, saved, rtol=0, atol=0)
    torch.testing.assert_close(
        result['forward_inputs']['model_action'].reshape_as(expected),
        expected,
        rtol=0,
        atol=0)
    torch.testing.assert_close(
        actions, policy.observation_adapter.env_actions(expected, 2, 7))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_smol_replay_and_real_ppo(smol_policy, env_obs, dtype):
    from test.test_rl.test_policy import \
        test_real_rlinf_ppo_updates_expert_and_critic

    policy = smol_policy
    policy.compute_dtype = dtype
    _, old = policy.predict_action_batch(
        env_obs, mode='train', rng=torch.Generator().manual_seed(20))
    chain = old['forward_inputs']['chains'].clone()
    policy.train()
    output = policy(forward_inputs=old['forward_inputs'], compute_entropy=True)
    torch.testing.assert_close(
        output['logprobs'], old['prev_logprobs'], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output['values'], old['prev_values'][:, 0])
    torch.testing.assert_close(
        chain, old['forward_inputs']['chains'], rtol=0, atol=0)
    assert all(
        torch.isfinite(v).all() and v.dtype == torch.float32
        for v in output.values())
    assert chain.shape == (3, 4, 3, 32)
    assert not policy.vlm_backbone.training
    assert not any(p.requires_grad for p in policy.state_proj.parameters())
    test_real_rlinf_ppo_updates_expert_and_critic(policy, env_obs, dtype)


def test_smol_observation_matches_flux(smol_assets):
    from test.test_rl.test_observation import \
        test_raw_flux_and_canonical_rlinf_preprocessing_match
    test_raw_flux_and_canonical_rlinf_preprocessing_match(smol_assets[:4])


def test_smol_trajectory_and_bootstrap(smol_policy, env_obs):
    from test.test_rl.test_backend import \
        test_policy_output_split_merge_and_trajectory_alignment
    test_policy_output_split_merge_and_trajectory_alignment(
        smol_policy, env_obs)
    before = torch.random.get_rng_state().clone()
    values = smol_policy.get_values(env_obs)
    assert torch.equal(before, torch.random.get_rng_state())
    _, output = smol_policy.predict_action_batch(env_obs)
    torch.testing.assert_close(values, output['prev_values'])


@pytest.mark.parametrize('mapped', [False, True])
def test_smol_loading_and_restore(smol_policy, smol_cfg, env_obs, tmp_path,
                                  mapped):
    state = {
        key: value
        for key, value in smol_policy.state_dict().items()
        if not key.startswith('value_head.')
    }
    if mapped:
        state = {
            smol_policy._mapped_name_candidates(key)[0][1]: value
            for key, value in state.items()
        }
    path = tmp_path / 'weights.pt'
    torch.save({'model': state}, path)
    smol_cfg.model_path = str(path)
    restored = build_smolvla_policy(smol_cfg)
    restored.load_state_dict(smol_policy.state_dict(), strict=True)
    noise = torch.randn(3, 3, 32)
    for key, value in smol_policy.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[key], value, rtol=0, atol=0)
    left = smol_policy.predict_action_batch(env_obs, noise=noise)[0]
    right = restored.predict_action_batch(env_obs, noise=noise)[0]
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    state.pop(next(iter(state)))
    torch.save(state, path)
    with pytest.raises(ValueError, match='checkpoint'):
        build_smolvla_policy(smol_cfg)


@pytest.mark.parametrize('phase', ['ppo', 'eval'])
def test_smol_config_cli_and_registration(smol_assets, phase):
    prepare_environment()
    register()
    name = f'benchmarks/libero/smolvla/{phase}'
    overrides = [
        f'actor.model.model_path={smol_assets[2]}',
        f'actor.model.fluxvla.tokenizer_path={smol_assets[3]}',
        f'actor.model.fluxvla.norm_stats_path={smol_assets[4]}'
    ]
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(config_name=name, overrides=overrides)
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg, evaluation=phase == 'eval')
    assert cfg.actor.model.model_type == 'fluxvla_smolvla'
    assert cfg.actor.model.fluxvla.action_horizon == 50
    assert cfg.actor.model.num_action_chunks == 10
    module = 'train' if phase == 'ppo' else 'eval'
    result = subprocess.run([
        sys.executable, '-m', f'fluxvla.rl.{module}', f'--config-name={name}',
        '--cfg', 'job', '--resolve', *overrides
    ],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    cfg.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap = [
        'LlamaDecoderLayer'
    ]
    with pytest.raises(ValueError, match='interleaved'):
        validate_frontend_cfg(cfg, evaluation=phase == 'eval')


def test_smol_worker_registration(smol_cfg):
    register()
    from rlinf.models import get_model
    assert type(get_model(smol_cfg)).__name__ == 'FluxSmolVLARLPolicy'
    code = '''
from rlinf.scheduler.cluster.utils import load_user_extension_module
load_user_extension_module()
from rlinf.models import _MODEL_REGISTRY
from rlinf.config import SupportedModel, EMBODIED_MODEL
assert 'fluxvla_smolvla' in _MODEL_REGISTRY
assert SupportedModel('fluxvla_smolvla') in EMBODIED_MODEL
'''
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=ROOT,
                            env=dict(
                                os.environ,
                                RLINF_EXT_MODULE='fluxvla.rl.rlinf_registry'),
                            capture_output=True,
                            text=True,
                            timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_smol_mode_and_microbatch(smol_policy, env_obs):
    import inspect
    from types import SimpleNamespace

    from rlinf.workers.rollout.hf.huggingface_worker import \
        MultiStepRolloutWorker

    from fluxvla.rl.workers.rollout import FluxRolloutWorker

    register()
    worker = object.__new__(FluxRolloutWorker)
    worker.cfg = OmegaConf.create({'rollout': {}, 'algorithm': {}})
    worker.algorithm_cfg = worker.cfg.algorithm
    worker.model_cfg = OmegaConf.create({'model_type': 'fluxvla_smolvla'})
    worker.expert_model = None
    worker.enable_dagger = False
    worker.setup_sample_params()
    modes = []

    def predict(env_obs, **kwargs):
        modes.append(kwargs['mode'])
        return torch.zeros(3, 2, 7), {}

    worker.hf_model = SimpleNamespace(predict_action_batch=predict)
    for mode in ('train', 'eval'):
        inspect.unwrap(MultiStepRolloutWorker.predict)(
            worker, env_obs, mode=mode)
    assert modes == ['train', 'eval']
    smol_policy.rollout_micro_batch_size = 2
    smol_policy.compute_dtype = torch.bfloat16
    _, old = smol_policy.predict_action_batch(env_obs, mode='train')
    for start in (0, 2):
        inputs = {
            key: tensor[start:start + 2]
            for key, tensor in old['forward_inputs'].items()
        }
        out = smol_policy(forward_inputs=inputs)
        torch.testing.assert_close(
            out['logprobs'],
            old['prev_logprobs'][start:start + 2],
            rtol=1e-5,
            atol=1e-6)
    worker.hf_model = smol_policy
    torch.testing.assert_close(
        worker.get_bootstrap_values(env_obs), old['prev_values'])


def test_smol_value_mask_and_prefix_freeze(smol_policy, env_obs):
    obs = smol_policy._prepare_obs(env_obs)
    hidden, mask, cache = smol_policy._prefix(obs)
    assert not hidden.requires_grad
    assert all(not value.requires_grad for pair in cache for value in pair)
    changed = hidden.clone()
    changed[~mask] = 1e6
    torch.testing.assert_close(
        smol_policy._value(hidden, mask),
        smol_policy._value(changed, mask),
        rtol=0,
        atol=0)


@pytest.mark.skipif(
    os.environ.get('RUN_SMOLVLA_GPU_PROBE') != '1',
    reason='Opt in explicitly to the two-GPU NCCL probe')
def test_smol_gpu_fsdp(smol_cfg, tmp_path):
    config_path = tmp_path / 'model.yaml'
    OmegaConf.save(smol_cfg, config_path)
    result = subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2',
        str(ROOT / 'test/test_rl/helpers/smolvla_fsdp.py'), '--model-config',
        str(config_path), '--output',
        str(tmp_path)
    ],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        report = json.loads((tmp_path / f'rank{rank}.json').read_text())
        assert report['passed'] and report['checkpoint_restored']
        assert report['preupdate_logprob_max_abs_drift'] <= 1e-5
