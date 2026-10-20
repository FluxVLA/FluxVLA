"""Tiny real SmolVLA model through the Flux adapter and RLinf PPO API."""

import copy
import json

import pytest
import torch
from mmengine import Config
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.vlas.pi05_rl_builder import load_sft_weights
from fluxvla.models.vlas.smolvla_rl_builder import build_smolvla_policy


@pytest.fixture(scope='session')
def smol_assets(tmp_path_factory, tiny_assets):
    root = tmp_path_factory.mktemp('smolvla_rl')
    cfg = Config.fromfile('configs/smolvla/smolvla_libero_10_finetune.py')
    model = cfg.model
    model.pretrained_name_or_path = None
    model.chunk_size = 3
    model.num_steps = 3
    model.enable_mixed_precision_training = False
    model.vlm_backbone.vision_config.update(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=4,
        intermediate_size=32)
    model.vlm_backbone.text_config.update(
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
            vocab={word: index
                   for index, word in enumerate(words)},
            unk_token='[UNK]'))
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
def smol_policy(smol_assets):
    _, config, weights, tokenizer, stats = smol_assets
    model_cfg = OmegaConf.create({
        'model_type': 'fluxvla_smolvla',
        'model_path': str(weights),
        'precision': 'fp32',
        'load_to_device': False,
        'is_lora': False,
        'num_action_chunks': 2,
        'action_dim': 7,
        'num_steps': 3,
        'joint_logprob': False,
        'add_value_head': True,
        'fluxvla': {
            'config_path': str(config),
            'tokenizer_path': str(tokenizer),
            'norm_stats_path': str(stats),
            'observation_adapter': {
                'type': 'LiberoRLAdapter'
            },
            'compute_dtype': 'fp32',
            'noise_level': 0.5
        }
    })
    return build_smolvla_policy(model_cfg)


def test_smol_eval_and_prefix_match_native(smol_policy, smol_assets, env_obs):
    original = build_vla_from_cfg(smol_assets[0].model).float().eval()
    load_sft_weights(original, smol_assets[2])
    obs = smol_policy._prepare_obs(env_obs)
    noise = torch.randn(3, 3, 32)
    saved = noise.clone()
    expected = original.predict_action(**obs, noise=noise.clone())
    actions, result = smol_policy.predict_action_batch(env_obs, noise=noise)
    torch.testing.assert_close(noise, saved, rtol=0, atol=0)
    torch.testing.assert_close(
        result['forward_inputs']['model_action'].reshape_as(expected),
        expected,
        rtol=0,
        atol=0)
    torch.testing.assert_close(
        actions, smol_policy.observation_adapter.env_actions(expected, 2, 7))


def test_smol_replay_and_real_ppo(smol_policy, env_obs):
    from test.test_rl.test_policy import \
        test_real_rlinf_ppo_updates_expert_and_critic

    _, old = smol_policy.predict_action_batch(
        env_obs, mode='train', rng=torch.Generator().manual_seed(20))
    smol_policy.train()
    output = smol_policy(
        forward_inputs=old['forward_inputs'], compute_entropy=True)
    torch.testing.assert_close(
        output['logprobs'], old['prev_logprobs'], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output['values'], old['prev_values'][:, 0])
    assert not smol_policy.vlm_backbone.training
    assert not any(p.requires_grad
                   for p in smol_policy.state_proj.parameters())
    test_real_rlinf_ppo_updates_expert_and_critic(smol_policy, env_obs,
                                                  torch.float32)
