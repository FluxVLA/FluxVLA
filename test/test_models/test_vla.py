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

import gc
import json
import os
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import (VLM_BACKBONES, build_vla_from_cfg,
                             set_seed_everywhere)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_CKPT_PATH = str(PROJECT_ROOT /
                        'checkpoints/openvla-7b-finetuned-libero-10')
LLAMA2_CKPT_PATH = str(PROJECT_ROOT / 'checkpoints/Llama-2-7b-hf')
DINO_CKPT_PATH = str(
    PROJECT_ROOT /
    'checkpoints/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors')
SIGLIP_CKPT_PATH = str(
    PROJECT_ROOT /
    'checkpoints/ViT-SO400M-14-SigLIP/open_clip_model.safetensors')
PI0_CKPT_PATH = str(PROJECT_ROOT / 'checkpoints/pi0_base/model.safetensors')
PI05_CKPT_PATH = str(PROJECT_ROOT / 'checkpoints/pi05_base/model.safetensors')
GR00T_CKPT_PATH = str(PROJECT_ROOT / 'checkpoints/GR00T-N1.5-3B')
DREAMZERO_CKPT_PATH = str(PROJECT_ROOT / 'checkpoints/DreamZero-AgiBot')
SMOLVLA_CKPT_PATH = str(PROJECT_ROOT /
                        'checkpoints/smolvla_base/model.safetensors')
DIT4DIT_CKPT_PATH = str(
    PROJECT_ROOT /
    'checkpoints/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt')
OPENVLA_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/vlas/openvla')
LLAVAVLA_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/vlas/llavavla')
GR00T_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/vlas/gr00t')
PI0_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/vlas/pi0')
PI05_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/vlas/pi05')
DREAMZERO_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/vlas/dreamzero')
DIT4DIT_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/vlas/dit4dit')
DREAMZERO_NUM_INFERENCE_STEPS = 2


def _tiny_vision_llm_config(vocab_size=64):
    return dict(
        vision_backbone=dict(
            type='SigLIPViTBackbone',
            vision_backbone_id='siglip_224',
            vision_config=dict(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
                attention_dropout=0.0)),
        llm_backbone=dict(
            type='LLaMa2LLMBackbone',
            llm_backbone_id='llama2-7b-pure_causal',
            llm_family='llama',
            llm_path=None,
            tokenizer_length=vocab_size,
            llm_config=dict(
                vocab_size=vocab_size,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=64,
                attention_dropout=0.0,
                output_hidden_states=True)),
        projector=dict(type='LinearProjector', in_dim=16, out_dim=16),
        enable_mixed_precision_training=False,
        freeze_vision_backbone=False,
        freeze_llm_backbone=False)


def test_tiny_openvla_forward_backward_and_predict_action(
        monkeypatch, request):
    from fluxvla.models.backbones.visions.configs import \
        VISION_BACKBONE_CONFIGS

    # The isolated CPU job uses native timm attention. The normal GPU suite
    # retains the production RADIO/CUDA attention path and BF16 coverage.
    device = 'cpu' if request.config.getoption('--cpu-model-tests') else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('The production import path requires CUDA.')
    # Match OpenVLA's dual-timm vision contract with smaller real ViTs. Only
    # architecture IDs and external tokenizer loading are substituted.
    for name in ('test_dino', 'test_siglip'):
        monkeypatch.setitem(VISION_BACKBONE_CONFIGS, name,
                            dict(model_id='vit_tiny_patch16_224'))
    monkeypatch.setattr(
        'fluxvla.tokenizers.action_tokenizer.AutoTokenizer.from_pretrained',
        lambda *args, **kwargs: SimpleNamespace(vocab_size=32000))
    backbone_cfg = _tiny_vision_llm_config(vocab_size=32064)
    backbone_cfg.update(
        vision_backbone=dict(
            type='DinoSigLIPViTBackbone',
            vision_backbone_id='dinosiglip-vit-so-224px',
            dino_config=dict(model_id='test_dino'),
            siglip_config=dict(model_id='test_siglip'),
            pretrained=False,
            img_size=16),
        projector=dict(type='LinearProjector', in_dim=384, out_dim=16),
        enable_mixed_precision_training=device == 'cuda')
    policy = build_vla_from_cfg(
        dict(
            backbone_cfg,
            type='OpenVLA',
            vla_head=dict(
                type='OpenVLAHead', vocab_size=32000, norm_stats=None),
            tokenizer=dict(
                type='ActionTokenizer', model_path='unused', bins=256),
            norm_stats={
                'test': {
                    'action': {
                        'q01': [-1.] * 3,
                        'q99': [1.] * 3
                    }
                }
            },
        )).to(device).eval()
    images = torch.linspace(
        -1, 1, 6 * 16 * 16, device=device).reshape(1, 6, 16, 16)
    tokens = torch.tensor([[1, 7, 31990, 31991, 31992]], device=device)
    labels = tokens.clone()
    labels[:, :2] = -100
    with torch.autocast(
            device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        output = policy(
            images=images,
            lang_tokens=tokens,
            lang_masks=torch.ones_like(tokens),
            labels=labels,
            dataset_names=['test'])
    assert policy.vision_backbone.num_patches == 1
    assert output['predictions'].shape == (1, 6, 32064)
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    for module in (policy.vision_backbone, policy.llm_backbone,
                   policy.projector):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    with torch.no_grad(), torch.autocast(
            device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        first = policy.predict_action(images=images, lang_tokens=tokens[:, :2])
        second = policy.predict_action(
            images=images, lang_tokens=tokens[:, :2])
    assert first.shape == (1, 3)
    assert torch.isfinite(first).all() and (first.abs() <= 1).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_tiny_llava_flow_matching_forward_backward_and_predict_action():
    """GR00T-style action DiT and wrapper with tiny real backbones."""
    backbone_cfg = _tiny_vision_llm_config()
    # Continuous actions consume hidden states, not a causal LM's logits.
    backbone_cfg['llm_backbone'].update(
        type='Qwen2LLMBackbone',
        llm_backbone_id='qwen2-0.5b',
        llm_family='qwen2')
    policy = build_vla_from_cfg(
        dict(
            backbone_cfg,
            type='LlavaVLA',
            vla_head=dict(
                type='FlowMatchingHead',
                state_dim=4,
                hidden_size=16,
                input_embedding_dim=16,
                backbone_embedding_dim=16,
                num_inference_timesteps=2,
                num_steps=3,
                action_dim=4,
                ori_action_dim=3,
                max_num_embodiments=1,
                num_target_vision_tokens=2,
                max_seq_len=16,
                vl_self_attention_cfg=dict(
                    attention_head_dim=8,
                    num_attention_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    final_dropout=False),
                diffusion_model_cfg=dict(
                    attention_head_dim=8,
                    num_attention_heads=2,
                    num_layers=2,
                    cross_attention_dim=16,
                    output_dim=16,
                    dropout=0.0,
                    final_dropout=False)),
        )).eval()
    inputs = dict(
        images=torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8),
        lang_tokens=torch.tensor([[1, 5, 2]]),
        lang_masks=torch.ones(1, 3, dtype=torch.bool),
        states=torch.zeros(1, 4),
        embodiment_ids=torch.zeros(1, dtype=torch.long))
    actions = torch.linspace(-0.5, 0.5, 12).reshape(1, 3, 4)
    result = policy(
        **inputs,
        actions=actions,
        action_masks=torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(result['loss'])
    assert result['pred_actions'].shape == (1, 3, 3)
    result['loss'].backward()
    for module in (policy.vision_backbone, policy.llm_backbone,
                   policy.vla_head):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    results = []
    for _ in range(2):
        torch.manual_seed(23)
        with torch.no_grad():
            results.append(policy.predict_action(**inputs))
    assert results[0].shape == (1, 3, 3)
    assert torch.isfinite(results[0]).all()
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.fixture(params=['PI0FlowMatching', 'PI05FlowMatching'])
def tiny_pi_policy(request):
    """Actual SigLIP + Gemma backbone/expert, including cached inference."""
    gemma = dict(
        type='ConditionGemmaModel',
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        adarms_cond_dim=None,
        hidden_activation='gelu_pytorch_tanh')

    def linear(in_dim, out_dim):
        return dict(type='LinearProjector', in_dim=in_dim, out_dim=out_dim)

    cfg = dict(
        type=request.param,
        llm_backbone=gemma,
        llm_expert=dict(gemma),
        vision_backbone=dict(
            type='SigLIPViTBackbone',
            vision_backbone_id='siglip_224',
            vision_config=dict(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
                attention_dropout=0.0)),
        projector=linear(16, 16),
        proj_width=16,
        action_in_proj=linear(4, 16),
        action_out_proj=linear(16, 4),
        max_action_dim=4,
        ori_action_dim=3,
        n_action_steps=3,
        num_steps=2,
        enable_mixed_precision_training=False,
        freeze_vision_backbone=False,
        freeze_llm_backbone=False)
    if request.param == 'PI0FlowMatching':
        cfg.update(
            state_proj=linear(4, 16),
            action_time_mlp_in=linear(32, 16),
            action_time_mlp_out=linear(16, 16))
    else:
        cfg.update(time_mlp_in=linear(16, 16), time_mlp_out=linear(16, 16))
        cfg['llm_expert']['adarms_cond_dim'] = 16
    policy = build_vla_from_cfg(cfg).float().eval()
    inputs = dict(
        images=torch.linspace(-1, 1, 2 * 6 * 8 * 8).reshape(2, 6, 8, 8),
        img_masks=torch.tensor([[True, True], [True, False]]),
        lang_tokens=torch.tensor([[1, 5, 2, 0], [1, 7, 3, 2]]),
        lang_masks=torch.tensor([[True, True, True, False], [True] * 4]),
        states=torch.linspace(-0.5, 0.5, 8).reshape(2, 4),
        noise=torch.linspace(-1, 1, 24).reshape(2, 3, 4),
    )
    return policy, inputs


def test_tiny_pi_forward_backward(tiny_pi_policy):
    policy, inputs = tiny_pi_policy
    actions = torch.linspace(-0.5, 0.5, 24).reshape(2, 3, 4)
    masks = torch.tensor([[True, True, False], [True, False, False]])
    time = torch.tensor([0.25, 0.75])
    rng = torch.random.get_rng_state().clone()
    # PI attention masks are BF16; exercise mixed precision with FP32 masters.
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = policy(
            **inputs, actions=actions, action_masks=masks, time=time)
    assert result['predictions'].shape == (2, 3, 3)
    target = (inputs['noise'] - actions)[..., :3]
    errors = (result['predictions'].float() - target).square()
    torch.testing.assert_close(result['loss'], errors[masks].mean())
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    result['loss'].backward()
    for module in (policy.vision_backbone, policy.llm_backbone,
                   policy.llm_expert, policy.action_out_proj):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


def test_tiny_pi_predict_action(tiny_pi_policy):
    policy, inputs = tiny_pi_policy
    noise = inputs['noise'].clone()
    rng = torch.random.get_rng_state().clone()
    # The Euler solver updates its initial noise in place. Each invocation
    # needs its own copy of the same starting state.
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        first = policy.predict_action(**{
            **inputs, 'noise': noise.clone()
        }).clone()
        second = policy.predict_action(**{**inputs, 'noise': noise.clone()})
    assert first.shape == (2, 3, 4)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(inputs['noise'], noise, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def _load_dit4dit_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def _load_dit4dit_action_state_dict(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
    state_dict = checkpoint.get('model', checkpoint)
    prefixes = ('action_model.', 'module.action_model.')
    action_state = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                action_state[key[len(prefix):]] = value
                break
    if not action_state:
        raise KeyError('Could not find action_model weights in checkpoint.')
    return action_state


def _dit4dit_artifacts_exist():
    return all(
        os.path.exists(path) for path in (
            DIT4DIT_CKPT_PATH,
            os.path.join(DIT4DIT_DATA_DIR, 'inputs.pkl'),
            os.path.join(DIT4DIT_DATA_DIR, 'source_outputs.pkl'),
            os.path.join(DIT4DIT_DATA_DIR, 'metadata.json'),
        ))


class DiT4DiTParityBackbone(nn.Module):

    def __init__(self, inputs_path):
        super().__init__()
        self.inputs_path = inputs_path

    @property
    def transformer_layer_cls(self):
        return nn.Identity

    def forward(self, images=None, **kwargs):
        data = _load_dit4dit_pickle(self.inputs_path)
        hidden_states = data['vl_embs']
        if torch.is_tensor(images):
            hidden_states = hidden_states.to(
                device=images.device, dtype=images.dtype)
        return SimpleNamespace(hidden_states=[hidden_states])


@pytest.fixture
def dit4dit_parity_backbone(monkeypatch):
    monkeypatch.setitem(VLM_BACKBONES.module_dict, 'DiT4DiTParityBackbone',
                        DiT4DiTParityBackbone)


@pytest.mark.skipif(
    not os.path.exists(OPENVLA_CKPT_PATH)
    or not os.path.exists(LLAMA2_CKPT_PATH)
    or not os.path.exists(DINO_CKPT_PATH)
    or not os.path.exists(SIGLIP_CKPT_PATH),
    reason=f'Checkpoint not found: {OPENVLA_CKPT_PATH}')
@pytest.mark.checkpoint
class TestOpenVLA(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='OpenVLA',
            vision_backbone=dict(
                type='DinoSigLIPViTBackbone',
                vision_backbone_id='dinosiglip-vit-so-224px',
                dino_config=dict(
                    model_id='dino',
                    file=  # noqa: E251
                    DINO_CKPT_PATH),
                siglip_config=dict(
                    model_id='siglip_224',
                    file=  # noqa: E251
                    SIGLIP_CKPT_PATH)),
            llm_backbone=dict(
                type='LLaMa2LLMBackbone',
                llm_backbone_id='llama2-7b-pure_causal',
                llm_family='llama',
                llm_path=  # noqa: E251
                LLAMA2_CKPT_PATH,  # noqa: E501
                llm_max_length=2048,
                hf_token=None,
                inference_mode=False),
            projector=dict(
                type='FusedMLPProjector', fused_vision_dim=2176, llm_dim=4096),
            tokenizer=dict(
                type='ActionTokenizer',
                model_path=  # noqa: E251
                OPENVLA_CKPT_PATH,  # noqa: E501
                bins=256,
                min_action=-1,
                max_action=1,
            ),
            pretrained_name_or_path=None,  # noqa: E501
            vla_head=dict(
                type='OpenVLAHead', norm_stats=None, vocab_size=32000),
            freeze_vision_backbone=False,
            freeze_llm_backbone=False,
            freeze_projector=False)
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_forward(self):
        input_ids = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR, 'input_ids.npy'))).cuda()
        attention_mask = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR,
                                 'attention_mask.npy'))).cuda()
        pixel_values_dino = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR,
                                 'pixel_values_dino.npy'))).cuda()
        pixel_values_siglip = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR,
                                 'pixel_values_siglip.npy'))).cuda()
        pixel_values = torch.cat([pixel_values_dino, pixel_values_siglip],
                                 dim=1)
        labels = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR, 'labels.npy'))).cuda()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                output, _ = self.vla.forward_model(input_ids, attention_mask,
                                                   pixel_values.bfloat16(),
                                                   labels)
        expected_loss = np.load(os.path.join(OPENVLA_DATA_DIR, 'loss.npy'))
        expected_logits = np.load(os.path.join(OPENVLA_DATA_DIR, 'logits.npy'))
        self.assertAlmostEqual(
            output['loss'].cpu().detach().numpy(), expected_loss, delta=1e-2)
        np.testing.assert_allclose(
            output['logits'].cpu().float().detach().numpy()[:, ::10, ::10],
            expected_logits,
            rtol=1e-3,
            atol=1e-1,
            equal_nan=False)


@pytest.mark.skipif(
    not os.path.exists(GR00T_CKPT_PATH),
    reason=f'Checkpoint not found: {GR00T_CKPT_PATH}')
@pytest.mark.checkpoint
class TestGr00t(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='LlavaVLA',
            pretrained_name_or_path=  # noqa: E251
            './checkpoints/GR00T-N1.5-3B',
            vlm_backbone=dict(
                type='EagleBackbone',
                vlm_path=  # noqa: E251
                'fluxvla/models/third_party_models/eagle2_hg_model'),
            vla_head=dict(
                type='FlowMatchingHead',
                state_dim=64,
                hidden_size=1024,
                input_embedding_dim=1536,
                num_inference_timesteps=4,
                num_steps=10,
                zero_padded_action_dims=False,
                clamp_sample_time=False,
                action_dim=32,
                ori_action_dim=7),
            freeze_vlm_backbone=False,
            name_mapping={
                'vlm_backbone.vlm': 'backbone.eagle_model',
                'vla_head': 'action_head'
            },
            freeze_projector=False)
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def test_forward(self):
        images = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda().repeat_interleave(
            8, dim=2).repeat_interleave(
                8, dim=3)
        img_masks = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'img_masks.npy'),
            allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = torch.from_numpy(
            np.load(os.path.join(LLAVAVLA_DATA_DIR, 'states.npy'))).cuda()
        states = F.pad(states, (0, 64 - states.shape[-1]))
        actions = torch.from_numpy(
            np.load(os.path.join(LLAVAVLA_DATA_DIR, 'actions.npy'))).cuda()
        actions = F.pad(actions, (0, 32 - actions.shape[-1]))
        embodiment_ids = torch.ones((2)).cuda().long()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                output = self.vla.forward(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states.bfloat16(),
                    actions=actions.bfloat16(),
                    action_masks=torch.ones((1, 10)).cuda(),
                    embodiment_ids=embodiment_ids)
        np.testing.assert_allclose(
            output['loss'].cpu().detach().numpy(),
            0.5135,
            atol=1e-2,
            rtol=1e-5,
            equal_nan=False)

    def test_predict_action(self):
        images = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda().repeat_interleave(
            8, dim=2).repeat_interleave(
                8, dim=3)
        img_masks = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'img_masks.npy'),
            allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = torch.from_numpy(
            np.load(os.path.join(LLAVAVLA_DATA_DIR, 'states.npy'))).cuda()
        states = F.pad(states, (0, 64 - states.shape[-1]))
        pred_actions_target = np.load(
            os.path.join(GR00T_DATA_DIR, 'pred_actions.npy'),
            allow_pickle=True)
        embodiment_ids = torch.ones((2)).cuda().long()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                pred_actions = self.vla.predict_action(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                    embodiment_ids=embodiment_ids)
        np.testing.assert_allclose(
            pred_actions.cpu().detach().numpy(),
            pred_actions_target,
            atol=1e-2,
            rtol=1e-5,
            equal_nan=False)


@pytest.mark.skipif(
    not os.path.exists(PI0_CKPT_PATH),
    reason=f'Checkpoint not found: {PI0_CKPT_PATH}')
@pytest.mark.checkpoint
class TestPI0FlowMatching(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='PI0FlowMatching',
            llm_backbone=dict(
                type='ConditionGemmaModel',
                adarms_cond_dim=None,
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act='gelu_pytorch_tanh',
                hidden_activation='gelu_pytorch_tanh',
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=16384,
                max_position_embeddings=8192,
                model_type='gemma',
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                use_cache=True,
                vocab_size=257152,
            ),
            vision_backbone=dict(
                type='SigLIPViTBackbone',
                vision_backbone_id='siglip_224',
                vision_config=dict(
                    attention_dropout=0.0,
                    hidden_act='gelu_pytorch_tanh',
                    hidden_size=1152,
                    image_size=224,
                    intermediate_size=4304,
                    layer_norm_eps=1e-06,
                    model_type='siglip_vision_model',
                    num_attention_heads=16,
                    num_channels=3,
                    num_hidden_layers=27,
                    patch_size=14,
                    projection_dim=2048,
                    projector_hidden_act='gelu_fast',
                    torch_dtype='float32',
                    vision_use_head=False,
                ),
            ),
            projector=dict(
                type='LinearProjector',
                in_dim=1152,
                out_dim=2048,
            ),
            proj_width=1024,
            n_action_steps=50,
            state_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
            action_in_proj=dict(
                type='LinearProjector', in_dim=32, out_dim=1024),
            action_out_proj=dict(
                type='LinearProjector', in_dim=1024, out_dim=32),
            action_time_mlp_in=dict(
                type='LinearProjector', in_dim=2048, out_dim=1024),
            action_time_mlp_out=dict(
                type='LinearProjector', in_dim=1024, out_dim=1024),
            max_action_dim=32,
            llm_expert=dict(
                type='ConditionGemmaModel',
                attention_bias=False,
                adarms_cond_dim=None,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act='gelu_pytorch_tanh',
                hidden_activation='gelu_pytorch_tanh',
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type='gemma',
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                transformers_version='4.48.1',
                use_adarms=False,
                use_cache=True,
                vocab_size=257152),
            freeze_llm_backbone=False,
            freeze_vision_backbone=False,
            pretrained_name_or_path=  # noqa: E251
            './checkpoints/pi0_base/model.safetensors',  # noqa: E501
            name_mapping={
                'llm_backbone':
                'paligemma_with_expert.paligemma.model.language_model',
                'vision_backbone.vision':
                'paligemma_with_expert.paligemma.model.vision_tower',
                'projector.projector':
                'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',  # noqa: E501
                'llm_expert':
                'paligemma_with_expert.gemma_expert.model',
                'action_time_mlp_in.projector':
                'action_time_mlp_in',
                'action_time_mlp_out.projector':
                'action_time_mlp_out',
                'state_proj.projector':
                'state_proj',
                'action_in_proj.projector':
                'action_in_proj',
                'action_out_proj.projector':
                'action_out_proj',
                'llm_backbone.embed_tokens':
                'paligemma_with_expert.paligemma.lm_head',
            },
            params_to_change_dtype=[
                'llm_expert.llm.model.layers',
                'vlm_backbone.vlm.model.language_model.layers',
                'vlm_backbone.vlm.model.vision_tower',
                'vlm_backbone.vlm.model.multi_modal_projector',
            ],
            ori_action_dim=7,
        )
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def test_prefix_forward(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        embs_target = np.load(
            os.path.join(PI0_DATA_DIR, 'prefix_embs.npy'), allow_pickle=True)
        pad_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'prefix_pad_masks.npy'),
            allow_pickle=True)
        att_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'prefix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                embs, pad_masks, att_masks = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks)
        np.testing.assert_allclose(
            embs.float().cpu().detach().numpy()[:, ::10, ::10],
            embs_target,
            atol=1e-1,
            rtol=1e-5,
            equal_nan=False)
        np.testing.assert_allclose(
            pad_masks.float().cpu().detach().numpy(),
            pad_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)
        np.testing.assert_allclose(
            att_masks.float().cpu().detach().numpy(),
            att_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)

    def test_suffix_forward(self):
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        suffix_embs_target = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_embs.npy'), allow_pickle=True)
        suffix_pad_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_pad_masks.npy'),
            allow_pickle=True)
        suffix_att_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

        np.testing.assert_allclose(
            suffix_embs.float().cpu().detach().numpy()[:, :, ::10],
            suffix_embs_target,
            atol=1e-2,
            rtol=1e-5,
            equal_nan=False)
        np.testing.assert_allclose(
            suffix_pad_masks.float().cpu().detach().numpy(),
            suffix_pad_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)
        np.testing.assert_allclose(
            suffix_att_masks.float().cpu().detach().numpy(),
            suffix_att_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)

    def test_forward(self):
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        actions_target = np.load(
            os.path.join(PI0_DATA_DIR, 'actions.npy'), allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                ) = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                )

                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1

                att_2d_masks_4d = self.vla._prepare_attention_masks_4d(
                    att_2d_masks)

                suffix_out, _ = self.vla.forward_model(
                    inputs_embeds=[prefix_embs, suffix_embs],
                    attention_masks=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    fill_kv_cache=None,
                    adarms_cond=[None, adarms_cond],
                    time=time)

                actions = self.vla.action_out_proj(
                    suffix_out[:, -self.vla.n_action_steps:])

                np.testing.assert_allclose(
                    actions.float().cpu().detach().numpy(),
                    actions_target,
                    atol=1e-1,
                    rtol=1e-5,
                    equal_nan=False)

    def test_predict_actions(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        noise = np.load(
            os.path.join(PI0_DATA_DIR, 'noise.npy'), allow_pickle=True)
        noise = torch.from_numpy(noise).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        pred_actions_target = np.load(
            os.path.join(PI0_DATA_DIR, 'pred_actions.npy'), allow_pickle=True)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float32, enabled=True):
                actions = self.vla.predict_action(
                    images=images,
                    states=states,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    noise=noise)

        np.testing.assert_allclose(
            actions.float().cpu().detach().numpy(),
            pred_actions_target,
            atol=5e-1,
            rtol=1e-5,
            equal_nan=False)


@pytest.mark.skipif(
    not os.path.exists(PI05_CKPT_PATH),
    reason=f'Checkpoint not found: {PI05_CKPT_PATH}')
@pytest.mark.checkpoint
class TestPI05FlowMatching(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='PI05FlowMatching',
            llm_backbone=dict(
                type='ConditionGemmaModel',
                adarms_cond_dim=None,
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act='gelu_pytorch_tanh',
                hidden_activation='gelu_pytorch_tanh',
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=16384,
                max_position_embeddings=8192,
                model_type='gemma',
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                use_cache=True,
                vocab_size=257152,
            ),
            vision_backbone=dict(
                type='SigLIPViTBackbone',
                vision_backbone_id='siglip_224',
                vision_config=dict(
                    attention_dropout=0.0,
                    hidden_act='gelu_pytorch_tanh',
                    hidden_size=1152,
                    image_size=224,
                    intermediate_size=4304,
                    layer_norm_eps=1e-06,
                    model_type='siglip_vision_model',
                    num_attention_heads=16,
                    num_channels=3,
                    num_hidden_layers=27,
                    patch_size=14,
                    projection_dim=2048,
                    projector_hidden_act='gelu_fast',
                    torch_dtype='float32',
                    vision_use_head=False,
                ),
            ),
            projector=dict(
                type='LinearProjector',
                in_dim=1152,
                out_dim=2048,
            ),
            proj_width=1024,
            n_action_steps=10,
            action_in_proj=dict(
                type='LinearProjector', in_dim=32, out_dim=1024),
            action_out_proj=dict(
                type='LinearProjector', in_dim=1024, out_dim=32),
            time_mlp_in=dict(
                type='LinearProjector', in_dim=1024, out_dim=1024),
            time_mlp_out=dict(
                type='LinearProjector', in_dim=1024, out_dim=1024),
            max_action_dim=32,
            llm_expert=dict(
                type='ConditionGemmaModel',
                attention_bias=False,
                adarms_cond_dim=1024,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act='gelu_pytorch_tanh',
                hidden_activation='gelu_pytorch_tanh',
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type='gemma',
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                transformers_version='4.48.1',
                use_adarms=True,
                use_cache=True,
                vocab_size=257152),
            freeze_llm_backbone=False,
            freeze_vision_backbone=False,
            pretrained_name_or_path=  # noqa: E251
            PI05_CKPT_PATH,  # noqa: E501
            name_mapping={
                'llm_backbone':
                'paligemma_with_expert.paligemma.model.language_model',
                'llm_backbone.embed_tokens':
                'paligemma_with_expert.paligemma.lm_head',
                'vision_backbone.vision':
                'paligemma_with_expert.paligemma.model.vision_tower',
                'projector.projector': 'paligemma_with_expert.paligemma.model.'
                'multi_modal_projector.linear',
                'llm_expert': ('paligemma_with_expert.gemma_expert.'
                               'model'),
                'time_mlp_in.projector': 'time_mlp_in',
                'time_mlp_out.projector': 'time_mlp_out',
                'action_in_proj.projector': 'action_in_proj',
                'action_out_proj.projector': 'action_out_proj',
            },
            params_to_change_dtype=[
                'llm_expert.llm.model.layers',
                'vlm_backbone.vlm.model.language_model.layers',
                'vlm_backbone.vlm.model.vision_tower',
                'vlm_backbone.vlm.model.multi_modal_projector',
            ])
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def test_prefix_forward(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        embs_target = np.load(
            os.path.join(PI05_DATA_DIR, 'prefix_embs.npy'), allow_pickle=True)
        pad_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'prefix_pad_masks.npy'),
            allow_pickle=True)
        att_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'prefix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                embs, pad_masks, att_masks = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks)
        np.testing.assert_allclose(
            embs.float().cpu().detach().numpy()[:, ::10, ::10],
            embs_target,
            atol=5e-1,
            rtol=1e-5,
            equal_nan=False)
        np.testing.assert_allclose(
            pad_masks.float().cpu().detach().numpy(),
            pad_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)
        np.testing.assert_allclose(
            att_masks.float().cpu().detach().numpy(),
            att_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)

    def test_suffix_forward(self):
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        suffix_embs_target = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_embs.npy'), allow_pickle=True)
        suffix_pad_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_pad_masks.npy'),
            allow_pickle=True)
        suffix_att_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

        np.testing.assert_allclose(
            suffix_embs.float().cpu().detach().numpy()[:, :, ::10],
            suffix_embs_target,
            atol=1e-2,
            rtol=1e-5,
            equal_nan=False)
        np.testing.assert_allclose(
            suffix_pad_masks.float().cpu().detach().numpy(),
            suffix_pad_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)
        np.testing.assert_allclose(
            suffix_att_masks.float().cpu().detach().numpy(),
            suffix_att_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)

    def test_forward(self):
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        actions_target = np.load(
            os.path.join(PI05_DATA_DIR, 'actions.npy'), allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                ) = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                )

                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1

                att_2d_masks_4d = self.vla._prepare_attention_masks_4d(
                    att_2d_masks)

                suffix_out, _ = self.vla.forward_model(
                    inputs_embeds=[prefix_embs, suffix_embs],
                    attention_masks=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    fill_kv_cache=None,
                    adarms_cond=[None, adarms_cond],
                    time=time)

                actions = self.vla.action_out_proj(suffix_out)

                np.testing.assert_allclose(
                    actions.float().cpu().detach().numpy(),
                    actions_target,
                    atol=1e-1,
                    rtol=1e-5,
                    equal_nan=False)

    def test_predict_actions(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        noise = np.load(
            os.path.join(PI05_DATA_DIR, 'noise.npy'), allow_pickle=True)
        noise = torch.from_numpy(noise).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        pred_actions_target = np.load(
            os.path.join(PI05_DATA_DIR, 'pred_actions.npy'), allow_pickle=True)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                actions = self.vla.predict_action(
                    images=images,
                    states=states,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    noise=noise)

        np.testing.assert_allclose(
            actions.float().cpu().detach().numpy(),
            pred_actions_target,
            atol=1e-1,
            rtol=1e-5,
            equal_nan=False)


@unittest.skipUnless(
    torch.cuda.is_available() and os.path.exists(DREAMZERO_CKPT_PATH),
    'DreamZero checkpoint not available or CUDA is not available')
@pytest.mark.checkpoint
class TestDreamZero(unittest.TestCase):
    """Compare DreamZero forward outputs with reference implementation IO."""

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.cfg = dict(
            type='DreamZeroVLA',
            num_views=2,
            frame_window_size=9,
            pretrained_name_or_path=DREAMZERO_CKPT_PATH,
            vlm_backbone=dict(
                type='Wan21Backbone',
                text_encoder_path=None,
                image_encoder_path=None,
                vae_path=None,
                tiled=False,
                skip_pretrained_loading=True,
            ),
            vla_head=dict(
                type='DreamZeroHead',
                action_dim=7,
                max_action_dim=32,
                action_horizon=10,
                max_state_dim=64,
                num_frames=9,
                num_frame_per_block=2,
                num_action_per_block=10,
                num_state_per_block=1,
                frame_seqlen=8,
                hidden_size=1024,
                input_embedding_dim=1536,
                dit_dim=5120,
                dit_ffn_dim=13824,
                dit_num_heads=40,
                dit_num_layers=40,
                dit_freq_dim=256,
                dit_in_dim=36,
                dit_out_dim=16,
                max_num_embodiments=32,
                noise_beta_alpha=1.5,
                noise_beta_beta=1.0,
                noise_s=0.999,
                num_inference_steps=DREAMZERO_NUM_INFERENCE_STEPS,
                train_architecture='full',
                skip_pretrained_loading=True,
                wan_model_path=None,
                use_gradient_checkpointing=True,
            ),
            name_mapping={
                'vla_head.model': 'action_head.model',
                'vlm_backbone.text_encoder': 'action_head.text_encoder',
                'vlm_backbone.image_encoder': 'action_head.image_encoder',
                'vlm_backbone.vae': 'action_head.vae',
            },
            strict_mapping=False,
            freeze_llm_backbone=True,
            freeze_vlm_backbone=True,
            freeze_projector=True,
        )

        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).bfloat16().cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def _load_tensor(self, root, name):
        return np.load(os.path.join(root, f'{name}.npy'), allow_pickle=True)

    def test_forward(self):
        if not os.path.isdir(DREAMZERO_DATA_DIR):
            self.skipTest(
                f'DreamZero data dir not found: {DREAMZERO_DATA_DIR}')

        images = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'images')).cuda().to(torch.bfloat16)
        lang_tokens = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'lang_tokens')).cuda().long()
        lang_masks = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR, 'lang_masks')).cuda().long()
        states = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'states')).cuda().to(torch.bfloat16)
        actions = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'actions')).cuda().to(torch.bfloat16)
        action_masks = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'action_masks')).cuda().bool()
        embodiment_ids = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'embodiment_ids')).cuda().long()

        loss_ref = self._load_tensor(DREAMZERO_DATA_DIR, 'loss')
        dynamics_loss_ref = self._load_tensor(DREAMZERO_DATA_DIR,
                                              'dynamics_loss')
        action_loss_ref = self._load_tensor(DREAMZERO_DATA_DIR, 'action_loss')

        set_seed_everywhere(0)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                output = self.vla.forward(
                    images=images,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                    actions=actions,
                    action_masks=action_masks,
                    embodiment_ids=embodiment_ids,
                )

        np.testing.assert_allclose(
            output['loss'].float().cpu().numpy(),
            loss_ref,
            atol=1e-3,
            rtol=1e-5,
            equal_nan=False)
        np.testing.assert_allclose(
            output['dynamics_loss'].float().cpu().numpy(),
            dynamics_loss_ref,
            atol=1e-3,
            rtol=1e-5,
            equal_nan=False)
        np.testing.assert_allclose(
            output['action_loss'].float().cpu().numpy(),
            action_loss_ref,
            atol=1e-3,
            rtol=1e-5,
            equal_nan=False)

    def test_predict_action(self):
        if not os.path.isdir(DREAMZERO_DATA_DIR):
            self.skipTest(
                f'DreamZero data dir not found: {DREAMZERO_DATA_DIR}')

        pred_actions_path = os.path.join(DREAMZERO_DATA_DIR,
                                         'pred_actions.npy')
        if not os.path.exists(pred_actions_path):
            self.skipTest(
                f'DreamZero pred_actions not found: {pred_actions_path}')

        images = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'images')).cuda().to(torch.bfloat16)
        lang_tokens = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'lang_tokens')).cuda().long()
        lang_masks = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR, 'lang_masks')).cuda().long()
        states = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'states')).cuda().to(torch.bfloat16)
        embodiment_ids = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'embodiment_ids')).cuda().long()

        pred_actions_ref = self._load_tensor(DREAMZERO_DATA_DIR,
                                             'pred_actions')

        set_seed_everywhere(0)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                pred_actions = self.vla.predict_action(
                    images=images,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                    embodiment_ids=embodiment_ids,
                )

        np.testing.assert_allclose(
            pred_actions.float().cpu().numpy(),
            pred_actions_ref,
            atol=1e-2,
            rtol=1e-5,
            equal_nan=False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _dit4dit_artifacts_exist(),
    reason='DiT4DiT checkpoint or test data not found.')
@pytest.mark.checkpoint
@pytest.mark.usefixtures('dit4dit_parity_backbone')
class TestDiT4DiT(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

        inputs_path = os.path.join(DIT4DIT_DATA_DIR, 'inputs.pkl')
        outputs_path = os.path.join(DIT4DIT_DATA_DIR, 'source_outputs.pkl')
        metadata_path = os.path.join(DIT4DIT_DATA_DIR, 'metadata.json')
        self.inputs = _load_dit4dit_pickle(inputs_path)
        self.expected = _load_dit4dit_pickle(outputs_path)
        with open(metadata_path, 'r') as handle:
            self.metadata = json.load(handle)

        self.cfg = dict(
            type='DiT4DiTVLA',
            image_layout='auto',
            repeated_diffusion_steps=1,
            vlm_backbone=dict(
                type='DiT4DiTParityBackbone',
                inputs_path=inputs_path,
            ),
            vla_head=self.metadata['head_cfg'],
            freeze_vlm_backbone=True,
        )
        action_state = _load_dit4dit_action_state_dict(DIT4DIT_CKPT_PATH)
        self.vla = build_vla_from_cfg(self.cfg).cuda().eval()
        self.vla.vla_head.load_state_dict(action_state, strict=True)

    def _cuda_inputs(self):
        return {
            key: value.cuda()
            for key, value in self.inputs.items() if torch.is_tensor(value)
        }

    def test_forward(self):
        data = self._cuda_inputs()
        set_seed_everywhere(self.metadata['seeds']['forward'])
        with torch.no_grad():
            output = self.vla.forward(
                images=data['images'],
                states=data['state'],
                actions=data['actions'],
                action_masks=data['action_mask'],
                task_description=self.inputs['task_description'],
            )

        np.testing.assert_allclose(
            output['loss'].cpu().numpy(),
            self.expected['loss'].cpu().numpy(),
            atol=1e-5,
            rtol=1e-4,
            equal_nan=False)

    def test_predict_action(self):
        data = self._cuda_inputs()
        set_seed_everywhere(self.metadata['seeds']['predict'])
        with torch.no_grad():
            pred_actions = self.vla.predict_action(
                images=data['images'],
                states=data['state'],
                task_description=self.inputs['task_description'],
            )

        np.testing.assert_allclose(
            pred_actions.cpu().numpy(),
            self.expected['pred_actions'].cpu().numpy(),
            atol=1e-5,
            rtol=1e-4,
            equal_nan=False)


@pytest.mark.skipif(
    not os.path.exists(SMOLVLA_CKPT_PATH),
    reason=f'Checkpoint not found: {SMOLVLA_CKPT_PATH}')
@pytest.mark.checkpoint
class TestSmolVLAFlowMatching(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='SmolVLAFlowMatching',
            vlm_backbone=dict(
                type='SmolVLMBackbone',
                vision_config=dict(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    image_size=512,
                    patch_size=16,
                    intermediate_size=3072,
                    hidden_act='gelu_pytorch_tanh',
                    layer_norm_eps=1e-6,
                ),
                text_config=dict(
                    hidden_size=960,
                    num_hidden_layers=32,
                    num_attention_heads=15,
                    num_key_value_heads=5,
                    head_dim=64,
                    intermediate_size=2560,
                    vocab_size=49280,
                    rms_norm_eps=1e-5,
                    hidden_act='silu',
                    max_position_embeddings=8192,
                ),
                scale_factor=4,
                num_vlm_layers=16,
            ),
            llm_expert=dict(
                type='SmolVLMExpert',
                hidden_size=720,
                num_hidden_layers=16,
                num_attention_heads=15,
                num_key_value_heads=5,
                head_dim=64,
                intermediate_size=-1,
                vocab_size=49280,
                attention_bias=False,
                rms_norm_eps=1e-5,
                hidden_act='silu',
                max_position_embeddings=8192,
                attention_mode='cross_attn',
                vlm_kv_dim=320,
                self_attn_every_n_layers=2,
            ),
            state_proj=dict(type='LinearProjector', in_dim=32, out_dim=960),
            action_in_proj=dict(
                type='LinearProjector', in_dim=32, out_dim=720),
            action_out_proj=dict(
                type='LinearProjector', in_dim=720, out_dim=32),
            action_time_mlp_in=dict(
                type='LinearProjector', in_dim=1440, out_dim=720),
            action_time_mlp_out=dict(
                type='LinearProjector', in_dim=720, out_dim=720),
            freeze_vlm_backbone=True,
            max_action_dim=32,
            ori_action_dim=7,
            chunk_size=50,
            num_steps=10,
            add_image_special_tokens=False,
            use_cache=True,
            pretrained_name_or_path=SMOLVLA_CKPT_PATH,
            name_mapping={
                'vlm_backbone.vlm': 'model.vlm_with_expert.vlm.model',
                'llm_expert.expert': 'model.vlm_with_expert.lm_expert',
                'state_proj.projector': 'model.state_proj',
                'action_in_proj.projector': 'model.action_in_proj',
                'action_out_proj.projector': 'model.action_out_proj',
                'action_time_mlp_in.projector': 'model.action_time_mlp_in',
                'action_time_mlp_out.projector': 'model.action_time_mlp_out',
            })
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def _load_images(self):
        """Load single uint8 image and reconstruct (B, N_cam*3, H, W)."""
        img = np.load(
            'test/data/models/vlas/smolvla/image.npy')  # (3,H,W) uint8
        img_t = torch.from_numpy(img).float() / 255.0  # (3, H, W)
        B, N_cam = 2, 2
        images = img_t[None].repeat(B * N_cam, 1, 1, 1)  # (B*N_cam, 3, H, W)
        return images.reshape(B, N_cam * 3, img.shape[1], img.shape[2]).cuda()

    def test_prefix_forward(self):
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        embs_target = np.load(
            'test/data/models/vlas/smolvla/prefix_embs.npy', allow_pickle=True)
        pad_masks_target = np.load(
            'test/data/models/vlas/smolvla/prefix_pad_masks.npy',
            allow_pickle=True)
        att_masks_target = np.load(
            'test/data/models/vlas/smolvla/prefix_att_masks.npy',
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                embs, pad_masks, att_masks = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states)
        embs_np = embs.float().cpu().detach().numpy()[:, ::10, ::10]
        embs_diff = np.abs(embs_np - embs_target)
        pad_np = pad_masks.float().cpu().detach().numpy()
        att_np = att_masks.float().cpu().detach().numpy()
        print('\n[prefix] embs max_diff={:.6f}, mean_diff={:.6f}'.format(
            embs_diff.max(), embs_diff.mean()))
        print('[prefix] pad_masks max_diff={:.6f}'.format(
            np.max(np.abs(pad_np - pad_masks_target))))
        print('[prefix] att_masks max_diff={:.6f}'.format(
            np.max(np.abs(att_np - att_masks_target))))
        np.testing.assert_allclose(
            embs_np, embs_target, atol=5e-1, rtol=1e-5, equal_nan=False)
        np.testing.assert_allclose(
            pad_masks.float().cpu().detach().numpy(),
            pad_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)
        np.testing.assert_allclose(
            att_masks.float().cpu().detach().numpy(),
            att_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)

    def test_suffix_forward(self):
        time = np.load(
            'test/data/models/vlas/smolvla/suffix_time.npy', allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            'test/data/models/vlas/smolvla/suffix_x_t.npy', allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        suffix_embs_target = np.load(
            'test/data/models/vlas/smolvla/suffix_embs.npy', allow_pickle=True)
        suffix_pad_masks_target = np.load(
            'test/data/models/vlas/smolvla/suffix_pad_masks.npy',
            allow_pickle=True)
        suffix_att_masks_target = np.load(
            'test/data/models/vlas/smolvla/suffix_att_masks.npy',
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                ) = self.vla.embed_suffix(x_t, time)

        suffix_embs_np = suffix_embs.float().cpu().detach().numpy()[:, :, ::10]
        s_diff = np.abs(suffix_embs_np - suffix_embs_target)
        s_pad_np = suffix_pad_masks.float().cpu().detach().numpy()
        s_att_np = suffix_att_masks.float().cpu().detach().numpy()
        print('\n[suffix] embs max_diff={:.6f}, mean_diff={:.6f}'.format(
            s_diff.max(), s_diff.mean()))
        print('[suffix] pad_masks max_diff={:.6f}'.format(
            np.max(np.abs(s_pad_np - suffix_pad_masks_target))))
        print('[suffix] att_masks max_diff={:.6f}'.format(
            np.max(np.abs(s_att_np - suffix_att_masks_target))))
        np.testing.assert_allclose(
            suffix_embs_np,
            suffix_embs_target,
            atol=1e-2,
            rtol=1e-5,
            equal_nan=False)
        np.testing.assert_allclose(
            suffix_pad_masks.float().cpu().detach().numpy(),
            suffix_pad_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)
        np.testing.assert_allclose(
            suffix_att_masks.float().cpu().detach().numpy(),
            suffix_att_masks_target,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=False)

    def test_forward(self):
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            'test/data/models/vlas/smolvla/suffix_time.npy', allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            'test/data/models/vlas/smolvla/suffix_x_t.npy', allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        actions_target = np.load(
            'test/data/models/vlas/smolvla/actions.npy', allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                ) = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                )

                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                ) = self.vla.embed_suffix(x_t, time)

                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1

                suffix_out, _ = self.vla.forward_model(
                    attention_mask=att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False)

                suffix_out = suffix_out[:, -self.vla.chunk_size:]
                suffix_out = suffix_out.to(dtype=torch.float32)
                actions = self.vla.action_out_proj(suffix_out)

                actions_np = actions.float().cpu().detach().numpy()
                a_diff = np.abs(actions_np - actions_target)
                print('\n[forward] actions max_diff={:.6f}, '
                      'mean_diff={:.6f}'.format(a_diff.max(), a_diff.mean()))
                np.testing.assert_allclose(
                    actions_np,
                    actions_target,
                    atol=1e-1,
                    rtol=1e-5,
                    equal_nan=False)

    def test_predict_actions(self):
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        noise = np.load(
            'test/data/models/vlas/smolvla/noise.npy', allow_pickle=True)
        noise = torch.from_numpy(noise).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        pred_actions_target = np.load(
            'test/data/models/vlas/smolvla/pred_actions.npy',
            allow_pickle=True)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                actions = self.vla.predict_action(
                    images=images,
                    states=states,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    noise=noise)

        pred_np = actions.float().cpu().detach().numpy()
        p_diff = np.abs(pred_np - pred_actions_target)
        print('\n[predict] pred_actions max_diff={:.6f}, '
              'mean_diff={:.6f}'.format(p_diff.max(), p_diff.mean()))
        np.testing.assert_allclose(
            pred_np,
            pred_actions_target,
            atol=5e-1,
            rtol=1e-5,
            equal_nan=False)

    def test_vlm_output_consistency(self):
        """Verify that joint forward and prefill+decode produce consistent
        suffix outputs."""
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            'test/data/models/vlas/smolvla/suffix_time.npy', allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            'test/data/models/vlas/smolvla/suffix_x_t.npy', allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    self.vla.embed_prefix(images, img_masks, lang_tokens,
                                          lang_masks, states))
                suffix_embs, suffix_pad_masks, suffix_att_masks = (
                    self.vla.embed_suffix(x_t, time))

                # Path 1: joint forward (training path)
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)
                joint_att_2d = make_att_2d_masks(pad_masks, att_masks)
                joint_pos_ids = torch.cumsum(pad_masks, dim=1) - 1
                suffix_out_joint, _ = self.vla.forward_model(
                    joint_att_2d, joint_pos_ids, [prefix_embs, suffix_embs])

                # Path 2: prefill + decode (inference path)
                prefix_att_2d = make_att_2d_masks(prefix_pad_masks,
                                                  prefix_att_masks)
                prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                _, past_kv = self.vla.forward_model(
                    prefix_att_2d,
                    prefix_pos_ids, [prefix_embs, None],
                    use_cache=True)

                suffix_len = suffix_pad_masks.shape[1]
                batch_size = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d = prefix_pad_masks[:, None, :].expand(
                    batch_size, suffix_len, prefix_len)
                suffix_att_2d = make_att_2d_masks(suffix_pad_masks,
                                                  suffix_att_masks)
                decode_att = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                decode_pos = (
                    prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1)
                suffix_out_decode, _ = self.vla.forward_model(
                    decode_att,
                    decode_pos, [None, suffix_embs],
                    past_key_values=past_kv,
                    use_cache=True)

        out_joint = suffix_out_joint.float().cpu().detach().numpy()
        out_decode = suffix_out_decode.float().cpu().detach().numpy()
        max_diff = np.max(np.abs(out_joint - out_decode))
        mean_diff = np.mean(np.abs(out_joint - out_decode))
        print('\nSuffix output consistency: max_diff={:.6f}, '
              'mean_diff={:.6f}'.format(max_diff, mean_diff))
        self.assertTrue(
            np.allclose(out_joint, out_decode, atol=1e-1),
            f'Suffix outputs diverge: max_diff={max_diff}')
