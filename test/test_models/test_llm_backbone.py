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
import os
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

from fluxvla.engines import build_llm_backbone_from_cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAMA2_CKPT_PATH = str(PROJECT_ROOT / 'checkpoints/Llama-2-7b-hf')
LLAMA_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/llm_backbones/llama')
GEMMA_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/llm_backbones/gemma')
QWEN_DATA_DIR = str(PROJECT_ROOT / 'test/data/models/llm_backbones/qwen')


@pytest.mark.parametrize('model_type,backbone_id,family', [
    ('LLaMa2LLMBackbone', 'llama2-7b-pure_causal', 'llama'),
    ('GemmaLLMBackbone', 'gemma-2b_causal', 'gemma'),
    ('Qwen2LLMBackbone', 'qwen2-0.5b_causal', 'qwen2'),
])
def test_tiny_llm_forward_backward_and_causal_mask(model_type, backbone_id,
                                                   family):
    """Real transformer layers, not mocked logits or downloaded weights."""
    backbone = build_llm_backbone_from_cfg(
        dict(
            type=model_type,
            llm_backbone_id=backbone_id,
            llm_family=family,
            llm_path=None,
            tokenizer_length=64,
            llm_config=dict(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=64,
                attention_dropout=0.0,
                pad_token_id=0),
        )).eval()
    tokens = torch.tensor([[1, 5, 9, 2], [1, 8, 3, 2]])
    masks = torch.ones_like(tokens)
    result = backbone(input_ids=tokens, attention_mask=masks, labels=tokens)
    assert result.logits.shape == (2, 4, 64)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    grads = [p.grad for p in backbone.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert backbone.llm.get_input_embeddings().weight.grad.abs().sum() > 0

    with torch.no_grad():
        embedded = backbone(
            inputs_embeds=backbone.embed_input_ids(tokens),
            attention_mask=masks)
        changed = tokens.clone()
        changed[:, -1] = 12
        future = backbone(input_ids=changed, attention_mask=masks)
    torch.testing.assert_close(embedded.logits, result.logits)
    torch.testing.assert_close(future.logits[:, :-1], result.logits[:, :-1])


@pytest.mark.parametrize('model_type,backbone_id,family', [
    ('LLaMa2LLMBackbone', 'llama2-7b-pure_causal', 'llama'),
    ('GemmaLLMBackbone', 'gemma-2b_causal', 'gemma'),
    ('Qwen2LLMBackbone', 'qwen2-0.5b_causal', 'qwen2'),
])
def test_tiny_llm_kv_cache_matches_full_sequence(model_type, backbone_id,
                                                 family):
    backbone = build_llm_backbone_from_cfg(
        dict(
            type=model_type,
            llm_backbone_id=backbone_id,
            llm_family=family,
            llm_path=None,
            tokenizer_length=64,
            llm_config=dict(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=64,
                attention_dropout=0.0,
                pad_token_id=0))).eval()
    tokens = torch.tensor([[1, 5, 9, 2], [1, 8, 3, 2]])
    with torch.no_grad():
        full = backbone(input_ids=tokens, use_cache=False)
        prefix = backbone(input_ids=tokens[:, :3], use_cache=True)
        assert prefix.past_key_values.get_seq_length() == 3
        cached = backbone(
            input_ids=tokens[:, 3:],
            attention_mask=torch.ones_like(tokens),
            past_key_values=prefix.past_key_values,
            use_cache=True)
    assert cached.past_key_values.get_seq_length() == 4
    torch.testing.assert_close(cached.logits, full.logits[:, 3:])


@pytest.mark.skipif(
    not os.path.exists(LLAMA2_CKPT_PATH),
    reason=f'Checkpoint not found: {LLAMA2_CKPT_PATH}')
@pytest.mark.checkpoint
class TestLLaMaLLMBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = {
            'type': 'LLaMa2LLMBackbone',
            'llm_backbone_id': 'llama2-7b-pure_causal',
            'llm_family': 'llama',
            'llm_path': LLAMA2_CKPT_PATH,
            'llm_max_length': 2048,
            'hf_token': None,
            'inference_mode': False
        }
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        # The saved reference uses FP32 weights/inputs; HF now preserves
        # the checkpoint's FP16 dtype unless explicitly converted.
        self.llm_backbone = build_llm_backbone_from_cfg(
            self.cfg).float().cuda().eval()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_llama_backbone_forward(self):
        attention_mask = np.load(
            os.path.join(LLAMA_DATA_DIR, 'attention_mask.npy'),
            allow_pickle=True)
        inputs_embeds = np.load(
            os.path.join(LLAMA_DATA_DIR, 'inputs_embeds.npy'),
            allow_pickle=True)
        labels = np.load(
            os.path.join(LLAMA_DATA_DIR, 'labels.npy'), allow_pickle=True)
        logits = np.load(
            os.path.join(LLAMA_DATA_DIR, 'logits.npy'), allow_pickle=True)
        inputs_embeds = torch.from_numpy(inputs_embeds).cuda()
        outputs = self.llm_backbone.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.from_numpy(attention_mask).cuda(),
            labels=torch.from_numpy(labels).cuda())
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(logits).mean(),
                outputs['logits'].cpu().mean(),
                rtol=1e-2))


@pytest.mark.checkpoint
class TestGemmaLLMBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = {
            'type':
            'GemmaLLMBackbone',
            'llm_backbone_id':
            'gemma-2b_causal',
            'llm_family':
            'gemma',
            'llm_path':
            None,  # noqa: E501
            'llm_config':
            dict(
                vocab_size=257152,
                hidden_size=2048,
                intermediate_size=16384,
                num_hidden_layers=18,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=256,
                max_position_embeddings=4096,
                hidden_act='gelu',
                rms_norm_eps=1e-6,
                tie_word_embeddings=False),
            'llm_max_length':
            2048,
            'hf_token':
            None,
            'inference_mode':
            False,
            'tokenizer_length':
            257152
        }
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.llm_backbone = build_llm_backbone_from_cfg(self.cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_llama_backbone_forward(self):
        input_ids = np.load(
            os.path.join(GEMMA_DATA_DIR, 'input_ids.npy'), allow_pickle=True)
        input_ids = torch.from_numpy(input_ids).cuda().unsqueeze(0)
        outputs = self.llm_backbone(input_ids=input_ids)
        self.assertEqual(outputs['logits'].shape, (1, 180, 257152))


@pytest.mark.checkpoint
class TestQWen2LLMBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = {
            'type':
            'Qwen2LLMBackbone',
            'llm_backbone_id':
            'qwen2-0.5b',
            'llm_family':
            'qwen2',
            'llm_path':
            None,
            'llm_config':
            dict(
                vocab_size=151936,
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=24,
                num_attention_heads=14,
                num_key_value_heads=2,
                max_position_embeddings=32768,
                hidden_act='silu',
                output_hidden_states=True,
                rms_norm_eps=1e-6),
        }
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.llm_backbone = build_llm_backbone_from_cfg(self.cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_qwen_backbone_forward(self):
        inputs_embeds = np.load(
            os.path.join(QWEN_DATA_DIR, 'inputs_embeds.npy'),
            allow_pickle=True)
        inputs_embeds = torch.from_numpy(inputs_embeds).cuda()
        outputs = self.llm_backbone(
            inputs_embeds=inputs_embeds, output_hidden_states=True)
        self.assertEqual(outputs['hidden_states'][-1].shape, (4, 390, 896))
