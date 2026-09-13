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

from fluxvla.engines import build_vision_backbone_from_cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINO_CKPT_PATH = str(
    PROJECT_ROOT /
    'checkpoints/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors')
SIGLIP_CKPT_PATH = str(
    PROJECT_ROOT /
    'checkpoints/ViT-SO400M-14-SigLIP/open_clip_model.safetensors')
VISION_BACKBONE_DATA_DIR = str(PROJECT_ROOT /
                               'test/data/models/vision_backbones')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA.')
def test_radio_attention_cuda_bf16_forward_backward():
    # RADIO installs a global CUDA/BF16 attention implementation at import.
    # Test that implementation, not an instance-scoped or CPU-only variant.
    from timm.models.vision_transformer import Attention

    from fluxvla.models.third_party_models.eagle2_hg_model import radio_model

    attention = Attention(dim=16, num_heads=2).cuda().bfloat16().eval()
    assert attention.forward.__func__ is radio_model.forward
    # Exactly representable weights and nonnegative inputs avoid cancellation
    # around zero; the BF16 comparison keeps its standard tolerance.
    with torch.no_grad():
        eye = torch.eye(16, device='cuda', dtype=torch.bfloat16)
        attention.qkv.weight.copy_(torch.cat([eye / 4, eye / 2, eye]))
        attention.proj.weight.copy_(eye.roll(1, dims=0) / 2)
        attention.proj.bias.fill_(0.125)
    inputs = torch.linspace(0, 1, 64, device='cuda').reshape(1, 4, 16)
    inputs = inputs.bfloat16().requires_grad_()
    actual = attention(inputs)
    with torch.no_grad():
        q, k, v = attention.qkv(inputs).reshape(1, 4, 3, 2,
                                                8).permute(2, 0, 3, 1,
                                                           4).float()
        weights = ((q @ k.transpose(-2, -1)) / 8**0.5).softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(1, 4, 16)
        expected = attention.proj(context.bfloat16())
    torch.testing.assert_close(actual, expected)
    actual.float().square().mean().backward()
    for grad in (inputs.grad, attention.qkv.weight.grad,
                 attention.proj.weight.grad):
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    with torch.no_grad():
        torch.testing.assert_close(attention(inputs), actual, rtol=0, atol=0)


@pytest.mark.parametrize('openpi_stem_fp32', [False, True])
def test_tiny_siglip_multiview_forward_backward(openpi_stem_fp32):
    backbone = build_vision_backbone_from_cfg(
        dict(
            type='SigLIPViTBackbone',
            vision_backbone_id='siglip_224',
            openpi_stem_fp32=openpi_stem_fp32,
            vision_config=dict(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
                attention_dropout=0.0),
        )).eval()
    assert backbone.vision.config.image_size == 8
    assert backbone.vision.config.hidden_size == 16
    assert backbone.vision.config.patch_size == 4
    images = torch.linspace(-1, 1, 2 * 6 * 8 * 8).reshape(2, 6, 8, 8)
    features = backbone(images)
    assert features.shape == (2, 8, 16)
    expected = torch.cat([
        backbone.vision(view).last_hidden_state
        for view in images.split(3, dim=1)
    ],
                         dim=1)
    torch.testing.assert_close(features, expected)
    features[..., 0].sum().backward()
    grad = backbone.vision.vision_model.embeddings.patch_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


@pytest.mark.skipif(
    not os.path.exists(DINO_CKPT_PATH),
    reason=f'Checkpoint not found: {DINO_CKPT_PATH}')
@pytest.mark.checkpoint
class TestHFCausalVisionBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='DinoSigLIPViTBackbone',
            vision_backbone_id='dinosiglip-vit-so-224px',
            dino_config=dict(
                model_id='dino',
                file=  # noqa: E251
                DINO_CKPT_PATH),
            siglip_config=dict(
                model_id='siglip_224',
                file=  # noqa: E251
                SIGLIP_CKPT_PATH))
        self.siglip_vit = build_vision_backbone_from_cfg(self.cfg).cuda().to(
            torch.bfloat16)

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_siglip_vit_forward(self):
        input_dino = torch.from_numpy(
            np.load(os.path.join(VISION_BACKBONE_DATA_DIR,
                                 'input_dino.npy'))).cuda().to(torch.bfloat16)

        input_siglip = torch.from_numpy(
            np.load(
                os.path.join(VISION_BACKBONE_DATA_DIR,
                             'input_siglip.npy'))).cuda().to(torch.bfloat16)
        pixel_values = torch.cat([input_dino, input_siglip], dim=1)
        output = self.siglip_vit(pixel_values).float()
        expected_output = torch.from_numpy(
            np.load(
                os.path.join(VISION_BACKBONE_DATA_DIR,
                             'output_dinosiglipvit.npy'))).cuda()
        assert torch.allclose(
            output[:, ::10, ::10], expected_output, rtol=1e-3, atol=1e-1)
