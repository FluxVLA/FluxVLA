"""Real-tensor tests for shared SigLIP and projector calculation.

Run with the export package installed and its Torch/Transformers dependencies
available; missing dependencies are errors, not skipped test cases.
"""

import copy
import unittest

import torch
import torch.nn.functional as F
from fluxvla_export.components.projector import LinearProjector
from fluxvla_export.components.siglip import SigLIPViTBackbone
from transformers import SiglipVisionConfig, SiglipVisionModel


def make_vision() -> SiglipVisionModel:
    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=8,
        patch_size=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = 'eager'
    return SiglipVisionModel(config)


class SharedVisionProjectorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(101)

    def assert_finite_nonzero(self, value):
        self.assertIsNotNone(value)
        self.assertTrue(torch.isfinite(value).all().item())
        self.assertGreater(value.float().abs().sum().item(), 0)

    def test_vision_registration_rng_and_state_roundtrip(self):
        vision = make_vision()
        rng_state = torch.random.get_rng_state().clone()
        model = SigLIPViTBackbone(vision)
        self.assertIs(model.vision, vision)
        self.assertEqual(list(model._modules), ['vision'])
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
        self.assertEqual(
            list(model.state_dict()),
            ['vision.' + name for name in vision.state_dict()],
        )
        self.assertFalse(
            any(name.startswith('core.') for name in model.state_dict()))
        for name, value in vision.state_dict().items():
            torch.testing.assert_close(model.state_dict()['vision.' + name],
                                       value)

        restored = SigLIPViTBackbone(make_vision())
        result = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        pixels = torch.randn(2, 6, 8, 8)
        torch.testing.assert_close(
            model(pixels), restored(pixels), rtol=0, atol=0)

    def test_packed_views_match_separate_views_and_gradients(self):
        # Rectangular inputs preserve the configured total patch count. No
        # interpolation or support for a different patch count is introduced.
        for dtype in (torch.float32, torch.bfloat16):
            for view_count, height, width in ((1, 8, 8), (2, 8, 8), (3, 4,
                                                                     16)):
                for stem, preserve in ((False, False), (False, True),
                                       (True, False), (True, True)):
                    with self.subTest(
                            dtype=dtype,
                            views=view_count,
                            height=height,
                            width=width,
                            stem=stem,
                            preserve=preserve):
                        model = SigLIPViTBackbone(
                            make_vision(),
                            openpi_stem_fp32=stem,
                            preserve_fp32_residuals=preserve)
                        reference = copy.deepcopy(model)
                        pixels = torch.randn(
                            2,
                            3 * view_count,
                            height,
                            width,
                            dtype=dtype,
                            requires_grad=True)
                        separate_pixels = pixels.detach().clone(
                        ).requires_grad_()
                        with torch.autocast(
                                'cpu',
                                dtype=torch.bfloat16,
                                enabled=dtype == torch.bfloat16):
                            actual = model(pixels)
                            expected_views = []
                            for view in separate_pixels.split(3, dim=1):
                                if stem:
                                    output = reference._forward_openpi_stem(
                                        view)
                                else:
                                    output = reference.vision(
                                        view).last_hidden_state
                                expected_views.append(output)
                            expected = torch.cat(expected_views, dim=1)
                            self.assertEqual(actual.shape,
                                             (2, 16 * view_count, 16))
                            torch.testing.assert_close(
                                actual, expected, rtol=0, atol=0)
                            # A nonuniform loss avoids LayerNorm's zero-sum
                            # output making the backward check vacuous.
                            weights = torch.randn_like(actual)
                            actual_loss = (actual.float() *
                                           weights.float()).sum()
                            expected_loss = (expected.float() *
                                             weights.float()).sum()
                        actual_loss.backward()
                        expected_loss.backward()
                        self.assert_finite_nonzero(pixels.grad)
                        torch.testing.assert_close(
                            pixels.grad, separate_pixels.grad, rtol=0, atol=0)
                        reference_parameters = dict(
                            reference.named_parameters())
                        nonzero_gradients = 0
                        for name, parameter in model.named_parameters():
                            reference_gradient = reference_parameters[
                                name].grad
                            if reference_gradient is None:
                                self.assertIsNone(parameter.grad, name)
                            else:
                                self.assertIsNotNone(parameter.grad, name)
                                self.assertTrue(
                                    torch.isfinite(parameter.grad).all(), name)
                                torch.testing.assert_close(
                                    parameter.grad,
                                    reference_gradient,
                                    rtol=0,
                                    atol=0,
                                    msg=name)
                                nonzero_gradients += int(
                                    parameter.grad.abs().sum() > 0)
                        self.assertGreater(nonzero_gradients, 10)

    def test_fp32_stem_and_residual_precision_boundaries(self):
        for autocast_enabled in (False, True):
            for preserve in (False, True):
                with self.subTest(
                        autocast=autocast_enabled, preserve=preserve):
                    model = SigLIPViTBackbone(
                        make_vision(),
                        openpi_stem_fp32=True,
                        preserve_fp32_residuals=preserve)
                    embedding_inputs = []
                    embedding_outputs = []
                    encoder_inputs = []

                    def record_embeddings(_module, args, output):
                        embedding_inputs.append(args[0].dtype)
                        embedding_outputs.append(output.dtype)

                    def record_encoder(_module, _args, kwargs):
                        encoder_inputs.append(kwargs['inputs_embeds'].dtype)

                    handles = [
                        model.vision.vision_model.embeddings.
                        register_forward_hook(record_embeddings),
                        model.vision.vision_model.encoder.
                        register_forward_pre_hook(
                            record_encoder, with_kwargs=True),
                    ]
                    pixels = torch.randn(
                        2, 6, 8, 8, dtype=torch.bfloat16, requires_grad=True)
                    try:
                        with torch.autocast(
                                'cpu',
                                dtype=torch.bfloat16,
                                enabled=autocast_enabled):
                            output = model(pixels)
                            loss = (output.float() *
                                    torch.randn_like(output).float()).sum()
                        loss.backward()
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(embedding_inputs, [torch.float32] * 2)
                    self.assertEqual(embedding_outputs, [torch.float32] * 2)
                    expected_dtype = (
                        torch.bfloat16 if autocast_enabled and not preserve
                        else torch.float32)
                    self.assertEqual(encoder_inputs, [expected_dtype] * 2)
                    self.assert_finite_nonzero(pixels.grad)
                    self.assert_finite_nonzero(
                        model.vision.vision_model.embeddings.patch_embedding.
                        weight.grad)

    def test_standard_forward_with_bf16_parameters(self):
        model = SigLIPViTBackbone(make_vision().to(dtype=torch.bfloat16))
        pixels = torch.randn(
            2, 6, 8, 8, dtype=torch.bfloat16, requires_grad=True)
        actual = model(pixels)
        expected = torch.cat([
            model.vision(view).last_hidden_state
            for view in pixels.split(3, dim=1)
        ],
                             dim=1)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (actual.float() * torch.randn_like(actual).float()).sum().backward()
        self.assert_finite_nonzero(pixels.grad)

    def test_invalid_channel_pack_raises(self):
        model = SigLIPViTBackbone(make_vision())
        with self.assertRaises(RuntimeError):
            model(torch.randn(2, 5, 8, 8))

    def test_projector_initialization_keys_and_roundtrip(self):
        torch.manual_seed(53)
        reference = torch.nn.Linear(16, 12, bias=True)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(53)
        projector = LinearProjector(16, 12)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng)
        self.assertEqual(
            list(projector.state_dict()),
            ['projector.weight', 'projector.bias'])
        torch.testing.assert_close(
            projector.projector.weight, reference.weight, rtol=0, atol=0)
        torch.testing.assert_close(
            projector.projector.bias, reference.bias, rtol=0, atol=0)
        restored = LinearProjector(16, 12)
        result = restored.load_state_dict(projector.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        features = torch.randn(2, 32, 16)
        torch.testing.assert_close(
            projector(features), restored(features), rtol=0, atol=0)

    def test_projector_real_tensor_shapes_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ((16, ), (2, 16), (2, 3, 16), (2, 3, 4, 16)):
                with self.subTest(dtype=dtype, shape=shape):
                    projector = LinearProjector(16, 12).to(dtype=dtype)
                    features = torch.randn(
                        *shape, dtype=dtype, requires_grad=True)
                    reference_features = features.detach().clone(
                    ).requires_grad_()
                    reference_weight = projector.projector.weight.detach(
                    ).clone().requires_grad_()
                    reference_bias = projector.projector.bias.detach().clone(
                    ).requires_grad_()
                    actual = projector(features)
                    expected = F.linear(reference_features, reference_weight,
                                        reference_bias)
                    self.assertEqual(actual.shape, (*shape[:-1], 12))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0)
                    weights = torch.randn_like(actual)
                    (actual.float() * weights.float()).sum().backward()
                    (expected.float() * weights.float()).sum().backward()
                    self.assert_finite_nonzero(features.grad)
                    torch.testing.assert_close(
                        features.grad, reference_features.grad, rtol=0, atol=0)
                    torch.testing.assert_close(
                        projector.projector.weight.grad,
                        reference_weight.grad,
                        rtol=0,
                        atol=0)
                    torch.testing.assert_close(
                        projector.projector.bias.grad,
                        reference_bias.grad,
                        rtol=0,
                        atol=0)


if __name__ == '__main__':
    unittest.main()
