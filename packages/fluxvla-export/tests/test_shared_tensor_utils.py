"""Real tensor regression tests for the shared PI numerical helpers."""

import unittest

import torch

from fluxvla_export.components.tensor_utils import (
    apply_rope, apply_rotary_pos_emb, create_sinusoidal_pos_embedding,
    eager_attention_forward, gated_residual, make_att_2d_masks, repeat_kv,
    rotate_half, sdpa_attention_forward, sdpa_math_fp32_attention_forward)


class TensorHelpersTest(unittest.TestCase):

    def setUp(self):
        self.generator = torch.Generator().manual_seed(47)

    def random(self, *shape):
        return torch.randn(*shape, generator=self.generator)

    def test_sinusoidal_embedding_preserves_batch_and_timestep_axes(self):
        for shape in ((2,), (2, 3)):
            time = torch.zeros(shape, requires_grad=True)
            result = create_sinusoidal_pos_embedding(time, 8, 0.004, 4.0)
            self.assertEqual(result.shape, (*shape, 8))
            torch.testing.assert_close(result[..., :4], torch.zeros(*shape, 4))
            torch.testing.assert_close(result[..., 4:], torch.ones(*shape, 4))
            result.sum().backward()
            self.assertTrue(torch.isfinite(time.grad).all())
            self.assertTrue((time.grad != 0).all())

    def test_sinusoidal_embedding_rejects_odd_width(self):
        with self.assertRaisesRegex(ValueError, 'divisible by 2'):
            create_sinusoidal_pos_embedding(torch.zeros(2), 7, 0.004, 4.0)

    def test_attention_mask_preserves_holes_and_block_visibility(self):
        pad = torch.tensor([[True, False, True, True]])
        boundary = torch.tensor([[False, False, True, False]])
        result = make_att_2d_masks(pad, boundary)
        expected = torch.tensor([[
            [True, False, False, False],
            [False, False, False, False],
            [True, False, True, True],
            [True, False, True, True],
        ]])
        torch.testing.assert_close(result, expected)
        for bad_pad, bad_boundary in ((pad[0], boundary), (pad, boundary[0])):
            with self.assertRaises(ValueError):
                make_att_2d_masks(bad_pad, bad_boundary)

    def test_gated_residual_preserves_none_semantics_and_gradients(self):
        x = torch.ones(2, 3, requires_grad=True)
        y = torch.full((2, 3), 2.0, requires_grad=True)
        gate = torch.full((2, 1), 3.0, requires_grad=True)
        self.assertIsNone(gated_residual(None, None, None))
        self.assertIs(gated_residual(x, None, gate), x)
        self.assertIs(gated_residual(None, y, gate), y)
        torch.testing.assert_close(gated_residual(x, y, None), x + y)
        result = gated_residual(x, y, gate)
        torch.testing.assert_close(result, torch.full((2, 3), 7.0))
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(y.grad, torch.full_like(y, 3.0))
        torch.testing.assert_close(gate.grad, torch.full_like(gate, 6.0))

    def test_repeat_kv_matches_repeat_interleave_and_accumulates_gradient(self):
        value = self.random(2, 2, 3, 4).requires_grad_()
        self.assertIs(repeat_kv(value, 1), value)
        result = repeat_kv(value, 3)
        torch.testing.assert_close(result, value.repeat_interleave(3, dim=1))
        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, 3.0))

    def test_rotary_helpers_keep_zero_positions_and_rotate_halves(self):
        value = self.random(2, 3, 2, 4).requires_grad_()
        rotated = apply_rope(value, torch.zeros(2, 3))
        torch.testing.assert_close(rotated, value, rtol=0, atol=0)
        rotated.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))
        torch.testing.assert_close(
            rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])),
            torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    def test_rotary_qk_embeddings_and_gradients(self):
        query = self.random(2, 2, 3, 4).requires_grad_()
        key = self.random(2, 1, 3, 4).requires_grad_()
        # cos=0/sin=1 rotates a quarter turn without approximating angles.
        cos, sin = torch.zeros(2, 3, 4), torch.ones(2, 3, 4)
        q_rotated, k_rotated = apply_rotary_pos_emb(query, key, cos, sin)
        torch.testing.assert_close(q_rotated, rotate_half(query))
        torch.testing.assert_close(k_rotated, rotate_half(key))
        (q_rotated.square().sum() + k_rotated.square().sum()).backward()
        torch.testing.assert_close(query.grad, 2 * query)
        torch.testing.assert_close(key.grad, 2 * key)

    def attention_case(self, function, inputs, mask, *, autocast=False):
        values = [value.clone().requires_grad_() for value in inputs]
        module = torch.nn.Module().eval()
        module.num_key_value_groups = 2
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
            output, weights = function(module, *values, mask, scaling=0.5)
            loss = output.float().square().mean()
        loss.backward()
        return output.detach(), weights, [value.grad for value in values]

    def test_attention_backends_match_real_outputs_and_qkv_gradients(self):
        inputs = [self.random(2, 4, 3, 4), self.random(2, 2, 5, 4),
                  self.random(2, 2, 5, 4)]
        # Distinct per-batch masks, including an interior hole.
        mask = torch.zeros(2, 1, 3, 7)
        mask[0, :, :, 1] = torch.finfo(torch.bfloat16).min
        mask[1, :, :, 3] = torch.finfo(torch.bfloat16).min
        reference, weights, grads = self.attention_case(
            eager_attention_forward, inputs, mask)
        self.assertEqual(weights.shape, (2, 4, 3, 5))
        for function in (sdpa_attention_forward,
                         sdpa_math_fp32_attention_forward):
            with self.subTest(backend=function.__name__):
                actual, actual_weights, actual_grads = self.attention_case(
                    function, inputs, mask)
                self.assertIsNone(actual_weights)
                torch.testing.assert_close(actual, reference)
                for value, expected in zip(actual_grads, grads):
                    torch.testing.assert_close(value, expected)
                    self.assertTrue(torch.isfinite(value).all())

    def test_protected_attention_keeps_fp32_under_bfloat16_autocast(self):
        inputs = [self.random(2, 4, 3, 4), self.random(2, 2, 5, 4),
                  self.random(2, 2, 5, 4)]
        expected, _, expected_grads = self.attention_case(
            sdpa_math_fp32_attention_forward, inputs, None)
        actual, _, actual_grads = self.attention_case(
            sdpa_math_fp32_attention_forward, inputs, None, autocast=True)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for value, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
