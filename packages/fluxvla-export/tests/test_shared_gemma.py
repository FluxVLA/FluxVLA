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
"""Standalone tensor tests; run with unittest, without the training package."""

import io
import subprocess
import sys
import unittest

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from fluxvla_export.components import condition_gemma
from fluxvla_export.components.condition_gemma import (ConditionGemmaModel,
                                                       GemmaRMSNorm)


def tiny_model(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        adarms_cond_dim=12,
        attention_dropout=0.0,
        use_cache=False)
    kwargs.update(overrides)
    return ConditionGemmaModel(**kwargs)


class SharedGemmaTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(920)

    def assert_same_tensors(self, actual, expected):
        self.assertEqual(tuple(actual), tuple(expected))
        for name in actual:
            with self.subTest(tensor=name):
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                torch.testing.assert_close(
                    actual[name], expected[name], rtol=0, atol=0)

    def assert_finite_gradients(self, model):
        gradients = {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(gradients)
        for name, gradient in gradients.items():
            with self.subTest(gradient=name):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertTrue(
            any(gradient.abs().sum().item() > 0
                for gradient in gradients.values()))

    def test_standalone_import_has_no_training_dependencies(self):
        script = '''
import sys
from fluxvla_export.components import condition_gemma
for name in ('fluxvla', 'mmengine', 'accelerate', 'timm', 'triton'):
    assert not any(key == name or key.startswith(name + '.')
                   for key in sys.modules), name
assert condition_gemma.GemmaModel is condition_gemma.ConditionGemmaModel
'''
        result = subprocess.run([sys.executable, '-c', script],
                                capture_output=True,
                                text=True,
                                check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ('get_fsdp_wrapping_policy', 'transformer_layer_cls',
                     'half_precision_dtype', 'last_layer_finetune_modules'):
            self.assertNotIn(name, ConditionGemmaModel.__dict__)
        for name in ('GemmaForCausalLM', 'GemmaForSequenceClassification',
                     'GemmaForTokenClassification', 'GenerationMixin',
                     'KwargsForCausalLM'):
            self.assertFalse(hasattr(condition_gemma, name), name)

    def test_seeded_initialization_and_strict_state_roundtrip(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                torch.manual_seed(54)
                original = tiny_model(use_adarms=adaptive)
                state = original.state_dict()
                torch.manual_seed(54)
                repeated = tiny_model(use_adarms=adaptive)
                self.assert_same_tensors(state, repeated.state_dict())
                expected_shapes = {'embed_tokens.weight': (32, 16)}
                norm_shape = (48, 12) if adaptive else (16, )
                for index in range(2):
                    prefix = f'layers.{index}.'
                    for projection, shape in (
                        ('self_attn.q_proj', (16, 16)),
                        ('self_attn.k_proj', (8, 16)),
                        ('self_attn.v_proj', (8, 16)),
                        ('self_attn.o_proj', (16, 16)),
                        ('mlp.gate_proj', (32, 16)),
                        ('mlp.up_proj', (32, 16)),
                        ('mlp.down_proj', (16, 32)),
                    ):
                        expected_shapes[prefix + projection +
                                        '.weight'] = shape
                    for norm in ('input_layernorm',
                                 'post_attention_layernorm'):
                        name = prefix + norm
                        if adaptive:
                            expected_shapes[name +
                                            '.dense.weight'] = norm_shape
                            expected_shapes[name + '.dense.bias'] = (48, )
                        else:
                            expected_shapes[name + '.weight'] = norm_shape
                if adaptive:
                    expected_shapes['norm.dense.weight'] = norm_shape
                    expected_shapes['norm.dense.bias'] = (48, )
                else:
                    expected_shapes['norm.weight'] = norm_shape
                self.assertEqual(
                    {
                        name: tuple(value.shape)
                        for name, value in state.items()
                    }, expected_shapes)
                for name, value in state.items():
                    expected_dtype = (
                        torch.bfloat16
                        if not adaptive and 'norm' in name else torch.float32)
                    self.assertEqual(value.dtype, expected_dtype, name)
                    if name.endswith('.bias'):
                        self.assertEqual(torch.count_nonzero(value).item(), 0)
                    if not adaptive and 'norm' in name:
                        torch.testing.assert_close(value,
                                                   torch.ones_like(value))
                self.assertEqual(
                    torch.count_nonzero(
                        state['embed_tokens.weight'][0]).item(), 0)
                buffer = io.BytesIO()
                torch.save(state, buffer)
                buffer.seek(0)
                restored = tiny_model(
                    use_adarms=adaptive,
                    adarms_fp32=True,
                    attention_math_fp32=True,
                    fp32_layers=(0, ))
                status = restored.load_state_dict(
                    torch.load(buffer, weights_only=True), strict=True)
                self.assertEqual(status.missing_keys, [])
                self.assertEqual(status.unexpected_keys, [])
                self.assert_same_tensors(state, restored.state_dict())
                missing = dict(state)
                missing.pop('embed_tokens.weight')
                with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                    restored.load_state_dict(missing, strict=True)

    def test_actual_forward_backward_for_ids_and_condition_shapes(self):
        for adaptive, token_condition in ((False, False), (True, False),
                                          (True, True)):
            with self.subTest(
                    adaptive=adaptive, token_condition=token_condition):
                model = tiny_model(use_adarms=adaptive).float().train()
                ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
                shape = (2, 4, 12) if token_condition else (2, 12)
                condition = torch.randn(shape, requires_grad=True)
                output = model(
                    input_ids=ids,
                    adarms_cond=condition if adaptive else None,
                    output_hidden_states=True)
                self.assertEqual(output.last_hidden_state.shape, (2, 4, 16))
                self.assertEqual(len(output.hidden_states), 3)
                self.assertIsNone(output.past_key_values)
                self.assertTrue(
                    torch.isfinite(output.last_hidden_state).all().item())
                weights = torch.randn_like(output.last_hidden_state)
                (output.last_hidden_state * weights).sum().backward()
                self.assert_finite_gradients(model)
                if adaptive:
                    self.assertTrue(
                        torch.isfinite(condition.grad).all().item())
                    self.assertGreater(condition.grad.abs().sum().item(), 0)
                with torch.no_grad():
                    embedded = model(
                        inputs_embeds=model.get_input_embeddings()(ids),
                        adarms_cond=condition if adaptive else None)
                torch.testing.assert_close(
                    embedded.last_hidden_state,
                    output.last_hidden_state,
                    rtol=0,
                    atol=0)

    def test_prefix_cache_matches_joint_and_expert_does_not_mutate(self):
        for amp in (False, True):
            with self.subTest(autocast=amp):
                model = tiny_model(
                    adarms_fp32=True,
                    attention_math_fp32=True,
                    fp32_layers=(0, )).float().eval()
                hidden = torch.randn(2, 7, 16)
                condition = torch.randn(2, 7, 12)
                mask = torch.zeros(2, 1, 7, 7)
                mask[:, :, :3, 3:] = float('-inf')
                positions = torch.arange(7)[None].expand(2, -1)
                with torch.no_grad(), torch.autocast(
                        'cpu', dtype=torch.bfloat16, enabled=amp):
                    joint = model(
                        inputs_embeds=hidden,
                        adarms_cond=condition,
                        attention_mask=mask,
                        position_ids=positions).last_hidden_state[:, 3:]
                    prefix = model(
                        inputs_embeds=hidden[:, :3],
                        adarms_cond=condition[:, :3],
                        attention_mask=mask[:, :, :3, :3],
                        position_ids=positions[:, :3],
                        use_cache=True)
                    cache = prefix.past_key_values
                    self.assertIsInstance(cache, DynamicCache)
                    self.assertEqual(cache.get_seq_length(), 3)
                    self.assertEqual(len(cache.layers), 2)
                    snapshots = [(layer.keys.clone(), layer.values.clone(),
                                  layer.keys.data_ptr(),
                                  layer.values.data_ptr())
                                 for layer in cache.layers]
                    for attempt in range(2):
                        suffix = model(
                            inputs_embeds=hidden[:, 3:],
                            adarms_cond=condition[:, 3:],
                            attention_mask=mask[:, :, 3:],
                            position_ids=positions[:, 3:],
                            past_key_values=cache,
                            use_cache=False)
                        self.assertIsNone(suffix.past_key_values)
                        torch.testing.assert_close(
                            suffix.last_hidden_state,
                            joint,
                            rtol=0.002,
                            atol=0.0002)
                    # A distinct conditional expert must also treat the prefix
                    # cache as read-only on every denoising iteration.
                    expert = tiny_model(
                        adarms_fp32=True,
                        attention_math_fp32=True,
                        fp32_layers=(0, )).float().eval()
                    for attempt in range(2):
                        expert_output = expert(
                            inputs_embeds=hidden[:, 3:] + attempt,
                            adarms_cond=condition[:, 3:] + attempt,
                            attention_mask=mask[:, :, 3:],
                            position_ids=positions[:, 3:],
                            past_key_values=cache,
                            use_cache=False)
                        self.assertIsNone(expert_output.past_key_values)
                        self.assertTrue(
                            torch.isfinite(
                                expert_output.last_hidden_state).all().item())
                        self.assertEqual(cache.get_seq_length(), 3)
                        for layer, saved in zip(cache.layers, snapshots):
                            keys, values, key_ptr, value_ptr = saved
                            self.assertEqual(layer.keys.data_ptr(), key_ptr)
                            self.assertEqual(layer.values.data_ptr(),
                                             value_ptr)
                            torch.testing.assert_close(
                                layer.keys, keys, rtol=0, atol=0)
                            torch.testing.assert_close(
                                layer.values, values, rtol=0, atol=0)
                    appended = model(
                        inputs_embeds=hidden[:, 3:],
                        adarms_cond=condition[:, 3:],
                        attention_mask=mask[:, :, 3:],
                        position_ids=positions[:, 3:],
                        past_key_values=cache,
                        use_cache=True)
                    self.assertIs(appended.past_key_values, cache)
                    self.assertEqual(cache.get_seq_length(), 7)
                    for layer, saved in zip(cache.layers, snapshots):
                        torch.testing.assert_close(
                            layer.keys[:, :, :3], saved[0], rtol=0, atol=0)
                        torch.testing.assert_close(
                            layer.values[:, :, :3], saved[1], rtol=0, atol=0)

    def test_adaptive_norm_protects_projection_under_autocast(self):
        norm = GemmaRMSNorm(16, cond_dim=12, adarms_fp32=True)
        nn.init.normal_(norm.dense.weight, std=0.1)
        x = torch.randn(2, 3, 16, requires_grad=True)
        condition = torch.randn(2, 3, 12, requires_grad=True)
        modulation = nn.functional.linear(condition, norm.dense.weight,
                                          norm.dense.bias)
        scale, shift, reference_gate = modulation.chunk(3, dim=-1)
        reference = x * torch.rsqrt(x.square().mean(-1, keepdim=True) +
                                    norm.eps)
        reference = reference * (1 + scale) + shift
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual, gate = norm(x, condition)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(gate.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(gate, reference_gate, rtol=0, atol=0)
        variables = (x, condition, norm.dense.weight, norm.dense.bias)
        expected_gradients = torch.autograd.grad(
            reference.square().sum() + reference_gate.square().sum(),
            variables)
        actual_gradients = torch.autograd.grad(
            actual.square().sum() + gate.square().sum(), variables)
        for actual_gradient, expected in zip(actual_gradients,
                                             expected_gradients):
            torch.testing.assert_close(
                actual_gradient, expected, rtol=0, atol=0)

    def test_selected_layers_keep_fp32_projection_and_mlp(self):
        model = tiny_model(
            adarms_fp32=True, attention_math_fp32=True,
            fp32_layers=(0, )).float().train()
        dtypes = []
        handles = []
        for layer in model.layers:
            for module in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                           layer.self_attn.v_proj, layer.self_attn.o_proj,
                           layer.mlp.gate_proj, layer.mlp.up_proj,
                           layer.mlp.down_proj):
                handles.append(
                    module.register_forward_hook(lambda module, inputs, output:
                                                 dtypes.append(output.dtype)))
        try:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = model(
                    input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    adarms_cond=torch.randn(2, 3, 12))
            output.last_hidden_state.square().mean().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(dtypes, [torch.float32] * 7 + [torch.bfloat16] * 7)
        self.assert_finite_gradients(model)

    def test_math_attention_keeps_implicit_causal_mask(self):
        model = tiny_model(
            use_adarms=False, attention_math_fp32=True).float().eval()
        hidden = torch.randn(2, 4, 16)
        mask = torch.full((2, 1, 4, 4), float('-inf')).triu(1)
        with torch.no_grad():
            implicit = model(inputs_embeds=hidden).last_hidden_state
            explicit = model(
                inputs_embeds=hidden, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(implicit, explicit, rtol=1e-6, atol=1e-6)

    def test_invalid_fp32_layers_are_rejected_before_initialization(self):
        for indices in ((-1, ), (2, ), (True, ), (0.5, )):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, 'fp32_layers'):
                    tiny_model(fp32_layers=indices)


if __name__ == '__main__':
    unittest.main()
