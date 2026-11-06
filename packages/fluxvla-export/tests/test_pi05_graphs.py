# Copyright 2026 Limx Dynamics
"""Real PyTorch flat-graph parity, precision and state invariants for PI0.5."""

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from fluxvla_export.components.euler import plain_euler
from fluxvla_export.components.tensor_utils import make_att_2d_masks
from fluxvla_export.errors import ContractError
from fluxvla_export.recipes.pi05.graphs import (PI05DenoiseGraph,
                                                PI05PrefillGraph,
                                                cache_tensor_names)
from _pi05_test_utils import Artifacts
from test_config import config_document


class PI05GraphTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def reference(self,
                  *,
                  amp=False,
                  flow=False,
                  protected=False,
                  states_bf16=False,
                  mixed_cache=False,
                  backend='sdpa'):
        document = config_document()
        for branch in ('backbone', 'expert'):
            document['model'][branch]['num_hidden_layers'] = 2
        document['precision'].update(
            autocast_dtype='bfloat16' if amp else None,
            flow_fp32=flow,
            residual_fp32=protected and not mixed_cache,
            vision_stem_fp32=protected,
            vision_residual_fp32=protected,
            adarms_fp32=protected,
            attention_math_fp32=protected and backend == 'sdpa',
            backbone_fp32_layers=[0] if protected else [],
            expert_fp32_layers=[1] if protected else [],
            attention_backend=backend)
        if states_bf16:
            document['inputs']['tensors']['states']['dtype'] = 'bfloat16'
        return Artifacts(self.directory, document).prepare()

    def graphs(self, reference):
        return (PI05PrefillGraph(reference.model),
                PI05DenoiseGraph(
                    reference.model,
                    states_dtype=reference.inputs['states'].dtype))

    def assert_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, expected.dtype)
        else:
            self.assertEqual(len(actual), len(expected))
            for value, target in zip(actual, expected):
                self.assert_equal(value, target)

    def test_flat_contract_names_and_weight_identity(self):
        reference = self.reference()
        prefill, denoise = self.graphs(reference)
        cache_names = ('key_0', 'value_0', 'key_1', 'value_1')
        self.assertEqual(cache_tensor_names(2), cache_names)
        self.assertEqual(prefill.name, 'prefill')
        self.assertEqual(prefill.input_names,
                         ('images', 'img_masks', 'lang_tokens', 'lang_masks'))
        self.assertEqual(prefill.output_names,
                         ('prefix_pad_masks', ) + cache_names)
        self.assertEqual(denoise.name, 'denoise')
        self.assertEqual(denoise.input_names,
                         ('noisy_actions', 'timestep', 'prefix_pad_masks') +
                         cache_names)
        self.assertEqual(denoise.output_names, ('velocity', ))
        self.assertIs(prefill.model, reference.model)
        self.assertIs(denoise.model, reference.model)
        expected = {id(value) for value in reference.model.parameters()}
        self.assertEqual({id(value)
                          for value in prefill.parameters()}, expected)
        self.assertEqual({id(value)
                          for value in denoise.parameters()}, expected)
        self.assertFalse(prefill.training)
        self.assertFalse(denoise.training)

    def test_prefill_matches_original_cache_preparation_in_all_precision_modes(
            self):
        for amp in (False, True):
            for flow in (False, True):
                for protected in (False, True):
                    with self.subTest(amp=amp, flow=flow, protected=protected):
                        reference = self.reference(
                            amp=amp, flow=flow, protected=protected)
                        prefill, _ = self.graphs(reference)
                        inputs, model = reference.inputs, reference.model
                        with torch.no_grad():
                            actual = prefill(
                                *(inputs[name]
                                  for name in prefill.input_names))
                            with torch.autocast(
                                    'cpu', dtype=torch.bfloat16, enabled=amp):
                                embeddings, padding, blocks = model.embed_prefix(
                                    images=inputs['images'],
                                    img_masks=inputs['img_masks'],
                                    lang_tokens=inputs['lang_tokens'],
                                    lang_masks=inputs['lang_masks'])
                                embeddings = model._cast_gemma_input(
                                    embeddings)
                                masks = model._prepare_attention_masks_4d(
                                    make_att_2d_masks(padding, blocks))
                                _, cache = model.forward_model(
                                    [embeddings, None],
                                    masks,
                                    torch.cumsum(padding, dim=1) - 1,
                                    use_cache=True)
                        expected = (padding, ) + tuple(
                            value for layer in cache.layers
                            for value in (layer.keys, layer.values))
                        self.assert_equal(actual, expected)
                        self.assertEqual(actual[0].dtype, torch.bool)
                        self.assertEqual(actual[1].shape,
                                         (2, 1, padding.shape[1], 8))

    def test_states_free_denoise_matches_scalar_and_per_token_time(self):
        for amp in (False, True):
            for flow in (False, True):
                for states_bf16 in ((False, True) if amp else (False, )):
                    for token_time in (False, True):
                        with self.subTest(
                                amp=amp,
                                flow=flow,
                                states_bf16=states_bf16,
                                token_time=token_time):
                            reference = self.reference(
                                amp=amp,
                                flow=flow,
                                protected=True,
                                states_bf16=states_bf16)
                            prefill, denoise = self.graphs(reference)
                            model, inputs = reference.model, reference.inputs
                            timestep = (
                                torch.tensor([[0.0, 0.25, 0.75],
                                              [0.5, 0.0, 1.0]])
                                if token_time else torch.tensor([0.25, 0.75]))
                            with torch.no_grad():
                                flat = prefill(
                                    *(inputs[name]
                                      for name in prefill.input_names))
                                actual = denoise(inputs['noise'], timestep,
                                                 *flat)
                                with torch.autocast(
                                        'cpu', dtype=torch.bfloat16,
                                        enabled=amp):
                                    padding, cache = model.prefill_prefix(
                                        inputs['images'],
                                        inputs['lang_tokens'],
                                        inputs['img_masks'],
                                        inputs['lang_masks'])
                                    expected = model.denoise_step(
                                        inputs['states'], padding, cache,
                                        inputs['noise'], timestep)
                                    # State values are genuinely unused, not a
                                    # constant-zero edge inserted for an exporter.
                                    changed_states = model.denoise_step(
                                        inputs['states'] + 50, padding, cache,
                                        inputs['noise'], timestep)
                            self.assert_equal(actual, expected)
                            self.assert_equal(actual, changed_states)

    def test_graph_euler_matches_reference_without_mutating_inputs_or_weights(
            self):
        for amp, backend in ((False, 'sdpa'), (True, 'sdpa'), (False,
                                                               'eager')):
            with self.subTest(amp=amp, backend=backend):
                reference = self.reference(
                    amp=amp, flow=True, protected=True, backend=backend)
                model, inputs = reference.model, reference.inputs
                prefill, denoise = self.graphs(reference)
                state = {
                    name: value.clone()
                    for name, value in model.state_dict().items()
                }
                fixtures = {
                    name: value.clone()
                    for name, value in inputs.items()
                }
                configs = [
                    copy.deepcopy(module.config.to_dict())
                    for module in (model.llm_backbone, model.llm_expert,
                                   model.vision_backbone.vision)
                ]
                rng = torch.random.get_rng_state().clone()
                with torch.no_grad():
                    flat = prefill(*(inputs[name]
                                     for name in prefill.input_names))
                    cache_before = tuple(value.clone() for value in flat)
                    noise = inputs['noise'].clone()
                    actual = plain_euler(noise,
                                         lambda x, t: denoise(x, t, *flat),
                                         noise.shape[0],
                                         torch.tensor(-1.0 / model.num_steps),
                                         noise.device)
                    expected = reference.run_reference()['actions']
                    repeated = plain_euler(
                        inputs['noise'].clone(),
                        lambda x, t: denoise(x, t, *flat), noise.shape[0],
                        torch.tensor(-1.0 / model.num_steps), noise.device)
                self.assertEqual(actual.data_ptr(), noise.data_ptr())
                self.assert_equal(actual, expected)
                self.assert_equal(actual, repeated)
                self.assert_equal(flat, cache_before)
                for name in inputs:
                    self.assert_equal(inputs[name], fixtures[name])
                for name, value in model.state_dict().items():
                    self.assert_equal(value, state[name])
                self.assert_equal(torch.random.get_rng_state(), rng)
                self.assertEqual(configs, [
                    module.config.to_dict()
                    for module in (model.llm_backbone, model.llm_expert,
                                   model.vision_backbone.vision)
                ])

    def test_denoise_preserves_mixed_per_layer_cache_dtypes(self):
        reference = self.reference(
            amp=True, flow=True, protected=True, mixed_cache=True)
        prefill, denoise = self.graphs(reference)
        inputs, model = reference.inputs, reference.model
        with torch.no_grad():
            flat = prefill(*(inputs[name] for name in prefill.input_names))
            self.assertEqual(
                tuple(value.dtype for value in flat),
                (torch.bool, torch.float32, torch.float32, torch.bfloat16,
                 torch.bfloat16))
            before = tuple(value.clone() for value in flat)
            time = torch.tensor([0.2, 0.8])
            actual = denoise(inputs['noise'], time, *flat)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                padding, cache = model.prefill_prefix(inputs['images'],
                                                      inputs['lang_tokens'],
                                                      inputs['img_masks'],
                                                      inputs['lang_masks'])
                expected = model.denoise_step(inputs['states'], padding, cache,
                                              inputs['noise'], time)
        self.assert_equal(actual, expected)
        self.assert_equal(flat, before)

    def test_fp32_wrappers_disable_ambient_autocast(self):
        reference = self.reference()
        prefill, denoise = self.graphs(reference)
        inputs = reference.inputs
        timestep = torch.tensor([0.2, 0.8])
        with torch.no_grad():
            expected_prefix = prefill(*(inputs[n]
                                        for n in prefill.input_names))
            expected_velocity = denoise(inputs['noise'], timestep,
                                        *expected_prefix)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual_prefix = prefill(*(inputs[n]
                                          for n in prefill.input_names))
                actual_velocity = denoise(inputs['noise'], timestep,
                                          *actual_prefix)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assert_equal(actual_prefix, expected_prefix)
        self.assert_equal(actual_velocity, expected_velocity)

    def test_denoise_preserves_real_input_gradients(self):
        reference = self.reference(flow=True)
        prefill, denoise = self.graphs(reference)
        inputs = reference.inputs
        with torch.no_grad():
            flat = prefill(*(inputs[n] for n in prefill.input_names))
        noise = inputs['noise'].clone().requires_grad_(True)
        timestep = torch.tensor([0.2, 0.8], requires_grad=True)
        output = denoise(noise, timestep, *flat)
        gradients = torch.autograd.grad(output.square().mean(),
                                        (noise, timestep))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_wrappers_reject_invalid_models_metadata_and_incomplete_cache(
            self):
        reference = self.reference()
        with self.assertRaisesRegex(ContractError, 'PI05Model'):
            PI05PrefillGraph(torch.nn.Linear(2, 2))
        for invalid in (0, -1, True):
            with self.assertRaisesRegex(ContractError, 'layer count'):
                cache_tensor_names(invalid)
        with self.assertRaisesRegex(ContractError, 'states_dtype'):
            PI05DenoiseGraph(reference.model, states_dtype=torch.float64)
        with self.assertRaisesRegex(ContractError, 'autocast'):
            PI05DenoiseGraph(reference.model, states_dtype=torch.bfloat16)
        prefill, denoise = self.graphs(reference)
        with torch.no_grad():
            flat = prefill(*(reference.inputs[n] for n in prefill.input_names))
        with self.assertRaisesRegex(ContractError, 'cache tensors'):
            denoise(reference.inputs['noise'], torch.ones(2), *flat[:-1])
        reference.model.llm_expert.train()
        with self.assertRaisesRegex(ContractError, 'eval'):
            PI05PrefillGraph(reference.model)
        self.assertTrue(reference.model.llm_expert.training)

    def test_euler_runtime_helper_has_no_model_or_transformers_dependency(
            self):
        code = '''
import sys
from fluxvla_export.components.euler import plain_euler
for prefix in ("transformers", "fluxvla", "mmengine", "accelerate", "timm"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)
'''
        result = subprocess.run([sys.executable, '-c', code],
                                env=os.environ.copy(),
                                capture_output=True,
                                text=True,
                                check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
