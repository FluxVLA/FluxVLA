# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Standalone PI0.5 tests with real tiny HF models and real safetensors."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from transformers import SiglipVisionModel

from fluxvla_export.components.condition_gemma import ConditionGemmaModel
from fluxvla_export.components.pi_policy import PI05Computation
from fluxvla_export.components.projector import LinearProjector
from fluxvla_export.components.siglip import SigLIPViTBackbone
from fluxvla_export.errors import ArtifactError, ConfigError, ContractError
from fluxvla_export.recipes.pi05.model import PI05Model, build_pi05_model
from _pi05_test_utils import Artifacts, fixture_tensors, write_config
from test_config import config_document


class PI05ReferenceTests(unittest.TestCase):

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

    def config(self, document=None):
        return write_config(
            self.directory / 'deployment.json',
            config_document() if document is None else document)

    def assert_tensors_equal(self, actual, expected):
        self.assertEqual(tuple(actual), tuple(expected))
        for name in expected:
            with self.subTest(tensor=name):
                torch.testing.assert_close(
                    actual[name], expected[name], rtol=0, atol=0)
                self.assertEqual(actual[name].dtype, expected[name].dtype)

    def test_factory_constructs_real_modules_and_native_complete_state(self):
        model = build_pi05_model(self.config())
        self.assertIsInstance(model, PI05Model)
        self.assertIsInstance(model, PI05Computation)
        self.assertIsInstance(model.vision_backbone, SigLIPViTBackbone)
        self.assertIsInstance(model.vision_backbone.vision, SiglipVisionModel)
        self.assertIsInstance(model.llm_backbone, ConditionGemmaModel)
        self.assertIsInstance(model.llm_expert, ConditionGemmaModel)
        for name in ('projector', 'action_in_proj', 'action_out_proj',
                     'time_mlp_in', 'time_mlp_out'):
            self.assertIsInstance(getattr(model, name), LinearProjector)
        self.assertEqual(
            tuple(model._modules),
            ('vision_backbone', 'llm_backbone', 'projector', 'action_in_proj',
             'action_out_proj', 'time_mlp_in', 'time_mlp_out', 'llm_expert'))
        for name in ('vlm_backbone', 'vla_head', 'state_proj',
                     'action_time_mlp_in', 'action_time_mlp_out'):
            self.assertIsNone(getattr(model, name))
        self.assertIn('llm_expert.embed_tokens.weight', model.state_dict())
        self.assertIn('projector.projector.bias', model.state_dict())
        self.assertIn(
            'vision_backbone.vision.vision_model.embeddings.'
            'patch_embedding.weight', model.state_dict())
        self.assertTrue(
            all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertIs(PI05Model.embed_suffix, PI05Computation.embed_suffix)
        self.assertIs(PI05Model.predict_action, PI05Computation.predict_action)

    def test_raw_builder_consumes_rng_normally_without_reseeding(self):
        config = self.config()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(512)
            before = torch.random.get_rng_state().clone()
            first = build_pi05_model(config)
            after = torch.random.get_rng_state().clone()
            self.assertFalse(torch.equal(before, after))
            torch.random.set_rng_state(before)
            second = build_pi05_model(config)
            self.assert_tensors_equal(first.state_dict(), second.state_dict())
            torch.testing.assert_close(torch.random.get_rng_state(), after)
            third = build_pi05_model(config)
            self.assertFalse(
                torch.equal(first.action_in_proj.projector.weight,
                            third.action_in_proj.projector.weight))

    def test_raw_builder_rejects_non_fp32_default_without_changing_it(self):
        config = self.config()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with self.assertRaisesRegex(ContractError,
                                        'default dtype float32'):
                build_pi05_model(config)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(previous)

    def test_factory_maps_every_architecture_and_precision_field(self):
        document = config_document()
        document['model']['vision'].update(
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=3,
            layer_norm_eps=1e-5)
        document['model']['projector']['in_dim'] = 24
        for branch in ('backbone', 'expert'):
            document['model'][branch].update(
                intermediate_size=48,
                num_hidden_layers=2,
                vocab_size=81,
                max_position_embeddings=80,
                rms_norm_eps=1e-5,
                rope_theta=12000.0,
                pad_token_id=None,
                bos_token_id=3,
                eos_token_id=4)
        document['precision'].update(
            autocast_dtype='bfloat16',
            flow_fp32=True,
            vision_stem_fp32=True,
            residual_fp32=True,
            vision_residual_fp32=True,
            adarms_fp32=True,
            attention_math_fp32=True,
            backbone_fp32_layers=[1],
            expert_fp32_layers=[0])
        config = self.config(document)
        model = build_pi05_model(config)
        vision = model.vision_backbone.vision.config
        for name, value in vars(config.model.vision).items():
            if name != 'type':
                hf_name = 'hidden_act' if name == 'hidden_activation' else name
                self.assertEqual(getattr(vision, hf_name), value)
        self.assertTrue(model.vision_backbone.openpi_stem_fp32)
        self.assertTrue(model.vision_backbone.preserve_fp32_residuals)
        self.assertTrue(model.enable_mixed_precision_training)
        self.assertTrue(model.openpi_fp32_flow)
        self.assertTrue(model.preserve_fp32_residuals)
        self.assertEqual(model.num_steps, config.inference.num_steps)
        self.assertEqual(model.n_action_steps, config.model.action_horizon)
        for name, module in (('backbone', model.llm_backbone),
                             ('expert', model.llm_expert)):
            for field, value in vars(getattr(config.model, name)).items():
                if field == 'rope_theta':
                    self.assertEqual(module.config.rope_parameters[field],
                                     value)
                elif field != 'type':
                    self.assertEqual(getattr(module.config, field), value)
            self.assertEqual(module.config._attn_implementation, 'sdpa')
            self.assertTrue(module.config.adarms_fp32)
            self.assertTrue(module.config.attention_math_fp32)
            self.assertEqual(module.config.fp32_layers,
                             getattr(config.precision, f'{name}_fp32_layers'))
            for index, layer in enumerate(module.layers):
                self.assertEqual(layer.self_attn.compute_fp32, index
                                 in module.config.fp32_layers)
                self.assertEqual(layer.mlp.compute_fp32, index
                                 in module.config.fp32_layers)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            output = model(**fixture_tensors(config))
        self.assertEqual(tuple(output.shape), (2, 3, 4))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_factory_revalidates_unsupported_replaced_config_flags(self):
        config = self.config()
        invalid = [
            ('recipe', 'other'),
            ('schema_version', True),
            ('inference.rtc', True),
            ('inference.mode', 'rtc'),
            ('inference.solver', 'heun'),
            ('precision.parameter_dtype', 'bfloat16'),
            ('precision.attention_backend', 'flash'),
            ('model.state_projection', True),
            ('model.vision.vision_use_head', True),
            ('model.projector.bias', False),
            ('model.expert.use_adarms', False),
        ]

        def replaced(record, path, value):
            name, *tail = path
            child = replaced(getattr(record, name), tail,
                             value) if tail else value
            return replace(record, **{name: child})

        for path, value in invalid:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ConfigError, path.split('.')[-1]):
                    build_pi05_model(replaced(config, path.split('.'), value))
        with self.assertRaisesRegex(ConfigError, 'DeploymentConfig'):
            build_pi05_model({})

    def test_six_input_forward_uses_shared_plain_prediction_and_clones_noise(
            self):
        config = self.config()
        model = build_pi05_model(config)
        inputs = fixture_tensors(config)
        before = {key: value.clone() for key, value in inputs.items()}
        with torch.no_grad():
            actual = model(*(inputs[spec.name] for spec in config.input_specs))
            expected = model.predict_action(
                **dict(inputs, noise=inputs['noise'].clone()),
                prev_actions=None,
                prefix_len=0,
                rtc_config=None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertNotEqual(actual.data_ptr(), inputs['noise'].data_ptr())
        self.assertFalse(torch.equal(actual, inputs['noise']))
        self.assert_tensors_equal(inputs, before)

    def test_independent_model_reports_training_only_extensions(self):
        config = self.config()
        model = build_pi05_model(config)
        inputs = fixture_tensors(config)
        with self.assertRaisesRegex(NotImplementedError, 'noise|sample_noise'):
            model.predict_action(**dict(inputs, noise=None))
        for method in ('prefix', 'guidance', 'unknown'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(NotImplementedError, 'RTC|rtc'):
                    model.predict_action(
                        **inputs,
                        prev_actions=torch.zeros_like(inputs['noise']),
                        prefix_len=1,
                        rtc_config={'method': method})
        with self.assertRaisesRegex(NotImplementedError, 'joint|transformer'):
            model.forward_model(
                inputs_embeds=[torch.zeros(2, 4, 16),
                               torch.zeros(2, 3, 16)],
                attention_masks=torch.ones(2, 1, 7, 7, dtype=torch.bool),
                position_ids=torch.arange(7).expand(2, -1))

    def test_prepare_strictly_loads_complete_checkpoint_and_returns_real_model(
            self):
        artifacts = Artifacts(self.directory)
        state = dict(artifacts.model.state_dict())
        state['llm_expert.embed_tokens.weight'] = torch.full_like(
            state['llm_expert.embed_tokens.weight'], 7.0)
        artifacts.save_checkpoint(state)
        reference = artifacts.prepare()
        self.assertIsInstance(reference.model, PI05Model)
        self.assertEqual(reference.config, artifacts.config)
        self.assertEqual(reference.device, torch.device('cpu'))
        self.assert_tensors_equal(reference.model.state_dict(), state)
        self.assert_tensors_equal(reference.inputs, artifacts.tensors)
        with torch.no_grad():
            artifacts.model.load_state_dict(state, strict=True)
            expected = artifacts.model(**artifacts.tensors)
        result = reference.run_reference()
        self.assertEqual(tuple(result), ('actions', ))
        torch.testing.assert_close(result['actions'], expected, rtol=0, atol=0)

    def test_repeated_reference_preserves_inputs_model_state_configs_and_rng(
            self):
        artifacts = Artifacts(self.directory)
        rng_before = torch.random.get_rng_state().clone()
        python_before = random.getstate()
        reference = artifacts.prepare()
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertEqual(random.getstate(), python_before)
        state = {
            name: value.clone()
            for name, value in reference.model.state_dict().items()
        }
        buffers = {
            name: value.clone()
            for name, value in reference.model.named_buffers()
        }
        inputs = {
            name: value.clone()
            for name, value in reference.inputs.items()
        }
        configs = [
            copy.deepcopy(module.config.to_dict())
            for module in (reference.model.llm_backbone,
                           reference.model.llm_expert,
                           reference.model.vision_backbone.vision)
        ]
        first = reference.run_reference()['actions']
        second = reference.run_reference()['actions']
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(first.requires_grad)
        self.assertIsNone(first.grad_fn)
        self.assertTrue(
            all(p.grad is None for p in reference.model.parameters()))
        self.assertTrue(all(not m.training for m in reference.model.modules()))
        self.assert_tensors_equal(reference.model.state_dict(), state)
        self.assert_tensors_equal(
            dict(reference.model.named_buffers()), buffers)
        self.assert_tensors_equal(reference.inputs, inputs)
        self.assertEqual(configs, [
            module.config.to_dict()
            for module in (reference.model.llm_backbone,
                           reference.model.llm_expert,
                           reference.model.vision_backbone.vision)
        ])
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertEqual(random.getstate(), python_before)
        first.zero_()
        torch.testing.assert_close(
            reference.run_reference()['actions'], second, rtol=0, atol=0)

    def test_reference_inputs_mapping_is_read_only(self):
        reference = Artifacts(self.directory).prepare()
        with self.assertRaises(TypeError):
            reference.inputs['noise'] = reference.inputs['noise'].clone()
        with self.assertRaises(TypeError):
            del reference.inputs['images']
        self.assertEqual(
            tuple(reference.inputs),
            tuple(spec.name for spec in reference.config.input_specs))

    def test_reference_rejects_caller_changed_training_mode_without_mutation(
            self):
        reference = Artifacts(self.directory).prepare()
        reference.model.llm_expert.train()
        with self.assertRaisesRegex(ContractError, 'eval mode'):
            reference.run_reference()
        self.assertTrue(reference.model.llm_expert.training)
        self.assertFalse(reference.model.training)

    def test_reference_rejects_real_overflow_from_finite_weights_and_inputs(
            self):
        artifacts = Artifacts(self.directory)
        state = dict(artifacts.model.state_dict())
        key = 'action_in_proj.projector.weight'
        state[key] = torch.full_like(state[key],
                                     torch.finfo(torch.float32).max)
        artifacts.save_checkpoint(state)
        inputs = dict(artifacts.tensors)
        inputs['noise'] = torch.ones_like(inputs['noise'])
        artifacts.save_inputs(inputs)
        reference = artifacts.prepare()
        with self.assertRaisesRegex(ContractError, 'actions.*finite'):
            reference.run_reference()
        self.assert_tensors_equal(reference.inputs, inputs)
        self.assert_tensors_equal(reference.model.state_dict(), state)

    def test_reference_checks_real_output_shape_after_caller_tensor_edit(self):
        reference = Artifacts(self.directory).prepare()
        reference.inputs['noise'].resize_(2, 2, 4)
        with self.assertRaisesRegex(ContractError, 'actions.*dimension'):
            reference.run_reference()

    def test_fp32_reference_explicitly_disables_ambient_autocast(self):
        reference = Artifacts(self.directory).prepare()
        expected = reference.run_reference()['actions']
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = reference.run_reference()['actions']
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bfloat16_reference_preserves_fp32_masters_and_all_protected_modes(
            self):
        for protected in (False, True):
            for flow in (False, True):
                with self.subTest(protected=protected, flow=flow):
                    document = config_document()
                    document['precision'].update(
                        autocast_dtype='bfloat16',
                        flow_fp32=flow,
                        vision_stem_fp32=protected,
                        residual_fp32=protected,
                        vision_residual_fp32=protected,
                        adarms_fp32=protected,
                        attention_math_fp32=protected,
                        backbone_fp32_layers=[0] if protected else [],
                        expert_fp32_layers=[0] if protected else [])
                    document['inputs']['tensors']['images'][
                        'dtype'] = 'bfloat16'
                    document['inputs']['tensors']['states'][
                        'dtype'] = 'bfloat16'
                    artifacts = Artifacts(self.directory, document)
                    rng = torch.random.get_rng_state().clone()
                    reference = artifacts.prepare()
                    before = {
                        k: v.clone()
                        for k, v in reference.inputs.items()
                    }
                    first = reference.run_reference()['actions']
                    second = reference.run_reference()['actions']
                    with torch.no_grad(), torch.autocast(
                            'cpu', dtype=torch.bfloat16):
                        expected = artifacts.model(**artifacts.tensors)
                    torch.testing.assert_close(first, expected, rtol=0, atol=0)
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                    self.assertEqual(first.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(first).all().item())
                    self.assertTrue(
                        all(p.dtype == torch.float32
                            for p in reference.model.parameters()))
                    self.assert_tensors_equal(reference.inputs, before)
                    torch.testing.assert_close(torch.random.get_rng_state(),
                                               rng)

    def test_eager_backend_is_explicit_and_repeatable(self):
        document = config_document()
        document['precision']['attention_backend'] = 'eager'
        reference = Artifacts(self.directory, document).prepare()
        for gemma in (reference.model.llm_backbone,
                      reference.model.llm_expert):
            self.assertEqual(gemma.config._attn_implementation, 'eager')
        torch.testing.assert_close(
            reference.run_reference()['actions'],
            reference.run_reference()['actions'],
            rtol=0,
            atol=0)

    def test_checkpoint_rejects_missing_unused_embedding_extra_head_and_remapping(
            self):
        artifacts = Artifacts(self.directory)
        original = dict(artifacts.model.state_dict())
        for variant in ('missing', 'head', 'remapped'):
            with self.subTest(variant=variant):
                state = dict(original)
                if variant == 'missing':
                    del state['llm_expert.embed_tokens.weight']
                elif variant == 'head':
                    state[
                        'vision_backbone.vision.vision_model.head.probe'] = torch.zeros(
                            1, 1, 16)
                else:
                    state = {
                        f'module.{key}': value
                        for key, value in state.items()
                    }
                artifacts.save_checkpoint(state)
                rng = torch.random.get_rng_state().clone()
                with self.assertRaisesRegex(ArtifactError, 'keys'):
                    artifacts.prepare()
                torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_checkpoint_rejects_shape_dtype_and_nonfinite_without_casting(
            self):
        artifacts = Artifacts(self.directory)
        original = dict(artifacts.model.state_dict())
        key = 'action_in_proj.projector.weight'
        for variant in ('shape', 'dtype', 'finite'):
            with self.subTest(variant=variant):
                state = dict(original)
                if variant == 'shape':
                    state[key] = state[key][:-1].clone()
                elif variant == 'dtype':
                    state[key] = state[key].to(torch.bfloat16)
                else:
                    state[key] = state[key].clone()
                    state[key][0, 0] = float('nan')
                artifacts.save_checkpoint(state)
                with self.assertRaisesRegex(ArtifactError, variant):
                    artifacts.prepare()

    def test_checkpoint_rejects_native_bfloat16_norm_storage(self):
        artifacts = Artifacts(self.directory)
        state = dict(artifacts.model.state_dict())
        name = 'llm_backbone.norm.weight'
        state[name] = state[name].to(torch.bfloat16)
        artifacts.save_checkpoint(state)
        with self.assertRaisesRegex(ArtifactError, 'norm.weight.*dtype'):
            artifacts.prepare()

    def test_missing_checkpoint_fails_before_model_construction(self):
        from fluxvla_export.recipes.pi05.reference import prepare_reference

        artifacts = Artifacts(self.directory)
        previous = torch.get_default_dtype()
        rng = torch.random.get_rng_state().clone()
        try:
            # Real construction would reject this dtype, without a mock.
            torch.set_default_dtype(torch.float64)
            with self.assertRaisesRegex(ArtifactError,
                                        'Checkpoint.*not found'):
                prepare_reference(
                    config=artifacts.config_path,
                    checkpoint=self.directory / 'missing.safetensors',
                    inputs=artifacts.inputs_path,
                    device='cpu')
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            torch.testing.assert_close(torch.random.get_rng_state(), rng)
        finally:
            torch.set_default_dtype(previous)

    def test_bad_checkpoint_hash_fails_before_model_construction(self):
        artifacts = Artifacts(self.directory)
        document = copy.deepcopy(artifacts.document)
        document['checkpoint']['sha256'] = '0' * 64
        write_config(artifacts.config_path, document)
        previous = torch.get_default_dtype()
        rng = torch.random.get_rng_state().clone()
        try:
            torch.set_default_dtype(torch.float64)
            with self.assertRaisesRegex(ArtifactError, 'Checkpoint.*SHA-256'):
                artifacts.prepare()
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            torch.testing.assert_close(torch.random.get_rng_state(), rng)
        finally:
            torch.set_default_dtype(previous)

    def test_reference_rechecks_both_artifact_hashes(self):
        artifacts = Artifacts(self.directory)
        for field in ('checkpoint', 'inputs'):
            with self.subTest(field=field):
                document = copy.deepcopy(artifacts.document)
                document[field]['sha256'] = '0' * 64
                write_config(artifacts.config_path, document)
                with self.assertRaisesRegex(ArtifactError, 'SHA-256'):
                    artifacts.prepare()
        write_config(artifacts.config_path, artifacts.document)

    def test_fixture_rejects_out_of_vocabulary_tokens_even_when_masked(self):
        artifacts = Artifacts(self.directory)
        for token in (-1, artifacts.config.model.vocab_size):
            with self.subTest(token=token):
                inputs = {
                    key: value.clone()
                    for key, value in artifacts.tensors.items()
                }
                inputs['lang_tokens'][0, -1] = token
                self.assertFalse(inputs['lang_masks'][0, -1].item())
                artifacts.save_inputs(inputs)
                with self.assertRaisesRegex(ArtifactError, 'lang_tokens'):
                    artifacts.prepare()

    def test_fixture_reader_rejects_invalid_real_tensors(self):
        artifacts = Artifacts(self.directory)
        for variant in ('keys', 'shape', 'dtype', 'finite'):
            with self.subTest(variant=variant):
                inputs = dict(artifacts.tensors)
                if variant == 'keys':
                    inputs['extra'] = torch.zeros(1)
                elif variant == 'shape':
                    inputs['noise'] = inputs['noise'][:, :-1].contiguous()
                elif variant == 'dtype':
                    inputs['noise'] = inputs['noise'].to(torch.float64)
                else:
                    inputs['images'] = inputs['images'].clone()
                    inputs['images'][0, 0, 0, 0] = float('inf')
                artifacts.save_inputs(inputs)
                message = 'dimension' if variant == 'shape' else variant
                with self.assertRaisesRegex(ArtifactError, message):
                    artifacts.prepare()

    def test_prepare_uses_strict_json_reader_and_rejects_unsupported_head(
            self):
        artifacts = Artifacts(self.directory)
        document = copy.deepcopy(artifacts.document)
        document['model']['vision']['vision_use_head'] = True
        artifacts.config_path.write_text(
            json.dumps(document), encoding='utf-8')
        with self.assertRaisesRegex(ConfigError, 'vision_use_head'):
            artifacts.prepare()
        artifacts.config_path.write_text(
            '{"recipe":"pi05-v1","recipe":"pi05-v1"}', encoding='utf-8')
        with self.assertRaisesRegex(ConfigError, 'duplicate'):
            artifacts.prepare()

    def test_device_errors_are_explicit_without_fallback(self):
        artifacts = Artifacts(self.directory)
        for device in ('mps', 'cpu:0', 'cuda:-1', 'cuda:01', 'CUDA', '', None):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ContractError, 'device'):
                    artifacts.prepare(device=device)
        invalid = f'cuda:{torch.cuda.device_count()}'
        with self.assertRaisesRegex(ContractError, 'CUDA'):
            artifacts.prepare(device=invalid)

    def test_registered_recipe_keeps_reference_imports_lazy(self):
        code = '''
import sys
import fluxvla_export.recipes.pi05
import fluxvla_export.recipes.pi05.reference
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
from fluxvla_export.registry import available_recipes
assert available_recipes() == ("pi05-v1",)
'''
        result = subprocess.run([sys.executable, '-c', code],
                                env=os.environ.copy(),
                                text=True,
                                capture_output=True,
                                check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
