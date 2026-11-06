# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Preparation of real native PI0.5 modules and explicit flat graph contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from fluxvla_export.contracts import PreparedDeployment, TensorSpec
from fluxvla_export.errors import ArtifactError, ConfigError
from fluxvla_export.recipes.pi05.graphs import PI05DenoiseGraph, PI05PrefillGraph
from fluxvla_export.recipes.pi05.recipe import PI05Recipe
from fluxvla_export.recipes.pi05.reference import prepare_reference_from_config
from _pi05_test_utils import Artifacts
from test_config import config_document
from test_pi05_executor import graph_engines


class PI05RecipeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def prepare(self, artifacts, *, config=None):
        return PI05Recipe().prepare(
            config=artifacts.config if config is None else config,
            checkpoint=artifacts.checkpoint_path,
            inputs=artifacts.inputs_path,
            device='cpu')

    def test_prepared_deployment_contains_actual_shared_model_graphs(self):
        artifacts = Artifacts(self.directory)
        rng = torch.random.get_rng_state().clone()
        prepared = self.prepare(artifacts)
        self.assertIsInstance(prepared, PreparedDeployment)
        self.assertEqual(prepared.recipe, 'pi05-v1')
        self.assertEqual(tuple(prepared.graphs), ('prefill', 'denoise'))
        prefill = prepared.graphs['prefill']
        denoise = prepared.graphs['denoise']
        expected_export_options = {
            'opset': 18,
            'dynamo': False,
            'do_constant_folding': False,
        }
        self.assertEqual(prefill.export_options, expected_export_options)
        self.assertEqual(denoise.export_options, expected_export_options)
        self.assertIsInstance(prefill.module, PI05PrefillGraph)
        self.assertIsInstance(denoise.module, PI05DenoiseGraph)
        self.assertIs(prefill.module.model, denoise.module.model)
        self.assertFalse(prefill.module.training)
        self.assertFalse(denoise.module.training)
        self.assertEqual(
            tuple(prefill.inputs),
            ('images', 'img_masks', 'lang_tokens', 'lang_masks'))
        self.assertEqual(
            tuple(spec.name for spec in prefill.output_specs),
            ('prefix_pad_masks', 'key_0', 'value_0'))
        self.assertEqual(
            tuple(denoise.inputs), ('noisy_actions', 'timestep',
                                    'prefix_pad_masks', 'key_0', 'value_0'))
        self.assertEqual(denoise.output_specs,
                         (TensorSpec('velocity', 'float32', (2, 3, 4)), ))
        self.assertEqual(prepared.output_specs,
                         (TensorSpec('actions', 'float32', (2, 3, 4)), ))
        self.assertEqual(prepared.policy_input_specs,
                         artifacts.config.input_specs)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        with torch.no_grad():
            expected = artifacts.model(**artifacts.tensors)
        torch.testing.assert_close(
            prepared.run_reference()['actions'], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            prepared.create_executor(graph_engines(prepared)).infer(
                prepared.policy_inputs)['actions'],
            expected,
            rtol=0,
            atol=0)
        for name, expected_tensor in artifacts.model.state_dict().items():
            torch.testing.assert_close(
                prefill.module.model.state_dict()[name],
                expected_tensor,
                rtol=0,
                atol=0)
        self.assertIn('llm_expert.embed_tokens.weight',
                      prefill.module.model.state_dict())

    def test_metadata_is_json_safe_self_contained_and_preserves_precision(
            self):
        document = config_document()
        document['precision'].update(
            autocast_dtype='bfloat16', flow_fp32=True, adarms_fp32=True)
        artifacts = Artifacts(self.directory, document)
        prepared = self.prepare(artifacts)
        encoded = json.dumps(prepared.metadata, allow_nan=False)
        restored = json.loads(encoded)
        self.assertEqual(restored, prepared.metadata)
        self.assertEqual(restored['precision_policy'], document['precision'])
        self.assertEqual(restored['checkpoint_sha256'],
                         artifacts.config.checkpoint_sha256)
        self.assertEqual(restored['inputs_sha256'],
                         artifacts.config.inputs_sha256)
        runtime = restored['runtime']
        self.assertEqual(
            runtime['entrypoint'],
            'fluxvla_export.recipes.pi05.executor:build_executor')
        self.assertEqual(runtime['num_steps'], 2)
        self.assertEqual(runtime['vocab_size'], 64)
        self.assertNotIn('checkpoint', runtime)
        self.assertNotIn('model', runtime)
        self.assertEqual(runtime['velocity_spec']['dtype'], 'float32')

    def test_mixed_layer_cache_dtypes_are_measured_from_real_graph(self):
        document = config_document()
        for branch in ('backbone', 'expert'):
            document['model'][branch]['num_hidden_layers'] = 2
        document['precision'].update(
            autocast_dtype='bfloat16',
            flow_fp32=True,
            residual_fp32=False,
            backbone_fp32_layers=[0],
            expert_fp32_layers=[0])
        artifacts = Artifacts(self.directory, document)
        prepared = self.prepare(artifacts)
        for graph in prepared.graphs.values():
            self.assertEqual(
                graph.export_options, {
                    'opset': 22,
                    'dynamo': True,
                    'optimize': True,
                    'do_constant_folding': False,
                })
        specs = prepared.graphs['prefill'].output_specs
        self.assertEqual(
            tuple(spec.name for spec in specs),
            ('prefix_pad_masks', 'key_0', 'value_0', 'key_1', 'value_1'))
        self.assertEqual(
            tuple(spec.dtype for spec in specs[1:]),
            ('float32', 'float32', 'bfloat16', 'bfloat16'))
        runtime = prepared.metadata['runtime']
        self.assertEqual(
            tuple(spec['dtype'] for spec in runtime['prefill_output_specs']),
            tuple(spec.dtype for spec in specs))
        actual = prepared.create_executor(graph_engines(prepared)).infer(
            prepared.policy_inputs)
        torch.testing.assert_close(
            actual['actions'],
            prepared.run_reference()['actions'],
            rtol=0,
            atol=0)

    def test_typed_reference_path_does_not_reparse_changed_json_file(self):
        artifacts = Artifacts(self.directory)
        config = artifacts.config
        artifacts.config_path.write_text('{invalid-json', encoding='utf-8')
        reference = prepare_reference_from_config(
            config=config,
            checkpoint=artifacts.checkpoint_path,
            inputs=artifacts.inputs_path,
            device='cpu')
        prepared = self.prepare(artifacts, config=config)
        torch.testing.assert_close(
            prepared.run_reference()['actions'],
            reference.run_reference()['actions'],
            rtol=0,
            atol=0)

    def test_typed_reference_revalidates_full_config_and_input_cross_dimensions(
            self):
        artifacts = Artifacts(self.directory)
        config = artifacts.config
        invalid_inputs = list(config.input_specs)
        invalid_inputs[-1] = TensorSpec('noise', 'float32', (2, 2, 4))
        invalid = (
            replace(config, input_specs=tuple(invalid_inputs)),
            replace(config, inference=replace(config.inference, rtc=True)),
            replace(config, checkpoint_sha256='not-a-digest'),
            replace(
                config,
                preprocessing=replace(
                    config.preprocessing, view_order=('front', ))),
        )
        for item in invalid:
            with self.subTest(config=item):
                with self.assertRaises(ConfigError):
                    self.prepare(artifacts, config=item)

    def test_typed_reference_preserves_checkpoint_and_fixture_hash_binding(
            self):
        artifacts = Artifacts(self.directory)
        for field in ('checkpoint_sha256', 'inputs_sha256'):
            with self.subTest(field=field):
                wrong = replace(artifacts.config, **{field: '0' * 64})
                with self.assertRaisesRegex(ArtifactError, 'SHA-256'):
                    self.prepare(artifacts, config=wrong)

    def test_preparation_leaves_fixture_noise_and_global_rng_unchanged(self):
        artifacts = Artifacts(self.directory)
        fixture_bytes = artifacts.inputs_path.read_bytes()
        rng = torch.random.get_rng_state().clone()
        first = self.prepare(artifacts)
        second = self.prepare(artifacts)
        self.assertEqual(artifacts.inputs_path.read_bytes(), fixture_bytes)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        torch.testing.assert_close(
            first.policy_inputs['noise'],
            artifacts.tensors['noise'],
            rtol=0,
            atol=0)
        torch.testing.assert_close(
            first.run_reference()['actions'],
            second.run_reference()['actions'],
            rtol=0,
            atol=0)


if __name__ == '__main__':
    unittest.main()
