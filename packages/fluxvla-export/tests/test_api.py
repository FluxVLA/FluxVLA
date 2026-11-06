"""API preflight/dispatch tests with real tensor files, not a PI0.5 model."""

from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from fluxvla_export import (ArtifactError, ConfigError, ContractError,
                            DeploymentConfig, PreparedDeployment,
                            PreparedGraph, RecipeNotFoundError, TensorSpec,
                            prepare_tensorrt, registry)
from safetensors.torch import save_file
from test_config import config_document


class IdentityExecutor:
    """Test-only executor for the Identity dispatch harness, not TensorRT."""

    def infer(self, inputs, *, synchronize=False):
        return {'actions': inputs['noise'].clone()}


class ApiTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / 'model.safetensors'
        self.inputs = self.root / 'inputs.safetensors'
        self.config = self.root / 'export_config.json'
        self.document = config_document()
        self.tensors = {}
        for name, spec in self.document['inputs']['tensors'].items():
            dtype = getattr(torch, spec['dtype'])
            self.tensors[name] = torch.ones(spec['shape'], dtype=dtype)
        self.tensors['img_masks'][1, 1] = False
        self.tensors['lang_masks'][0, -1] = False
        # A real small module state validates file dispatch only; this is not
        # a PI0.5 state or a substitute for future model compatibility tests.
        save_file(torch.nn.Linear(4, 4).state_dict(), self.checkpoint)
        self.document['checkpoint']['sha256'] = self.digest(self.checkpoint)
        self.save_inputs()
        self.arguments = []
        self.prepared = self.deployment()
        test = self

        class DispatchHarness:
            name = 'pi05-v1'

            def prepare(self, *, config, checkpoint, inputs, device):
                test.arguments.append((config, checkpoint, inputs, device))
                return test.prepared

        # Scoped registration only: the distributed package ships no model.
        previous = dict(registry._RECIPE_FACTORIES)
        self.addCleanup(self.restore_registry, previous)
        registry._RECIPE_FACTORIES.pop('pi05-v1', None)
        registry.register_recipe('pi05-v1')(DispatchHarness)

    @staticmethod
    def restore_registry(previous):
        registry._RECIPE_FACTORIES.clear()
        registry._RECIPE_FACTORIES.update(previous)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_config(self):
        self.config.write_text(json.dumps(self.document), encoding='utf-8')

    def save_inputs(self):
        save_file(self.tensors, self.inputs)
        self.document['inputs']['sha256'] = self.digest(self.inputs)
        self.save_config()

    def deployment(self):
        spec = TensorSpec('noise', 'float32', (2, 3, 4))
        output = TensorSpec('actions', 'float32', (2, 3, 4))
        module = torch.nn.Identity().eval()
        inputs = {'noise': self.tensors['noise']}
        graph = PreparedGraph(
            name='identity',
            module=module,
            inputs=inputs,
            input_specs=(spec, ),
            output_specs=(output, ),
        )
        return PreparedDeployment(
            recipe='pi05-v1',
            graphs={'identity': graph},
            policy_inputs=inputs,
            policy_input_specs=(spec, ),
            output_specs=(output, ),
            reference_runner=lambda:
            {'actions': module(inputs['noise'].clone())},
            executor_factory=lambda engines: IdentityExecutor(),
        )

    def prepare(self, **overrides):
        arguments = dict(
            recipe='pi05-v1',
            config=self.config,
            checkpoint=self.checkpoint,
            inputs=self.inputs,
            device='cpu',
        )
        arguments.update(overrides)
        return prepare_tensorrt(**arguments)

    def test_dispatches_validated_config_and_resolved_paths(self):
        result = self.prepare()
        self.assertIs(result, self.prepared)
        config, checkpoint, inputs, device = self.arguments[0]
        self.assertIsInstance(config, DeploymentConfig)
        self.assertEqual(config.model.vocab_size, 64)
        self.assertEqual(config.checkpoint_sha256, self.digest(checkpoint))
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(inputs, self.inputs.resolve())
        self.assertEqual(device, 'cpu')
        torch.testing.assert_close(result.run_reference()['actions'],
                                   self.tensors['noise'])

    def test_rejects_unknown_recipe_without_manufacturing_model(self):
        registry._RECIPE_FACTORIES.pop('pi05-v1')
        with self.assertRaisesRegex(RecipeNotFoundError, 'pi05-v1'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_config_parameter_is_required(self):
        with self.assertRaisesRegex(TypeError, 'config'):
            prepare_tensorrt(
                recipe='pi05-v1',
                checkpoint=self.checkpoint,
                inputs=self.inputs,
                device='cpu')

    def test_config_recipe_mismatch_prevents_dispatch(self):
        with self.assertRaises(ConfigError):
            self.prepare(recipe='different-v1')
        self.assertEqual(self.arguments, [])

    def test_rejects_invalid_device_before_reading_files(self):
        for device in ('', ' ', ' cuda:0', 'cuda:0 ', 0, None, 'cuda:-1',
                       'cuda:one', 'not-a-device'):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ContractError, 'device'):
                    self.prepare(device=device, config='missing.json')
        self.assertEqual(self.arguments, [])

    def test_rejects_missing_artifact(self):
        with self.assertRaisesRegex(ArtifactError, 'not found'):
            self.prepare(checkpoint=self.root / 'missing.safetensors')
        self.assertEqual(self.arguments, [])

    def test_rejects_pickle_extension(self):
        with self.assertRaisesRegex(ArtifactError, 'safetensors'):
            self.prepare(checkpoint=self.root / 'model.pt')
        self.assertEqual(self.arguments, [])

    def test_rejects_checkpoint_hash_mismatch(self):
        self.document['checkpoint']['sha256'] = '0' * 64
        self.save_config()
        with self.assertRaisesRegex(ArtifactError, 'SHA-256'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_fixture_hash_mismatch(self):
        self.document['inputs']['sha256'] = '0' * 64
        self.save_config()
        with self.assertRaisesRegex(ArtifactError, 'SHA-256'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_corrupt_checkpoint_with_matching_hash(self):
        self.checkpoint.write_bytes(b'not really a safetensors file')
        self.document['checkpoint']['sha256'] = self.digest(self.checkpoint)
        self.save_config()
        with self.assertRaisesRegex(ArtifactError, 'safetensors'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_missing_fixture_tensor(self):
        self.tensors.pop('states')
        self.save_inputs()
        with self.assertRaisesRegex(ArtifactError, 'states'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_unknown_fixture_tensor(self):
        self.tensors['action_masks'] = torch.ones((2, 3), dtype=torch.bool)
        self.save_inputs()
        with self.assertRaisesRegex(ArtifactError, 'action_masks'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_wrong_fixture_dtype(self):
        self.tensors['img_masks'] = self.tensors['img_masks'].float()
        self.save_inputs()
        with self.assertRaisesRegex(ArtifactError, 'dtype'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_wrong_fixture_shape(self):
        self.tensors['noise'] = torch.ones((2, 4, 4))
        self.save_inputs()
        with self.assertRaises(ArtifactError):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_nonfinite_fixture(self):
        self.tensors['noise'][0, 0, 0] = float('nan')
        self.save_inputs()
        with self.assertRaisesRegex(ArtifactError, 'finite'):
            self.prepare()
        self.assertEqual(self.arguments, [])

    def test_checks_all_tokens_even_masked_positions(self):
        for value in (-1, self.document['model']['backbone']['vocab_size']):
            with self.subTest(token=value):
                self.tensors['lang_tokens'][0, -1] = value
                self.save_inputs()
                with self.assertRaisesRegex(ArtifactError, 'lang_tokens'):
                    self.prepare()
        self.assertEqual(self.arguments, [])

    def test_rejects_wrong_return_type(self):
        self.prepared = None
        with self.assertRaisesRegex(ContractError, 'PreparedDeployment'):
            self.prepare()

    def test_rejects_wrong_deployment_recipe(self):
        self.prepared.recipe = 'other-v1'
        with self.assertRaisesRegex(ContractError, 'returned deployment'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
