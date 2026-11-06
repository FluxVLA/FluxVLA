"""Explicit storage precision using real safetensors and native Torch loads."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from _pi05_test_utils import Artifacts
from fluxvla_export.checkpoint import load_checkpoint
from fluxvla_export.config import (
    config_from_document, config_to_document, load_config, validate_config)
from fluxvla_export.errors import ArtifactError, ConfigError
from fluxvla_export.recipes.pi05.artifacts import produce_pi05_artifacts
from fluxvla_export.recipes.pi05.reference import prepare_reference
from test_config import config_document


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bf16_state(state):
    return {name: value.detach().bfloat16() if value.is_floating_point()
            else value.detach().clone() for name, value in state.items()}


def _canonical(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':')).encode()


class StorageConfigTests(unittest.TestCase):

    def test_default_and_explicit_float32_preserve_legacy_canonical_hash(self):
        document = config_document()
        original = _canonical(document)
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                if explicit:
                    document['checkpoint']['storage_dtype'] = 'float32'
                config = config_from_document(document, recipe='pi05-v1')
                self.assertEqual(config.checkpoint_storage_dtype, 'float32')
                actual = _canonical(config_to_document(config))
                self.assertEqual(actual, original)
                self.assertEqual(hashlib.sha256(actual).digest(),
                                 hashlib.sha256(original).digest())

    def test_bfloat16_roundtrip_changes_provenance_not_checkpoint_digest(self):
        document = config_document()
        original = config_from_document(document, recipe='pi05-v1')
        document['checkpoint']['storage_dtype'] = 'bfloat16'
        config = config_from_document(document, recipe='pi05-v1')
        self.assertEqual(config.checkpoint_storage_dtype, 'bfloat16')
        self.assertEqual(config.precision.parameter_dtype, 'float32')
        self.assertEqual(config.checkpoint_sha256, original.checkpoint_sha256)
        self.assertEqual(config_to_document(config), document)
        self.assertEqual(validate_config(config), config)
        self.assertNotEqual(_canonical(config_to_document(original)),
                            _canonical(config_to_document(config)))

    def test_invalid_storage_declarations_and_replaced_configs_fail(self):
        original = config_from_document(config_document(), recipe='pi05-v1')
        for invalid in (None, True, 16, 'float16', 'BF16', [], {}, 'mixed'):
            with self.subTest(storage_dtype=invalid):
                document = config_document()
                document['checkpoint']['storage_dtype'] = invalid
                with self.assertRaisesRegex(ConfigError, 'storage_dtype'):
                    config_from_document(document, recipe='pi05-v1')
                with self.assertRaisesRegex(ConfigError, 'storage_dtype'):
                    validate_config(replace(
                        original, checkpoint_storage_dtype=invalid))


class StorageLoaderTests(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'weights.safetensors'

    def load(self, expected, storage_dtype='bfloat16', digest=None):
        return load_checkpoint(
            self.path, expected_sha256=digest or _digest(self.path),
            expected_state=expected, storage_dtype=storage_dtype)

    def test_widening_matches_native_load_preserving_source_and_buffers(self):
        model = torch.nn.BatchNorm1d(3)
        model.register_buffer('mask', torch.tensor([True, False]))
        model.register_buffer('empty', torch.empty(0, 3))
        model.register_buffer('scalar', torch.tensor(1.003))
        expected = model.state_dict()
        original = {name: value.clone() for name, value in expected.items()}
        stored = _bf16_state(expected)
        save_file(stored, self.path)
        before = self.path.read_bytes()
        stamp = self.path.stat().st_mtime_ns
        with torch.device('meta'):
            loaded = self.load(expected)
        for name, value in loaded.items():
            self.assertEqual(value.device.type, 'cpu')
            self.assertEqual(value.dtype, expected[name].dtype)
            self.assertTrue(torch.equal(expected[name], original[name]))
        # Original Torch copy semantics are the independent widening reference.
        model.load_state_dict(load_file(self.path), strict=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
        self.assertEqual(loaded['mask'].dtype, torch.bool)
        self.assertEqual(loaded['num_batches_tracked'].dtype, torch.int64)
        self.assertEqual(loaded['scalar'].shape, ())
        self.assertEqual(loaded['empty'].shape, (0, 3))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.stat().st_mtime_ns, stamp)

    def test_absent_or_misdeclared_storage_never_enables_widening(self):
        save_file({'x': torch.ones(2, dtype=torch.bfloat16)}, self.path)
        for declaration in (None, 'float32'):
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(ArtifactError, "'x' dtype"):
                    self.load({'x': torch.zeros(2)}, declaration)
        for invalid in ('float16', 'BF16', True, 16, {}, []):
            with self.subTest(declaration=invalid):
                with self.assertRaisesRegex(ArtifactError, 'storage_dtype'):
                    self.load({'x': torch.zeros(2)}, invalid)

    def test_rejects_mixed_float_storage_and_non_float_buffer_casting(self):
        expected = {'x': torch.zeros(2), 'y': torch.zeros(2),
                    'count': torch.tensor(1), 'mask': torch.tensor(True)}
        original = _bf16_state(expected)
        for key, dtype in (('x', torch.float32), ('y', torch.float16),
                           ('count', torch.int32), ('mask', torch.int64)):
            with self.subTest(key=key, dtype=dtype):
                stored = dict(original)
                stored[key] = stored[key].to(dtype)
                save_file(stored, self.path)
                with self.assertRaisesRegex(ArtifactError, f"'{key}' dtype"):
                    self.load(expected)
        save_file(original, self.path)
        for dtype in (torch.bfloat16, torch.float16, torch.float64):
            with self.subTest(target=dtype):
                target = dict(expected, x=expected['x'].to(dtype))
                with self.assertRaisesRegex(ArtifactError, 'target dtype'):
                    self.load(target)

    def test_complete_artifact_checks_precede_widening(self):
        expected = {'x': torch.zeros(2)}
        cases = (
            ({}, 'tensor keys'),
            ({'x': torch.ones(3, dtype=torch.bfloat16)}, 'shape'),
            ({'x': torch.ones(2, dtype=torch.bfloat16),
              'extra': torch.ones(1)}, 'tensor keys'),
        )
        for stored, error in cases:
            save_file(stored, self.path)
            with self.assertRaisesRegex(ArtifactError, error):
                self.load(expected)
        for number in (float('nan'), float('inf'), -float('inf')):
            save_file({'x': torch.full((2,), number, dtype=torch.bfloat16)},
                      self.path)
            with self.assertRaisesRegex(ArtifactError, 'non-finite'):
                self.load(expected)
        save_file({'x': torch.ones(2, dtype=torch.bfloat16)}, self.path)
        with self.assertRaisesRegex(ArtifactError, 'SHA-256 mismatch'):
            self.load(expected, digest='0' * 64)
        self.assertTrue(torch.equal(expected['x'], torch.zeros(2)))


class StorageProducerReferenceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.assets = Artifacts(self.root)
        self.output = self.root / 'produced'

    def produce(self, **options):
        return produce_pi05_artifacts(
            model=self.assets.model, inputs=self.assets.tensors,
            checkpoint=self.assets.checkpoint_path,
            preprocessing=self.assets.config.preprocessing,
            output_dir=self.output, **options)

    def test_native_bfloat16_load_producer_reference_and_hash_agree(self):
        stored = _bf16_state(self.assets.model.state_dict())
        self.assets.save_checkpoint(stored)
        before = self.assets.checkpoint_path.read_bytes()
        # Do not round model parameters in place: load the actual stored file
        # using the original framework's strict FP32 copy semantics.
        self.assets.model.load_state_dict(
            load_file(self.assets.checkpoint_path), strict=True)
        state_before = {
            name: value.clone()
            for name, value in self.assets.model.state_dict().items()
        }
        with torch.no_grad():
            baseline = self.assets.model(**self.assets.tensors)
        result = self.produce(checkpoint_storage_dtype='bfloat16')
        config = load_config(result.config, recipe='pi05-v1')
        self.assertEqual(config.checkpoint_storage_dtype, 'bfloat16')
        self.assertEqual(config.checkpoint_sha256,
                         hashlib.sha256(before).hexdigest())
        self.assertEqual(result.checkpoint, self.assets.checkpoint_path)
        reference = prepare_reference(
            config=result.config, checkpoint=result.checkpoint,
            inputs=result.inputs, device='cpu')
        torch.testing.assert_close(reference.run_reference()['actions'],
                                   baseline, rtol=0, atol=0)
        for name, value in reference.model.state_dict().items():
            self.assertEqual(value.dtype, state_before[name].dtype)
            torch.testing.assert_close(
                value, state_before[name], rtol=0, atol=0)
            self.assertTrue(torch.equal(
                self.assets.model.state_dict()[name], state_before[name]))
        self.assertEqual(self.assets.checkpoint_path.read_bytes(), before)

    def test_unrounded_live_float32_is_not_the_checkpoint_reference(self):
        with torch.no_grad():
            self.assets.model.action_out_proj.projector.bias.fill_(1.003)
            unrounded_actions = self.assets.model(**self.assets.tensors)
        live = self.assets.model.state_dict()
        stored = _bf16_state(live)
        key = 'action_out_proj.projector.bias'
        self.assertFalse(torch.equal(live[key], stored[key].float()))
        self.assertTrue(torch.equal(live[key].bfloat16(), stored[key]))
        self.assets.save_checkpoint(stored)
        before = self.assets.checkpoint_path.read_bytes()
        result = self.produce(checkpoint_storage_dtype='bfloat16')
        self.assertTrue(torch.equal(
            live[key], torch.full_like(live[key], 1.003)))
        reference = prepare_reference(
            config=result.config, checkpoint=result.checkpoint,
            inputs=result.inputs, device='cpu')
        self.assertFalse(torch.equal(reference.model.state_dict()[key],
                                     live[key]))
        self.assets.model.load_state_dict(load_file(result.checkpoint),
                                          strict=True)
        with torch.no_grad():
            baseline = self.assets.model(**self.assets.tensors)
        self.assertFalse(torch.equal(baseline, unrounded_actions))
        torch.testing.assert_close(reference.run_reference()['actions'],
                                   baseline, rtol=0, atol=0)
        self.assertEqual(self.assets.checkpoint_path.read_bytes(), before)

    def test_both_storage_types_accept_authoritative_ema_values(self):
        for storage_dtype in ('float32', 'bfloat16'):
            with self.subTest(storage_dtype=storage_dtype):
                self.output = self.root / storage_dtype
                weights = {name: value.detach().clone()
                           for name, value in self.assets.model.state_dict()
                           .items()}
                key = 'action_out_proj.projector.bias'
                weights[key].fill_(2.5)
                if storage_dtype == 'bfloat16':
                    weights = _bf16_state(weights)
                self.assets.save_checkpoint(weights)
                live = self.assets.model.state_dict()[key].clone()
                result = self.produce(checkpoint_storage_dtype=storage_dtype)
                reference = prepare_reference(
                    config=result.config, checkpoint=result.checkpoint,
                    inputs=result.inputs, device='cpu')
                self.assertTrue(torch.equal(reference.model.state_dict()[key],
                                            weights[key].float()))
                self.assertTrue(torch.equal(
                    self.assets.model.state_dict()[key], live))
                self.assertFalse(torch.equal(live, weights[key].float()))

    def test_bfloat16_producer_and_reference_require_declaration(self):
        self.assets.save_checkpoint(
            _bf16_state(self.assets.model.state_dict()))
        with self.assertRaisesRegex(ArtifactError, 'dtype'):
            self.produce()
        with self.assertRaisesRegex(ArtifactError, 'dtype'):
            self.assets.prepare()
        self.assertFalse(self.output.exists())

    def test_bfloat16_storage_and_compute_keep_separate_bound_contracts(self):
        from fluxvla_export import prepare_tensorrt

        directory = self.root / 'autocast'
        directory.mkdir()
        document = config_document()
        document['precision']['autocast_dtype'] = 'bfloat16'
        self.assets = Artifacts(directory, document)
        self.assets.save_checkpoint(
            _bf16_state(self.assets.model.state_dict()))
        result = self.produce(checkpoint_storage_dtype='bfloat16')
        prepared = prepare_tensorrt(
            recipe='pi05-v1', config=result.config,
            checkpoint=result.checkpoint, inputs=result.inputs, device='cpu')
        config = load_config(result.config, recipe='pi05-v1')
        self.assertEqual(prepared.metadata['source_config_sha256'],
                         hashlib.sha256(_canonical(
                             config_to_document(config))).hexdigest())
        self.assertEqual(prepared.metadata['checkpoint_sha256'],
                         _digest(result.checkpoint))
        self.assets.model.load_state_dict(load_file(result.checkpoint),
                                          strict=True)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            expected = self.assets.model(**self.assets.tensors)
        torch.testing.assert_close(prepared.run_reference()['actions'],
                                   expected, rtol=0, atol=0)
        self.assertEqual(config.precision.parameter_dtype, 'float32')
        self.assertEqual(config.precision.autocast_dtype, 'bfloat16')
        self.assertEqual(config.checkpoint_storage_dtype, 'bfloat16')

    def test_bfloat16_declaration_rejects_float32_file_before_output(self):
        with self.assertRaisesRegex(ArtifactError, 'dtype'):
            self.produce(checkpoint_storage_dtype='bfloat16')
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
