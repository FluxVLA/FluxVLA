from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from fluxvla_export.contracts import TensorSpec
from fluxvla_export.errors import ArtifactError
from fluxvla_export.fixture import load_fixture


class FixtureTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / 'inputs.safetensors'

    def save(self, tensors, *, metadata=None):
        save_file(tensors, self.path, metadata=metadata)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def load(self, specs):
        return load_fixture(
            self.path,
            expected_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            specs=specs,
        )

    def test_real_policy_fixture_uses_spec_order_and_exact_tensor_types(self):
        stored = {
            'noise': torch.randn(2, 3, 4),
            'lang_tokens': torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
            'states': torch.randn(2, 4),
            'lang_masks': torch.ones(2, 4, dtype=torch.bool),
            'img_masks': torch.tensor([[True, False], [False, True]]),
            'images': torch.randn(2, 6, 8, 8),
        }
        specs = (
            TensorSpec('images', 'float32', (2, 6, 8, 8)),
            TensorSpec('img_masks', 'bool', (2, 2)),
            TensorSpec('lang_tokens', 'int64', (2, 4)),
            TensorSpec('lang_masks', 'bool', (2, 4)),
            TensorSpec('states', 'float32', (2, 4)),
            TensorSpec('noise', 'float32', (2, 3, 4)),
        )
        digest = self.save(stored)
        loaded = load_fixture(self.path, expected_sha256=digest, specs=specs)
        self.assertEqual(tuple(loaded), tuple(spec.name for spec in specs))
        for name, value in loaded.items():
            self.assertEqual(value.device.type, 'cpu')
            torch.testing.assert_close(value, stored[name], rtol=0, atol=0)

    def test_real_bfloat16_is_preserved(self):
        stored = {'images': torch.tensor([[1.0, -2.5]], dtype=torch.bfloat16)}
        self.save(stored)
        loaded = self.load((TensorSpec('images', 'bfloat16', (1, 2)), ))
        self.assertEqual(loaded['images'].dtype, torch.bfloat16)
        torch.testing.assert_close(loaded['images'], stored['images'])

    def test_empty_fixture_with_empty_specs(self):
        self.save({})
        self.assertEqual(self.load(()), {})

    def test_missing_and_extra_names_rejected(self):
        specs = (TensorSpec('images', 'float32', (1, 3)), )
        for stored in ({}, {'image': torch.zeros(1, 3)},
                       {'images': torch.zeros(1, 3), 'noise': torch.zeros(1)}):
            with self.subTest(names=tuple(stored)):
                self.save(stored)
                with self.assertRaisesRegex(
                        ArtifactError, 'tensor keys.*missing=.*unexpected='):
                    self.load(specs)

    def test_spec_names_must_be_unique(self):
        self.save({'x': torch.zeros(1)})
        spec = TensorSpec('x', 'float32', (1, ))
        with self.assertRaisesRegex(ArtifactError, 'names must be unique'):
            self.load((spec, spec))

    def test_spec_values_must_be_tensor_specs(self):
        self.save({'x': torch.zeros(1)})
        with self.assertRaisesRegex(ArtifactError, 'TensorSpec'):
            self.load(({'name': 'x', 'shape': (1, ), 'dtype': 'float32'}, ))

    def test_exact_shapes_and_rank_are_required(self):
        for shape in ((2, 3), (1, 2), (3, ), (1, 3, 1)):
            with self.subTest(shape=shape):
                self.save({'images': torch.ones(shape)})
                with self.assertRaisesRegex(ArtifactError, 'shape|dimension'):
                    self.load((TensorSpec('images', 'float32', (1, 3)), ))

    def test_exact_dtype_without_bool_int_or_float_casting(self):
        for stored_dtype, spec_dtype in (
            (torch.int32, 'int64'),
            (torch.int64, 'bool'),
            (torch.bool, 'int64'),
            (torch.bfloat16, 'float32'),
            (torch.float32, 'bfloat16'),
        ):
            with self.subTest(stored=stored_dtype, expected=spec_dtype):
                self.save({'x': torch.ones(2, dtype=stored_dtype)})
                with self.assertRaisesRegex(ArtifactError, 'dtype'):
                    self.load((TensorSpec('x', spec_dtype, (2, )), ))

    def test_nonfinite_fp32_and_bfloat16_rejected(self):
        for dtype, spec_dtype in ((torch.float32, 'float32'),
                                 (torch.bfloat16, 'bfloat16')):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(dtype=dtype, value=value):
                    self.save({'x': torch.tensor([value], dtype=dtype)})
                    with self.assertRaisesRegex(
                            ArtifactError, "'x'.*non-finite"):
                        self.load((TensorSpec('x', spec_dtype, (1, )), ))

    def test_dynamic_profiles_accept_boundaries_and_intermediate_shapes(self):
        spec = TensorSpec('tokens', 'int64', (1, -1),
                          min_shape=(1, 2), opt_shape=(1, 4), max_shape=(1, 8))
        for length in (2, 3, 4, 8):
            with self.subTest(length=length):
                self.save({'tokens': torch.arange(length).reshape(1, length)})
                self.assertEqual(
                    self.load((spec, ))['tokens'].shape, (1, length))

    def test_dynamic_profiles_reject_outside_bounds(self):
        spec = TensorSpec('tokens', 'int64', (1, -1),
                          min_shape=(1, 2), opt_shape=(1, 4), max_shape=(1, 8))
        for length in (1, 9):
            with self.subTest(length=length):
                self.save({
                    'tokens': torch.zeros(1, length, dtype=torch.int64)})
                with self.assertRaisesRegex(
                        ArtifactError, r'outside \[2, 8\]'):
                    self.load((spec, ))

    def test_zero_sized_and_scalar_inputs_cannot_bypass_graph_contract(self):
        for value in (torch.empty(0), torch.tensor(1.0)):
            with self.subTest(shape=value.shape):
                self.save({'x': value})
                with self.assertRaises(ArtifactError):
                    self.load((TensorSpec('x', 'float32', (1, )), ))

    def test_metadata_cannot_relax_fixture_contract(self):
        self.save({'wrong_name': torch.ones(1)}, metadata={
            'input_names': '["images"]',
            'images.dtype': 'float32',
            'images.shape': '[1]',
        })
        with self.assertRaisesRegex(ArtifactError, 'tensor keys'):
            self.load((TensorSpec('images', 'float32', (1, )), ))

    def test_hash_mismatch_rejects_changed_payload(self):
        original_hash = self.save({'x': torch.tensor([1.0])})
        self.save({'x': torch.tensor([2.0])})
        with self.assertRaisesRegex(ArtifactError, 'SHA-256 mismatch'):
            load_fixture(self.path, expected_sha256=original_hash,
                         specs=(TensorSpec('x', 'float32', (1, )), ))

    def test_renamed_torch_pickle_is_not_a_fixture(self):
        torch.save({'images': torch.zeros(1, 3)}, self.path)
        with self.assertRaises(ArtifactError):
            self.load((TensorSpec('images', 'float32', (1, 3)), ))


if __name__ == '__main__':
    unittest.main()
