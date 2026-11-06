from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load as safe_load
from safetensors.torch import save_file

from fluxvla_export._safetensors import _read_verified, validate_safetensors
from fluxvla_export.checkpoint import load_checkpoint
from fluxvla_export.errors import ArtifactError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_file(header: object, payload: bytes = b'') -> bytes:
    if not isinstance(header, bytes):
        header = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header += b' ' * (-len(header) % 8)
    return len(header).to_bytes(8, 'little') + header + payload


class _UnsafePayload:

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return Path.touch, (self.marker, )


class SafetensorsPreflightTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / 'model.safetensors'

    def validate(self, raw):
        self.path.write_bytes(raw)
        return validate_safetensors(
            self.path, expected_sha256=_sha256(self.path),
            label='Test artifact')

    def test_public_import_does_not_load_optional_dependencies(self):
        code = '''
import sys
import fluxvla_export
import fluxvla_export.checkpoint
import fluxvla_export.fixture
import fluxvla_export._safetensors
assert 'torch' not in sys.modules
assert 'safetensors' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_valid_format_with_scalar_and_empty_tensor(self):
        raw = _raw_file({
            'empty': {'dtype': 'F32', 'shape': [0, 3], 'data_offsets': [0, 0]},
            'scalar': {'dtype': 'I64', 'shape': [], 'data_offsets': [0, 8]},
        }, b'\x00' * 8)
        self.assertEqual(self.validate(raw), self.path)

    def test_preflight_returns_canonical_relative_and_symlink_paths(self):
        self.path.write_bytes(_raw_file({}))
        link = self.path.with_name('link.safetensors')
        link.symlink_to(self.path.name)
        for value in (os.path.relpath(self.path), link):
            with self.subTest(value=value):
                verified = validate_safetensors(
                    value, expected_sha256=_sha256(self.path),
                    label='Checkpoint')
                self.assertEqual(verified, self.path)
                self.assertTrue(verified.is_absolute())

    def test_preflight_expands_home_in_a_real_subprocess(self):
        self.path.write_bytes(_raw_file({}))
        code = '''
import sys
from pathlib import Path
from fluxvla_export._safetensors import validate_safetensors
actual = validate_safetensors('~/model.safetensors',
    expected_sha256=sys.argv[1], label='Checkpoint')
assert actual == Path(sys.argv[2]), actual
'''
        subprocess.run(
            [sys.executable, '-c', code, _sha256(self.path), str(self.path)],
            env={**os.environ, 'HOME': str(self.path.parent)}, check=True)

    def test_invalid_path_and_symlink_loop_are_artifact_errors(self):
        link = self.path.with_name('loop.safetensors')
        link.symlink_to(link.name)
        for value in ('bad\x00.safetensors', link):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ArtifactError, 'resolve artifact path'):
                    validate_safetensors(
                        value, expected_sha256='0' * 64, label='Checkpoint')

    def test_missing_file_and_wrong_extension(self):
        with self.assertRaisesRegex(ArtifactError, 'not found'):
            validate_safetensors(
                self.path, expected_sha256='0' * 64, label='Checkpoint')
        path = self.path.with_suffix('.pt')
        path.write_bytes(b'not a safe tensor')
        with self.assertRaisesRegex(ArtifactError, '.safetensors format'):
            validate_safetensors(
                path, expected_sha256='0' * 64, label='Checkpoint')

    def test_directory_is_not_a_regular_artifact(self):
        self.path.mkdir()
        with self.assertRaisesRegex(ArtifactError, 'cannot read|regular file'):
            validate_safetensors(
                self.path, expected_sha256='0' * 64, label='Checkpoint')

    def test_fifo_is_rejected_without_waiting_for_a_writer(self):
        os.mkfifo(self.path)
        code = '''
import sys
from fluxvla_export._safetensors import validate_safetensors
from fluxvla_export.checkpoint import load_checkpoint
from fluxvla_export.errors import ArtifactError
for validate in (
    lambda: validate_safetensors(sys.argv[1], expected_sha256='0' * 64,
                                label='Checkpoint'),
    lambda: load_checkpoint(sys.argv[1], expected_sha256='0' * 64,
                            expected_state={}),
):
    try:
        validate()
    except ArtifactError as error:
        assert 'regular file' in str(error), error
    else:
        raise AssertionError('FIFO must not be accepted as an artifact')
'''
        subprocess.run(
            [sys.executable, '-c', code, str(self.path)],
            check=True, timeout=10)

    def test_expected_digest_must_be_sha256_hex(self):
        self.path.write_bytes(_raw_file({}))
        for digest in ('', 'a' * 63, 'g' * 64, '0' * 65, None, 123):
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(ArtifactError, '64 hex digits'):
                    validate_safetensors(
                        self.path, expected_sha256=digest, label='Checkpoint')

    def test_hash_binds_exact_header_padding_and_payload_bytes(self):
        original = _raw_file({
            'x': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]},
        }, b'\x01')
        self.path.write_bytes(original)
        expected = _sha256(self.path)
        self.assertEqual(validate_safetensors(
            self.path, expected_sha256=expected.upper(), label='Checkpoint'),
            self.path)
        for replacement in (
                original[:-1] + b'\x02',
                _raw_file({
                    'x': {'dtype': 'I8', 'shape': [1], 'data_offsets': [0, 1]},
                }, b'\x01'),
                (original[:8] + original[8:-1].replace(b' ', b'\t')
                 + original[-1:]),
        ):
            with self.subTest(replacement=replacement):
                self.path.write_bytes(replacement)
                with self.assertRaisesRegex(ArtifactError, 'SHA-256 mismatch'):
                    validate_safetensors(
                        self.path, expected_sha256=expected,
                        label='Checkpoint')

    def test_rejects_non_safe_truncated_or_invalid_length(self):
        for raw in (b'', b'1234567', b'not a safe tensor', b'\x00' * 8,
                    (100_000_001).to_bytes(8, 'little'),
                    (16).to_bytes(8, 'little') + b'{}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ArtifactError):
                    self.validate(raw)

    def test_rejects_invalid_header_json_and_duplicate_keys(self):
        for header in (
                b'[]', b' {}', b'{', b'{"x":NaN}', b'{"x":Infinity}',
                b'{"x":-Infinity}', b'{"\xff":{}}',
                b'{"__metadata__":{},"__metadata__":{}}',
                b'{"x":{"dtype":"F32","dtype":"I32",'
                b'"shape":[1],"data_offsets":[0,4]}}',
        ):
            with self.subTest(header=header):
                with self.assertRaises(ArtifactError):
                    self.validate(_raw_file(header, b'\x00' * 4))

    def test_preflight_rejects_json_that_the_safe_decoder_rejects(self):
        from safetensors import SafetensorError, deserialize

        for header in (
            b'{"x":{"dtype":"U8","shape":[-0],"data_offsets":[0,0]}}',
            b'{"__metadata__":{"x":"\\ud800"}}',
            b'{"\\ud800":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}',
            b'{"x":{"dtype":"U8","shape":[0,18446744073709551616],'
            b'"data_offsets":[0,0]}}',
        ):
            with self.subTest(header=header):
                raw = _raw_file(header)
                with self.assertRaises(SafetensorError):
                    deserialize(raw)
                with self.assertRaises(ArtifactError):
                    self.validate(raw)

    def test_rejects_invalid_metadata_fields(self):
        for metadata in ([], 'trusted', {'sha256': 123}, {'shape': [1]}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ArtifactError, 'metadata'):
                    self.validate(_raw_file({'__metadata__': metadata}))

    def test_rejects_invalid_tensor_fields_shapes_dtypes_and_offsets(self):
        normal = {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}
        entries = [
            None,
            {'dtype': 'F32', 'shape': [1]},
            dict(normal, unexpected='ignored'),
            dict(normal, dtype='float32'),
            dict(normal, dtype=['F32']),
            dict(normal, shape=[True]),
            dict(normal, shape=[-1]),
            dict(normal, shape=[1.0]),
            dict(normal, shape='1'),
            dict(normal, data_offsets=[False, 4]),
            dict(normal, data_offsets=[0.0, 4]),
            dict(normal, data_offsets=[0, -4]),
            dict(normal, data_offsets=[4, 0]),
            dict(normal, data_offsets=[0, 8]),
            dict(normal, data_offsets=[0, 3]),
            dict(normal, data_offsets=[0, 4, 4]),
            dict(normal, shape=[2]),
            dict(normal, shape=[0, 2**64]),
            dict(normal, shape=[2**32, 2**32, 0]),
        ]
        for entry in entries:
            with self.subTest(entry=entry):
                with self.assertRaises(ArtifactError):
                    self.validate(_raw_file({'x': entry}, b'\x00' * 4))

    def test_rejects_gaps_overlap_and_unindexed_trailing_bytes(self):
        for contents, data_size in (
            ({'x': {'dtype': 'U8', 'shape': [1], 'data_offsets': [1, 2]}}, 2),
            ({'x': {'dtype': 'U8', 'shape': [2], 'data_offsets': [0, 2]},
              'y': {'dtype': 'U8', 'shape': [1], 'data_offsets': [1, 2]}}, 2),
            ({'x': {'dtype': 'U8', 'shape': [1], 'data_offsets': [0, 1]}}, 2),
            ({}, 1),
        ):
            with self.subTest(contents=contents):
                with self.assertRaises(ArtifactError):
                    self.validate(_raw_file(contents, b'\x00' * data_size))

    def test_pickle_renamed_safetensors_is_never_executed(self):
        marker = self.path.parent / 'unpickle-executed'
        with self.assertRaises(ArtifactError):
            self.validate(pickle.dumps(_UnsafePayload(marker)))
        self.assertFalse(marker.exists())


class CheckpointTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / 'model.safetensors'

    def save(self, tensors, *, metadata=None):
        save_file(tensors, self.path, metadata=metadata)
        return _sha256(self.path)

    def load(self, expected):
        return load_checkpoint(
            self.path, expected_sha256=_sha256(self.path),
            expected_state=expected)

    def test_real_complete_native_state_including_scalar_and_empty(self):
        module = torch.nn.BatchNorm1d(3)
        module.register_buffer('empty', torch.empty((0, 3)))
        expected = module.state_dict()
        original = {name: value.clone() for name, value in expected.items()}
        stored = {name: value.clone() + 1 for name, value in expected.items()}
        digest = self.save(stored)
        loaded = load_checkpoint(
            self.path, expected_sha256=digest, expected_state=expected)
        self.assertEqual(tuple(loaded), tuple(expected))
        self.assertEqual(loaded['num_batches_tracked'].shape, ())
        self.assertEqual(loaded['empty'].shape, (0, 3))
        for name in stored:
            self.assertEqual(loaded[name].device.type, 'cpu')
            torch.testing.assert_close(loaded[name], stored[name])
            torch.testing.assert_close(expected[name], original[name])
        loaded['weight'].fill_(9)
        torch.testing.assert_close(expected['weight'], original['weight'])

    def test_preserves_bool_int64_fp32_and_bfloat16_exactly(self):
        stored = {
            'bool': torch.tensor([True, False]),
            'int64': torch.tensor([-(2**63), 2**63 - 1], dtype=torch.int64),
            'fp32': torch.tensor([0.0, -2.5], dtype=torch.float32),
            'bf16': torch.tensor([1.0, 2.5], dtype=torch.bfloat16),
        }
        self.save(stored)
        loaded = self.load({name: torch.empty_like(value)
                            for name, value in stored.items()})
        for name, value in loaded.items():
            torch.testing.assert_close(value, stored[name], rtol=0, atol=0)
            self.assertEqual(value.dtype, stored[name].dtype)

    def test_empty_state(self):
        self.save({})
        self.assertEqual(self.load({}), {})

    def test_loader_ignores_callers_default_device(self):
        expected = {'x': torch.zeros(2), 'empty': torch.empty(0)}
        self.save(expected)
        with torch.device('meta'):
            loaded = self.load(expected)
        for value in loaded.values():
            self.assertEqual(value.device.type, 'cpu')

    def test_missing_unexpected_and_prefixed_keys_are_not_remapped(self):
        expected = {'weight': torch.zeros(2), 'counter': torch.tensor(0)}
        for stored in (
            {'weight': torch.zeros(2)},
            dict(expected, extra=torch.zeros(1)),
            {'module.' + name: value for name, value in expected.items()},
        ):
            with self.subTest(keys=tuple(stored)):
                self.save(stored)
                with self.assertRaisesRegex(
                        ArtifactError, 'tensor keys.*missing=.*unexpected='):
                    self.load(expected)

    def test_rejects_wrong_shape_without_mutating_expected_state(self):
        expected = {'weight': torch.ones(2, 3)}
        self.save({'weight': torch.ones(3, 2)})
        with self.assertRaisesRegex(ArtifactError, "'weight' shape"):
            self.load(expected)
        torch.testing.assert_close(expected['weight'], torch.ones(2, 3))

    def test_rejects_dtype_casting_even_when_shapes_match(self):
        for stored_dtype, expected_dtype in (
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.int64, torch.float32),
            (torch.bool, torch.int64),
        ):
            with self.subTest(stored=stored_dtype, expected=expected_dtype):
                self.save({'x': torch.ones(2, dtype=stored_dtype)})
                with self.assertRaisesRegex(ArtifactError, "'x' dtype"):
                    self.load({'x': torch.zeros(2, dtype=expected_dtype)})

    def test_rejects_nan_and_both_infinities_in_fp32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            for number in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(dtype=dtype, number=number):
                    self.save({'x': torch.tensor([number], dtype=dtype)})
                    with self.assertRaisesRegex(
                            ArtifactError, "'x'.*non-finite"):
                        self.load({'x': torch.zeros(1, dtype=dtype)})

    def test_rejects_invalid_expected_state(self):
        self.save({'x': torch.zeros(1)})
        for expected in ([], {'x': [0]}, {0: torch.zeros(1)}):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ArtifactError, 'expected_state'):
                    self.load(expected)

    def test_metadata_cannot_override_external_contract_or_hash(self):
        self.save({'x': torch.ones(2, dtype=torch.bfloat16)}, metadata={
            'sha256': '0' * 64,
            'dtype': 'float32',
            'shape': '[2]',
            'keys': '["x"]',
        })
        with self.assertRaisesRegex(ArtifactError, 'SHA-256 mismatch'):
            load_checkpoint(self.path, expected_sha256='0' * 64,
                            expected_state={'x': torch.zeros(2)})
        with self.assertRaisesRegex(ArtifactError, 'dtype'):
            self.load({'x': torch.zeros(2)})
        self.load({'x': torch.zeros(2, dtype=torch.bfloat16)})

    def test_loaded_bytes_remain_bound_when_the_path_is_replaced(self):
        self.save({'x': torch.tensor([1.0])})
        original_digest = _sha256(self.path)
        _, snapshot = _read_verified(
            self.path, expected_sha256=original_digest,
            label='Checkpoint', retain_bytes=True)
        replacement = self.path.with_name('replacement.safetensors')
        save_file({'x': torch.tensor([2.0])}, replacement)
        replacement.replace(self.path)
        torch.testing.assert_close(
            safe_load(snapshot)['x'], torch.tensor([1.0]))
        with self.assertRaisesRegex(ArtifactError, 'SHA-256 mismatch'):
            load_checkpoint(self.path, expected_sha256=original_digest,
                            expected_state={'x': torch.zeros(1)})

    def test_torch_serialization_never_falls_back_to_unpickling(self):
        marker = self.path.parent / 'unpickle-executed'
        torch.save(_UnsafePayload(marker), self.path)
        with self.assertRaises(ArtifactError):
            self.load({})
        self.assertFalse(marker.exists())

    def test_preflight_streams_multiple_chunks(self):
        self.save({'x': torch.arange(300_000, dtype=torch.float32)})
        self.assertEqual(validate_safetensors(
            self.path, expected_sha256=_sha256(self.path), label='Checkpoint'),
            self.path)


if __name__ == '__main__':
    unittest.main()
