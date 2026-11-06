# Copyright 2026 Limx Dynamics
"""Common preflight rules must match for dispatch and real reference loading."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch

import fluxvla_export
from fluxvla_export._validation import (
    validate_device_syntax, validate_token_vocabulary)
from fluxvla_export.errors import ArtifactError, ContractError


class RuntimeValidationTest(unittest.TestCase):
    def test_device_syntax_does_not_require_available_hardware(self):
        for device in ('cpu', 'cuda', 'cuda:0', 'cuda:17'):
            with self.subTest(device=device):
                self.assertIsNone(validate_device_syntax(device))

    def test_invalid_device_spellings_are_rejected(self):
        for device in ('', 'CPU', 'cpu:0', 'cuda:-1', 'cuda:01', 'cuda:+1',
                       'cuda:1.0', 'cuda:0\n', ' cuda', 'mps', None, 0, b'cpu'):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ContractError, 'device must be'):
                    validate_device_syntax(device)

    def test_device_object_does_not_bypass_string_contract(self):
        with self.assertRaises(ContractError):
            validate_device_syntax(torch.device('cpu'))

    def test_vocabulary_includes_both_valid_endpoints(self):
        tokens = torch.tensor([[0, 63], [17, 4]], dtype=torch.int64)
        self.assertIsNone(validate_token_vocabulary(
            tokens, vocab_size=64, label='fixture'))

    def test_negative_tokens_are_rejected(self):
        with self.assertRaisesRegex(ArtifactError, r'fixture: lang_tokens'):
            validate_token_vocabulary(
                torch.tensor([[0, -1]], dtype=torch.int64),
                vocab_size=64, label='fixture')

    def test_vocabulary_upper_bound_is_exclusive(self):
        # Every stored position is checked; a padding mask cannot exempt it.
        with self.assertRaisesRegex(ArtifactError, r'\[0, 64\)'):
            validate_token_vocabulary(
                torch.tensor([[0, 64]], dtype=torch.int64),
                vocab_size=64, label='fixture')

    def test_vocabulary_validation_does_not_modify_tensor(self):
        tokens = torch.tensor([[3, 1], [5, 2]], dtype=torch.int64)
        before = tokens.clone()
        validate_token_vocabulary(tokens, vocab_size=64, label='fixture')
        torch.testing.assert_close(tokens, before, rtol=0, atol=0)

    def test_import_is_stdlib_only(self):
        package_root = str(Path(fluxvla_export.__file__).resolve().parent.parent)
        code = (
            'import sys\n'
            f'sys.path.insert(0, {package_root!r})\n'
            'from fluxvla_export._validation import validate_device_syntax\n'
            "validate_device_syntax('cuda:0')\n"
            "assert not any(name in sys.modules for name in "
            "('torch', 'transformers', 'safetensors', 'fluxvla', 'mmengine'))\n"
        )
        result = subprocess.run(
            [sys.executable, '-I', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
