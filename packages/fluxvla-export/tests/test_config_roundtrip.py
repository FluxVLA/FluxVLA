"""Strict in-memory schema handoff uses the same validation as JSON files."""

import copy
import json
import unittest
from dataclasses import replace

from fluxvla_export.config import (config_from_document, config_to_document,
                                   validate_config)
from fluxvla_export.contracts import TensorSpec
from fluxvla_export.errors import ConfigError
from test_config import config_document


class ConfigRoundtripTest(unittest.TestCase):

    def setUp(self):
        self.document = config_document()
        self.config = config_from_document(self.document, recipe='pi05-v1')

    def test_roundtrip_preserves_every_schema_field(self):
        document = config_to_document(self.config)
        self.assertEqual(document, self.document)
        restored = config_from_document(
            json.loads(json.dumps(document)), recipe='pi05-v1')
        self.assertEqual(restored, self.config)
        self.assertEqual(validate_config(self.config), self.config)

    def test_document_validation_does_not_mutate_or_alias(self):
        before = copy.deepcopy(self.document)
        self.assertEqual(self.document, before)
        output = config_to_document(self.config)
        output['precision']['backbone_fp32_layers'].append(0)
        output['model']['backbone']['hidden_size'] = 999
        self.assertEqual(self.document, before)
        self.assertEqual(config_to_document(self.config), before)

    def test_replaced_invalid_dataclass_groups_are_rejected(self):
        cases = (
            replace(self.config, checkpoint_sha256='invalid'),
            replace(self.config, schema_version=True),
            replace(
                self.config,
                inference=replace(self.config.inference, num_steps=0)),
            replace(
                self.config,
                precision=replace(
                    self.config.precision, parameter_dtype='float16')),
            replace(self.config, model=None),
        )
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ConfigError):
                    validate_config(config)
                with self.assertRaises(ConfigError):
                    config_to_document(config)

    def test_specs_cannot_be_silently_reordered_or_overwritten(self):
        specs = self.config.input_specs
        for invalid in (specs[::-1], specs + specs[:1], specs[:-1],
                        list(specs)):
            with self.subTest(specs=invalid):
                with self.assertRaisesRegex(ConfigError, 'input_specs'):
                    validate_config(replace(self.config, input_specs=invalid))

    def test_profile_fields_cannot_be_silently_discarded(self):
        first = self.config.input_specs[0]
        profiled = TensorSpec(
            first.name,
            first.dtype,
            first.shape,
            min_shape=first.shape,
            opt_shape=first.shape,
            max_shape=first.shape)
        with self.assertRaisesRegex(ConfigError, 'without profiles'):
            config_to_document(
                replace(
                    self.config,
                    input_specs=(profiled, *self.config.input_specs[1:])))

    def test_document_rejects_non_string_keys_and_recipe_mismatch(self):
        document = copy.deepcopy(self.document)
        document[17] = 'bad key'
        with self.assertRaisesRegex(ConfigError, 'keys must be strings'):
            config_from_document(document, recipe='pi05-v1')
        with self.assertRaisesRegex(ConfigError, 'does not match'):
            config_from_document(self.document, recipe='other-v1')

    def test_canonical_revalidation_checks_input_cross_dimensions(self):
        specs = list(self.config.input_specs)
        specs[-1] = TensorSpec('noise', 'float32', (1, 200, 32))
        with self.assertRaisesRegex(ConfigError, 'inputs.tensors.noise.shape'):
            validate_config(replace(self.config, input_specs=tuple(specs)))


if __name__ == '__main__':
    unittest.main()
