# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Real JSON parsing tests; no torch, safetensors, or model instantiation."""

from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from fluxvla_export.config import DeploymentConfig, load_config
from fluxvla_export.contracts import TensorSpec
from fluxvla_export.errors import ConfigError


def config_document() -> dict:
    """Complete schema-1 example using the existing tiny PI0.5 test dimensions.

    Architecture dimensions come from test/test_models/test_vla.py's real
    SigLIP/Gemma fixture. These digests are syntax examples only: artifact
    tests must replace checkpoint.sha256 and inputs.sha256 with file hashes.
    This helper performs no I/O and constructs no model or output tensors.
    """
    gemma = {
        'type': 'condition_gemma',
        'hidden_size': 16,
        'intermediate_size': 32,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'num_key_value_heads': 1,
        'head_dim': 8,
        'vocab_size': 64,
        'max_position_embeddings': 64,
        'hidden_activation': 'gelu_pytorch_tanh',
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'attention_bias': False,
        'attention_dropout': 0.0,
        'use_adarms': True,
        'adarms_cond_dim': None,
        'use_cache': True,
        'pad_token_id': 0,
        'bos_token_id': 2,
        'eos_token_id': 1,
    }

    def linear(in_dim, out_dim):
        return {
            'type': 'linear',
            'in_dim': in_dim,
            'out_dim': out_dim,
            'bias': True
        }

    def tensor(dtype, shape):
        return {'dtype': dtype, 'shape': shape}

    return {
        'schema_version': 1,
        'recipe': 'pi05-v1',
        'checkpoint': {
            'format': 'fluxvla-native',
            'kind': 'dense',
            'sha256': 'a' * 64,
        },
        'inputs': {
            'format': 'safetensors',
            'sha256': 'b' * 64,
            'tensors': {
                'images': tensor('float32', [2, 6, 8, 8]),
                'img_masks': tensor('bool', [2, 2]),
                'lang_tokens': tensor('int64', [2, 4]),
                'lang_masks': tensor('bool', [2, 4]),
                'states': tensor('float32', [2, 4]),
                'noise': tensor('float32', [2, 3, 4]),
            },
        },
        'model': {
            'type': 'pi05',
            'vision': {
                'type': 'siglip',
                'hidden_size': 16,
                'intermediate_size': 32,
                'num_hidden_layers': 1,
                'num_attention_heads': 2,
                'num_channels': 3,
                'image_size': 8,
                'patch_size': 4,
                'hidden_activation': 'gelu_pytorch_tanh',
                'layer_norm_eps': 1e-6,
                'attention_dropout': 0.0,
                'vision_use_head': False,
            },
            'backbone': gemma,
            'expert': dict(gemma, adarms_cond_dim=16),
            'projector': linear(16, 16),
            'action_in_proj': linear(4, 16),
            'action_out_proj': linear(16, 4),
            'time_mlp_in': linear(16, 16),
            'time_mlp_out': linear(16, 16),
            'proj_width': 16,
            'state_dim': 4,
            'action_dim': 4,
            'action_horizon': 3,
            'state_projection': False,
        },
        'inference': {
            'mode': 'plain',
            'solver': 'euler',
            'num_steps': 2,
            'rtc': False,
        },
        'precision': {
            'parameter_dtype': 'float32',
            'autocast_dtype': None,
            'attention_backend': 'sdpa',
            'flow_fp32': False,
            'vision_stem_fp32': False,
            'residual_fp32': False,
            'vision_residual_fp32': False,
            'adarms_fp32': False,
            'attention_math_fp32': False,
            'backbone_fp32_layers': [],
            'expert_fp32_layers': [],
        },
        'preprocessing': {
            'id': 'tiny-pi05-fixture-v1',
            'tokenizer_id': 'training-tokenizer-v1',
            'tokenizer_sha256': 'c' * 64,
            'prompt_id': 'task-and-state-prompt-v1',
            'normalization_id': 'training-stats-v1',
            'normalization_sha256': 'd' * 64,
            'state_semantics': 'normalized-padded-joints-v1',
            'action_semantics': 'normalized-padded-joint-deltas-v1',
            'image_transform_id': 'training-image-transform-v1',
            'view_order': ['front', 'wrist'],
            'state_in_prompt': True,
        },
    }


def _replace(document, path, value):
    target = document
    keys = path.split('.')
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def _objects(document, path=()):
    yield path, document
    for key, value in document.items():
        if isinstance(value, dict):
            yield from _objects(value, path + (key, ))


class ConfigTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / 'deployment.json'

    def load(self, document=None, *, recipe='pi05-v1'):
        if document is None:
            document = config_document()
        self.path.write_text(json.dumps(document), encoding='utf-8')
        return load_config(self.path, recipe=recipe)

    def reject(self, path, value, *, message=None):
        document = config_document()
        _replace(document, path, value)
        with self.assertRaises(ConfigError) as raised:
            self.load(document)
        self.assertIn(str(self.path), str(raised.exception))
        self.assertIn(message or path, str(raised.exception))

    def test_loads_real_json_and_typed_immutable_contract(self):
        config = self.load()
        self.assertIsInstance(config, DeploymentConfig)
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.recipe, 'pi05-v1')
        self.assertEqual(config.checkpoint_sha256, 'a' * 64)
        self.assertEqual(config.inputs_sha256, 'b' * 64)
        self.assertEqual(config.model.vocab_size, 64)
        self.assertEqual(config.model.backbone.adarms_cond_dim, None)
        self.assertEqual(config.model.expert.adarms_cond_dim, 16)
        self.assertEqual(config.model.action_horizon, 3)
        self.assertEqual(config.inference.num_steps, 2)
        self.assertEqual(config.preprocessing.view_order, ('front', 'wrist'))
        self.assertEqual(
            tuple(s.name for s in config.input_specs),
            ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'states',
             'noise'))
        self.assertTrue(
            all(isinstance(s, TensorSpec) for s in config.input_specs))
        self.assertEqual(config.input_specs[-1].shape, (2, 3, 4))
        with self.assertRaises(FrozenInstanceError):
            config.model.action_dim = 9
        with self.assertRaises(FrozenInstanceError):
            config.precision.flow_fp32 = True

    def test_config_document_is_independent_between_calls(self):
        first = config_document()
        first['model']['expert']['hidden_size'] = 100
        self.assertEqual(first['model']['backbone']['hidden_size'], 16)
        self.assertEqual(config_document()['model']['expert']['hidden_size'],
                         16)

    def test_every_object_rejects_missing_and_unknown_fields(self):
        for path, original in _objects(config_document()):
            for mutation in ('missing', 'unknown'):
                with self.subTest(path=path, mutation=mutation):
                    document = config_document()
                    target = document
                    for key in path:
                        target = target[key]
                    if mutation == 'missing':
                        key = next(iter(original))
                        del target[key]
                    else:
                        key = 'unrecognized_field'
                        target[key] = None
                    with self.assertRaises(ConfigError) as raised:
                        self.load(document)
                    self.assertIn('.'.join(path) or 'config',
                                  str(raised.exception))
                    self.assertIn(key, str(raised.exception))

    def test_every_nested_object_rejects_non_objects(self):
        for path, _ in _objects(config_document()):
            if not path:
                continue
            for invalid in (None, [], 'dict', True):
                with self.subTest(path=path, invalid=invalid):
                    self.reject('.'.join(path), invalid)
        with self.assertRaisesRegex(ConfigError,
                                    'config: expected a JSON object'):
            self.load([])

    def test_integer_fields_reject_bools_strings_floats_and_nonpositive(self):
        document = config_document()
        for path, record in _objects(document):
            for key, value in record.items():
                if type(value) is int and not key.endswith('_token_id'):
                    field = '.'.join(path + (key, ))
                    for invalid in (True, str(value), float(value), 0, -1):
                        with self.subTest(field=field, invalid=invalid):
                            self.reject(field, invalid)

    def test_boolean_fields_reject_numbers_and_strings(self):
        document = config_document()
        for path, record in _objects(document):
            for key, value in record.items():
                if type(value) is bool:
                    field = '.'.join(path + (key, ))
                    for invalid in (0, 1, 'true', None):
                        with self.subTest(field=field, invalid=invalid):
                            self.reject(field, invalid)

    def test_duplicate_json_keys_at_root_and_nested_levels(self):
        text = json.dumps(config_document())
        for original, replacement, key in (
            ('"schema_version": 1', '"schema_version": 1, "schema_version": 1',
             'schema_version'), ('"head_dim": 8',
                                 '"head_dim": 8, "head_dim": 8', 'head_dim')):
            with self.subTest(key=key):
                self.path.write_text(
                    text.replace(original, replacement, 1), encoding='utf-8')
                with self.assertRaisesRegex(ConfigError,
                                            f'duplicate object key.*{key}'):
                    load_config(self.path, recipe='pi05-v1')

    def test_rejects_nonfinite_json_numbers_and_overflow(self):
        for number in ('NaN', 'Infinity', '-Infinity', '1e999'):
            with self.subTest(number=number):
                text = json.dumps(config_document()).replace(
                    '10000.0', number, 1)
                self.path.write_text(text, encoding='utf-8')
                with self.assertRaisesRegex(ConfigError, 'non-finite'):
                    load_config(self.path, recipe='pi05-v1')

    def test_schema_recipe_and_requested_recipe_are_strict(self):
        for version in (0, 2, '1', True):
            self.reject('schema_version', version)
        self.reject('recipe', 'pi0-v1')
        with self.assertRaisesRegex(ConfigError,
                                    'does not match requested recipe'):
            self.load(recipe='different-recipe-v1')
        for recipe in ('', '   ', 1, None):
            with self.subTest(recipe=recipe):
                with self.assertRaisesRegex(ConfigError, 'recipe argument'):
                    self.load(recipe=recipe)

    def test_rejects_rtc_lora_external_checkpoints_and_solvers(self):
        for path, value in (('inference.mode', 'rtc'), ('inference.rtc', True),
                            ('inference.solver', 'heun'), ('checkpoint.kind',
                                                           'lora'),
                            ('checkpoint.kind',
                             'quantized'), ('checkpoint.format', 'openpi'),
                            ('inputs.format', 'pickle'), ('model.type', 'pi0'),
                            ('model.state_projection', True)):
            with self.subTest(path=path, value=value):
                self.reject(path, value)

    def test_all_digests_are_required_sha256_not_paths(self):
        for path in ('checkpoint.sha256', 'inputs.sha256',
                     'preprocessing.tokenizer_sha256',
                     'preprocessing.normalization_sha256'):
            for invalid in ('a' * 63, 'A' * 64, 'g' * 64, '', '/model.bin',
                            42):
                with self.subTest(path=path, invalid=invalid):
                    self.reject(path, invalid)

    def test_architecture_semantics_not_arbitrary_dicts(self):
        for path, value in (('model.vision.type',
                             'clip'), ('model.vision.num_channels', 1),
                            ('model.vision.hidden_activation',
                             'relu'), ('model.vision.hidden_size',
                                       15), ('model.vision.image_size', 9),
                            ('model.vision.vision_use_head',
                             True), ('model.vision.attention_dropout',
                                     0.1), ('model.backbone.type', 'gemma2'),
                            ('model.backbone.hidden_activation',
                             'silu'), ('model.backbone.attention_bias', True),
                            ('model.backbone.attention_dropout',
                             0.1), ('model.backbone.num_key_value_heads',
                                    3), ('model.backbone.head_dim', 7),
                            ('model.backbone.use_cache',
                             False), ('model.backbone.adarms_cond_dim',
                                      16), ('model.expert.use_adarms', False),
                            ('model.expert.adarms_cond_dim',
                             8), ('model.expert.hidden_size',
                                  8), ('model.proj_width',
                                       15), ('model.projector.type',
                                             'mlp'), ('model.projector.bias',
                                                      False)):
            with self.subTest(path=path, value=value):
                self.reject(
                    path,
                    value,
                    message=path.rsplit('.', 1)[0]
                    if path.endswith('num_key_value_heads') else path)

    def test_norm_epsilon_and_rope_numbers_are_strict(self):
        for path in ('model.vision.layer_norm_eps',
                     'model.backbone.rms_norm_eps',
                     'model.expert.rms_norm_eps', 'model.backbone.rope_theta',
                     'model.expert.rope_theta'):
            for invalid in (True, '0.001', 0, -1, None):
                with self.subTest(path=path, invalid=invalid):
                    self.reject(path, invalid)

    def test_cross_branch_cache_compatibility(self):
        for field, value in (('num_hidden_layers', 2), ('num_key_value_heads',
                                                        2), ('head_dim', 16),
                             ('rope_theta', 20000), ('vocab_size', 128)):
            self.reject(f'model.expert.{field}', value)

    def test_every_projection_dimension_matches_its_boundary(self):
        for name in ('projector', 'action_in_proj', 'action_out_proj',
                     'time_mlp_in', 'time_mlp_out'):
            for dim in ('in_dim', 'out_dim'):
                with self.subTest(name=name, dim=dim):
                    self.reject(
                        f'model.{name}.{dim}', 999, message=f'model.{name}')

    def test_special_tokens_are_in_vocabulary_and_may_be_null(self):
        for branch in ('backbone', 'expert'):
            for token in ('pad_token_id', 'bos_token_id', 'eos_token_id'):
                for invalid in (True, -1, 64, '0'):
                    self.reject(f'model.{branch}.{token}', invalid)
        document = config_document()
        document['model']['backbone']['pad_token_id'] = None
        self.assertIsNone(self.load(document).model.backbone.pad_token_id)

    def test_bfloat16_autocast_preserves_protected_flags(self):
        document = config_document()
        precision = document['precision']
        precision.update(
            autocast_dtype='bfloat16',
            flow_fp32=True,
            vision_stem_fp32=True,
            residual_fp32=True,
            vision_residual_fp32=True,
            adarms_fp32=True,
            attention_math_fp32=True,
            backbone_fp32_layers=[0],
            expert_fp32_layers=[0])
        document['inputs']['tensors']['images']['dtype'] = 'bfloat16'
        document['inputs']['tensors']['states']['dtype'] = 'bfloat16'
        config = self.load(document)
        self.assertEqual(config.precision.parameter_dtype, 'float32')
        self.assertEqual(config.precision.autocast_dtype, 'bfloat16')
        self.assertEqual(config.precision.expert_fp32_layers, (0, ))
        self.assertTrue(config.precision.adarms_fp32)
        self.assertTrue(config.precision.attention_math_fp32)
        self.assertEqual(config.input_specs[0].dtype, 'bfloat16')

    def test_unsupported_precision_and_layer_selections(self):
        for path, value in (('precision.parameter_dtype',
                             'bfloat16'), ('precision.parameter_dtype',
                                           'float16'),
                            ('precision.autocast_dtype',
                             'float16'), ('precision.autocast_dtype', False),
                            ('precision.attention_backend',
                             'flash_attention_2'),
                            ('precision.backbone_fp32_layers',
                             [1]), ('precision.expert_fp32_layers',
                                    [-1]), ('precision.expert_fp32_layers',
                                            [True]),
                            ('precision.expert_fp32_layers',
                             [0, 0]), ('precision.expert_fp32_layers', ['0']),
                            ('precision.expert_fp32_layers',
                             {}), ('inputs.tensors.images.dtype',
                                   'bfloat16'), ('inputs.tensors.states.dtype',
                                                 'bfloat16')):
            with self.subTest(path=path, value=value):
                self.reject(path, value)
        document = config_document()
        document['precision'].update(
            attention_math_fp32=True, attention_backend='eager')
        with self.assertRaisesRegex(ConfigError,
                                    'precision.attention_backend'):
            self.load(document)

    def test_all_inputs_reject_wrong_dtype_rank_and_dynamic_shapes(self):
        for name, tensor in config_document()['inputs']['tensors'].items():
            path = f'inputs.tensors.{name}'
            self.reject(f'{path}.dtype', 'float64')
            for shape in ([], [1], tensor['shape'] + [1], '2,4'):
                self.reject(f'{path}.shape', shape)
            for invalid in (-1, 0, True, 1.0, '2'):
                self.reject(f'{path}.shape', [invalid] + tensor['shape'][1:])
        self.reject('inputs.tensors.lang_tokens.dtype', 'int32')
        self.reject('inputs.tensors.img_masks.dtype', 'int64')
        self.reject('inputs.tensors.lang_masks.dtype', 'float32')
        self.reject('inputs.tensors.noise.dtype', 'bfloat16')

    def test_batch_view_image_token_state_and_noise_dimensions(self):
        for path, shape in (('images', [2, 3, 8, 8]), ('images', [2, 6, 4, 8]),
                            ('img_masks', [1, 2]), ('img_masks', [2, 1]),
                            ('lang_tokens', [1, 4]), ('lang_masks', [2, 5]),
                            ('states', [2, 3]), ('states',
                                                 [3, 4]), ('noise', [2, 4, 4]),
                            ('noise', [2, 3, 3]), ('noise', [1, 3, 4])):
            with self.subTest(path=path, shape=shape):
                self.reject(f'inputs.tensors.{path}.shape', shape)

    def test_positions_cover_unmasked_full_prefix_and_suffix(self):
        self.reject('model.backbone.max_position_embeddings', 11)
        self.reject('model.expert.max_position_embeddings', 14)
        document = config_document()
        document['model']['backbone']['max_position_embeddings'] = 12
        document['model']['expert']['max_position_embeddings'] = 15
        self.load(document)

    def test_preprocessing_is_explicit_and_view_order_unique(self):
        for key in ('id', 'tokenizer_id', 'prompt_id', 'normalization_id',
                    'state_semantics', 'action_semantics',
                    'image_transform_id'):
            for invalid in ('', '   ', 1, {}):
                self.reject(f'preprocessing.{key}', invalid)
        for invalid in ([], ['front', 'front'], [1, 'wrist'], [''], 'front'):
            self.reject('preprocessing.view_order', invalid)
        self.reject(
            'preprocessing.view_order', ['front'],
            message='inputs.tensors.images.shape')

    def test_supports_other_batch_views_horizon_and_state_dimensions(self):
        document = config_document()
        document['model'].update(state_dim=7, action_dim=6, action_horizon=10)
        document['model']['action_in_proj']['in_dim'] = 6
        document['model']['action_out_proj']['out_dim'] = 6
        document['preprocessing'].update(
            view_order=['overhead'], state_in_prompt=False)
        shapes = {
            'images': [1, 3, 8, 8],
            'img_masks': [1, 1],
            'lang_tokens': [1, 5],
            'lang_masks': [1, 5],
            'states': [1, 7],
            'noise': [1, 10, 6]
        }
        for name, shape in shapes.items():
            document['inputs']['tensors'][name]['shape'] = shape
        config = self.load(document)
        self.assertEqual(config.input_specs[-1].shape, (1, 10, 6))
        self.assertEqual(config.model.state_dim, 7)
        self.assertFalse(config.preprocessing.state_in_prompt)

    def test_accepts_real_aloha_expert_head_width_relationship(self):
        document = config_document()
        model = document['model']
        for branch, width, intermediate in (('backbone', 2048, 16384),
                                            ('expert', 1024, 4096)):
            model[branch].update(
                hidden_size=width,
                intermediate_size=intermediate,
                num_hidden_layers=18,
                num_attention_heads=8,
                head_dim=256,
                vocab_size=257152,
                max_position_embeddings=8192)
        model['expert']['adarms_cond_dim'] = 1024
        model.update(
            proj_width=1024, action_horizon=50, action_dim=32, state_dim=32)
        model['vision'].update(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14)
        for name, in_dim, out_dim in (('projector', 1152,
                                       2048), ('action_in_proj', 32, 1024),
                                      ('action_out_proj', 1024,
                                       32), ('time_mlp_in', 1024, 1024),
                                      ('time_mlp_out', 1024, 1024)):
            model[name].update(in_dim=in_dim, out_dim=out_dim)
        document['inputs']['tensors']['images']['shape'] = [2, 6, 224, 224]
        document['inputs']['tensors']['states']['shape'] = [2, 32]
        document['inputs']['tensors']['noise']['shape'] = [2, 50, 32]
        document['precision'].update(
            autocast_dtype='bfloat16', expert_fp32_layers=[0, 1, 2, 3, 4, 5])
        config = self.load(document)
        self.assertEqual(config.model.expert.hidden_size, 1024)
        self.assertEqual(
            config.model.expert.num_attention_heads *
            config.model.expert.head_dim, 2048)
        self.assertEqual(config.precision.expert_fp32_layers, tuple(range(6)))

    def test_missing_directory_invalid_encoding_and_malformed_json(self):
        with self.assertRaisesRegex(ConfigError, 'cannot read UTF-8 config'):
            load_config(self.path, recipe='pi05-v1')
        with self.assertRaisesRegex(ConfigError, 'cannot read UTF-8 config'):
            load_config(self.path.parent, recipe='pi05-v1')
        self.path.write_bytes(b'\xff')
        with self.assertRaisesRegex(ConfigError, 'cannot read UTF-8 config'):
            load_config(self.path, recipe='pi05-v1')
        for text in ('{"recipe":', '{"a":1,}', 'null', '[]'):
            self.path.write_text(text, encoding='utf-8')
            with self.assertRaises(ConfigError) as raised:
                load_config(self.path, recipe='pi05-v1')
            self.assertIn(str(self.path), str(raised.exception))
        for path in ('', '   ', None, 1):
            with self.assertRaisesRegex(ConfigError, 'config path'):
                load_config(path, recipe='pi05-v1')

    def test_resolves_relative_and_symlink_paths(self):
        expected = self.load()
        relative = os.path.relpath(self.path, Path.cwd())
        self.assertEqual(load_config(relative, recipe='pi05-v1'), expected)
        link = self.path.with_name('link.json')
        link.symlink_to(self.path)
        self.assertEqual(load_config(link, recipe='pi05-v1'), expected)
        self.path.write_text('{}', encoding='utf-8')
        with self.assertRaises(ConfigError) as raised:
            load_config(link, recipe='pi05-v1')
        self.assertIn(str(self.path), str(raised.exception))

    def test_expands_home_in_a_real_subprocess(self):
        import fluxvla_export

        self.load()
        package_root = Path(fluxvla_export.__file__).resolve().parent.parent
        script = ('from fluxvla_export.config import load_config\n'
                  'assert load_config("~/deployment.json", '
                  'recipe="pi05-v1").model.vocab_size == 64\n')
        result = subprocess.run([sys.executable, '-S', '-c', script],
                                env=dict(
                                    os.environ,
                                    HOME=str(self.path.parent),
                                    PYTHONPATH=str(package_root)),
                                capture_output=True,
                                text=True,
                                check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_rejects_fifo_without_blocking(self):
        import fluxvla_export

        fifo = self.path.with_name('fifo.json')
        os.mkfifo(fifo)
        package_root = Path(fluxvla_export.__file__).resolve().parent.parent
        script = ('import sys\n'
                  'from fluxvla_export.config import load_config\n'
                  'from fluxvla_export.errors import ConfigError\n'
                  'try:\n'
                  '    load_config(sys.argv[1], recipe="pi05-v1")\n'
                  'except ConfigError as error:\n'
                  '    assert "regular file" in str(error), str(error)\n'
                  'else:\n'
                  '    raise AssertionError("FIFO must be rejected")\n')
        result = subprocess.run(
            [sys.executable, '-S', '-c', script,
             str(fifo)],
            env=dict(os.environ, PYTHONPATH=str(package_root)),
            capture_output=True,
            text=True,
            check=False,
            timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unresolvable_paths_report_config_error(self):
        loop = self.path.with_name('loop.json')
        loop.symlink_to(loop)
        for path in (loop, '\x00invalid.json',
                     '~fluxvla-nonexistent-user-5d927f/config.json'):
            with self.subTest(path=path):
                with self.assertRaises(ConfigError) as raised:
                    load_config(path, recipe='pi05-v1')
                self.assertIn('cannot resolve config path',
                              str(raised.exception))
                self.assertIn(str(path), str(raised.exception))

    def test_does_not_execute_python_or_read_neighbor_artifacts(self):
        self.path.write_text(
            'raise RuntimeError("must not execute")', encoding='utf-8')
        with self.assertRaisesRegex(ConfigError, 'invalid JSON'):
            load_config(self.path, recipe='pi05-v1')
        self.load()
        self.assertEqual(tuple(self.path.parent.iterdir()), (self.path, ))

    def test_stdlib_only_import_and_parse_in_clean_interpreter(self):
        import fluxvla_export

        self.load()
        package_dir = Path(fluxvla_export.__file__).resolve().parent
        script = (
            'import sys\n'
            'from pathlib import Path\n'
            'import fluxvla_export\n'
            'from fluxvla_export.config import load_config\n'
            'assert Path(fluxvla_export.__file__).resolve().parent '
            '== Path(sys.argv[2])\n'
            'config = load_config(sys.argv[1], recipe="pi05-v1")\n'
            'assert config.model.vocab_size == 64\n'
            'assert not any(name.split(".")[0] in '
            '{"torch", "safetensors", "transformers", "mmengine", "fluxvla"} '
            'for name in sys.modules)\n')
        result = subprocess.run([
            sys.executable, '-S', '-c', script,
            str(self.path),
            str(package_dir)
        ],
                                env=dict(
                                    os.environ,
                                    PYTHONPATH=str(package_dir.parent)),
                                capture_output=True,
                                text=True,
                                check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
