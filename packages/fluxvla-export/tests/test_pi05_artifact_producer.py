# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Real canonical models and safetensors exercise the training-side handoff."""

from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import fluxvla_export
import torch
from _pi05_test_utils import Artifacts
from fluxvla_export.config import config_to_document, load_config
from fluxvla_export.errors import ArtifactError, ConfigError, ContractError
from fluxvla_export.recipes.pi05.artifacts import produce_pi05_artifacts
from fluxvla_export.recipes.pi05.reference import prepare_reference
from safetensors.torch import load_file, save_file
from test_config import config_document


class PI05ArtifactProducerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.assets = Artifacts(self.directory)
        self.output = self.directory / 'export'

    def produce(self, **changes):
        arguments = {
            'model': self.assets.model,
            'inputs': self.assets.tensors,
            'checkpoint': self.assets.checkpoint_path,
            'preprocessing': self.assets.config.preprocessing,
            'output_dir': self.output,
        }
        arguments.update(changes)
        return produce_pi05_artifacts(**arguments)

    def assert_rejected(self, text, **changes):
        with self.assertRaises((ArtifactError, ConfigError)) as raised:
            self.produce(**changes)
        self.assertIn(text, str(raised.exception))
        self.assertFalse(self.output.exists())

    def test_actual_artifacts_roundtrip_and_reference_exact(self):
        expected = self.assets.prepare().run_reference()['actions']
        checkpoint_before = self.assets.checkpoint_path.read_bytes()
        result = self.produce()
        self.assertEqual(result.checkpoint, self.assets.checkpoint_path)
        self.assertEqual(result.config.name, 'export_config.json')
        self.assertEqual(result.inputs.name, 'export_inputs.safetensors')
        self.assertEqual({p.name
                          for p in self.output.iterdir()},
                         {'export_config.json', 'export_inputs.safetensors'})
        config = load_config(result.config, recipe='pi05-v1')
        self.assertEqual(config.model, self.assets.config.model)
        self.assertEqual(config.precision, self.assets.config.precision)
        self.assertEqual(config.inference, self.assets.config.inference)
        self.assertEqual(config.preprocessing,
                         self.assets.config.preprocessing)
        self.assertEqual(config.checkpoint_sha256,
                         hashlib.sha256(checkpoint_before).hexdigest())
        self.assertEqual(
            config.inputs_sha256,
            hashlib.sha256(result.inputs.read_bytes()).hexdigest())
        self.assertEqual(
            json.loads(result.config.read_text()), config_to_document(config))
        for name, actual in load_file(result.inputs).items():
            torch.testing.assert_close(
                actual, self.assets.tensors[name], rtol=0, atol=0)
        prepared = prepare_reference(
            config=result.config,
            inputs=result.inputs,
            checkpoint=result.checkpoint,
            device='cpu')
        torch.testing.assert_close(
            prepared.run_reference()['actions'], expected, rtol=0, atol=0)
        self.assertEqual(self.assets.checkpoint_path.read_bytes(),
                         checkpoint_before)

    def test_bfloat16_protection_and_eager_variants_roundtrip(self):
        for protected, eager in ((True, False), (False, True)):
            with self.subTest(protected=protected, eager=eager):
                directory = self.directory / f'variant-{protected}'
                directory.mkdir()
                document = config_document()
                document['precision'].update(
                    autocast_dtype='bfloat16',
                    flow_fp32=protected,
                    residual_fp32=protected,
                    vision_stem_fp32=protected,
                    vision_residual_fp32=protected,
                    adarms_fp32=protected,
                    attention_math_fp32=protected,
                    expert_fp32_layers=[0] if protected else [],
                    backbone_fp32_layers=[0] if protected else [],
                    attention_backend='eager' if eager else 'sdpa')
                document['inputs']['tensors']['images']['dtype'] = 'bfloat16'
                document['inputs']['tensors']['states']['dtype'] = 'bfloat16'
                assets = Artifacts(directory, document)
                expected = assets.prepare().run_reference()['actions']
                result = self.produce(
                    model=assets.model,
                    inputs=assets.tensors,
                    checkpoint=assets.checkpoint_path,
                    preprocessing=assets.config.preprocessing,
                    output_dir=directory / 'emitted')
                config = load_config(result.config, recipe='pi05-v1')
                self.assertEqual(config.precision, assets.config.precision)
                self.assertEqual(config.input_specs, assets.config.input_specs)
                prepared = prepare_reference(
                    config=result.config,
                    inputs=result.inputs,
                    checkpoint=result.checkpoint,
                    device='cpu')
                torch.testing.assert_close(
                    prepared.run_reference()['actions'],
                    expected,
                    rtol=0,
                    atol=0)

    def test_derives_nondefault_dimensions_rope_and_view_order(self):
        directory = self.directory / 'different'
        directory.mkdir()
        document = config_document()
        document['model'].update(state_dim=9, action_horizon=5, action_dim=7)
        for branch in ('backbone', 'expert'):
            document['model'][branch]['rope_theta'] = 5000.0
        document['model']['action_in_proj']['in_dim'] = 7
        document['model']['action_out_proj']['out_dim'] = 7
        document['preprocessing']['view_order'] = ['wrist', 'overhead', 'side']
        document['inputs']['tensors']['images']['shape'] = [2, 9, 8, 8]
        document['inputs']['tensors']['img_masks']['shape'] = [2, 3]
        document['inputs']['tensors']['states']['shape'] = [2, 9]
        document['inputs']['tensors']['noise']['shape'] = [2, 5, 7]
        assets = Artifacts(directory, document)
        result = self.produce(
            model=assets.model,
            inputs=assets.tensors,
            checkpoint=assets.checkpoint_path,
            preprocessing=assets.config.preprocessing)
        config = load_config(result.config, recipe='pi05-v1')
        self.assertEqual(config.model, assets.config.model)
        self.assertEqual(config.model.expert.rope_theta, 5000.0)
        self.assertEqual(config.preprocessing.view_order,
                         ('wrist', 'overhead', 'side'))

    def test_checkpoint_is_authoritative_not_live_parameter_values(self):
        expected = self.assets.prepare().run_reference()['actions']
        parameter = self.assets.model.action_out_proj.projector.weight
        with torch.no_grad():
            parameter.add_(10)
        changed = parameter.detach().clone()
        result = self.produce()
        self.assertTrue(torch.equal(parameter, changed))
        prepared = prepare_reference(
            config=result.config,
            inputs=result.inputs,
            checkpoint=result.checkpoint,
            device='cpu')
        torch.testing.assert_close(
            prepared.run_reference()['actions'], expected, rtol=0, atol=0)
        self.assertFalse(
            torch.equal(prepared.model.action_out_proj.projector.weight,
                        parameter))

    def test_preserves_model_inputs_gradients_rng_and_noncontiguous_values(
            self):
        tensors = self.assets.tensors
        tensors['images'] = tensors['images'].transpose(-1, -2)
        tensors['states'].requires_grad_()
        first_parameter = next(self.assets.model.parameters())
        first_parameter.grad = torch.ones_like(first_parameter)
        states = {
            key: value.detach().clone()
            for key, value in self.assets.model.state_dict().items()
        }
        buffers = {
            key: value.detach().clone()
            for key, value in self.assets.model.named_buffers()
        }
        originals = {
            key: value.detach().clone()
            for key, value in tensors.items()
        }
        versions = {key: value._version for key, value in tensors.items()}
        rng = torch.random.get_rng_state().clone()
        grad = first_parameter.grad.clone()
        result = self.produce()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(torch.equal(first_parameter.grad, grad))
        for name, value in self.assets.model.state_dict().items():
            self.assertTrue(torch.equal(value, states[name]), name)
        for name, value in self.assets.model.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]), name)
        saved = load_file(result.inputs)
        for name, value in tensors.items():
            self.assertTrue(torch.equal(value, originals[name]), name)
            self.assertTrue(torch.equal(saved[name], originals[name]), name)
            self.assertEqual(value._version, versions[name])
        self.assertTrue(tensors['states'].requires_grad)
        self.assertFalse(tensors['images'].is_contiguous())

    def test_requires_explicit_complete_preprocessing(self):
        self.assert_rejected('PreprocessingConfig', preprocessing=None)
        self.assert_rejected('PreprocessingConfig', preprocessing={})
        for field, value in (('tokenizer_sha256', ''), ('prompt_id', ''),
                             ('view_order', ('front', )), ('state_in_prompt',
                                                           'true')):
            provenance = dataclasses.replace(self.assets.config.preprocessing,
                                             **{field: value})
            self.assert_rejected(
                'preprocessing' if field != 'view_order' else 'shape',
                preprocessing=provenance)

    def test_rejects_training_mode_in_any_module(self):
        self.assets.model.train()
        self.assert_rejected('eval mode')
        self.assets.model.eval()
        self.assets.model.llm_expert.layers[0].train()
        self.assert_rejected('eval mode')

    def test_rejects_wrapped_custom_and_pi0_models(self):
        self.assert_rejected(
            'unwrapped', model=torch.nn.Sequential(self.assets.model))
        from fluxvla_export.recipes.pi05.model import PI05Model

        class UnsupportedPolicy(PI05Model):
            pass

        with torch.random.fork_rng(devices=[]):
            model = UnsupportedPolicy(self.assets.config)
        self.assert_rejected('unwrap externally', model=model)

    def test_rejects_root_and_nested_extra_state(self):
        self.assets.model.register_buffer('extra', torch.ones(1))
        self.assert_rejected('root parameters or buffers')
        del self.assets.model.extra
        projection = self.assets.model.projector.projector
        projection.register_parameter(
            'lora_A',
            torch.nn.Parameter(torch.ones(1, projection.in_features)))
        self.assert_rejected('lora_A')

    def test_rejects_extra_stateless_module_or_replaced_projection(self):
        self.assets.model.unrecognized = torch.nn.Identity()
        self.assert_rejected('root modules')
        del self.assets.model.unrecognized
        self.assets.model.projector.projector = torch.nn.Sequential(
            self.assets.model.projector.projector)
        self.assert_rejected('projector')

    def test_rejects_overridden_shared_computation_and_hooks(self):

        def other_predict(self, *args, **kwargs):
            raise AssertionError('producer must not run forward')

        self.assets.model.predict_action = types.MethodType(
            other_predict, self.assets.model)
        self.assert_rejected('overridden computation')
        del self.assets.model.predict_action
        handle = self.assets.model.projector.register_forward_hook(
            lambda module, args, output: output)
        try:
            self.assert_rejected('_forward_hooks')
        finally:
            handle.remove()

    def test_rejects_config_module_dimension_disagreement(self):
        self.assets.model.llm_expert.config.intermediate_size += 4
        self.assert_rejected('intermediate_size')

    def test_rejects_per_layer_protection_disagreement(self):
        self.assets.model.llm_expert.layers[0].mlp.compute_fp32 = True
        self.assert_rejected('compute_fp32')

    def test_rejects_branch_precision_disagreement(self):
        self.assets.model.llm_backbone.config.adarms_fp32 = True
        self.assert_rejected('both Gemma branches must agree')

    def test_rejects_modified_nonpersistent_rope(self):
        self.assets.model.llm_expert.rotary_emb.inv_freq.add_(1)
        self.assert_rejected('inv_freq')

    def test_rejects_modified_nonpersistent_positions(self):
        vision = self.assets.model.vision_backbone.vision.vision_model
        value = vision.embeddings
        value.position_ids.add_(1)
        self.assert_rejected('position_ids')

    def test_rejects_nondefault_rope_and_rtc(self):
        rope = self.assets.model.llm_expert.config.rope_parameters
        rope['rope_type'] = 'dynamic'
        self.assert_rejected('rope_parameters')
        rope['rope_type'] = 'default'
        self.assets.model.rtc_training_config = {'enabled': True}
        self.assert_rejected('RTC')

    def test_rejects_non_fp32_live_parameters_without_casting(self):
        self.assets.model.projector.bfloat16()
        self.assert_rejected('materialized FP32')
        self.assertEqual(self.assets.model.projector.projector.weight.dtype,
                         torch.bfloat16)

    def test_rejects_non_fp32_checkpoint_storage(self):
        weights = dict(self.assets.model.state_dict())
        key = next(iter(weights))
        weights[key] = weights[key].bfloat16()
        save_file(weights, self.assets.checkpoint_path)
        self.assert_rejected('dtype')

    def test_rejects_checkpoint_missing_extra_shape_and_nonfinite(self):
        original = self.assets.model.state_dict()
        key = next(iter(original))
        for mode in ('missing', 'extra', 'shape', 'nonfinite'):
            with self.subTest(mode=mode):
                weights = {
                    name: value.detach().clone()
                    for name, value in original.items()
                }
                if mode == 'missing':
                    del weights[key]
                elif mode == 'extra':
                    weights['model.unexpected'] = torch.zeros(1)
                elif mode == 'shape':
                    weights[key] = torch.zeros(1)
                else:
                    weights[key].flatten()[0] = float('nan')
                save_file(weights, self.assets.checkpoint_path)
                self.assert_rejected('Checkpoint')

    def test_rejects_malformed_checkpoint_without_output(self):
        self.assets.checkpoint_path.write_bytes(b'not safetensors')
        self.assert_rejected('safetensors')

    def test_rejects_missing_noise_extra_input_dtype_shape_and_values(self):
        tensors = dict(self.assets.tensors)
        del tensors['noise']
        self.assert_rejected('including noise', inputs=tensors)
        tensors = dict(self.assets.tensors, action_masks=torch.ones(2, 3))
        self.assert_rejected('requires exactly', inputs=tensors)
        tensors = dict(self.assets.tensors)
        tensors['lang_tokens'] = tensors['lang_tokens'].int()
        self.assert_rejected('dtype', inputs=tensors)
        tensors = dict(self.assets.tensors)
        tensors['noise'] = torch.zeros(2, 4, 4)
        self.assert_rejected('noise.shape', inputs=tensors)
        tensors = dict(self.assets.tensors)
        tensors['images'] = tensors['images'].clone()
        tensors['images'][0, 0, 0, 0] = float('inf')
        self.assert_rejected('non-finite', inputs=tensors)
        tensors = dict(self.assets.tensors)
        tensors['lang_tokens'] = tensors['lang_tokens'].clone()
        tensors['lang_tokens'][0, -1] = self.assets.config.model.vocab_size
        self.assert_rejected('lang_tokens must be', inputs=tensors)

    def test_refuses_existing_directory_and_never_overwrites_files(self):
        self.output.mkdir()
        original = self.output / 'export_config.json'
        original.write_text('user-owned data')
        with self.assertRaisesRegex(ArtifactError, 'fresh'):
            self.produce()
        self.assertEqual(original.read_text(), 'user-owned data')
        self.assertEqual(tuple(self.output.iterdir()), (original, ))

    def test_failure_does_not_consume_rng(self):
        self.assets.model.llm_expert.layers[0].mlp.compute_fp32 = True
        rng = torch.random.get_rng_state().clone()
        self.assert_rejected('compute_fp32')
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_private_meta_constructor_checks_layout_without_real_weights(self):
        from fluxvla_export.recipes.pi05.model import PI05Model

        rng = torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            model = PI05Model(
                self.assets.config, _initialization_device='meta')
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        tensors = list(model.parameters()) + list(model.buffers())
        self.assertTrue(tensors)
        self.assertTrue(all(value.device.type == 'meta' for value in tensors))
        expected = self.assets.model.state_dict()
        self.assertEqual(tuple(model.state_dict()), tuple(expected))
        for name, value in model.state_dict().items():
            self.assertEqual(value.shape, expected[name].shape)
            self.assertEqual(value.dtype, expected[name].dtype)
        with self.assertRaisesRegex(ContractError, 'only cpu or meta'):
            PI05Model(self.assets.config, _initialization_device='cuda:0')

    def test_rejects_dangling_output_symlink(self):
        target = self.directory / 'must-not-create'
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ArtifactError, 'fresh'):
            self.produce()
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(target.exists())

    def test_rejects_independent_layer_config_and_vision_backend(self):
        import copy

        layer = self.assets.model.llm_expert.layers[0].self_attn
        layer.config = copy.deepcopy(layer.config)
        layer.config.attention_math_fp32 = True
        self.assert_rejected('share the canonical branch configuration')
        layer.config = self.assets.model.llm_expert.config
        vision = self.assets.model.vision_backbone.vision
        vision.config._attn_implementation = 'eager'
        self.assert_rejected('overridden vision backend')

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires POSIX FIFO support')
    def test_fifo_checkpoint_fails_without_blocking(self):
        fifo = self.directory / 'fifo.safetensors'
        os.mkfifo(fifo)
        script = (
            'import sys\n'
            'from safetensors.torch import load_file\n'
            'from fluxvla_export.config import load_config\n'
            'from fluxvla_export.errors import ArtifactError\n'
            'from fluxvla_export.recipes.pi05.model import build_pi05_model\n'
            'from fluxvla_export.recipes.pi05.artifacts import '
            'produce_pi05_artifacts\n'
            'c=load_config(sys.argv[1],recipe="pi05-v1")\n'
            'try:\n'
            '    produce_pi05_artifacts(model=build_pi05_model(c), '
            'inputs=load_file(sys.argv[2]), checkpoint=sys.argv[3], '
            'preprocessing=c.preprocessing, output_dir=sys.argv[4])\n'
            'except ArtifactError as error:\n'
            '    assert "regular file" in str(error), str(error)\n'
            'else:\n'
            '    raise AssertionError("FIFO must be rejected")\n')
        package_root = Path(fluxvla_export.__file__).resolve().parent.parent
        result = subprocess.run([
            sys.executable, '-c', script,
            str(self.assets.config_path),
            str(self.assets.inputs_path),
            str(fifo),
            str(self.output)
        ],
                                env=dict(
                                    os.environ, PYTHONPATH=str(package_root)),
                                capture_output=True,
                                text=True,
                                timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_producer_import_is_dependency_light_and_wheel_relative(self):
        package_dir = Path(fluxvla_export.__file__).resolve().parent
        script = ('import sys\n'
                  'from pathlib import Path\n'
                  'import fluxvla_export\n'
                  'from fluxvla_export.recipes.pi05.artifacts import '
                  'produce_pi05_artifacts\n'
                  'assert Path(fluxvla_export.__file__).resolve().parent '
                  '== Path(sys.argv[1])\n'
                  'assert not {"torch", "transformers", "safetensors", '
                  '"fluxvla", "mmengine", "timm"}.intersection(sys.modules)\n')
        result = subprocess.run(
            [sys.executable, '-S', '-c', script,
             str(package_dir)],
            env=dict(os.environ, PYTHONPATH=str(package_dir.parent)),
            capture_output=True,
            text=True,
            timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
