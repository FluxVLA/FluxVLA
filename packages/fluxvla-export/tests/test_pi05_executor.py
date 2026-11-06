# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Real graph-backed CPU engine execution, without model or output mocks."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from fluxvla_export.errors import ArtifactError, ContractError
from fluxvla_export.recipes.pi05.executor import build_executor
from fluxvla_export.recipes.pi05.recipe import PI05Recipe
from _pi05_test_utils import Artifacts
from test_config import config_document


class TorchGraphEngine:
    """An InferenceEngine executing the actual prepared torch graph."""

    def __init__(self, graph):
        self.graph = graph
        self.calls = []
        self.prefix_cache = None
        self.prefix_cache_before = None

    def infer(self, inputs, *, synchronize=False):
        self.calls.append({
            'synchronize':
            synchronize,
            'names':
            tuple(inputs),
            'cache_ids': {
                name: id(value)
                for name, value in inputs.items()
                if name.startswith(('key_', 'value_'))
            },
            'timestep':
            (inputs['timestep'].clone() if 'timestep' in inputs else None),
        })
        with torch.no_grad():
            actual = self.graph.module(*(inputs[spec.name]
                                         for spec in self.graph.input_specs))
        outputs = actual if isinstance(actual, tuple) else (actual, )
        names = tuple(spec.name for spec in self.graph.output_specs)
        if len(outputs) != len(names):
            raise AssertionError(
                'Actual prepared graph output arity mismatch.')
        result = dict(zip(names, outputs))
        if self.graph.name == 'prefill':
            self.prefix_cache = result
            self.prefix_cache_before = {
                name: tensor.clone()
                for name, tensor in result.items()
            }
        return result


def graph_engines(prepared):
    return {
        name: TorchGraphEngine(graph)
        for name, graph in prepared.graphs.items()
    }


def prepare_artifacts(directory, document=None):
    artifacts = Artifacts(directory, document)
    prepared = PI05Recipe().prepare(
        config=artifacts.config,
        checkpoint=artifacts.checkpoint_path,
        inputs=artifacts.inputs_path,
        device='cpu')
    return artifacts, prepared


class PI05ExecutorTests(unittest.TestCase):

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

    def assert_tensors_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            with self.subTest(tensor=name):
                torch.testing.assert_close(
                    actual[name], expected[name], rtol=0, atol=0)

    def test_real_engines_match_reference_for_fp32_bf16_and_protections(self):
        for amp in (False, True):
            for protected in (False, True):
                for flow in (False, True):
                    with self.subTest(amp=amp, protected=protected, flow=flow):
                        document = config_document()
                        document['precision'].update(
                            autocast_dtype='bfloat16' if amp else None,
                            flow_fp32=flow,
                            residual_fp32=protected,
                            vision_stem_fp32=protected,
                            vision_residual_fp32=protected,
                            adarms_fp32=protected,
                            attention_math_fp32=protected,
                            backbone_fp32_layers=[0] if protected else [],
                            expert_fp32_layers=[0] if protected else [])
                        if amp:
                            for name in ('images', 'states'):
                                document['inputs']['tensors'][name][
                                    'dtype'] = 'bfloat16'
                        _, prepared = prepare_artifacts(
                            self.directory, document)
                        engines = graph_engines(prepared)
                        actual = prepared.create_executor(engines).infer(
                            prepared.policy_inputs)
                        self.assert_tensors_equal(actual,
                                                  prepared.run_reference())
                        self.assertEqual(actual['actions'].dtype,
                                         torch.float32)
                        self.assertFalse(actual['actions'].requires_grad)

    def test_exact_time_sequence_synchronization_and_cache_reuse(self):
        document = config_document()
        document['inference']['num_steps'] = 7
        _, prepared = prepare_artifacts(self.directory, document)
        engines = graph_engines(prepared)
        before = {
            name: value.clone()
            for name, value in prepared.policy_inputs.items()
        }
        rng = torch.random.get_rng_state().clone()
        result = prepared.create_executor(engines).infer(
            prepared.policy_inputs, synchronize=True)
        self.assertEqual(len(engines['prefill'].calls), 1)
        self.assertEqual(len(engines['denoise'].calls), 7)
        time = torch.tensor(1.0, dtype=torch.float32)
        dt = torch.tensor(-1.0 / 7, dtype=torch.float32)
        prefix = engines['prefill'].prefix_cache
        expected_ids = {
            name: id(value)
            for name, value in prefix.items()
            if name.startswith(('key_', 'value_'))
        }
        for call in engines['denoise'].calls:
            self.assertTrue(call['synchronize'])
            torch.testing.assert_close(
                call['timestep'], time.expand(2), rtol=0, atol=0)
            self.assertEqual(call['cache_ids'], expected_ids)
            self.assertEqual(
                call['names'],
                tuple(spec.name
                      for spec in prepared.graphs['denoise'].input_specs))
            time += dt
        self.assertTrue(engines['prefill'].calls[0]['synchronize'])
        self.assert_tensors_equal(prefix,
                                  engines['prefill'].prefix_cache_before)
        self.assert_tensors_equal(prepared.policy_inputs, before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertNotEqual(result['actions'].data_ptr(),
                            before['noise'].data_ptr())
        self.assertNotEqual(result['actions'].data_ptr(),
                            prepared.policy_inputs['noise'].data_ptr())

    def test_json_metadata_reconstructs_without_model_weights(self):
        _, prepared = prepare_artifacts(self.directory)
        metadata_path = self.directory / 'runtime.json'
        metadata_path.write_text(
            json.dumps(prepared.metadata['runtime']), encoding='utf-8')
        restored = json.loads(metadata_path.read_text(encoding='utf-8'))
        executor = build_executor(graph_engines(prepared), metadata=restored)
        self.assertFalse(
            any(
                isinstance(value, torch.nn.Module)
                for value in vars(executor).values()))
        self.assert_tensors_equal(
            executor.infer(prepared.policy_inputs), prepared.run_reference())
        code = '''
import sys
import fluxvla_export.recipes.pi05.executor
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'fluxvla' not in sys.modules
assert 'fluxvla_export.recipes.pi05.model' not in sys.modules
'''
        completed = subprocess.run([sys.executable, '-c', code],
                                   env=os.environ.copy(),
                                   text=True,
                                   capture_output=True,
                                   check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_repeatability_model_input_immutability_and_unused_state_values(
            self):
        _, prepared = prepare_artifacts(self.directory)
        engines = graph_engines(prepared)
        executor = prepared.create_executor(engines)
        state = {
            name: tensor.clone()
            for name, tensor in
            prepared.graphs['prefill'].module.state_dict().items()
        }
        first = executor.infer(prepared.policy_inputs)
        second = executor.infer(prepared.policy_inputs)
        changed = dict(prepared.policy_inputs)
        changed['states'] = changed['states'] * 100 + 42
        third = executor.infer(changed)
        self.assert_tensors_equal(first, second)
        self.assert_tensors_equal(first, third)
        self.assert_tensors_equal(
            prepared.graphs['prefill'].module.state_dict(), state)
        self.assertTrue(
            all(not call['synchronize'] for engine in engines.values()
                for call in engine.calls))

    def test_input_contract_failures_precede_any_real_engine_execution(self):
        _, prepared = prepare_artifacts(self.directory)
        engines = graph_engines(prepared)
        executor = prepared.create_executor(engines)
        for variant in ('missing', 'extra', 'shape', 'dtype', 'finite',
                        'vocab'):
            with self.subTest(variant=variant):
                inputs = dict(prepared.policy_inputs)
                if variant == 'missing':
                    del inputs['states']
                elif variant == 'extra':
                    inputs['actions'] = inputs['noise']
                elif variant == 'shape':
                    inputs['noise'] = inputs['noise'][:, :-1]
                elif variant == 'dtype':
                    inputs['states'] = inputs['states'].double()
                elif variant == 'finite':
                    inputs['noise'] = inputs['noise'].clone()
                    inputs['noise'][0, 0, 0] = float('inf')
                else:
                    inputs['lang_tokens'] = inputs['lang_tokens'].clone()
                    inputs['lang_tokens'][0, -1] = 64
                with self.assertRaises((ContractError, ArtifactError)):
                    executor.infer(inputs)
        with self.assertRaisesRegex(ContractError, 'synchronize'):
            executor.infer(prepared.policy_inputs, synchronize=1)
        self.assertEqual(engines['prefill'].calls, [])
        self.assertEqual(engines['denoise'].calls, [])

    def test_metadata_rejects_unknown_missing_types_and_invalid_specs(self):
        _, prepared = prepare_artifacts(self.directory)
        original = prepared.metadata['runtime']
        edits = [
            (('schema_version', ), True),
            (('recipe', ), 'pi0'),
            (('entrypoint', ), 'untrusted:call'),
            (('num_steps', ), 0),
            (('vocab_size', ), True),
            (('input_specs', 0, 'shape', 0), -1),
            (('input_specs', 0, 'dtype'), 'float16'),
            (('input_specs', 4, 'name'), 'unknown'),
            (('input_specs', 4, 'shape', 0), 3),
            (('prefill_output_specs', 1, 'name'), 'value_0'),
            (('prefill_output_specs', 1, 'shape', 2), 100),
            (('velocity_spec', 'shape', 1), 99),
            (('output_spec', 'dtype'), 'bfloat16'),
        ]
        for path, value in edits:
            with self.subTest(path=path):
                metadata = copy.deepcopy(original)
                target = metadata
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ContractError):
                    build_executor(graph_engines(prepared), metadata=metadata)
        for edit in ('missing', 'unknown'):
            metadata = copy.deepcopy(original)
            if edit == 'missing':
                del metadata['num_steps']
            else:
                metadata['rtc'] = True
            with self.assertRaises(ContractError):
                build_executor(graph_engines(prepared), metadata=metadata)

    def test_engine_names_and_callable_contract_are_strict(self):
        _, prepared = prepare_artifacts(self.directory)
        original = graph_engines(prepared)
        for engines in ({
                'prefill': original['prefill']
        }, dict(original,
                extra=original['prefill']), dict(original, denoise=object())):
            with self.assertRaises(ContractError):
                build_executor(engines, metadata=prepared.metadata['runtime'])

    def test_actual_bfloat16_velocity_rejects_incorrect_metadata_dtype(self):
        document = config_document()
        document['precision']['autocast_dtype'] = 'bfloat16'
        _, prepared = prepare_artifacts(self.directory, document)
        metadata = copy.deepcopy(prepared.metadata['runtime'])
        self.assertEqual(metadata['velocity_spec']['dtype'], 'bfloat16')
        metadata['velocity_spec']['dtype'] = 'float32'
        executor = build_executor(graph_engines(prepared), metadata=metadata)
        with self.assertRaisesRegex(ContractError, 'denoise.velocity.*dtype'):
            executor.infer(prepared.policy_inputs)

    def test_real_prefill_output_rejects_incorrect_metadata_shape(self):
        _, prepared = prepare_artifacts(self.directory)
        metadata = copy.deepcopy(prepared.metadata['runtime'])
        metadata['prefill_output_specs'][0]['shape'][1] += 2
        for spec in metadata['prefill_output_specs'][1:]:
            spec['shape'][2] += 2
        executor = build_executor(graph_engines(prepared), metadata=metadata)
        with self.assertRaisesRegex(ContractError, 'prefill.*dimension'):
            executor.infer(prepared.policy_inputs)

    def test_real_engine_overflow_is_rejected_without_sanitizing(self):
        _, prepared = prepare_artifacts(self.directory)
        module = prepared.graphs['denoise'].module
        parameters = dict(module.named_parameters())
        name = next(
            name for name in parameters
            if name.endswith('action_in_proj.projector.weight'))
        with torch.no_grad():
            parameters[name].fill_(torch.finfo(torch.float32).max)
        inputs = dict(prepared.policy_inputs)
        inputs['noise'] = torch.ones_like(inputs['noise'])
        executor = prepared.create_executor(graph_engines(prepared))
        with self.assertRaisesRegex(ContractError, 'denoise.velocity.*finite'):
            executor.infer(inputs)
        self.assertTrue(torch.isfinite(parameters[name]).all().item())


if __name__ == '__main__':
    unittest.main()
