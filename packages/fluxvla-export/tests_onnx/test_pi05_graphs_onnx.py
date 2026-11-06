# Copyright 2026 Limx Dynamics
"""Mandatory real ONNX/ORT checks, invoked separately from torch-only tests.

Run with the export extra installed and ``PYTHONPATH=tests`` from the package
root. Missing ONNX/ORT dependencies are errors in this suite, never skips.
"""

import tempfile
import unittest
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import (NotImplemented as
                                                         OrtNotImplemented)

from fluxvla_export.components.euler import plain_euler
from fluxvla_export.recipes.pi05.graphs import PI05DenoiseGraph, PI05PrefillGraph
from _pi05_test_utils import Artifacts
from test_config import config_document


class PI05GraphOnnxTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def artifacts(self,
                  *,
                  amp=False,
                  flow=False,
                  protected=False,
                  backend='sdpa',
                  bf16_inputs=False,
                  mixed_cache=False):
        document = config_document()
        for branch in ('backbone', 'expert'):
            document['model'][branch]['num_hidden_layers'] = 2
        document['precision'].update(
            autocast_dtype='bfloat16' if amp else None,
            flow_fp32=flow,
            residual_fp32=protected and not mixed_cache,
            vision_stem_fp32=protected,
            vision_residual_fp32=protected,
            adarms_fp32=protected,
            attention_math_fp32=protected and backend == 'sdpa',
            backbone_fp32_layers=[0] if protected else [],
            expert_fp32_layers=[1] if protected else [],
            attention_backend=backend)
        if bf16_inputs:
            for name in ('images', 'states'):
                document['inputs']['tensors'][name]['dtype'] = 'bfloat16'
        return Artifacts(self.directory, document)

    def export(self, module, args):
        path = self.directory / (module.name + '.onnx')
        options = (
            dict(opset_version=22, dynamo=True, optimize=True)
            if module.autocast_enabled else dict(
                opset_version=18, dynamo=False, do_constant_folding=False))
        with torch.no_grad():
            program = torch.onnx.export(
                module,
                args,
                path,
                input_names=list(module.input_names),
                output_names=list(module.output_names),
                **options)
            if module.autocast_enabled:
                # Check the real captured tensor program as well as ONNX
                # schema validity; BF16 CPU ORT kernel coverage is separate.
                captured = program.exported_program.module()(*args)
                expected = module(*args)
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        onnx.checker.check_model(str(path), full_check=True)
        graph = onnx.load(str(path)).graph
        self.assertEqual(
            tuple(value.name for value in graph.input), module.input_names)
        self.assertEqual(
            tuple(value.name for value in graph.output), module.output_names)
        if module.autocast_enabled:
            initializers = {value.name for value in graph.initializer}
            reductions = [node for node in graph.node
                          if node.op_type.startswith('Reduce')]
            self.assertTrue(reductions)
            for node in reductions:
                # TensorRT 10.13 rejects even literal axes behind Reshape.
                self.assertGreaterEqual(len(node.input), 2, node.name)
                self.assertIn(node.input[1], initializers, node.name)
        return path, graph

    def session(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        return ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=['CPUExecutionProvider'])

    def infer(self, session, names, args):
        return tuple(
            torch.from_numpy(value) for value in session.run(
                None, {
                    name: value.detach().cpu().numpy()
                    for name, value in zip(names, args)
                }))

    def assert_close(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            expected = (expected, )
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected):
            self.assertEqual(value.dtype, target.dtype)
            torch.testing.assert_close(value, target, rtol=1e-4, atol=1e-5)

    def test_real_fp32_prefill_denoise_and_euler_with_changed_inputs(self):
        for backend, protected in (('sdpa', False), ('sdpa', True), ('eager',
                                                                     False)):
            with self.subTest(backend=backend, protected=protected):
                artifacts = self.artifacts(
                    backend=backend, protected=protected, flow=protected)
                reference = artifacts.prepare()
                model, inputs = reference.model, reference.inputs
                prefill = PI05PrefillGraph(model)
                denoise = PI05DenoiseGraph(
                    model, states_dtype=inputs['states'].dtype)
                pargs = tuple(inputs[name] for name in prefill.input_names)
                with torch.no_grad():
                    flat = prefill(*pargs)
                dargs = (inputs['noise'], torch.tensor([0.25, 0.75]), *flat)
                prefill_path, _ = self.export(prefill, pargs)
                denoise_path, _ = self.export(denoise, dargs)
                p_session, d_session = (self.session(prefill_path),
                                        self.session(denoise_path))
                variants = [dict(inputs)]
                changed = {
                    name: value.clone()
                    for name, value in inputs.items()
                }
                changed['images'] = changed['images'] * 0.4 + 0.1
                changed['img_masks'] = changed['img_masks'].flip(1)
                changed['lang_masks'] = changed['lang_masks'].flip(1)
                changed['lang_tokens'] = (changed['lang_tokens'] + 3) % (
                    reference.config.model.vocab_size)
                changed['noise'] = changed['noise'].flip(1) * -0.75
                variants.append(changed)
                for fixture in variants:
                    with torch.no_grad():
                        pa = tuple(fixture[name]
                                   for name in prefill.input_names)
                        cache = self.infer(p_session, prefill.input_names, pa)
                        self.assert_close(cache, prefill(*pa))
                        before_cache = tuple(value.clone() for value in cache)
                        time = torch.tensor([0.1, 0.9])
                        velocity = self.infer(d_session, denoise.input_names,
                                              (fixture['noise'], time, *cache))
                        self.assert_close(
                            velocity, denoise(fixture['noise'], time, *cache))
                        noise = fixture['noise'].clone()
                        actual = plain_euler(
                            noise, lambda x, t: self.infer(
                                d_session, denoise.input_names,
                                (x, t, *cache))[0], noise.shape[0],
                            torch.tensor(-1.0 / model.num_steps), noise.device)
                        expected = model(**fixture)
                    self.assert_close((actual, ), expected)
                    for value, before in zip(cache, before_cache):
                        torch.testing.assert_close(
                            value, before, rtol=0, atol=0)

    def test_bf16_cpu_provider_rejects_only_the_known_missing_conv_kernel(
            self):
        artifacts = self.artifacts(amp=True)
        prefill = PI05PrefillGraph(artifacts.model)
        args = tuple(artifacts.tensors[name] for name in prefill.input_names)
        # First prove that this is a valid, native-BF16 graph. Schema/export
        # failures must never be confused with runtime kernel availability.
        path, _ = self.export(prefill, args)
        with self.assertRaisesRegex(
                OrtNotImplemented,
                r'Could not find an implementation for Conv\(22\)'):
            self.session(path)

    def assert_native_bf16_exports(self, artifacts):
        # ORT CPU does not supply every BF16 kernel. Check actual export,
        # full schema, captured-program values and native tensor dtypes.
        dtypes = {
            torch.bool: onnx.TensorProto.BOOL,
            torch.int64: onnx.TensorProto.INT64,
            torch.float32: onnx.TensorProto.FLOAT,
            torch.bfloat16: onnx.TensorProto.BFLOAT16
        }
        model, inputs = artifacts.model, artifacts.tensors
        prefill = PI05PrefillGraph(model)
        denoise = PI05DenoiseGraph(model, states_dtype=inputs['states'].dtype)
        state = {
            name: value.clone()
            for name, value in model.state_dict().items()
        }
        pa = tuple(inputs[name] for name in prefill.input_names)
        with torch.no_grad():
            cache = prefill(*pa)
            da = (inputs['noise'], torch.tensor([0.25, 0.75]), *cache)
            velocity = denoise(*da)
        for module, args, outputs in ((prefill, pa, cache), (denoise, da,
                                                             (velocity, ))):
            _, graph = self.export(module, args)
            self.assertEqual(
                tuple(value.type.tensor_type.elem_type
                      for value in graph.input),
                tuple(dtypes[value.dtype] for value in args))
            self.assertEqual(
                tuple(value.type.tensor_type.elem_type
                      for value in graph.output),
                tuple(dtypes[value.dtype] for value in outputs))
        for name, value in model.state_dict().items():
            self.assertEqual(value.dtype, torch.float32)
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        return tuple(value.dtype for value in cache[1:]), velocity.dtype

    def test_bf16_exports_retain_native_precision_boundaries_and_fp32_masters(
            self):
        for protected in (False, True):
            for flow in (False, True):
                with self.subTest(protected=protected, flow=flow):
                    self.assert_native_bf16_exports(
                        self.artifacts(
                            amp=True, protected=protected, flow=flow))

    def test_bf16_storage_widens_before_native_bf16_graph_export(self):
        artifacts = self.artifacts(amp=True, protected=True, flow=True)
        artifacts.document['checkpoint']['storage_dtype'] = 'bfloat16'
        stored = {name: value.detach().bfloat16()
                  if value.is_floating_point() else value.detach().clone()
                  for name, value in artifacts.model.state_dict().items()}
        artifacts.save_checkpoint(stored)
        before = artifacts.checkpoint_path.read_bytes()
        reference = artifacts.prepare()
        for name, value in reference.model.state_dict().items():
            expected = (stored[name].float() if stored[name].is_floating_point()
                        else stored[name])
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        artifacts.model = reference.model
        self.assert_native_bf16_exports(artifacts)
        self.assertEqual(artifacts.checkpoint_path.read_bytes(), before)

    def test_bf16_images_and_states_export_with_native_flow(self):
        artifacts = self.artifacts(amp=True, bf16_inputs=True, flow=False)
        self.assertEqual(artifacts.tensors['images'].dtype, torch.bfloat16)
        self.assertEqual(artifacts.tensors['states'].dtype, torch.bfloat16)
        cache_dtypes, velocity_dtype = self.assert_native_bf16_exports(
            artifacts)
        print('BF16 images/states:', cache_dtypes, 'velocity:', velocity_dtype)

    def test_bf16_eager_attention_exports_without_fp32_math(self):
        artifacts = self.artifacts(amp=True, backend='eager', protected=False)
        self.assertEqual(artifacts.model.attention_implementation, 'eager')
        self.assertFalse(
            artifacts.model.llm_backbone.config.attention_math_fp32)
        self.assertFalse(artifacts.model.llm_expert.config.attention_math_fp32)
        cache_dtypes, velocity_dtype = self.assert_native_bf16_exports(
            artifacts)
        print('BF16 eager:', cache_dtypes, 'velocity:', velocity_dtype)

    def test_bf16_mixed_per_layer_cache_exports_without_residual_promotion(
            self):
        artifacts = self.artifacts(
            amp=True, flow=True, protected=True, mixed_cache=True)
        self.assertFalse(artifacts.model.preserve_fp32_residuals)
        cache_dtypes, velocity_dtype = self.assert_native_bf16_exports(
            artifacts)
        self.assertEqual(
            cache_dtypes,
            (torch.float32, torch.float32, torch.bfloat16, torch.bfloat16))
        print('BF16 mixed cache:', cache_dtypes, 'velocity:', velocity_dtype)


if __name__ == '__main__':
    unittest.main()
