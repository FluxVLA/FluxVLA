import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_wan21_backbone_module():

    class RegistryStub:

        @staticmethod
        def register_module():
            return lambda cls: cls

    base_module_name = 'fluxvla.models.backbones.vlms.wan_backbone'
    wan21_module_name = 'fluxvla.models.backbones.vlms.wan21_backbone'
    package_names = (
        'fluxvla',
        'fluxvla.models',
        'fluxvla.models.backbones',
        'fluxvla.models.backbones.vlms',
    )
    module_names = package_names + (
        'fluxvla.engines',
        base_module_name,
        wan21_module_name,
    )
    old_modules = {name: sys.modules.get(name) for name in module_names}
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    engines = types.ModuleType('fluxvla.engines')
    engines.VLM_BACKBONES = RegistryStub()
    sys.modules['fluxvla.engines'] = engines

    vlm_dir = (
        Path(__file__).resolve().parents[2] / 'fluxvla/models/backbones/vlms')
    base_spec = importlib.util.spec_from_file_location(
        base_module_name, vlm_dir / 'wan_backbone.py')
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules[base_module_name] = base_module
    assert base_spec.loader is not None
    base_spec.loader.exec_module(base_module)

    spec = importlib.util.spec_from_file_location(
        wan21_module_name, vlm_dir / 'wan21_backbone.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[wan21_module_name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


def _load_attention_module():
    optional_modules = ('flash_attn', 'flash_attn_interface',
                        'transformer_engine')
    old_modules = {name: sys.modules.get(name) for name in optional_modules}
    for name in optional_modules:
        sys.modules[name] = None

    module_path = Path(__file__).resolve().parents[2] / (
        'fluxvla/models/third_party_models/dreamzero/modules/'
        'wan2_1_attention.py')
    spec = importlib.util.spec_from_file_location('wan_attention_under_test',
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


class TestWanEncoderCompileOptions(unittest.TestCase):

    def setUp(self):
        self.module = _load_wan21_backbone_module()
        self.backbone = object.__new__(self.module.Wan21Backbone)
        self.backbone.torch_compile_mode = 'reduce-overhead'
        self.backbone.torch_compile_fullgraph = True
        self.backbone.torch_compile_dynamic = False
        self.backbone._torch_compile_applied = False
        self.backbone.text_encoder = types.SimpleNamespace(
            forward=mock.Mock(name='text_forward'))
        self.backbone.image_encoder = types.SimpleNamespace(
            model=types.SimpleNamespace(
                visual=types.SimpleNamespace(
                    forward=mock.Mock(name='image_forward'))))
        self.backbone.vae = types.SimpleNamespace(
            model=types.SimpleNamespace(encode=mock.Mock(name='vae_encode')))

    def test_encoder_compile_is_disabled_by_default(self):
        self.backbone.use_torch_compile_encoders = False
        with mock.patch.object(self.module.torch, 'compile') as compile_mock:
            self.backbone._compile_encoder_modules_if_configured()

        compile_mock.assert_not_called()
        self.assertFalse(self.backbone._torch_compile_applied)

    def test_enabled_option_wraps_exactly_three_encoder_entries(self):
        self.backbone.use_torch_compile_encoders = True

        def compile_stub(**kwargs):
            self.assertEqual(
                kwargs,
                dict(mode='reduce-overhead', fullgraph=True, dynamic=False))
            return lambda function: mock.Mock(wraps=function)

        with mock.patch.dict(os.environ, {'ENABLE_TENSORRT': 'False'}), \
                mock.patch.object(
                    self.module.torch,
                    'compile',
                    side_effect=compile_stub) as compile_mock:
            self.backbone._compile_encoder_modules_if_configured()

        self.assertEqual(compile_mock.call_count, 3)
        self.assertTrue(self.backbone._torch_compile_applied)

    def test_tensorrt_mode_does_not_compile_encoders(self):
        self.backbone.use_torch_compile_encoders = True
        with mock.patch.dict(os.environ, {'ENABLE_TENSORRT': 'true'}), \
                mock.patch.object(self.module.torch,
                                  'compile') as compile_mock:
            self.backbone._compile_encoder_modules_if_configured()

        compile_mock.assert_not_called()
        self.assertFalse(self.backbone._torch_compile_applied)


class TestDreamZeroAttentionBackendOptions(unittest.TestCase):

    def setUp(self):
        self.module = _load_attention_module()

    def _make_attention(self, environment):
        with mock.patch.dict(os.environ, environment, clear=True):
            return self.module.AttentionModule(num_heads=2, head_dim=8)

    def test_default_backend_remains_fa2_without_runtime_log(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            attention = self._make_attention({})

        self.assertEqual(attention.backend, 'FA2')
        self.assertEqual(output.getvalue(), '')

    def test_explicit_flash_and_torch_backends_are_preserved(self):
        for backend in ('FA2', 'FA3', 'torch', 'torch_onnx'):
            with self.subTest(backend=backend):
                attention = self._make_attention(
                    {'ATTENTION_BACKEND': backend})
                self.assertEqual(attention.backend, backend)

    def test_tensorrt_forces_export_compatible_torch_backend(self):
        attention = self._make_attention({
            'ATTENTION_BACKEND': 'FA3',
            'ENABLE_TENSORRT': 'true',
        })
        self.assertEqual(attention.backend, 'torch')

    def test_unavailable_transformer_engine_falls_back_to_fa2(self):
        with contextlib.redirect_stdout(io.StringIO()):
            attention = self._make_attention({'ATTENTION_BACKEND': 'TE'})
        self.assertEqual(attention.backend, 'FA2')


if __name__ == '__main__':
    unittest.main()
