import unittest
import importlib.util
import sys
import types
from pathlib import Path

import torch


class DummyTorchModel:

    def __init__(self):
        self.calls = 0

    def __call__(self, reference_latents, **kwargs):
        self.calls += 1
        return (
            reference_latents + 1,
            kwargs['action'] + 1,
            [torch.ones_like(kwargs['kv_cache'][0])],
        )


class DummyTrtEngine:

    def __init__(self):
        self.calls = 0

    def __call__(self, reference_latents, **kwargs):
        self.calls += 1
        assert 'kv_cache' in kwargs
        assert 'context' in kwargs
        return reference_latents + 2, kwargs['action'] + 2


class TestDreamZeroTensorRTRuntime(unittest.TestCase):

    @staticmethod
    def _load_head_class():
        class RegistryStub:

            @staticmethod
            def register_module():
                def decorator(cls):
                    return cls
                return decorator

        root = types.ModuleType('fluxvla')
        engines = types.ModuleType('fluxvla.engines')
        losses = types.ModuleType('fluxvla.engines.losses')
        engines.HEADS = RegistryStub()
        losses.reduce_action_bc_loss = lambda loss, sample_weight=None: loss
        old_modules = {
            name: sys.modules.get(name)
            for name in ('fluxvla', 'fluxvla.engines',
                         'fluxvla.engines.losses')
        }
        sys.modules['fluxvla'] = root
        sys.modules['fluxvla.engines'] = engines
        sys.modules['fluxvla.engines.losses'] = losses
        module_path = (
            Path(__file__).resolve().parents[2] /
            'fluxvla/models/heads/dreamzero_head.py')
        spec = importlib.util.spec_from_file_location(
            'dreamzero_head_under_test', module_path)
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
        return module.DreamZeroHead

    def _make_head(self):
        DreamZeroHead = self._load_head_class()
        head = object.__new__(DreamZeroHead)
        head.model = DummyTorchModel()
        head.trt_engine = DummyTrtEngine()
        head.trt_engine_path = '/tmp/fake.trt'
        return head

    def _make_inputs(self):
        reference_latents = torch.zeros(1, 16, 2, 8, 8)
        action = torch.zeros(1, 10, 32)
        kv_cache = [torch.zeros(2, 1, 4, 2, 8)]
        crossattn_cache = [torch.zeros(2, 1, 512, 2, 8)]
        return dict(
            reference_latents=reference_latents,
            timestep=torch.zeros(1, 2),
            clip_feas=torch.zeros(1, 257, 1280),
            ys=torch.zeros(1, 20, 2, 8, 8),
            prompt_emb=torch.zeros(1, 512, 4096),
            frame_seqlen=16,
            action=action,
            timestep_action=torch.zeros(1, 10),
            state=torch.zeros(1, 1, 64),
            embodiment_id=torch.zeros(1, dtype=torch.long),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            start_frame=1,
        )

    def test_cache_update_uses_torch_model(self):
        head = self._make_head()
        out_video, out_action, updated_cache, used_trt = (
            head._run_denoise_model_forward(
                update_cache=True,
                **self._make_inputs(),
            ))

        self.assertFalse(used_trt)
        self.assertEqual(head.model.calls, 1)
        self.assertEqual(head.trt_engine.calls, 0)
        self.assertIsNotNone(updated_cache)
        torch.testing.assert_close(out_video, torch.ones_like(out_video))
        torch.testing.assert_close(out_action, torch.ones_like(out_action))

    def test_cached_denoise_uses_trt_engine(self):
        head = self._make_head()
        out_video, out_action, updated_cache, used_trt = (
            head._run_denoise_model_forward(
                update_cache=False,
                **self._make_inputs(),
            ))

        self.assertTrue(used_trt)
        self.assertEqual(head.model.calls, 0)
        self.assertEqual(head.trt_engine.calls, 1)
        self.assertIsNone(updated_cache)
        torch.testing.assert_close(out_video, torch.full_like(out_video, 2))
        torch.testing.assert_close(out_action, torch.full_like(out_action, 2))


if __name__ == '__main__':
    unittest.main()
