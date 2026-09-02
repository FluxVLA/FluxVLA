import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

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
        engines.__path__ = []
        losses = types.ModuleType('fluxvla.engines.losses')
        utils = types.ModuleType('fluxvla.engines.utils')
        utils.__path__ = []
        fsdp_wrapping = types.ModuleType('fluxvla.engines.utils.fsdp_wrapping')
        engines.HEADS = RegistryStub()
        losses.reduce_action_bc_loss = lambda loss, sample_weight=None: loss
        fsdp_wrapping.build_module_wrap_policy = lambda *args, **kwargs: None
        old_modules = {
            name: sys.modules.get(name)
            for name in ('fluxvla', 'fluxvla.engines',
                         'fluxvla.engines.losses', 'fluxvla.engines.utils',
                         'fluxvla.engines.utils.fsdp_wrapping')
        }
        sys.modules['fluxvla'] = root
        sys.modules['fluxvla.engines'] = engines
        sys.modules['fluxvla.engines.losses'] = losses
        sys.modules['fluxvla.engines.utils'] = utils
        sys.modules['fluxvla.engines.utils.fsdp_wrapping'] = fsdp_wrapping
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
        out_video, out_action, updated_cache = (
            head._run_denoise_model_forward(
                update_cache=True,
                **self._make_inputs(),
            ))

        self.assertEqual(head.model.calls, 1)
        self.assertEqual(head.trt_engine.calls, 0)
        self.assertIsNotNone(updated_cache)
        torch.testing.assert_close(out_video, torch.ones_like(out_video))
        torch.testing.assert_close(out_action, torch.ones_like(out_action))

    def test_cached_denoise_uses_trt_engine(self):
        head = self._make_head()
        out_video, out_action, updated_cache = (
            head._run_denoise_model_forward(
                update_cache=False,
                **self._make_inputs(),
            ))

        self.assertEqual(head.model.calls, 0)
        self.assertEqual(head.trt_engine.calls, 1)
        self.assertIsNone(updated_cache)
        torch.testing.assert_close(out_video, torch.full_like(out_video, 2))
        torch.testing.assert_close(out_action, torch.full_like(out_action, 2))

    def test_official_dit_step_masks(self):
        DreamZeroHead = self._load_head_class()
        expected_indices = {
            5: (0, 1, 2, 7, 12),
            6: (0, 1, 5, 10, 14, 15),
            7: (0, 1, 2, 6, 10, 14, 15),
            8: (0, 1, 2, 6, 10, 13, 14, 15),
        }
        for compute_steps, indices in expected_indices.items():
            with self.subTest(compute_steps=compute_steps):
                mask = DreamZeroHead._build_dit_step_mask(16, compute_steps)
                self.assertEqual(
                    [i for i, enabled in enumerate(mask) if enabled],
                    list(indices))

    def test_dit_step_mask_without_skipping_runs_every_step(self):
        DreamZeroHead = self._load_head_class()
        self.assertEqual(
            DreamZeroHead._build_dit_step_mask(16, None), [True] * 16)
        self.assertEqual(
            DreamZeroHead._build_dit_step_mask(16, 16), [True] * 16)

    def test_invalid_dit_step_mask_configuration_is_rejected(self):
        DreamZeroHead = self._load_head_class()
        with self.assertRaisesRegex(ValueError, 'only defined'):
            DreamZeroHead._build_dit_step_mask(8, 5)
        for compute_steps in (0, 4, 9):
            with self.subTest(compute_steps=compute_steps), \
                    self.assertRaisesRegex(ValueError, 'Unsupported'):
                DreamZeroHead._build_dit_step_mask(16, compute_steps)

    def test_cfg_prediction_exchange_waits_for_all_requests(self):
        DreamZeroHead = self._load_head_class()
        head = object.__new__(DreamZeroHead)
        head.cfg_parallel = True
        head.cfg_scale = 5.0
        predictions = [(torch.zeros(1, 2), torch.ones(1, 3))]
        requests = [mock.Mock() for _ in range(4)]
        dist = DreamZeroHead._exchange_cfg_parallel_predictions.__globals__[
            'dist']

        with mock.patch.object(dist, 'is_available', return_value=True), \
                mock.patch.object(dist, 'is_initialized', return_value=True), \
                mock.patch.object(dist, 'get_world_size', return_value=2), \
                mock.patch.object(dist, 'get_rank', return_value=0), \
                mock.patch.object(dist, 'P2POp', side_effect=lambda *x: x), \
                mock.patch.object(dist, 'isend'), \
                mock.patch.object(dist, 'irecv'), \
                mock.patch.object(
                    dist, 'batch_isend_irecv', return_value=requests) as batch:
            exchanged = head._exchange_cfg_parallel_predictions(predictions)

        batch.assert_called_once()
        self.assertEqual(len(batch.call_args.args[0]), 4)
        for request in requests:
            request.wait.assert_called_once_with()
        self.assertIs(exchanged[0], predictions[0])
        self.assertEqual(len(exchanged), 2)


if __name__ == '__main__':
    unittest.main()
