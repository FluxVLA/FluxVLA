import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def _load_cfg_parallel_module():
    module_path = (
        Path(__file__).resolve().parents[2] /
        'fluxvla/engines/utils/cfg_parallel.py')
    spec = importlib.util.spec_from_file_location(
        'dreamzero_cfg_parallel_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDreamZeroCfgParallelProtocol(unittest.TestCase):

    def setUp(self):
        self.module = _load_cfg_parallel_module()

    @staticmethod
    def _make_predict_kwargs(image_frames=4, with_embodiment=False):
        kwargs = dict(
            images=torch.zeros(1, 3, image_frames, 256, 128),
            lang_tokens=torch.zeros(1, 2, 512, dtype=torch.int64),
            lang_masks=torch.ones(1, 2, 512, dtype=torch.bool),
            states=torch.zeros(1, 32, dtype=torch.bfloat16),
            reset_history=True,
            num_inference_steps=16,
            img_masks=torch.ones(1, 2, dtype=torch.bool),
            image_grid_thw=torch.ones(1, 3, dtype=torch.int64),
            unnorm_key='libero_10_no_noops',
            seed=7,
        )
        if with_embodiment:
            kwargs['embodiment_ids'] = torch.zeros(1, dtype=torch.int32)
        return kwargs

    def test_predict_header_contains_only_model_tensor_inputs(self):
        kwargs = self._make_predict_kwargs(with_embodiment=True)
        header = self.module.build_dreamzero_cfg_header(
            self.module.CFG_PARALLEL_PREDICT, kwargs)
        command, reset_history, num_inference_steps, specs = (
            self.module.decode_dreamzero_cfg_header(header))

        self.assertEqual(command, self.module.CFG_PARALLEL_PREDICT)
        self.assertTrue(reset_history)
        self.assertEqual(num_inference_steps, 16)
        self.assertEqual(
            tuple(specs), ('images', 'lang_tokens', 'lang_masks', 'states',
                           'embodiment_ids'))
        self.assertEqual(specs['images'],
                         (torch.float32, torch.Size([1, 3, 4, 256, 128])))
        self.assertEqual(specs['lang_tokens'],
                         (torch.int64, torch.Size([1, 2, 512])))
        self.assertEqual(specs['lang_masks'],
                         (torch.bool, torch.Size([1, 2, 512])))
        self.assertEqual(specs['states'],
                         (torch.bfloat16, torch.Size([1, 32])))

    def test_dynamic_image_shape_is_encoded_in_fixed_size_header(self):
        first = self.module.build_dreamzero_cfg_header(
            self.module.CFG_PARALLEL_PREDICT,
            self._make_predict_kwargs(image_frames=1),
        )
        later = self.module.build_dreamzero_cfg_header(
            self.module.CFG_PARALLEL_PREDICT,
            self._make_predict_kwargs(image_frames=4),
        )
        first_specs = self.module.decode_dreamzero_cfg_header(first)[3]
        later_specs = self.module.decode_dreamzero_cfg_header(later)[3]

        self.assertEqual(first.numel(), self.module.CFG_PARALLEL_HEADER_SIZE)
        self.assertEqual(later.numel(), self.module.CFG_PARALLEL_HEADER_SIZE)
        self.assertEqual(first_specs['images'][1],
                         torch.Size([1, 3, 1, 256, 128]))
        self.assertEqual(later_specs['images'][1],
                         torch.Size([1, 3, 4, 256, 128]))

    def test_optional_embodiment_and_default_controls(self):
        kwargs = self._make_predict_kwargs()
        kwargs.pop('num_inference_steps')
        kwargs['reset_history'] = False
        header = self.module.build_dreamzero_cfg_header(
            self.module.CFG_PARALLEL_PREDICT, kwargs)
        _, reset_history, num_inference_steps, specs = (
            self.module.decode_dreamzero_cfg_header(header))

        self.assertFalse(reset_history)
        self.assertIsNone(num_inference_steps)
        self.assertNotIn('embodiment_ids', specs)

    def test_stop_header_has_no_tensor_payload(self):
        header = self.module.build_dreamzero_cfg_header(
            self.module.CFG_PARALLEL_STOP)
        command, reset_history, num_inference_steps, specs = (
            self.module.decode_dreamzero_cfg_header(header))

        self.assertEqual(command, self.module.CFG_PARALLEL_STOP)
        self.assertFalse(reset_history)
        self.assertIsNone(num_inference_steps)
        self.assertEqual(specs, {})

    def test_missing_required_tensor_is_rejected(self):
        kwargs = self._make_predict_kwargs()
        kwargs.pop('states')
        with self.assertRaisesRegex(KeyError, 'states'):
            self.module.build_dreamzero_cfg_header(
                self.module.CFG_PARALLEL_PREDICT, kwargs)

    def test_more_than_five_tensor_dimensions_is_rejected(self):
        kwargs = self._make_predict_kwargs()
        kwargs['states'] = torch.zeros(1, 1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, 'at most 5'):
            self.module.build_dreamzero_cfg_header(
                self.module.CFG_PARALLEL_PREDICT, kwargs)


class TestDreamZeroCfgParallelValidation(unittest.TestCase):

    def setUp(self):
        self.module = _load_cfg_parallel_module()

    def test_disabled_mode_accepts_any_model_and_world_size(self):
        self.module.validate_dreamzero_cfg_parallel(False, 'gr00t', 8)

    def test_dreamzero_requires_exactly_two_ranks(self):
        for world_size in (1, 4, 8):
            with self.subTest(world_size=world_size), self.assertRaisesRegex(
                    ValueError, 'exactly 2 ranks'):
                self.module.validate_dreamzero_cfg_parallel(
                    True, 'dreamzero', world_size)

    def test_non_dreamzero_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'only.*DreamZero'):
            self.module.validate_dreamzero_cfg_parallel(True, 'gr00t', 2)

    def test_dreamzero_with_two_ranks_is_accepted(self):
        self.module.validate_dreamzero_cfg_parallel(True, 'dreamzero', 2)


if __name__ == '__main__':
    unittest.main()
