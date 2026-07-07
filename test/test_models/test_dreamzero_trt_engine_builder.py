import importlib.util
import sys
import unittest
from pathlib import Path


def _load_builder_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / 'tools' / 'build_dreamzero_trt_engine.py'
    spec = importlib.util.spec_from_file_location(
        'build_dreamzero_trt_engine', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDreamZeroTrtEngineBuilder(unittest.TestCase):

    def setUp(self):
        self.module = _load_builder_module()
        self.head_cfg = {
            'frame_seqlen': 128,
            'num_frame_per_block': 2,
            'max_chunk_size': 2,
            'dit_dim': 5120,
            'dit_num_heads': 40,
            'dit_num_layers': 40,
            'dit_in_dim': 36,
            'dit_out_dim': 16,
            'action_horizon': 10,
            'max_action_dim': 32,
            'max_state_dim': 64,
            'num_state_per_block': 1,
        }

    def test_libero_cache_profile_shapes(self):
        spec = self.module.build_shape_spec(
            self.head_cfg, latent_height=32, latent_width=16)

        self.assertEqual(spec.min_shape_arg,
                         'kv_cache_packed:40x2x1x128x40x128')
        self.assertEqual(spec.opt_shape_arg,
                         'kv_cache_packed:40x2x1x384x40x128')
        self.assertEqual(spec.max_shape_arg,
                         'kv_cache_packed:40x2x1x640x40x128')
        self.assertEqual(spec.video_channels, 16)
        self.assertEqual(spec.cond_channels, 20)
        self.assertEqual(spec.action_horizon, 10)

    def test_trtexec_command_uses_only_kv_cache_dynamic_profile(self):
        spec = self.module.build_shape_spec(
            self.head_cfg, latent_height=32, latent_width=16)
        cmd = self.module.build_trtexec_command(
            trtexec='/opt/trt/bin/trtexec',
            onnx_path='/tmp/model.onnx',
            engine_path='/tmp/model.trt',
            spec=spec,
        )

        joined = ' '.join(cmd)
        self.assertIn('--onnx=/tmp/model.onnx', cmd)
        self.assertIn('--saveEngine=/tmp/model.trt', cmd)
        self.assertIn('--useCudaGraph', cmd)
        self.assertIn('--minShapes=kv_cache_packed:40x2x1x128x40x128',
                      cmd)
        self.assertIn('--optShapes=kv_cache_packed:40x2x1x384x40x128',
                      cmd)
        self.assertIn('--maxShapes=kv_cache_packed:40x2x1x640x40x128',
                      cmd)
        self.assertNotIn('--fp16', joined)
        self.assertNotIn('--bf16', joined)
        self.assertNotIn('--fp8', joined)

    def test_legacy_precision_flags_are_optional(self):
        spec = self.module.build_shape_spec(
            self.head_cfg, latent_height=32, latent_width=16)
        cmd = self.module.build_trtexec_command(
            trtexec='/opt/trt/bin/trtexec',
            onnx_path='/tmp/model.onnx',
            engine_path='/tmp/model.trt',
            spec=spec,
            legacy_precision_flags=True,
        )

        self.assertIn('--fp16', cmd)
        self.assertIn('--bf16', cmd)

    def test_latent_shape_must_match_frame_seqlen(self):
        with self.assertRaises(ValueError):
            self.module.build_shape_spec(
                self.head_cfg, latent_height=44, latent_width=80)


if __name__ == '__main__':
    unittest.main()
