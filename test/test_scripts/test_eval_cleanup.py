import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_eval_module():

    class ConfigStub:
        pass

    class DictActionStub:
        pass

    mmengine = types.ModuleType('mmengine')
    mmengine.Config = ConfigStub
    mmengine.DictAction = DictActionStub

    engines = types.ModuleType('fluxvla.engines')
    engines.build_runner_from_cfg = mock.Mock()
    engines.initialize_overwatch = mock.Mock(return_value=mock.Mock())
    feishu_reporter = types.ModuleType('fluxvla.engines.utils.feishu_reporter')
    feishu_reporter.maybe_report_summary_to_feishu = mock.Mock()
    torch_utils = types.ModuleType('fluxvla.engines.utils.torch_utils')
    torch_utils.configure_inference_attention_defaults = mock.Mock()

    module_names = (
        'mmengine',
        'fluxvla',
        'fluxvla.engines',
        'fluxvla.engines.utils',
        'fluxvla.engines.utils.feishu_reporter',
        'fluxvla.engines.utils.torch_utils',
    )
    old_modules = {name: sys.modules.get(name) for name in module_names}
    sys.modules['mmengine'] = mmengine
    sys.modules['fluxvla'] = types.ModuleType('fluxvla')
    sys.modules['fluxvla.engines'] = engines
    sys.modules['fluxvla.engines.utils'] = types.ModuleType(
        'fluxvla.engines.utils')
    sys.modules['fluxvla.engines.utils.feishu_reporter'] = feishu_reporter
    sys.modules['fluxvla.engines.utils.torch_utils'] = torch_utils

    module_path = Path(__file__).resolve().parents[2] / 'scripts/eval.py'
    spec = importlib.util.spec_from_file_location('eval_under_test',
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


class TestEvalDistributedCleanup(unittest.TestCase):

    def setUp(self):
        self.module = _load_eval_module()

    def test_initialized_process_group_is_destroyed(self):
        with mock.patch.object(
                self.module.dist, 'is_available',
                return_value=True), mock.patch.object(
                    self.module.dist, 'is_initialized',
                    return_value=True), mock.patch.object(
                        self.module.dist, 'destroy_process_group') as destroy:
            self.module._destroy_distributed_process_group()

        destroy.assert_called_once_with()

    def test_uninitialized_process_group_is_ignored(self):
        with mock.patch.object(
                self.module.dist, 'is_available',
                return_value=True), mock.patch.object(
                    self.module.dist, 'is_initialized',
                    return_value=False), mock.patch.object(
                        self.module.dist, 'destroy_process_group') as destroy:
            self.module._destroy_distributed_process_group()

        destroy.assert_not_called()

    def test_main_cleans_up_when_eval_raises(self):
        failure = RuntimeError('eval failed')
        with mock.patch.object(
                self.module, '_run_main',
                side_effect=failure), mock.patch.object(
                    self.module,
                    '_destroy_distributed_process_group') as cleanup:
            with self.assertRaisesRegex(RuntimeError, 'eval failed'):
                self.module.main()

        cleanup.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
