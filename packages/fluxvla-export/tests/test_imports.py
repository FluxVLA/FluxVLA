import subprocess
import sys
import unittest
from pathlib import Path

import fluxvla_export


class ImportTest(unittest.TestCase):

    def test_public_surface_contains_only_handoff_contracts(self):
        self.assertEqual(
            set(fluxvla_export.__all__), {
                'ContractError',
                'FluxVLAExportError',
                'TensorLike',
                'TensorSpec',
                'ExportModule',
                'InferenceEngine',
                'DeploymentExecutor',
                'PreparedGraph',
                'PreparedDeployment',
            })
        for name in fluxvla_export.__all__:
            self.assertTrue(hasattr(fluxvla_export, name))
        for name in ('prepare_tensorrt', 'load_config', 'load_checkpoint',
                     'load_fixture', 'ExportRecipe'):
            self.assertFalse(hasattr(fluxvla_export, name))

    def test_import_does_not_load_model_or_backend_libraries(self):
        root = str(Path(fluxvla_export.__file__).resolve().parent.parent)
        code = (
            'import sys\n'
            f'sys.path.insert(0, {root!r})\n'
            'from fluxvla_export import TensorSpec\n'
            "assert TensorSpec('x', 'float32', (2, 3)).shape == (2, 3)\n"
            "assert not any(name in sys.modules for name in "
            "('torch', 'transformers', 'safetensors', 'fluxvla', 'fluxhermes', "
            "'mmengine', 'onnx', 'tensorrt'))\n")
        result = subprocess.run([sys.executable, '-I', '-c', code],
                                capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
