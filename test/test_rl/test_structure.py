"""Model/benchmark boundaries and lazy optional dependencies stay explicit."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from fluxvla.rl.benchmarks.registry import get_benchmark_spec
from fluxvla.rl.utils.imports import resolve_symbol

ROOT = Path(__file__).resolve().parents[2]
RL = ROOT / 'fluxvla/rl'


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_no_flat_bridge_or_experiment_modules():
    assert not (RL / 'bridge').exists()
    assert {p.name
            for p in RL.glob('*.py')
            } == {'__init__.py', 'train.py', 'eval.py', 'rlinf_registry.py'}
    assert (RL / 'benchmarks/libero/adapter.py').is_file()
    assert (ROOT / 'scripts/rl/run.sh').is_file()


def test_models_do_not_import_specific_benchmarks():
    for path in (RL / 'models').rglob('*.py'):
        for name in imported_modules(path):
            assert not name.startswith(
                ('fluxvla.rl.benchmarks.libero',
                 'fluxvla.rl.benchmarks.robotwin')), path


@pytest.mark.parametrize('benchmark', ['libero', 'robotwin'])
def test_benchmark_adapters_are_independent(benchmark):
    path = RL / 'benchmarks' / benchmark / 'adapter.py'
    other = 'robotwin' if benchmark == 'libero' else 'libero'
    for name in imported_modules(path):
        assert not name.startswith('fluxvla.rl.models'), name
        assert not name.startswith(f'fluxvla.rl.benchmarks.{other}'), name


def test_benchmark_registry_is_lazy_in_fresh_worker():
    code = '''
import sys
from fluxvla.rl.benchmarks.registry import get_benchmark_spec
from fluxvla.rl.models.registry import get_policy_spec
assert get_benchmark_spec('robotwin').audit_actor
assert get_policy_spec('fluxvla_pi05').model_type == 'fluxvla_pi05'
assert 'rlinf' not in sys.modules
assert 'sapien' not in sys.modules
assert 'fluxvla.rl.benchmarks.robotwin.env' not in sys.modules
assert 'fluxvla.rl.benchmarks.libero.adapter' not in sys.modules
assert 'fluxvla.rl.models.pi05.policy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_registry_selects_benchmark_workers_and_runner():
    from fluxvla.rl.benchmarks.robotwin.worker import RoboTwinEnvWorker
    from fluxvla.rl.workers.env import FluxEnvWorker

    libero = get_benchmark_spec('libero')
    robotwin = get_benchmark_spec('robotwin')
    assert libero.get_environment_class(object) is object
    assert resolve_symbol(libero.env_worker_class) is FluxEnvWorker
    assert resolve_symbol(robotwin.env_worker_class) is RoboTwinEnvWorker
    assert robotwin.pair_eval_rng and not libero.pair_eval_rng
    assert not hasattr(FluxEnvWorker, 'memory_report')
    assert hasattr(RoboTwinEnvWorker, 'memory_report')
    assert resolve_symbol(robotwin.runner_class).__name__ == 'RoboTwinRunner'
    with pytest.raises(ValueError, match='Unknown benchmark'):
        get_benchmark_spec('not_registered')
