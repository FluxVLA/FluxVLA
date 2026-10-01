"""User-facing launch and preparation tools work without private paths."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from fluxvla.rl.train import prepare_environment

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'scripts/rl/run.sh'


@pytest.fixture
def launch_env(tmp_path):
    prepare_environment()
    import rlinf
    capture = tmp_path / 'capture'
    capture.write_text(f'#!{sys.executable}\nimport json,sys\n'
                       'print(json.dumps(sys.argv[1:]))\n')
    capture.chmod(0o755)
    return dict(
        os.environ,
        TMUX='unit-test-session',
        RLINF_ROOT=str(Path(rlinf.__file__).resolve().parents[1]),
        FLUX_RL_PYTHON=str(capture),
        FLUX_RL_MODEL_PATH=str(tmp_path / 'weights with space=1.safetensors'),
        FLUX_RL_TOKENIZER_PATH=str(tmp_path / 'tokenizer'),
        FLUX_RL_STATS_PATH=str(tmp_path / 'stats.json'),
        FLUX_RL_RUN_NAME='unit_test',
        WANDB_API_KEY='unit-test-placeholder')


@pytest.mark.parametrize('mode', ['train', 'eval'])
def test_user_launcher_preserves_overrides_and_paths(launch_env, mode):
    result = subprocess.run([
        'bash',
        str(LAUNCHER), mode, 'benchmarks/libero/smolvla/ppo_8gpu',
        'runner.max_steps=7'
    ],
                            env=launch_env,
                            capture_output=True,
                            text=True,
                            timeout=15)
    assert result.returncode == 0, result.stderr
    assert launch_env['WANDB_API_KEY'] not in result.stdout + result.stderr
    args = json.loads(result.stdout)
    assert args[:3] == ['-u', '-m', f'fluxvla.rl.{mode}']
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        cfg = compose(config_name=args[3].split('=', 1)[1], overrides=args[4:])
    assert cfg.actor.model.model_path == launch_env['FLUX_RL_MODEL_PATH']
    assert cfg.runner.max_steps == 7
    assert cfg.runner.only_eval == (mode == 'eval')
    assert cfg.runner.logger.experiment_name == 'unit_test'


def test_launch_requires_tmux_but_inspection_does_not(launch_env):
    launch_env.pop('TMUX')
    command = ['bash', str(LAUNCHER), 'train', 'benchmarks/libero/smolvla/ppo']
    result = subprocess.run(
        command, env=launch_env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and 'tmux' in result.stderr
    result = subprocess.run(
        command + ['--cfg', 'job', '--resolve'],
        env=launch_env,
        capture_output=True,
        text=True,
        timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('script',
                         ['prepare_smolvla.py', 'prepare_robotwin.py'])
def test_preparation_help_is_available_without_download(script):
    result = subprocess.run(
        [sys.executable,
         str(ROOT / 'scripts/rl' / script), '--help'],
        capture_output=True,
        text=True,
        timeout=15)
    assert result.returncode == 0, result.stderr


def test_public_scripts_have_no_machine_specific_paths():
    for path in (ROOT / 'scripts/rl').iterdir():
        if path.is_file():
            source = path.read_text()
            assert '/mnt/data/cpfs/users/danny' not in source
            assert '/root/miniconda3' not in source


@pytest.mark.parametrize('config', ['../base/ppo', 'missing'])
def test_unknown_config_is_rejected(launch_env, config):
    result = subprocess.run(['bash', str(LAUNCHER), 'train', config],
                            env=launch_env,
                            capture_output=True,
                            text=True,
                            timeout=15)
    assert result.returncode == 2
