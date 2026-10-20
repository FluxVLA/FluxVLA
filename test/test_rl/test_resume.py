"""Checkpoint resume and validation-selection contracts."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from fluxvla.engines.runners.rlinf.experiment import choose_best
from fluxvla.engines.runners.rlinf_train_runner import RLinfTrainRunner

ROOT = Path(__file__).resolve().parents[2]


def _robotwin_cfg(tiny_assets):
    RLinfTrainRunner.prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        return compose(
            config_name='default',
            overrides=[
                'simulator=robotwin_adjust_bottle',
                f'actor.model.model_path={tiny_assets[2]}'
            ])


def test_validation_best_survives_resume_boundary():
    records = [
        {
            'step': 20,
            'split': 'validation',
            'success_rate': .9
        },
        {
            'step': 70,
            'split': 'validation',
            'success_rate': .8
        },
    ]
    assert choose_best(records)['step'] == 20


def test_resume_rejects_missing_checkpoint(tiny_assets, tmp_path):
    cfg = _robotwin_cfg(tiny_assets)
    cfg.runner.resume_dir = str(tmp_path / 'missing')
    with pytest.raises(FileNotFoundError, match='Resume checkpoint'):
        RLinfTrainRunner.validate_frontend_cfg(cfg)


def test_robotwin_resume_batch_contract(tiny_assets, tmp_path):
    cfg = _robotwin_cfg(tiny_assets)
    checkpoint = tmp_path / 'global_step_10'
    checkpoint.mkdir()
    cfg.runner.resume_dir = str(checkpoint)
    RLinfTrainRunner.validate_frontend_cfg(cfg)
    assert cfg.actor.global_batch_size == 64
    assert cfg.env.train.total_num_envs == 32
