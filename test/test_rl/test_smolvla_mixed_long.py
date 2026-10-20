"""The long mixed-task run changes budget, not PPO or sampling semantics."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.train import prepare_environment, validate_frontend_cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('evaluation', [False, True])
def test_long_recipe_preserves_mixed_control(tiny_assets, evaluation):
    prepare_environment()
    overrides = [
        f'actor.model.model_path={tiny_assets[2]}',
        f'actor.model.fluxvla.tokenizer_path={tiny_assets[3]}',
        'actor.model.fluxvla.norm_stats_path=/not-read.json',
        f'runner.only_eval={str(evaluation).lower()}'
    ]
    with initialize_config_dir(
            version_base='1.3', config_dir=str(ROOT / 'configs/rl')):
        short = compose(
            config_name='benchmarks/libero/smolvla/ppo_8gpu',
            overrides=overrides)
        long = compose(
            config_name='benchmarks/libero/smolvla/ppo_8gpu',
            overrides=overrides + [
                'runner.max_steps=1000', 'runner.max_epochs=1000',
                'runner.save_interval=25', 'runner.val_check_interval=25'
            ])
    validate_frontend_cfg(long, evaluation=evaluation)
    assert long.runner.max_steps == long.runner.max_epochs == 1000
    assert long.runner.save_interval == long.runner.val_check_interval == 25
    assert long.runner.resume_dir is None
    assert long.env.train.get('task_id_filter') is None
    assert long.env.eval.total_num_envs * long.env.eval.rollout_epoch == 100
    # Undo the deliberately changed budget/log identity and require all
    # interpolated settings to match, including scheduler and video paths.
    for field in ('max_steps', 'max_epochs', 'save_interval',
                  'val_check_interval'):
        long.runner[field] = short.runner[field]
    long.runner.logger.experiment_name = short.runner.logger.experiment_name
    assert OmegaConf.to_container(long, resolve=True) == \
        OmegaConf.to_container(short, resolve=True)
