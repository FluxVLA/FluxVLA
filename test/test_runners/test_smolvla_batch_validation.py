"""Check SmolVLA PPO batch geometry without launching distributed workers."""

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from fluxvla.engines.runners.rlinf_train_runner import (
    _validate_smolvla_batch_parameters, _validate_smolvla_batch_placement)


@pytest.fixture
def cfg():
    env = dict(
        total_num_envs=32,
        group_size=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=480)
    return OmegaConf.create(
        dict(
            runner=dict(
                task_type='embodied', only_eval=False, val_check_interval=10),
            rollout=dict(pipeline_stage_num=1),
            actor=dict(
                global_batch_size=64,
                micro_batch_size=2,
                model=dict(
                    num_action_chunks=10,
                    fluxvla=dict(rollout_micro_batch_size=4))),
            env=dict(train=env, eval=env)))


def placement(env_ranks, actor_ranks=None, rollout_ranks=1):
    sizes = {'env': env_ranks, 'actor': actor_ranks, 'rollout': rollout_ranks}
    return Mock(get_world_size=Mock(side_effect=sizes.__getitem__))


@pytest.mark.parametrize('env_ranks,actor_ranks,envs,micro', [(8, 8, 32, 2),
                                                              (32, 32, 32, 2),
                                                              (64, 64, 64, 1),
                                                              (4, 8, 32, 2)])
def test_valid_batch_geometry(cfg, env_ranks, actor_ranks, envs, micro):
    cfg.env.train.total_num_envs = cfg.env.eval.total_num_envs = envs
    cfg.actor.micro_batch_size = micro
    _validate_smolvla_batch_parameters(cfg)
    _validate_smolvla_batch_placement(
        cfg, placement(env_ranks, actor_ranks, rollout_ranks=env_ranks))


@pytest.mark.parametrize(
    'field,value', [('rollout.pipeline_stage_num', 0),
                    ('actor.micro_batch_size', 0),
                    ('actor.global_batch_size', -4),
                    ('actor.model.num_action_chunks', 0),
                    ('actor.model.fluxvla.rollout_micro_batch_size', 1.5),
                    ('env.eval.group_size', 0),
                    ('env.train.max_steps_per_rollout_epoch', False)])
def test_invalid_batch_parameter(cfg, field, value):
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        _validate_smolvla_batch_parameters(cfg)


@pytest.mark.parametrize('phase', ['train', 'eval'])
def test_pipeline_stage_remainder(cfg, phase):
    cfg.env[phase].total_num_envs = 12
    cfg.rollout.pipeline_stage_num = 2
    with pytest.raises(ValueError, match=f'env.{phase}.total_num_envs=12'):
        _validate_smolvla_batch_placement(cfg, placement(4, 4))


@pytest.mark.parametrize('phase', ['train', 'eval'])
def test_env_to_rollout_routing_uses_stage_batch(cfg, phase):
    cfg.rollout.pipeline_stage_num = 2
    cfg.env[phase].total_num_envs = 24
    ranks = placement(4, 4, rollout_ranks=8)
    with pytest.raises(
            ValueError,
            match=f'env.{phase}.*12 envs per stage.*rollout ranks'):
        _validate_smolvla_batch_placement(cfg, ranks)
    cfg.env[phase].total_num_envs = 32
    _validate_smolvla_batch_placement(cfg, ranks)


def test_env_to_actor_split_precedes_time_flattening(cfg):
    cfg.env.train.total_num_envs = 12
    with pytest.raises(ValueError, match='actor ranks'):
        _validate_smolvla_batch_placement(cfg, placement(4, 8))


def test_incomplete_ppo_batch(cfg):
    cfg.actor.global_batch_size = 320
    with pytest.raises(
            ValueError, match='48 rollout samples.*per-rank PPO batch 10'):
        _validate_smolvla_batch_placement(cfg, placement(32, 32))
    cfg.env.train.rollout_epoch = 5
    _validate_smolvla_batch_placement(cfg, placement(32, 32))


def test_inactive_phases(cfg):
    cfg.runner.val_check_interval = -1
    cfg.env.eval.total_num_envs = 0
    _validate_smolvla_batch_parameters(cfg)
    _validate_smolvla_batch_placement(cfg, placement(8, 8))
    cfg.runner.only_eval = True
    cfg.env.train.total_num_envs = 0
    cfg.env.eval.total_num_envs = 32
    _validate_smolvla_batch_parameters(cfg)
    _validate_smolvla_batch_placement(cfg, placement(8))
