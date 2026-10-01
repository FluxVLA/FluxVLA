"""Validate task isolation with actual RLinf LIBERO seed selection methods."""

from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.rlinf_registry import register
from fluxvla.rl.train import prepare_environment, validate_frontend_cfg


@pytest.mark.parametrize('only_eval', [False, True])
def test_single_task_config_and_actual_seed_selection(tiny_assets, monkeypatch,
                                                      only_eval):
    import rlinf.config as backend
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import get_benchmark_overridden

    root = prepare_environment()
    register()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(
            config_name='benchmarks/libero/smolvla/ppo_8gpu',
            overrides=[
                'runner.max_steps=50', 'runner.max_epochs=50',
                'runner.save_interval=5', 'runner.val_check_interval=5',
                '++env.train.task_id_filter=[6]',
                '++env.eval.task_id_filter=[6]', 'env.eval.rollout_epoch=5',
                f'actor.model.model_path={tiny_assets[2]}',
                f'actor.model.fluxvla.tokenizer_path={tiny_assets[3]}',
                'actor.model.fluxvla.norm_stats_path=/not-read.json',
                f'runner.only_eval={str(only_eval).lower()}'
            ])
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg, evaluation=only_eval)
    monkeypatch.setattr(backend, 'Cluster', lambda *a, **k: None)
    monkeypatch.setattr(
        backend, 'HybridComponentPlacement',
        lambda *a, **k: SimpleNamespace(get_world_size=lambda name: 4
                                        if name == 'actor' else 2))
    backend.validate_embodied_cfg(cfg)
    assert cfg.env.train.task_id_filter == cfg.env.eval.task_id_filter == [6]
    assert cfg.runner.max_steps == cfg.runner.max_epochs == 50
    assert cfg.runner.save_interval == cfg.runner.val_check_interval == 5
    assert cfg.runner.resume_dir is None
    assert cfg.actor.global_batch_size == 128
    assert cfg.actor.micro_batch_size == 2
    assert cfg.actor.optim.lr == 1e-6
    assert cfg.actor.optim.value_lr == 1e-4
    assert cfg.algorithm.update_epoch == 2
    assert cfg.actor.model.fluxvla.noise_level == 0.5
    assert cfg.env.train.total_num_envs == 32
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 50

    # Skip simulator construction, but exercise the backend's real methods.
    env = object.__new__(LiberoEnv)
    env.task_suite = get_benchmark_overridden('libero_10')()
    assert env.task_suite.get_task(6).name == (
        'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_'
        'chocolate_pudding_to_the_right_of_the_plate')
    env.task_id_filter = [6]
    env.specific_reset_id = None
    env._generator = np.random.default_rng(1234)
    env._compute_total_num_group_envs()
    train_ids = env._get_random_reset_state_ids(3200)
    tasks, trials = env._get_task_and_trial_ids_from_reset_state_ids(train_ids)
    assert set(tasks) == {6}
    assert min(trials) >= 0 and max(trials) < env.trial_id_bins[6]

    env.is_eval = True
    env.total_num_processes = 2
    pools = env.get_reset_state_ids_all()
    selected = []
    for rank in range(2):
        env._eval_reset_pool = pools[rank][pools[rank] >= 0]
        env.start_idx = 0
        for _ in range(cfg.env.eval.rollout_epoch):
            selected.extend(env._get_ordered_reset_state_ids(5).tolist())
    assert len(selected) == len(set(selected)) == 50
    tasks, trials = env._get_task_and_trial_ids_from_reset_state_ids(selected)
    assert set(tasks) == {6}
    assert sorted(trials) == list(range(50))
