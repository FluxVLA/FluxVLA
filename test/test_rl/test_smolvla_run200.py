"""Validate the eight-GPU recipe and repeatable 100-episode monitor."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fluxvla.rl.train import prepare_environment, validate_frontend_cfg
from fluxvla.rl.workers.rollout import FluxRolloutWorker


@pytest.mark.parametrize('only_eval', [False, True])
def test_backend_train_and_eval_schema(tiny_assets, monkeypatch, only_eval):
    import rlinf.config as backend

    from fluxvla.rl.rlinf_registry import register

    root = prepare_environment()
    register()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(
            config_name='benchmarks/libero/smolvla/ppo_8gpu',
            overrides=[
                f'actor.model.model_path={tiny_assets[2]}',
                f'runner.only_eval={str(only_eval).lower()}'
            ])
    OmegaConf.resolve(cfg)
    assert cfg.rollout.model == cfg.actor.model
    monkeypatch.setattr(backend, 'Cluster', lambda *a, **k: None)
    monkeypatch.setattr(
        backend, 'HybridComponentPlacement',
        lambda *a, **k: SimpleNamespace(get_world_size=lambda name: 4
                                        if name == 'actor' else 2))
    backend.validate_embodied_cfg(cfg)


def test_recipe_and_actual_libero_seed_coverage(tiny_assets):
    from rlinf.envs.libero.utils import (
        build_interleaved_eval_reset_state_ids,
        distribute_reset_state_ids_round_robin, get_benchmark_overridden)

    root = prepare_environment()
    with initialize_config_dir(
            version_base='1.3', config_dir=str(root / 'configs/rl')):
        cfg = compose(
            config_name='benchmarks/libero/smolvla/ppo_8gpu',
            overrides=[
                f'actor.model.model_path={tiny_assets[2]}',
                f'actor.model.fluxvla.tokenizer_path={tiny_assets[3]}',
                'actor.model.fluxvla.norm_stats_path=/not-read.json'
            ])
    OmegaConf.resolve(cfg)
    validate_frontend_cfg(cfg)
    assert dict(cfg.cluster.component_placement) == dict(
        actor='0-3', rollout='4-5', env='6-7')
    assert cfg.runner.max_steps == cfg.runner.max_epochs == 200
    assert cfg.actor.global_batch_size == 128
    assert cfg.algorithm.update_epoch == 2
    assert cfg.env.train.total_num_envs == 32
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch == 100
    suite = get_benchmark_overridden('libero_10')()
    counts = [len(suite.get_task_init_states(i)) for i in range(10)]
    cumulative = np.cumsum(counts)
    pool = distribute_reset_state_ids_round_robin(
        build_interleaved_eval_reset_state_ids(counts, cumulative), 2)
    selected = pool[:, :50].flatten()
    assert len(set(selected)) == 100
    for task in range(10):
        start = 0 if task == 0 else cumulative[task - 1]
        trials = sorted(
            int(item - start) for item in selected
            if start <= item < cumulative[task])
        assert trials == list(range(10))
    assert Path(cfg.actor.model.model_path).exists()


def test_paired_monitor_rng_does_not_change_training_rng(monkeypatch):
    from rlinf.workers.rollout.hf.huggingface_worker import \
        MultiStepRolloutWorker

    async def evaluate(self, input_channel, output_channel):
        return torch.rand(4)

    monkeypatch.setattr(MultiStepRolloutWorker, 'evaluate', evaluate)
    worker = object.__new__(FluxRolloutWorker)
    worker.cfg = OmegaConf.create(
        dict(
            actor=dict(seed=1234), experiment=dict(paired_eval_seed=20260920)))
    worker.model_cfg = OmegaConf.create(
        dict(fluxvla=dict(observation_adapter='libero')))
    worker._rank = 0
    before = torch.random.get_rng_state().clone()
    first = asyncio.run(worker.evaluate(None, None))
    second = asyncio.run(worker.evaluate(None, None))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())
