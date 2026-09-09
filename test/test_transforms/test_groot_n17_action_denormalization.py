# Copyright 2026 Limx Dynamics

import numpy as np
import pytest
from mmengine import Config

from fluxvla.engines.runners.robocasa_eval_runner import RobocasaEvalRunner
from fluxvla.transforms.normalize import DenormalizeDeltaAction

_CONFIG = ('configs/gr00tn17/'
           'gr00tn17_qwen3vl_2b_robocasa_full_finetune.py')


def _build_denormalizer(cfg, action_stats_key):
    statistic_name = cfg._STATISTIC_NAME
    return DenormalizeDeltaAction(
        norm_stats={
            statistic_name: {
                action_stats_key:
                cfg._DATASET_STATISTICS[statistic_name].actions.to_dict()
            }
        },
        statistic_name=statistic_name,
        norm_type='min_max',
        action_dim=29,
        delta_action_mask=[True] * 26 + [False] * 3,
        state_permutation=cfg._N17_DIMENSION_PERMUTATION,
        normalize_gripper_action=False,
        invert_gripper_action=False,
    )


def test_n17_robocasa_uses_indexed_action_denormalization():
    cfg = Config.fromfile(_CONFIG)

    assert cfg.eval.eval_chunk_size == 8
    assert 'denormalize_action_chunk' not in cfg.eval


@pytest.mark.parametrize('action_stats_key', ['action', 'actions'])
@pytest.mark.parametrize('chunk_size', [1, 4, 8])
def test_indexed_per_action_denormalization_matches_chunk(
        action_stats_key, chunk_size):
    cfg = Config.fromfile(_CONFIG)
    denormalizer = _build_denormalizer(cfg, action_stats_key)
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1.0, 1.0, size=(8, 132)).astype(np.float32)
    raw_state = rng.uniform(-0.5, 0.5, size=29).astype(np.float32)

    chunk_result = denormalizer({
        'action': actions,
        'state': raw_state,
    })[:chunk_size]
    indexed_result = np.stack([
        denormalizer({
            'action': action,
            'action_horizon_index': chunk_index,
            'state': raw_state,
        }) for chunk_index, action in enumerate(actions[:chunk_size])
    ])

    np.testing.assert_array_equal(indexed_result, chunk_result)


@pytest.mark.parametrize('action_stats_key', ['action', 'actions'])
def test_indexed_denormalization_rejects_stats_horizon_overflow(
        action_stats_key):
    cfg = Config.fromfile(_CONFIG)
    denormalizer = _build_denormalizer(cfg, action_stats_key)

    with pytest.raises(
            IndexError, match=r'action_horizon_index=8.*statistics horizon=8'):
        denormalizer({
            'action': np.zeros(132, dtype=np.float32),
            'action_horizon_index': 8,
            'state': np.zeros(29, dtype=np.float32),
        })


def test_action_horizon_validation_accepts_supported_chunks():
    for eval_chunk_size in (1, 4, 8):
        RobocasaEvalRunner._validate_action_horizons(
            eval_chunk_size=eval_chunk_size,
            model_output_horizon=40,
            action_stats_horizon=8,
        )


def test_action_horizon_validation_rejects_stats_overflow():
    with pytest.raises(
            ValueError,
            match=(r'eval_chunk_size=9 exceeds action statistics horizon=8, '
                   r'while model output horizon=40 is sufficient')):
        RobocasaEvalRunner._validate_action_horizons(
            eval_chunk_size=9,
            model_output_horizon=40,
            action_stats_horizon=8,
        )


def test_action_horizon_validation_rejects_model_overflow():
    with pytest.raises(
            ValueError,
            match=(r'eval_chunk_size=41 exceeds model output horizon=40.*'
                   r'increase the model action_horizon and retrain/adapt')):
        RobocasaEvalRunner._validate_action_horizons(
            eval_chunk_size=41,
            model_output_horizon=40,
            action_stats_horizon=8,
        )
