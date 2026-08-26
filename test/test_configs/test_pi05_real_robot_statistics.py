# Copyright 2026 Limx Dynamics

from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ('config_name', 'profile'),
    (
        ('pi05_paligemma_ur3_full_finetune.py', 'ur3'),
        ('pi05_paligemma_franka_dual_qpos_full_finetune.py', 'franka-qpos'),
        ('pi05_paligemma_franka_dual_eepose_full_finetune.py',
         'franka-eepose'),
        ('pi05_paligemma_tron2_full_finetune.py', 'tron2'),
    ),
)
def test_pi05_real_robot_configs_compute_transformed_statistics(
        config_name, profile):
    cfg = Config.fromfile(ROOT / 'configs/pi05' / config_name)
    dataset = cfg.train_dataloader.dataset

    assert 'dataset_statistics' not in dataset
    assert 'dataset_statistics_path' not in dataset
    assert dataset.auto_compute_statistics == dict(profile=profile)


def test_pi05_aloha_keeps_explicit_statistics():
    cfg = Config.fromfile(ROOT /
                          'configs/pi05/pi05_paligemma_aloha_full_finetune.py')
    dataset = cfg.train_dataloader.dataset

    assert 'dataset_statistics' in dataset
    assert 'auto_compute_statistics' not in dataset


def test_pi05_robocasa_keeps_explicit_statistics():
    cfg = Config.fromfile(
        ROOT /
        'configs/pi05/pi05_paligemma_robocasa_full_data_full_finetune.py')
    dataset = cfg.train_dataloader.dataset

    assert 'dataset_statistics' in dataset
    assert 'auto_compute_statistics' not in dataset
    stats = dataset.dataset_statistics[dataset.statistic_name]
    assert len(stats.proprio.q01) == 29
    assert len(stats.action.q01) == 29


def test_tron2_config_uses_16d_joint_delta_contract():
    cfg = Config.fromfile(ROOT /
                          'configs/pi05/pi05_paligemma_tron2_full_finetune.py')
    dataset = cfg.train_dataloader.dataset
    transform_types = [item.type for item in dataset.datasets[0].transforms]

    assert cfg.model.ori_action_dim == 16
    assert cfg.model.loss_action_dim == 32
    assert dataset.datasets[0].action_window_size == 50
    assert dataset.datasets[0].window_start_idx == 0
    assert dataset.datasets[0].supervise_terminal_padding is True
    assert 'RelativeActions' in transform_types
    relative = next(item for item in dataset.datasets[0].transforms
                    if item.type == 'RelativeActions')
    expected_mask = [True] * 7 + [False] + [True] * 7 + [False]
    assert relative.mask == expected_mask
    assert cfg.inference.denormalize_action.type == 'DenormalizeDeltaAction'
    assert cfg.inference.denormalize_action.action_dim == 16
    assert cfg.inference.denormalize_action.delta_action_mask == expected_mask
