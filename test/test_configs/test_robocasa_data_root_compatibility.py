# Copyright 2026 Limx Dynamics
"""Keep legacy RoboCasa paths stable without leaking them into new recipes."""

import copy
import os
from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
LEGACY_CONFIGS = [
    'pi05/pi05_paligemma_robocasa_full_data_full_finetune.py',
    'pi0/pi0_paligemma_robocasa_full_data_full_finetune.py',
]
MIXED_CONFIG = 'pi05/pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py'
LOCAL_ROOT = './datasets/robocasa_lerobot_V2.1'
SHARED_ROOT = ('/mnt/data/cpfs/mnt/data/yiming/fluxvla/upload_staging/'
               'robocasa_lerobot_V2.1')
LOCAL_MARKER = Path(LOCAL_ROOT) / 'PnPBottleToCabinetClose/videos'
ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'


@pytest.fixture(autouse=True)
def isolated_data_environment(monkeypatch, tmp_path):
    monkeypatch.delenv('ROBOCASA_DATA_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)


def assert_task_roots(cfg, expected_root):
    # Preserve all 24 task names and their order, not just the parent path.
    tasks = [
        task.split('/')[-1].removesuffix(ENV_SUFFIX)
        for task in cfg.eval.task_list
    ]
    assert len(tasks) == len(set(tasks)) == 24
    assert cfg.train_dataloader.dataset.datasets.data_root_path == [
        f'{expected_root}/{task}' for task in tasks
    ]


@pytest.mark.parametrize('config', LEGACY_CONFIGS)
@pytest.mark.parametrize('local_state',
                         ['missing', 'incomplete', 'marker_is_file', 'ready'])
def test_legacy_robocasa_default_root(config, local_state):
    if local_state == 'incomplete':
        LOCAL_MARKER.parent.mkdir(parents=True)
    elif local_state == 'marker_is_file':
        LOCAL_MARKER.parent.mkdir(parents=True)
        LOCAL_MARKER.touch()
    elif local_state == 'ready':
        LOCAL_MARKER.mkdir(parents=True)

    cfg = Config.fromfile(ROOT / 'configs' / config)
    expected_root = LOCAL_ROOT if local_state == 'ready' else SHARED_ROOT
    assert_task_roots(cfg, expected_root)


@pytest.mark.parametrize('config', LEGACY_CONFIGS + [MIXED_CONFIG])
@pytest.mark.parametrize('local_ready', [False, True])
@pytest.mark.parametrize('override',
                         ['./datasets/custom-robocasa', '/external/robocasa'])
def test_robocasa_explicit_root_has_priority(monkeypatch, config, local_ready,
                                             override):
    if local_ready:
        LOCAL_MARKER.mkdir(parents=True)
    monkeypatch.setenv('ROBOCASA_DATA_ROOT', override)

    cfg = Config.fromfile(ROOT / 'configs' / config)
    assert_task_roots(cfg, override)


@pytest.mark.parametrize('local_ready', [False, True])
def test_mixed_recipe_only_relocates_robot_data(local_ready):
    if local_ready:
        LOCAL_MARKER.mkdir(parents=True)
    base_path = ROOT / 'configs' / LEGACY_CONFIGS[0]
    base = Config.fromfile(base_path)
    mixed = Config.fromfile(ROOT / 'configs' / MIXED_CONFIG)

    expected_dataset = copy.deepcopy(base.train_dataloader.dataset.to_dict())
    expected_dataset['datasets']['data_root_path'] = [
        f'{LOCAL_ROOT}/{Path(path).name}'
        for path in expected_dataset['datasets']['data_root_path']
    ]
    # Statistics, transforms, sampler settings and task order stay inherited.
    assert mixed.train_dataloader.dataset.to_dict() == expected_dataset
    assert mixed.eval.task_list == base.eval.task_list
    assert_task_roots(mixed, LOCAL_ROOT)
    # Loading the child must not change environment defaults or later loads.
    assert 'ROBOCASA_DATA_ROOT' not in os.environ
    assert Config.fromfile(base_path).to_dict() == base.to_dict()
