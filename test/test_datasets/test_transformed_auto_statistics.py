# Copyright 2026 Limx Dynamics

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mmengine import ConfigDict

import scripts.train as train_script
from fluxvla.datasets.utils.transformed_statistics import (
    PROFILES, compute_statistics_from_dataset_config)


def _wrapper_config(dataset_root):
    return ConfigDict(
        type='DistributedRepeatingDataset',
        statistic_name='robot',
        statistic_keys=['observation.state', 'action'],
        auto_compute_statistics=dict(
            state_key='observation.state',
            action_key='action',
            delta_mask=[True, False],
        ),
        datasets=ConfigDict(
            type='ParquetDataset',
            data_root_path=str(dataset_root),
            action_window_size=2,
            window_start_idx=0,
            supervise_terminal_padding=True,
        ),
    )


def _write_dataset(dataset_root):
    (dataset_root / 'meta').mkdir(parents=True)
    (dataset_root / 'data').mkdir()
    (dataset_root / 'meta' / 'info.json').write_text('{}')
    table = pa.table({
        'observation.state': [[1.0, 10.0], [2.0, 20.0]],
        'action': [[2.0, 12.0], [5.0, 25.0]],
        'episode_index': [0, 0],
        'frame_index': [0, 1],
    })
    pq.write_table(table, dataset_root / 'data' / 'episode.parquet')


def test_compute_transformed_statistics_from_dataset_config(tmp_path):
    dataset_root = tmp_path / 'dataset'
    _write_dataset(dataset_root)

    config = _wrapper_config(dataset_root)
    stats, metadata = compute_statistics_from_dataset_config(
        config, config.auto_compute_statistics, default_temp_dir=tmp_path)

    action_stats = stats['robot']['action']
    np.testing.assert_allclose(action_stats['mean'], [2.75, 21.75])
    assert action_stats['count'] == 4
    assert metadata['action_horizon'] == 2
    assert metadata['repeat_terminal'] is True


def test_absolute_profile_keeps_qpos_actions_absolute(tmp_path):
    dataset_root = tmp_path / 'dataset'
    _write_dataset(dataset_root)

    config = _wrapper_config(dataset_root)
    stats, metadata = compute_statistics_from_dataset_config(
        config,
        options=dict(profile='absolute'),
        default_temp_dir=tmp_path,
    )

    np.testing.assert_allclose(stats['robot']['action']['mean'], [4.25, 21.75])
    assert metadata['delta_mask'] == []
    assert PROFILES['tron2'].delta_mask == ()


def test_inline_statistics_take_priority_over_automatic_computation(
        tmp_path, monkeypatch):
    config = _wrapper_config(tmp_path / 'unused')
    configured = {'robot': {'proprio': {}, 'action': {}}}
    config.dataset_statistics = configured
    cfg = ConfigDict(train_dataloader=ConfigDict(dataset=config))

    def fail_if_called(*args, **kwargs):
        raise AssertionError('automatic statistics should not run')

    monkeypatch.setattr(train_script, 'compute_statistics_from_dataset_config',
                        fail_if_called)
    train_script._prepare_automatic_dataset_statistics(cfg, tmp_path)

    assert cfg.train_dataloader.dataset.dataset_statistics == configured
    assert 'dataset_statistics_path' not in cfg.train_dataloader.dataset


def test_statistics_path_takes_priority_over_automatic_computation(
        tmp_path, monkeypatch):
    config = _wrapper_config(tmp_path / 'unused')
    config.dataset_statistics_path = str(tmp_path / 'configured.json')
    cfg = ConfigDict(train_dataloader=ConfigDict(dataset=config))

    def fail_if_called(*args, **kwargs):
        raise AssertionError('automatic statistics should not run')

    monkeypatch.setattr(train_script, 'compute_statistics_from_dataset_config',
                        fail_if_called)
    train_script._prepare_automatic_dataset_statistics(cfg, tmp_path)

    assert cfg.train_dataloader.dataset.dataset_statistics_path == str(
        tmp_path / 'configured.json')


def test_automatic_statistics_are_saved_and_attached(tmp_path, monkeypatch):
    config = _wrapper_config(tmp_path / 'unused')
    cfg = ConfigDict(train_dataloader=ConfigDict(dataset=config))
    expected = {'robot': {'proprio': {'mean': [0.0]}, 'action': {}}}
    metadata = {'profile': 'test'}

    monkeypatch.setattr(
        train_script,
        'compute_statistics_from_dataset_config',
        lambda *args, **kwargs: (expected, metadata),
    )
    train_script._prepare_automatic_dataset_statistics(cfg, tmp_path)

    stats_path = tmp_path / 'dataset_statistics.json'
    metadata_path = tmp_path / 'dataset_statistics_metadata.json'
    assert json.loads(stats_path.read_text()) == expected
    assert json.loads(metadata_path.read_text()) == metadata
    assert cfg.train_dataloader.dataset.dataset_statistics_path == str(
        stats_path.resolve())
