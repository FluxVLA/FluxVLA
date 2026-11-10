# Copyright 2026 Limx Dynamics

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from mmengine import Config

from fluxvla.datasets.parquet_dataset_v3 import ParquetDatasetV3
from fluxvla.datasets.utils.subtask_annotations import build_subtask_index
from fluxvla.datasets.utils.transformed_statistics import \
    compute_statistics_from_dataset_config
from fluxvla.engines.losses import reduce_action_bc_loss
from fluxvla.transforms.prompters import PreparePromptWithState
from fluxvla.transforms.transform_inputs import ProcessParquetInputs


def _episode():
    return dict(
        episode_index=0,
        length=4,
        dense_subtask_names=['Pick up the cup', 'Place the cup'],
        dense_subtask_start_frames=[0, 2],
        dense_subtask_end_frames=[2, 4])


def _write_dataset(root, *, record=None, action_offset=0):
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'meta/info.json').write_text(
        json.dumps(dict(fps=25, video_path='unused', chunks_size=1000)))
    (root / 'meta/stats.json').write_text('{}')
    pq.write_table(
        pa.Table.from_pylist([dict(task_index=0, task='Whole task')]),
        root / 'meta/tasks.parquet')
    pq.write_table(
        pa.Table.from_pylist([record or _episode()]),
        root / 'meta/episodes/chunk-000/file-000.parquet')
    actions = np.array([[2, 12], [5, 25], [13, 34], [19, 48]]) + action_offset
    pq.write_table(
        pa.table({
            'episode_index': [0] * 4,
            'frame_index': list(range(4)),
            'task_index': [0] * 4,
            'timestamp': [i / 25 for i in range(4)],
            'observation.state': [[1, 10], [2, 20], [3, 30], [4, 40]],
            'action': actions.tolist(),
        }), root / 'data/chunk-000/file-000.parquet')
    return root


def _dataset(roots, **kwargs):
    return ParquetDatasetV3(
        data_root_path=[str(p) for p in roots],
        transforms=[],
        action_key='action',
        action_window_size=3,
        window_start_idx=0,
        instruction_source=kwargs.pop('instruction_source', 'subtask'),
        **kwargs)


def test_subtask_selects_current_frame_and_masks_exclusive_end(tmp_path):
    dataset = _dataset([_write_dataset(tmp_path / 'data')])
    before = dataset.__getitem__(1, {'private': {}})
    after = dataset.__getitem__(2, {'private': {}})
    assert before['task_description'] == before['subtask'] == 'Pick up the cup'
    assert before['full_task_description'] == 'Whole task'
    assert before['subtask_index'] == 0
    assert before['subtask_start_frame'] == 0
    assert before['subtask_end_frame'] == 2
    assert after['task_description'] == 'Place the cup'
    assert after['subtask_index'] == 1
    np.testing.assert_array_equal(before['actions'], [[5, 25]] * 3)
    np.testing.assert_array_equal(before['action_masks'], [1, 0, 0])
    np.testing.assert_array_equal(after['actions'],
                                  [[13, 34], [19, 48], [19, 48]])
    np.testing.assert_array_equal(after['action_masks'], [1, 1, 0])
    assert len(dataset) == 4


def test_final_frame_not_resampled_and_roots_do_not_join(tmp_path):
    roots = [
        _write_dataset(tmp_path / 'a'),
        _write_dataset(tmp_path / 'b', action_offset=100)
    ]
    dataset = _dataset(roots, expose_index=True)
    for index, expected in [(3, [19, 48]), (7, [119, 148])]:
        sample = dataset.__getitem__(index, {'private': {}})
        assert sample['index'] == index
        np.testing.assert_array_equal(sample['actions'], [expected] * 3)
        np.testing.assert_array_equal(sample['action_masks'], [1, 0, 0])


def test_repeated_subtask_names_still_have_a_boundary(tmp_path):
    record = _episode()
    record['dense_subtask_names'] = ['move', 'move']
    dataset = _dataset([_write_dataset(tmp_path / 'data', record=record)])
    np.testing.assert_array_equal(
        dataset.__getitem__(1, {'private': {}})['action_masks'], [1, 0, 0])


def test_one_frame_segment_and_temporal_frames_are_padded(tmp_path):
    record = _episode()
    record['dense_subtask_start_frames'] = [0, 1]
    record['dense_subtask_end_frames'] = [1, 4]
    dataset = _dataset([_write_dataset(tmp_path / 'data', record=record)],
                       frame_window_size=3)
    sample = dataset.__getitem__(0, {'private': {}})
    np.testing.assert_array_equal(sample['action_masks'], [1, 0, 0])
    np.testing.assert_array_equal(sample['frame_masks'], [1, 0, 0])
    assert sample['frame_timestamps'] == [0, 0, 0]


@pytest.mark.parametrize('mutation,match', [
    ({
        'dense_subtask_names': None
    }, 'Missing'),
    ({
        'dense_subtask_names': []
    }, 'mismatched'),
    ({
        'dense_subtask_names': ['', 'place']
    }, 'Empty/non-string'),
    ({
        'dense_subtask_start_frames': [0, 3]
    }, 'no gaps/overlaps'),
    ({
        'dense_subtask_start_frames': [0, 1]
    }, 'no gaps/overlaps'),
    ({
        'dense_subtask_end_frames': [2, 3]
    }, 'Incomplete'),
    ({
        'dense_subtask_end_frames': [2, 5]
    }, 'Invalid subtask range'),
    ({
        'dense_subtask_end_frames': [2, 4.5]
    }, 'Invalid subtask range'),
])
def test_bad_annotations_fail_explicitly(mutation, match):
    record = _episode()
    record.update(mutation)
    with pytest.raises(ValueError, match=match):
        build_subtask_index([record])


def test_custom_prefix_and_duplicate_episode_detection():
    record = {
        key.replace('dense_', 'sparse_'): value
        for key, value in _episode().items()
    }
    assert build_subtask_index([record], 'sparse')[0][0].end == 2
    with pytest.raises(ValueError, match='Duplicate'):
        build_subtask_index([record, record], 'sparse')


def test_whole_task_mode_does_not_require_annotations(tmp_path):
    record = dict(episode_index=0, length=4)
    root = _write_dataset(tmp_path / 'data', record=record)
    dataset = _dataset([root], instruction_source='task')
    sample = dataset.__getitem__(1, {'private': {}})
    assert sample['task_description'] == 'Whole task'
    assert 'subtask' not in sample
    np.testing.assert_array_equal(sample['actions'],
                                  [[5, 25], [13, 34], [19, 48]])
    with pytest.raises(ValueError, match='Missing subtask annotations'):
        _dataset([root])


def test_subtask_mode_rejects_shifted_window():
    with pytest.raises(ValueError, match='window_start_idx=0'):
        ParquetDatasetV3('unused', [], instruction_source='subtask')


def test_noncontiguous_frame_rows_are_not_silently_mislabeled(tmp_path):
    dataset = _dataset([_write_dataset(tmp_path / 'data')])
    dataset.dataset = dataset.dataset.select([0, 2, 1, 3])
    with pytest.raises(ValueError, match='Non-contiguous'):
        dataset.__getitem__(0, {'private': {}})


def test_prompt_uses_subtask_not_episode_instruction(tmp_path):
    dataset = _dataset([_write_dataset(tmp_path / 'data')])
    sample = dataset.__getitem__(1, {'private': {}})
    inputs = ProcessParquetInputs(
        parquet_keys=[
            'observation.state', 'actions', 'action_masks', 'subtask'
        ],
        video_keys=[],
        name_mappings={'observation.state': ['states']})(
            sample)
    prompt = PreparePromptWithState()(inputs)['prompt']
    assert prompt.startswith('Task: Pick up the cup, State: ')
    assert prompt.endswith(';\nAction: ')
    assert 'Whole task' not in prompt


def test_subtask_statistics_match_only_supervised_delta_targets(tmp_path):
    root = _write_dataset(tmp_path / 'data')
    config = dict(
        statistic_name='private',
        datasets=[
            dict(
                type='ParquetDatasetV3',
                data_root_path=str(root),
                instruction_source='subtask',
                subtask_prefix='dense',
                window_start_idx=0,
                action_window_size=3)
        ])
    options = dict(delta_mask=[True, False])
    stats, meta = compute_statistics_from_dataset_config(
        config, options, default_temp_dir=tmp_path)
    dataset = _dataset([root])
    targets = []
    for index in range(len(dataset)):
        sample = dataset.__getitem__(index, {'private': {}})
        actions = sample['actions'][sample['action_masks'].astype(bool)].copy()
        actions[:, 0] -= sample['observation.state'][0]
        targets.extend(actions)
    targets = np.asarray(targets)
    result = stats['private']['action']
    assert result['count'] == 6
    np.testing.assert_allclose(result['mean'], targets.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(result['q01'],
                               np.quantile(targets, .01, axis=0))
    np.testing.assert_allclose(result['q99'],
                               np.quantile(targets, .99, axis=0))
    assert stats['private']['proprio']['count'] == 4
    assert meta['episodes'] == 1 and meta['subtask_segments'] == 2
    assert meta['repeat_terminal'] is False
    with pytest.raises(ValueError, match='repeat_terminal=False'):
        compute_statistics_from_dataset_config(
            config,
            dict(options, repeat_terminal=True),
            default_temp_dir=tmp_path)


def test_pi05_loss_reducer_ignores_masked_targets(tmp_path):
    dataset = _dataset([_write_dataset(tmp_path / 'data')])
    sample = dataset.__getitem__(1, {'private': {}})
    losses = torch.tensor([[[2., 4.], [100., 100.], [200., 200.]]],
                          requires_grad=True)
    loss = reduce_action_bc_loss(
        losses,
        torch.from_numpy(sample['action_masks'])[None])
    assert loss.item() == 3
    loss.backward()
    assert torch.count_nonzero(losses.grad[:, 1:]) == 0


def test_robodojo_subtask_config_contract(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/pi05/'
                          'pi05_paligemma_robodojo_subtask_full_finetune.py')
    dataset = cfg.train_dataloader.dataset
    source = dataset.datasets[0]
    assert source.type == 'ParquetDatasetV3'
    assert source.instruction_source == 'subtask'
    assert source.subtask_prefix == 'dense'
    assert source.action_window_size == cfg.model.n_action_steps == 50
    assert source.window_start_idx == 0 and not source.use_delta
    assert 'supervise_terminal_padding' not in source
    assert dataset.dataset_statistics is None
    assert dataset.auto_compute_statistics.delta_mask == ([True] * 6 +
                                                          [False] +
                                                          [True] * 6 + [False])
    assert cfg.model.ori_action_dim == 14 and cfg.model.max_action_dim == 32
    assert len(source.transforms[0].video_keys) == 3
    assert 'subtask' in source.transforms[0].parquet_keys
    assert 'subtask' in cfg.runner.collator.meta_keys
    assert cfg.eval is None and cfg.themis is None
    assert cfg.runner.grad_accumulation_steps == 1
    assert cfg.runner.grad_accumulation_steps * (
        cfg.train_dataloader.per_device_batch_size * 2) == 4
    # Loading/mutating the child recipe must not alter the original one.
    base = Config.fromfile(root / 'configs/pi05/'
                           'pi05_paligemma_robodojo_full_finetune.py')
    assert base.train_dataloader.dataset.dataset_statistics is not None
    assert base.train_dataloader.dataset.datasets[0].type == 'ParquetDataset'
    monkeypatch.setenv('ROBODOJO_SUBTASK_DATA_ROOT', '/custom/data')
    overridden = Config.fromfile(cfg.filename)
    assert overridden.train_dataloader.dataset.datasets[0].data_root_path == [
        '/custom/data'
    ]
