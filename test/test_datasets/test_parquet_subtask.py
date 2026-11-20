# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from mmengine import Config

from datasets import load_dataset
from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.datasets.parquet_dataset_v3 import (ParquetDatasetV3,
                                                 _load_parquet_records)
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


def _write_dataset(root, *, record=None, action_offset=0, legacy=False):
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'meta/info.json').write_text(
        json.dumps(dict(fps=25, video_path='unused', chunks_size=1000)))
    record = record or _episode()
    if legacy:
        for filename, row in (('episodes_stats.jsonl',
                               dict(episode_index=0, stats={})),
                              ('tasks.jsonl',
                               dict(task_index=0, task='Whole task')),
                              ('episodes.jsonl', record)):
            (root / 'meta' / filename).write_text(json.dumps(row) + '\n')
    else:
        (root / 'meta/stats.json').write_text('{}')
        pq.write_table(
            pa.Table.from_pylist([dict(task_index=0, task='Whole task')]),
            root / 'meta/tasks.parquet')
        pq.write_table(
            pa.Table.from_pylist([record]),
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


def _dataset(roots, dataset_class=ParquetDatasetV3, **kwargs):
    return dataset_class(
        data_root_path=[str(p) for p in roots],
        transforms=[],
        action_key='action',
        action_window_size=3,
        window_start_idx=0,
        instruction_source=kwargs.pop('instruction_source', 'subtask'),
        **kwargs)


def test_v3_metadata_does_not_prepare_hf_cache(tmp_path, monkeypatch):
    root = _write_dataset(tmp_path / 'data')

    def reject_hf_cache(*args, **kwargs):
        raise AssertionError('Local metadata must not generate HF caches')

    monkeypatch.setattr('datasets.builder.DatasetBuilder.download_and_prepare',
                        reject_hf_cache)
    stats, tasks, episodes = ParquetDatasetV3._load_metadata(root / 'meta')
    assert stats == [dict(stats={})]
    assert tasks == {0: 'Whole task'}
    assert episodes == [_episode()]


def test_parquet_metadata_keeps_file_and_row_group_order(tmp_path):
    paths = []
    expected = []
    for name, indices in [('b', [3, 1]), ('a', [2, 0])]:
        path = tmp_path / f'{name}.parquet'
        rows = [dict(_episode(), episode_index=index) for index in indices]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
        paths.append(path)
        expected.extend(rows)
    assert _load_parquet_records(paths) == expected
    assert _load_parquet_records([]) == []


@pytest.mark.parametrize('dataset_class', [ParquetDataset, ParquetDatasetV3])
@pytest.mark.parametrize('explicit_cache', [False, True])
def test_frame_data_cache_directory_is_forwarded(tmp_path, monkeypatch,
                                                 dataset_class,
                                                 explicit_cache):
    roots = [
        _write_dataset(
            tmp_path / name, legacy=dataset_class is ParquetDataset)
        for name in ('a', 'b')
    ]
    cache_dir = str(tmp_path / 'hf-cache') if explicit_cache else None
    observed = []

    def checked_load_dataset(*args, **kwargs):
        observed.append(kwargs['cache_dir'])
        return load_dataset(*args, **kwargs)

    monkeypatch.setattr('fluxvla.datasets.parquet_dataset.load_dataset',
                        checked_load_dataset)
    dataset = _dataset(roots, dataset_class=dataset_class, cache_dir=cache_dir)
    assert observed == [cache_dir, cache_dir]
    assert len(dataset) == 8
    assert dataset.__getitem__(
        1, {'private': {}})['subtask'] == ('Pick up the cup')
    if explicit_cache:
        assert dataset.dataset.cache_files
        assert all(
            Path(item['filename']).is_relative_to(cache_dir)
            for item in dataset.dataset.cache_files)


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


@pytest.mark.parametrize('legacy', [False, True])
def test_subtask_statistics_match_only_supervised_delta_targets(
        tmp_path, legacy):
    root = _write_dataset(tmp_path / 'data', legacy=legacy)
    dataset_class = ParquetDataset if legacy else ParquetDatasetV3
    config = dict(
        statistic_name='private',
        datasets=[
            dict(
                type=dataset_class.__name__,
                data_root_path=str(root),
                instruction_source='subtask',
                subtask_prefix='dense',
                window_start_idx=0,
                action_window_size=3)
        ])
    options = dict(delta_mask=[True, False])
    stats, meta = compute_statistics_from_dataset_config(
        config, options, default_temp_dir=tmp_path)
    dataset = _dataset([root], dataset_class=dataset_class)
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


def test_v3_inherits_shared_sampling_and_transform_assembly():
    assert ParquetDatasetV3.__getitem__ is ParquetDataset.__getitem__
    assert ParquetDatasetV3._getitem_subtask is ParquetDataset._getitem_subtask
    assert (ParquetDatasetV3._transform_sample is
            ParquetDataset._transform_sample)


@pytest.mark.parametrize('use_delta', [False, True])
@pytest.mark.parametrize('frame_count', [1, 4])
def test_legacy_and_v3_subtasks_have_identical_outputs(tmp_path, use_delta,
                                                       frame_count):
    datasets = []
    for legacy, dataset_class in ((True, ParquetDataset), (False,
                                                           ParquetDatasetV3)):
        roots = [
            _write_dataset(
                tmp_path / f'{legacy}-{index}',
                action_offset=index * 100,
                legacy=legacy) for index in range(2)
        ]
        datasets.append(
            _dataset(
                roots,
                dataset_class=dataset_class,
                use_delta=use_delta,
                frame_window_size=frame_count,
                frame_sample_stride=2,
                expose_index=True))
    # Include segment boundaries, episode tails and equal episode IDs in
    # different roots. No annotation/tokenization logic depends on format.
    for index in range(8):
        samples = [
            dataset.__getitem__(index, {'private': {}}) for dataset in datasets
        ]
        for key in ('actions', 'action_masks', 'task_description', 'subtask',
                    'full_task_description', 'subtask_index', 'index'):
            np.testing.assert_array_equal(samples[0][key], samples[1][key])
        assert samples[0]['actions'].dtype == np.float32
        if frame_count > 1:
            assert samples[0]['frame_timestamps'] == samples[1][
                'frame_timestamps']
            np.testing.assert_array_equal(samples[0]['frame_masks'],
                                          samples[1]['frame_masks'])
        if use_delta and index in (0, 4):
            np.testing.assert_array_equal(samples[0]['actions'][0], [0, 0])


def test_v3_whole_task_keeps_historical_tail_resampling(tmp_path, monkeypatch):
    root = _write_dataset(
        tmp_path / 'v3', record=dict(episode_index=0, length=4))
    dataset = _dataset([root], instruction_source='task', expose_index=True)
    monkeypatch.setattr(dataset, '_rand_another', lambda: 1)
    sample = dataset.__getitem__(3, {'private': {}})
    assert sample['index'] == 1 and sample['frame_index'] == 1
    assert sample['task_description'] == 'Whole task'
    np.testing.assert_array_equal(sample['action_masks'], [1, 1, 1])


def test_legacy_default_task_mode_keeps_terminal_padding(tmp_path):
    root = _write_dataset(
        tmp_path / 'legacy',
        record=dict(episode_index=0, length=4),
        legacy=True)
    assert not (root / 'meta/tasks.parquet').exists()
    assert not list((root / 'meta/episodes').rglob('*.parquet'))
    dataset = ParquetDataset(
        str(root), [],
        action_key='action',
        action_window_size=3,
        window_start_idx=0,
        supervise_terminal_padding=True)
    sample = dataset.__getitem__(3, {'private': {}})
    assert dataset.instruction_source == 'task'
    assert sample['task_description'] == 'Whole task'
    assert 'subtask' not in sample
    np.testing.assert_array_equal(sample['actions'], [[19, 48]] * 3)
    np.testing.assert_array_equal(sample['action_masks'], [1, 1, 1])
    with pytest.raises(ValueError, match='Missing subtask annotations'):
        _dataset([root], dataset_class=ParquetDataset)


@pytest.mark.parametrize('dataset_class', [ParquetDataset, ParquetDatasetV3])
def test_subtask_sample_split_repeats_only_selected_episodes(
        tmp_path, dataset_class):
    legacy = dataset_class is ParquetDataset
    root = _write_dataset(tmp_path / 'data', legacy=legacy)
    path = root / 'data/chunk-000/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    other = [
        dict(
            row,
            episode_index=1,
            action=[value + 100 for value in row['action']]) for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(rows + other), path)
    records = [_episode(), dict(_episode(), episode_index=1)]
    if legacy:
        (root / 'meta/episodes.jsonl').write_text(''.join(
            json.dumps(record) + '\n' for record in records))
    else:
        pq.write_table(
            pa.Table.from_pylist(records),
            root / 'meta/episodes/chunk-000/file-000.parquet')
    dataset = _dataset([root],
                       dataset_class=dataset_class,
                       expose_index=True,
                       train_episode_fraction=0.5,
                       repeat_to_full_length=True)
    assert len(dataset) == 8
    np.testing.assert_array_equal(dataset.sample_indices, [0, 1, 2, 3])
    sample = dataset.__getitem__(7, {'private': {}})
    assert sample['episode_index'] == 0 and sample['index'] == 3
    np.testing.assert_array_equal(sample['actions'], [[19, 48]] * 3)


@pytest.mark.parametrize('dataset_class', [ParquetDataset, ParquetDatasetV3])
def test_subtask_validates_video_rows_beyond_action_window(
        tmp_path, dataset_class):
    record = dict(
        _episode(),
        dense_subtask_names=['move'],
        dense_subtask_start_frames=[0],
        dense_subtask_end_frames=[4])
    root = _write_dataset(
        tmp_path / 'data',
        record=record,
        legacy=dataset_class is ParquetDataset)
    dataset = dataset_class(
        str(root), [],
        action_key='action',
        action_window_size=1,
        window_start_idx=0,
        instruction_source='subtask',
        frame_window_size=4)
    dataset.dataset = dataset.dataset.select([0, 1, 3, 2])
    with pytest.raises(ValueError, match='Non-contiguous'):
        dataset.__getitem__(0, {'private': {}})


@pytest.mark.parametrize('dataset_class', [ParquetDataset, ParquetDatasetV3])
def test_subtask_delta_validates_previous_row(tmp_path, dataset_class):
    root = _write_dataset(
        tmp_path / 'data', legacy=dataset_class is ParquetDataset)
    dataset = _dataset([root], dataset_class=dataset_class, use_delta=True)
    dataset.dataset = dataset.dataset.select([0, 0, 2, 3])
    with pytest.raises(ValueError, match='Non-contiguous subtask previous'):
        dataset.__getitem__(2, {'private': {}})


@pytest.mark.parametrize('dataset_class', [ParquetDataset, ParquetDatasetV3])
@pytest.mark.parametrize('options', [
    dict(window_start_idx=1),
    dict(supervise_terminal_padding=True),
    dict(image_frame_offset=1),
    dict(require_full_window=True),
])
def test_subtask_rejects_conflicting_window_options(dataset_class, options):
    settings = dict(window_start_idx=0, instruction_source='subtask')
    settings.update(options)
    with pytest.raises(ValueError, match='Subtask training requires'):
        dataset_class('unused', [], **settings)


@pytest.mark.parametrize('dataset_class', [ParquetDataset, ParquetDatasetV3])
def test_subtask_honors_action_dtype(tmp_path, dataset_class):
    root = _write_dataset(
        tmp_path / 'data', legacy=dataset_class is ParquetDataset)
    path = root / 'data/chunk-000/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    rows[1]['action'] = [5.0000000001, 25.0000000001]
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = _dataset([root], dataset_class=dataset_class, action_dtype=None)
    sample = dataset.__getitem__(1, {'private': {}})
    assert sample['actions'].dtype == np.float64
    np.testing.assert_array_equal(sample['actions'][0], rows[1]['action'])


def test_robodojo_subtask_config_contract(monkeypatch):
    monkeypatch.delenv('HF_DATASETS_CACHE', raising=False)
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/pi05/'
                          'pi05_paligemma_robodojo_subtask_full_finetune.py')
    dataset = cfg.train_dataloader.dataset
    source = dataset.datasets[0]
    assert source.type == 'ParquetDatasetV3'
    assert source.cache_dir == '/mnt/data/cpfs/.cache/fluxvla/hf-datasets'
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
    assert 'cache_dir' not in base.train_dataloader.dataset.datasets[0]
    monkeypatch.setenv('ROBODOJO_SUBTASK_DATA_ROOT', '/custom/data')
    monkeypatch.setenv('HF_DATASETS_CACHE', '/custom/cache')
    overridden = Config.fromfile(cfg.filename)
    assert overridden.train_dataloader.dataset.datasets[0].data_root_path == [
        '/custom/data'
    ]
    assert overridden.train_dataloader.dataset.datasets[0].cache_dir == (
        '/custom/cache')
    overridden.merge_from_dict(
        {'train_dataloader.dataset.datasets.0.cache_dir': '/cli/cache'})
    assert overridden.train_dataloader.dataset.datasets[0].cache_dir == (
        '/cli/cache')
