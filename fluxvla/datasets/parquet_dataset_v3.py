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
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import concatenate_datasets, load_dataset
from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.datasets.utils.subtask_annotations import build_subtask_index
from fluxvla.engines import DATASETS, build_transform_from_cfg


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_parquet_records(paths: List[Path]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    dataset = load_dataset(
        'parquet', data_files=[str(path) for path in paths], split='train')
    return [dict(record) for record in dataset]


def _load_episode_records(meta_root: Path) -> List[Dict[str, Any]]:
    legacy_path = meta_root / 'episodes.jsonl'
    if legacy_path.exists():
        return _load_jsonl_records(legacy_path)

    episodes_dir = meta_root / 'episodes'
    parquet_files = sorted(episodes_dir.rglob('*.parquet'))
    if parquet_files:
        return _load_parquet_records(parquet_files)

    raise FileNotFoundError(f'Episodes metadata not found under {meta_root}')


def _load_stats_records(meta_root: Path) -> List[Dict[str, Any]]:
    v3_stats_path = meta_root / 'stats.json'
    if v3_stats_path.exists():
        return [{'stats': _load_json(v3_stats_path)}]

    legacy_path = meta_root / 'episodes_stats.jsonl'
    if legacy_path.exists():
        return _load_jsonl_records(legacy_path)

    raise FileNotFoundError(f'Statistics metadata not found under {meta_root}')


def _extract_task_text(record: Dict[str, Any]) -> str:
    for key in ('task', 'tasks', '__index_level_0__'):
        value = record.get(key)
        if value is not None:
            return str(value)

    for key, value in record.items():
        if key == 'task_index' or value is None:
            continue
        if isinstance(value, str):
            return value

    for key, value in record.items():
        if key != 'task_index' and value is not None:
            return str(value)

    return ''


def _load_task_mapping(meta_root: Path) -> Dict[int, str]:
    v3_path = meta_root / 'tasks.parquet'
    if v3_path.exists():
        records = _load_parquet_records([v3_path])
    else:
        legacy_path = meta_root / 'tasks.jsonl'
        if not legacy_path.exists():
            raise FileNotFoundError(
                f'Tasks metadata not found under {meta_root}')
        records = _load_jsonl_records(legacy_path)

    tasks = {}
    for fallback_index, record in enumerate(records):
        task_index = record.get('task_index', fallback_index)
        if isinstance(task_index, np.ndarray) and task_index.ndim == 0:
            task_index = task_index.item()
        if isinstance(task_index, torch.Tensor) and task_index.numel() == 1:
            task_index = task_index.item()
        tasks[int(task_index)] = _extract_task_text(record)
    return tasks


@DATASETS.register_module()
class ParquetDatasetV3(ParquetDataset):
    """ParquetDataset variant for LeRobot v3 metadata layouts.

    This subclass keeps the original ``ParquetDataset`` unchanged while adding
    support for v3 metadata files such as ``tasks.parquet``,
    ``meta/episodes/*.parquet`` and ``stats.json``.
    """

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 transforms: List[Dict],
                 action_window_size: int = 9,
                 action_key: str = 'observation.state',
                 use_delta: bool = False,
                 statistic_name: str = 'private',
                 window_start_idx: int = 1,
                 frame_window_size: int = 1,
                 frame_sample_stride: int = 1,
                 expose_index: bool = False,
                 instruction_source: str = 'task',
                 subtask_prefix: str = 'dense') -> None:
        """Initialize a parquet dataset backed by LeRobot v3 metadata.

        Args:
            data_root_path (Union[str, List[str]]): One dataset root, or a list
                of dataset roots, each containing ``meta`` and ``data``.
            transforms (List[Dict]): Transform config dictionaries applied in
                order after sample assembly.
            action_window_size (int): Number of future actions to return.
            action_key (str): Dataset key used as the action/state source.
            use_delta (bool): Whether actions are converted to frame-to-frame
                deltas.
            statistic_name (str): Statistics entry read from
                ``dataset_statistics``.
            window_start_idx (int): First offset used when building the action
                window.
            frame_window_size (int): Number of timestamps to expose for
                frame-sequence consumers.
            frame_sample_stride (int): Stride (in dataset rows) between sampled
                video frames. Defaults to 1. Increase this when the sampled
                frames should span a longer temporal window.
            expose_index (bool): Whether to expose the concatenated row index
                to transforms for offline sample weighting.
            instruction_source (str): ``'task'`` preserves whole-task training.
                ``'subtask'`` uses the current frame's episode annotation as
                ``task_description`` before prompt construction/tokenization.
                Action windows stop at the annotated end (exclusive); repeated
                tail actions have zero loss mask, including episode tails.
                This mode requires window_start_idx=0 so every sample has at
                least one supervised action, even in one-frame subtasks.
            subtask_prefix (str): Episode metadata prefix, e.g. ``'dense'``
                reads dense_subtask_names/start_frames/end_frames. Annotations
                must fully cover every episode without gaps or overlaps.
        """
        Dataset.__init__(self)
        if instruction_source not in ('task', 'subtask'):
            raise ValueError('instruction_source must be task or subtask')
        if instruction_source == 'subtask' and window_start_idx != 0:
            raise ValueError('Subtask training requires window_start_idx=0')
        if action_window_size <= 0:
            raise ValueError('action_window_size must be positive')
        self.instruction_source = instruction_source
        self.subtask_prefix = subtask_prefix
        self.action_window_size = action_window_size
        if isinstance(data_root_path, str):
            data_root_path = [data_root_path]
        self.data_root_path = data_root_path

        meta_root = [os.path.join(path, 'meta') for path in data_root_path]
        data_root = [os.path.join(path, 'data') for path in data_root_path]

        all_stats = []
        all_tasks = []
        all_episodes = []
        episodes_by_dataset = []
        info_list = []

        for root in meta_root:
            info_path = os.path.join(root, 'info.json')
            assert os.path.exists(info_path), \
                f'Metadata file not found at {info_path}'
            info = _load_json(Path(info_path))
            if 'features' in info:
                for feature in info['features'].values():
                    if isinstance(feature.get('shape'), list):
                        feature['shape'] = tuple(feature['shape'])
            info_list.append(info)

            meta_path = Path(root)
            all_stats.extend(_load_stats_records(meta_path))
            all_tasks.append(_load_task_mapping(meta_path))

            episode_records = _load_episode_records(meta_path)
            episodes_by_dataset.append(episode_records)
            all_episodes.extend(episode_records)

        self.info = info_list
        self.stats = all_stats
        self.tasks = all_tasks
        self.episodes = all_episodes
        self.episodes_by_dataset = episodes_by_dataset
        self.subtasks_by_dataset = ([
            build_subtask_index(records, subtask_prefix)
            for records in episodes_by_dataset
        ] if instruction_source == 'subtask' else None)

        datasets = []
        dataset_sizes = []
        for root in data_root:
            hf_dataset = load_dataset('parquet', data_dir=root, split='train')
            dataset_sizes.append(len(hf_dataset))
            datasets.append(hf_dataset)
        self.dataset_cumulative_sizes = np.cumsum([0] + dataset_sizes)
        self.dataset = concatenate_datasets(datasets)
        self.full_length = len(self.dataset)
        self.sample_indices = np.arange(self.full_length, dtype=np.int64)
        self.effective_length = self.full_length
        self.transforms = list()
        self.action_key = action_key
        self.use_delta = use_delta
        self.statistic_name = statistic_name
        self.window_start_idx = window_start_idx
        self.frame_window_size = frame_window_size
        self.frame_sample_stride = frame_sample_stride
        self.expose_index = expose_index
        for transform in transforms:
            self.transforms.append(build_transform_from_cfg(transform))

    def _resolve_task_description(self, dataset_idx: int, data: Dict) -> str:
        raw_task = data.get('task')
        if isinstance(raw_task, str):
            return raw_task
        if isinstance(raw_task, (list, tuple)) and len(raw_task) == 1:
            only_value = raw_task[0]
            if isinstance(only_value, str):
                return only_value
            if isinstance(only_value, (int, np.integer)):
                return self.tasks[dataset_idx].get(int(only_value), '')
        if isinstance(raw_task, np.ndarray) and raw_task.ndim == 0:
            raw_task = raw_task.item()
        if isinstance(raw_task, torch.Tensor) and raw_task.numel() == 1:
            raw_task = raw_task.item()
        if isinstance(raw_task, (int, np.integer)):
            return self.tasks[dataset_idx].get(int(raw_task), '')

        raw_task_index = data.get('task_index')
        if isinstance(raw_task_index, np.ndarray) and raw_task_index.ndim == 0:
            raw_task_index = raw_task_index.item()
        if isinstance(raw_task_index,
                      torch.Tensor) and raw_task_index.numel() == 1:
            raw_task_index = raw_task_index.item()
        if isinstance(raw_task_index, (int, np.integer)):
            return self.tasks[dataset_idx].get(int(raw_task_index), '')

        return ''

    def __getitem__(self, index, dataset_statistics):
        """Build one transformed sample from a parquet row.

        Args:
            index (int): Dataset row index.
            dataset_statistics (Dict): Statistics mapping supplied by the
                runner or collator.

        Returns:
            Dict: Sample dictionary containing task text, actions, masks,
            metadata, and transform outputs.
        """
        if self.instruction_source == 'subtask':
            return self._getitem_subtask(index, dataset_statistics)
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        while True:
            if index == len(self.dataset) - 1:
                needs_resample = True
            else:
                next_data = self.dataset[index + 1]
                next_task = self._resolve_task_description(
                    dataset_idx, next_data)
                needs_resample = (
                    data['episode_index'] != next_data['episode_index']
                    or self._get_dataset_index(index + 1) != dataset_idx
                    or next_task in ('empty', 'static'))

            if not needs_resample:
                break
            index = self._rand_another()
            data = self.dataset[index]
            dataset_idx = self._get_dataset_index(index)
        actions = list()
        action_masks = list()
        window_idx = self.window_start_idx
        while len(actions) < self.action_window_size:
            future_idx = index + window_idx
            future_in_range = future_idx < len(self.dataset)
            if future_in_range:
                future_data = self.dataset[future_idx]
                future_task = self._resolve_task_description(
                    dataset_idx, future_data)
                future_dataset_idx = self._get_dataset_index(future_idx)
            else:
                future_data = None
                future_task = ''
                future_dataset_idx = None

            valid_future = (
                future_in_range
                and data['episode_index'] == future_data['episode_index']
                and future_dataset_idx == dataset_idx
                and future_task not in ('empty', 'static'))
            if valid_future:
                if self.use_delta:
                    actions.append(
                        np.array(self.dataset[future_idx][self.action_key]) -
                        np.array(self.dataset[index + window_idx -
                                              1][self.action_key]).tolist())
                else:
                    actions.append(self.dataset[future_idx][self.action_key])
                action_masks.append(1)
            elif not future_in_range or future_task == 'empty':
                for _ in range(self.action_window_size - len(actions)):
                    actions.append(actions[-1])
                    action_masks.append(0)
                break
            elif future_task == 'static':
                window_idx += 1
                continue
            else:
                if len(actions) > 0:
                    actions.append(actions[-1])
                else:
                    actions.append(data[self.action_key])
                action_masks.append(0)
            window_idx += 1

        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi * self.frame_sample_stride
                if (future_idx < len(self.dataset)
                        and self.dataset[future_idx]['episode_index']
                        == data['episode_index'] and
                        self._get_dataset_index(future_idx) == dataset_idx):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            data['frame_timestamps'] = frame_timestamps
            data['frame_masks'] = np.array(frame_masks, dtype=np.float32)

        return self._transform_sample(
            data, index, dataset_idx, actions, action_masks,
            self._resolve_task_description(dataset_idx, data),
            dataset_statistics)

    def _getitem_subtask(self, index, dataset_statistics):
        index = self._resolve_index(index)
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        episode = int(data['episode_index'])
        segments = self.subtasks_by_dataset[dataset_idx].get(episode)
        if segments is None:
            raise ValueError(
                f'Missing subtask annotations for episode '
                f'{episode} in {self.data_root_path[dataset_idx]}')
        frame = int(data['frame_index'])
        segment_index = bisect_right([s.end for s in segments], frame)
        if frame < 0 or segment_index >= len(segments):
            raise ValueError(f'Frame {frame} outside annotated episode '
                             f'{episode}')
        segment = segments[segment_index]
        valid_length = min(self.action_window_size, segment.end - frame)
        rows = self.dataset[index:index + valid_length]
        # Do not silently align annotations to unordered, missing or foreign
        # rows, including equal episode IDs in different data roots.
        if (index + valid_length >
                self.dataset_cumulative_sizes[dataset_idx + 1]
                or rows['episode_index'] != [episode] * valid_length
                or rows['frame_index'] != list(
                    range(frame, frame + valid_length))):
            raise ValueError(f'Non-contiguous subtask action rows for episode '
                             f'{episode}, frame {frame}')
        actions = np.asarray(rows[self.action_key], dtype=np.float32)
        if self.use_delta:
            # Match the existing frame-to-frame mode without reading the last
            # action of a previous episode for frame zero.
            previous = (
                self.dataset[index - 1][self.action_key]
                if frame > 0 else data[self.action_key])
            actions = actions - np.concatenate(
                [np.asarray(previous, dtype=np.float32)[None], actions[:-1]])
        actions = np.concatenate([
            actions,
            np.repeat(
                actions[-1:], self.action_window_size - valid_length, axis=0)
        ])
        action_masks = np.zeros(self.action_window_size, dtype=np.float32)
        action_masks[:valid_length] = 1
        if self.frame_window_size > 1:
            offsets = np.arange(
                self.frame_window_size) * self.frame_sample_stride
            frame_masks = offsets < segment.end - frame
            offsets = np.minimum(offsets, segment.end - frame - 1)
            data['frame_timestamps'] = [
                self.dataset[index + int(offset)]['timestamp']
                for offset in offsets
            ]
            data['frame_masks'] = frame_masks.astype(np.float32)
        data.update(
            full_task_description=self._resolve_task_description(
                dataset_idx, data),
            subtask=segment.name,
            subtask_index=segment_index,
            subtask_start_frame=segment.start,
            subtask_end_frame=segment.end)
        return self._transform_sample(data, index, dataset_idx, actions,
                                      action_masks, segment.name,
                                      dataset_statistics)

    def _transform_sample(self, data, index, dataset_idx, actions,
                          action_masks, task_description, dataset_statistics):
        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        if self.expose_index:
            data['index'] = np.array(index, dtype=np.int64)
        data['task_description'] = task_description
        data['data_root'] = self.data_root_path[dataset_idx]
        for transform in self.transforms:
            data = transform(data)

        return data
