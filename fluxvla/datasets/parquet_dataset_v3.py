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
from typing import Any, Dict, List, Union

import numpy as np
import torch

from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.datasets.utils.lerobot_metadata import \
    load_episode_records as _load_episode_records
from fluxvla.datasets.utils.lerobot_metadata import \
    load_jsonl_records as _load_jsonl_records
from fluxvla.datasets.utils.lerobot_metadata import \
    load_parquet_records as _load_parquet_records
from fluxvla.engines import DATASETS


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


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
    """LeRobot v3 metadata adapter with shared ParquetDataset sampling.

    Reads tasks.parquet, meta/episodes/**/*.parquet and stats.json, with
    legacy JSONL fallbacks. Instruction selection, subtask clipping and
    transform assembly are inherited unchanged from the parent.
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
                 subtask_prefix: str = 'dense',
                 **kwargs) -> None:
        # Preserve V3's positional API; common options belong to the parent.
        super().__init__(
            data_root_path=data_root_path,
            transforms=transforms,
            action_window_size=action_window_size,
            action_key=action_key,
            use_delta=use_delta,
            statistic_name=statistic_name,
            window_start_idx=window_start_idx,
            frame_window_size=frame_window_size,
            frame_sample_stride=frame_sample_stride,
            expose_index=expose_index,
            instruction_source=instruction_source,
            subtask_prefix=subtask_prefix,
            **kwargs)

    @staticmethod
    def _load_info(meta_root: Path) -> Dict:
        info = ParquetDataset._load_info(meta_root)
        for feature in info.get('features', {}).values():
            if isinstance(feature.get('shape'), list):
                feature['shape'] = tuple(feature['shape'])
        return info

    @staticmethod
    def _load_metadata(meta_root: Path):
        return (_load_stats_records(meta_root), _load_task_mapping(meta_root),
                _load_episode_records(meta_root))

    def _invalid_start_index(self, index: int, dataset_idx: int,
                             data: Dict[str, Any]) -> bool:
        # Keep V3's historical whole-task start selection, including its
        # next-row check when window_start_idx=0. Subtask mode never resamples.
        next_index = index + 1
        if not self._same_episode_and_dataset(next_index, dataset_idx, data):
            return True
        if self._get_task_name(dataset_idx, next_index) in ('empty', 'static'):
            return True
        if self.require_full_window or self.image_frame_offset > 0:
            return super()._invalid_start_index(index, dataset_idx, data)
        return False

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
