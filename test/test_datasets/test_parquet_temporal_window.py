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

import numpy as np
import pytest

from datasets import Dataset
from fluxvla.datasets.parquet_dataset import ParquetDataset


def _make_dataset(boundary='episode', **settings):
    """In-memory rows; sampling, boundary checks and assembly stay real."""
    dataset = ParquetDataset.__new__(ParquetDataset)
    dataset.dataset = Dataset.from_dict({
        'episode_index':
        [0] * 5 + ([1] * 5 if boundary == 'episode' else [0] * 5),
        'task_index': [0] * 10,
        'timestamp':
        list(range(5)) + list(range(5)),
        'action': [[float(i)] for i in range(10)],
    })
    sizes = [0, 5, 10] if boundary == 'dataset' else [0, 10]
    dataset.dataset_cumulative_sizes = np.array(sizes)
    dataset.sample_indices = np.arange(10)
    dataset.tasks = [[dict(task='move')]] * (len(sizes) - 1)
    dataset.info = [{}] * (len(sizes) - 1)
    dataset.data_root_path = ['unused'] * (len(sizes) - 1)
    dataset.statistic_name = 'test'
    dataset.transforms = []
    dataset.action_key = 'action'
    dataset.action_dtype = np.dtype('float32')
    dataset.use_delta = False
    dataset.expose_index = True
    dataset.supervise_terminal_padding = False
    dataset.require_full_window = False
    dataset.frame_window_size = 3
    dataset.frame_sample_stride = 2
    dataset.window_start_idx = 0
    dataset.action_window_size = 3
    for name, value in settings.items():
        setattr(dataset, name, value)
    return dataset


@pytest.mark.parametrize('boundary', ['episode', 'dataset'])
@pytest.mark.parametrize('frame_count,action_offset,action_count', [(4, 0, 2),
                                                                    (2, 2, 2)])
def test_full_contiguous_window_rejects_crossing_boundary(
        boundary, frame_count, action_offset, action_count):
    dataset = _make_dataset(
        boundary,
        require_full_window=True,
        frame_sample_stride=1,
        frame_window_size=frame_count,
        window_start_idx=action_offset,
        action_window_size=action_count)
    assert not dataset._invalid_start_index(1, 0, dataset.dataset[1])
    assert dataset._invalid_start_index(2, 0, dataset.dataset[2])


def test_strided_frames_cover_complete_window():
    dataset = _make_dataset(require_full_window=True, action_window_size=5)
    sample = dataset.__getitem__(0, {'test': {}})
    assert sample['frame_timestamps'] == [0, 2, 4]
    np.testing.assert_array_equal(sample['frame_masks'], [1, 1, 1])
    np.testing.assert_array_equal(sample['actions'], [[0], [1], [2], [3], [4]])
    np.testing.assert_array_equal(sample['action_masks'], [1, 1, 1, 1, 1])


@pytest.mark.parametrize('boundary', ['episode', 'dataset'])
@pytest.mark.parametrize('supervise_padding', [False, True])
def test_strided_frames_and_actions_pad_only_within_episode(
        boundary, supervise_padding):
    # Padding is supported with require_full_window=False. In particular,
    # same episode IDs in different roots must not join two trajectories.
    dataset = _make_dataset(
        boundary, supervise_terminal_padding=supervise_padding)
    sample = dataset.__getitem__(3, {'test': {}})
    assert sample['index'] == 3
    assert sample['frame_timestamps'] == [3, 3, 3]
    np.testing.assert_array_equal(sample['frame_masks'], [1, 0, 0])
    np.testing.assert_array_equal(sample['actions'], [[3], [4], [4]])
    np.testing.assert_array_equal(sample['action_masks'],
                                  [1, 1, int(supervise_padding)])
