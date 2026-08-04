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
"""Task-balanced repeating datasets for multi-source robot data."""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from fluxvla.engines import DATASETS
from .dataset_wrapper import DistributedRepeatingDataset


@DATASETS.register_module()
class DistributedBalancedRepeatingDataset(DistributedRepeatingDataset):
    """Repeat multiple sources with equal source probability.

    ``DistributedRepeatingDataset`` samples the concatenated frame stream, so
    longer task roots are seen more often. This opt-in wrapper instead gives
    every source the same number of virtual samples per epoch and repeats a
    shorter source as needed. It leaves the underlying dataset and its
    normalization statistics unchanged.

    A source can be either an item in a dataset list or one root of a single
    multi-root :class:`ParquetDataset`. Supporting the latter avoids building
    a tokenizer and transform pipeline once per RoboCasa task.

    Args:
        epoch_size: Number of virtual samples in one epoch. By default each
            source contributes the length of the longest source.
    """

    def __init__(
        self,
        datasets: Union[Dict, List[Dict]],
        statistic_keys: List[str],
        name_mappings: Optional[Dict] = None,
        shuffle: bool = True,
        reshuffle_each_epoch: bool = True,
        seed: int = 42,
        statistic_name: str = 'private',
        dim: Optional[int] = None,
        dataset_statistics: Optional[Dict] = None,
        statistics_overrides: Optional[Dict] = None,
        dataset_statistics_path: Optional[str] = None,
        epoch_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            datasets=datasets,
            statistic_keys=statistic_keys,
            name_mappings=name_mappings,
            shuffle=shuffle,
            reshuffle_each_epoch=reshuffle_each_epoch,
            seed=seed,
            statistic_name=statistic_name,
            dim=dim,
            dataset_statistics=dataset_statistics,
            statistics_overrides=statistics_overrides,
            dataset_statistics_path=dataset_statistics_path,
        )
        if self.is_grouped:
            raise ValueError(
                'DistributedBalancedRepeatingDataset does not support '
                'grouped datasets.')

        self._source_positions = self._build_source_positions()
        self.source_lengths = [
            len(indices) for indices in self._source_positions
        ]
        if not self.source_lengths or any(length <= 0
                                          for length in self.source_lengths):
            raise ValueError(
                'Every balanced dataset source must be non-empty.')

        default_epoch_size = len(self.source_lengths) * max(
            self.source_lengths)
        if epoch_size is None:
            epoch_size = default_epoch_size
        if int(epoch_size) <= 0:
            raise ValueError('epoch_size must be positive.')
        self.total_len = int(epoch_size)

    def _build_source_positions(self) -> List[np.ndarray]:
        if self.is_list:
            return [
                np.arange(length, dtype=np.int64)
                for length in self.dataset_lens
            ]

        cumulative_sizes = getattr(self.dataset, 'dataset_cumulative_sizes',
                                   None)
        sample_indices = getattr(self.dataset, 'sample_indices', None)
        if cumulative_sizes is None or sample_indices is None:
            return [np.arange(len(self.dataset), dtype=np.int64)]

        cumulative_sizes = np.asarray(cumulative_sizes, dtype=np.int64)
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        if cumulative_sizes.ndim != 1 or len(cumulative_sizes) < 2:
            raise ValueError('Invalid dataset_cumulative_sizes.')

        # Store positions into ``sample_indices`` rather than resolved global
        # frame indices. ParquetDataset.__getitem__ applies that indirection.
        positions = np.arange(len(sample_indices), dtype=np.int64)
        return [
            positions[(sample_indices >= start) & (sample_indices < end)]
            for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:])
        ]

    @lru_cache(maxsize=8)
    def _epoch_source_order_and_offsets(self, epoch: int):
        rng = np.random.default_rng(self.seed + 104729 * int(epoch))
        source_order = rng.permutation(len(self.source_lengths))
        source_offsets = np.asarray(
            [rng.integers(length) for length in self.source_lengths],
            dtype=np.int64)
        return source_order, source_offsets

    def _sample_dataset_and_index(self, epoch: int, virtual_index: int):
        """Map a virtual index to a source and a source-local index."""
        source_order, source_offsets = self._epoch_source_order_and_offsets(
            epoch)
        source_slot = int(virtual_index) % len(self.source_lengths)
        source_index = int(source_order[source_slot])
        cycle = int(virtual_index) // len(self.source_lengths)
        sample_index = int((source_offsets[source_index] + cycle) %
                           self.source_lengths[source_index])
        return source_index, sample_index

    def _affine_permutation(self, epoch: int, indices: np.ndarray):
        if not self.shuffle or self.total_len <= 1:
            return indices

        rng = np.random.default_rng(self.seed + 130363 * int(epoch) + 17)
        multiplier = int(rng.integers(1, self.total_len))
        while math.gcd(multiplier, self.total_len) != 1:
            multiplier = (multiplier + 1) % self.total_len
            if multiplier == 0:
                multiplier = 1
        offset = int(rng.integers(self.total_len))
        return (multiplier * indices + offset) % self.total_len

    def _ordered_virtual_indices(self, epoch: int) -> np.ndarray:
        indices = np.arange(self.total_len, dtype=np.int64)
        return self._affine_permutation(epoch, indices)

    def _shard_virtual_indices(self, epoch: int, rank: int,
                               world_size: int) -> np.ndarray:
        # Apply a bijective affine permutation after sharding the source
        # positions. This is equivalent to sharding a global permutation but
        # avoids materializing the full epoch in every dataloader worker.
        positions = np.arange(rank, self.total_len, world_size, dtype=np.int64)
        return self._affine_permutation(epoch, positions)

    def _get_balanced_item(self, source_index: int, sample_index: int):
        source_position = int(
            self._source_positions[source_index][sample_index])
        if self.is_list:
            dataset = self.datasets[source_index]
            return dataset.__getitem__(source_position,
                                       self.dataset_statistics)
        return self.dataset.__getitem__(source_position,
                                        self.dataset_statistics)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        total_world = self.world_size * num_workers
        total_rank = self.rank * num_workers + worker_id

        while True:
            epoch = self._epoch
            if self.reshuffle_each_epoch:
                self._epoch += 1
            virtual_indices = self._shard_virtual_indices(
                epoch, total_rank, total_world)
            for virtual_index in virtual_indices:
                source_index, sample_index = self._sample_dataset_and_index(
                    epoch, int(virtual_index))
                yield self._get_balanced_item(source_index, sample_index)
