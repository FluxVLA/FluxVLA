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

import hashlib
import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from fluxvla.engines import DATASETS
from .dataset_wrapper import DistributedRepeatingDataset


@DATASETS.register_module()
class DistributedBalancedRepeatingDataset(DistributedRepeatingDataset):
    """Repeat multiple sources with deterministic balanced sampling.

    By default, every source contributes exactly once per source cycle, with a
    deterministic per-epoch source order and source-local offset. Supplying
    ``sampling_weights`` switches to the DiT4DiT-compatible deterministic
    sampling-with-replacement rule.

    A source can be either an item in a dataset list or one root of a single
    multi-root :class:`ParquetDataset`. Supporting the latter avoids building
    a tokenizer and transform pipeline once per RoboCasa task.

    Args:
        datasets: Dataset configs, built datasets, or a multi-root dataset.
        statistic_keys: Keys used by inherited statistics aggregation.
        name_mappings: Optional statistics key mappings.
        sampling_weights: Optional positive source weights. When set, preserve
            deterministic weighted sampling with replacement.
        epoch_size: Number of virtual samples in one epoch. By default this is
            ``num_sources * max(source_lengths)`` for balanced cycling, or
            ``max(source_length / probability)`` for weighted sampling.
        shuffle: Whether to shuffle virtual indices before sharding.
        reshuffle_each_epoch: Whether ``epoch`` changes after each pass.
        seed: Base seed used by source mapping and virtual-index ordering.
    """

    def __init__(
        self,
        datasets: Union[Dict, List[Dict]],
        statistic_keys: List[str],
        name_mappings: Optional[Dict] = None,
        sampling_weights: Optional[Sequence[float]] = None,
        shuffle: bool = True,
        reshuffle_each_epoch: bool = True,
        seed: int = 42,
        statistic_name: str = 'private',
        dim: Optional[int] = None,
        dataset_statistics: Optional[Dict] = None,
        statistics_overrides: Optional[Dict] = None,
        dataset_statistics_path: Optional[str] = None,
        auto_compute_statistics: Optional[Dict] = None,
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
            auto_compute_statistics=auto_compute_statistics,
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

        self.sampling_probabilities = self._normalize_sampling_weights(
            sampling_weights)
        self.source_total_len = self.total_len
        if epoch_size is None:
            if self.sampling_probabilities is None:
                epoch_size = len(self.source_lengths) * max(
                    self.source_lengths)
            else:
                lengths = np.asarray(self.source_lengths, dtype=np.float64)
                epoch_size = int(np.max(lengths / self.sampling_probabilities))
        if int(epoch_size) <= 0:
            raise ValueError('`epoch_size` must be a positive integer.')
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

    def _normalize_sampling_weights(
            self, sampling_weights: Optional[Sequence[float]]):
        if sampling_weights is None:
            return None
        weights = np.asarray(sampling_weights, dtype=np.float64)
        if weights.shape != (len(self.source_lengths), ):
            raise ValueError(
                '`sampling_weights` must contain one value per source, got '
                f'{weights.shape} for {len(self.source_lengths)} sources.')
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(
                '`sampling_weights` must contain finite positive values, got '
                f'{weights.tolist()}.')
        return weights / weights.sum()

    @staticmethod
    def _mapping_seed(epoch: int, index: int, seed: int) -> int:
        """Return the stable seed used by weighted replacement sampling."""
        value = repr((int(epoch), int(index), int(seed))).encode('utf-8')
        digest = hashlib.sha256(value).hexdigest()
        return int(digest, 16) & ((1 << 128) - 1)

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
        if virtual_index < 0 or virtual_index >= self.total_len:
            raise IndexError(f'Virtual index {virtual_index} is outside '
                             f'[0, {self.total_len}).')

        if self.sampling_probabilities is not None:
            rng = np.random.default_rng(
                self._mapping_seed(epoch, virtual_index, self.seed))
            source_index = int(
                rng.choice(
                    len(self.source_lengths), p=self.sampling_probabilities))
            sample_index = int(rng.choice(self.source_lengths[source_index]))
            return source_index, sample_index

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
        if self.sampling_probabilities is not None:
            if self.shuffle:
                epoch_offset = epoch if self.reshuffle_each_epoch else 0
                rng = np.random.default_rng(self.seed + epoch_offset)
                rng.shuffle(indices)
            return indices
        return self._affine_permutation(epoch, indices)

    def _shard_virtual_indices(self, epoch: int, rank: int,
                               world_size: int) -> np.ndarray:
        """Shard virtual indices with the legacy round-robin policy."""
        if world_size <= 0:
            raise ValueError('`world_size` must be positive.')
        if rank < 0 or rank >= world_size:
            raise ValueError(f'rank must be in [0, {world_size}), got {rank}.')
        if self.sampling_probabilities is not None:
            return self._ordered_virtual_indices(epoch)[rank::world_size]
        positions = np.arange(rank, self.total_len, world_size, dtype=np.int64)
        return self._affine_permutation(epoch, positions)

    def _get_worker_virtual_indices(
        self,
        epoch: int,
        worker_id: int,
        num_workers: int,
    ) -> np.ndarray:
        """Shard balanced virtual indices across ranks and workers.

        Round-robin preserves the original sample-wise sharding behavior.
        Blockwise first forms distributed batches, assigns each rank its
        contiguous local batch, and then assigns complete local batches to
        DataLoader workers.
        """
        if self._sharding_strategy == 'round_robin':
            total_world = self.world_size * num_workers
            total_rank = self.rank * num_workers + worker_id
            return self._shard_virtual_indices(epoch, total_rank, total_world)

        if self._sharding_strategy == 'blockwise':
            indices = torch.as_tensor(
                self._ordered_virtual_indices(epoch), dtype=torch.int64)
            shard = self._get_blockwise_shard(indices, worker_id, num_workers)
            return np.asarray(shard, dtype=np.int64)

        raise RuntimeError('Unsupported dataset sharding strategy: '
                           f'{self._sharding_strategy!r}.')

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

        shared_epoch = getattr(self, '_runner_epoch', None)
        runner_epoch = (-1
                        if shared_epoch is None else int(shared_epoch.item()))

        while True:
            if runner_epoch >= 0:
                epoch = runner_epoch
            else:
                epoch = self._epoch
            if runner_epoch < 0 and self.reshuffle_each_epoch:
                self._epoch += 1
            virtual_indices = self._get_worker_virtual_indices(
                epoch, worker_id, num_workers)
            for virtual_index in virtual_indices:
                source_index, sample_index = self._sample_dataset_and_index(
                    epoch, int(virtual_index))
                yield self._get_balanced_item(source_index, sample_index)


__all__ = ['DistributedBalancedRepeatingDataset']
