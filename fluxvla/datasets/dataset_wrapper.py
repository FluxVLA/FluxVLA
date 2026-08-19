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
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from fluxvla.engines import (DATASETS, build_dataset_from_cfg,
                             initialize_overwatch)

overwatch = initialize_overwatch(__name__)


@DATASETS.register_module()
class DistributedRepeatingDataset(IterableDataset):
    """A distributed dataset that repeats indefinitely.
    This dataset is designed to be used in a distributed setting,
    where each worker gets a shard of the dataset. It supports
    shuffling and can be used with PyTorch's DataLoader.

    Now supports three formats:
    1. Single dataset (dict): Single dataset configuration
    2. List of datasets (list of dict): Treats all datasets as one
        concatenated dataset
    3. Grouped datasets (dict of list of dict): Groups datasets
        by keys, each group has separate statistics and is
        treated as a separate dataset

    Args:
        datasets (dict, list of dict, or dict of list of dict):
            Configuration for the dataset(s) to be wrapped.
        statistic_keys (list[str]): Keys for which
            to collect statistics.
        name_mappings (dict, optional): Mappings for
            statistic names. Defaults to None.
        shuffle (bool): Whether to shuffle the dataset.
        reshuffle_each_epoch (bool): Whether to change the shuffle order
            after each full pass over the local shard. Defaults to False.
        seed (int): Seed for random number generation.
        statistic_name (str): Name for the statistics collection.
        dim (int, optional): Target dimension for padding/copying data.
            If provided, data will be padded/copied to be an integer
            multiple of this dimension. Defaults to None.
        statistics_overrides (dict, optional): Nested statistic values used to
            override collected statistics. For grouped datasets, the outer
            keys may select individual dataset groups.
    """

    def __init__(self,
                 datasets: Union[Dict, List[Dict], Dict[str, List[Dict]]],
                 statistic_keys: List[str],
                 name_mappings: Dict = None,
                 shuffle: bool = True,
                 reshuffle_each_epoch: bool = False,
                 seed: int = 42,
                 statistic_name: str = 'private',
                 dim: Optional[int] = None,
                 dataset_statistics: Optional[Dict] = None,
                 statistics_overrides: Optional[Dict] = None,
                 dataset_statistics_path: Optional[str] = None) -> None:
        if (dataset_statistics is not None
                and dataset_statistics_path is not None):
            raise ValueError(
                'dataset_statistics and dataset_statistics_path are mutually '
                'exclusive')
        if dataset_statistics_path is not None:
            with open(dataset_statistics_path, 'r', encoding='utf-8') as f:
                dataset_statistics = json.load(f)
        self.shuffle = shuffle
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.seed = seed
        self.statistic_name = statistic_name
        self.dim = dim
        # Determine the dataset format and initialize accordingly
        if isinstance(datasets, dict) and not (isinstance(
                list(datasets.values())[0], list) if datasets else False):
            # Case 1: Single dataset (dict)
            if isinstance(datasets, dict):
                self.dataset = build_dataset_from_cfg(datasets)
            else:
                # Already built dataset object
                self.dataset = datasets

            assert hasattr(self.dataset, '__getitem__') and hasattr(self.dataset, '__len__'), \
                'The wrapped dataset must implement __getitem__ and __len__ methods.'  # noqa: E501

            stats = self.dataset.stats
            self.total_len = len(self.dataset)
            self.is_grouped = False
            self.is_list = False

            if dataset_statistics is None:
                self.dataset_statistics = self.get_dataset_statistics(
                    stats, statistic_keys, name_mappings)
            else:
                self.dataset_statistics = dataset_statistics

        elif isinstance(datasets, list):
            # Case 2: List of datasets (list of dict)
            self.datasets = []
            for ds_cfg in datasets:
                ds = build_dataset_from_cfg(ds_cfg) if isinstance(
                    ds_cfg, dict) else ds_cfg
                self.datasets.append(ds)

            # Validate all datasets
            for i, ds in enumerate(self.datasets):
                assert hasattr(ds, '__getitem__') and hasattr(ds, '__len__'), \
                    f'Dataset {i} must implement __getitem__ and __len__ methods.'  # noqa: E501

            # Calculate cumulative lengths for efficient indexing
            self.dataset_lens = [len(ds) for ds in self.datasets]
            self.cumulative_lens = np.cumsum([0] + self.dataset_lens).tolist()
            self.total_len = self.cumulative_lens[-1]

            # Collect dataset statistics if available
            stats = []
            for ds in self.datasets:
                assert hasattr(ds, 'stats'), \
                    'Each dataset must have a stats attribute for statistics collection.'  # noqa: E501
                if hasattr(ds, 'stats'):
                    stats.extend(ds.stats)

            self.is_grouped = False
            self.is_list = True

            if dataset_statistics is None:
                self.dataset_statistics = self.get_dataset_statistics(
                    stats, statistic_keys, name_mappings)
            else:
                self.dataset_statistics = dataset_statistics

        else:
            if dataset_statistics is not None:
                raise ValueError(
                    'dataset_statistics_path is only supported for '
                    'single/list dataset configs; grouped datasets should use '
                    'grouped stats.')
            # Case 3: Grouped datasets (dict of list of dict)
            self.grouped_datasets = {}
            self.grouped_dataset_lens = {}
            self.grouped_cumulative_lens = {}
            self.grouped_dataset_statistics = {}

            for group_name, group_datasets in datasets.items():
                # Build all datasets in this group
                group_ds_list = []
                for ds_cfg in group_datasets:
                    ds = build_dataset_from_cfg(ds_cfg) if isinstance(
                        ds_cfg, dict) else ds_cfg
                    group_ds_list.append(ds)

                # Validate all datasets in this group
                for i, ds in enumerate(group_ds_list):
                    assert hasattr(ds, '__getitem__') and hasattr(ds, '__len__'), \
                        f'Dataset {i} in group {group_name} must implement __getitem__ and __len__ methods.'  # noqa: E501

                # Calculate cumulative lengths for this group
                group_lens = [len(ds) for ds in group_ds_list]
                group_cumulative_lens = np.cumsum([0] + group_lens).tolist()

                # Collect dataset statistics for this group
                group_stats = []
                for ds in group_ds_list:
                    assert hasattr(ds, 'stats'), \
                        f'Each dataset in group {group_name} must have a stats attribute for statistics collection.'  # noqa: E501
                    if hasattr(ds, 'stats'):
                        group_stats.extend(ds.stats)

                # Store group information
                self.grouped_datasets[group_name] = group_ds_list
                self.grouped_dataset_lens[group_name] = group_lens
                self.grouped_cumulative_lens[
                    group_name] = group_cumulative_lens

                # Calculate statistics for this group
                self.grouped_dataset_statistics[
                    group_name] = self.get_dataset_statistics(
                        group_stats, statistic_keys, name_mappings)

            # Calculate total length across all groups
            self.total_len = sum(group_cumulative_lens[-1]
                                 for group_cumulative_lens in
                                 self.grouped_cumulative_lens.values())

            self.is_grouped = True
            self.is_list = False

        if statistics_overrides is not None:
            self._apply_configured_statistics_overrides(statistics_overrides)

        # Get the rank and world size from the overwatch
        self.rank = overwatch.rank()
        self.world_size = overwatch.world_size()
        self._epoch = 0

    def _apply_statistics_overrides(self, statistics: Dict,
                                    overrides: Dict) -> None:
        for key, value in overrides.items():
            if (isinstance(value, dict) and key in statistics
                    and isinstance(statistics[key], dict)):
                self._apply_statistics_overrides(statistics[key], value)
            else:
                statistics[key] = value

    def _apply_configured_statistics_overrides(self, overrides: Dict) -> None:
        if not self.is_grouped:
            self._apply_statistics_overrides(self.dataset_statistics,
                                             overrides)
            return

        group_names = set(self.grouped_dataset_statistics)
        if set(overrides).issubset(group_names):
            for group_name, group_overrides in overrides.items():
                self._apply_statistics_overrides(
                    self.grouped_dataset_statistics[group_name],
                    group_overrides)
            return

        for statistics in self.grouped_dataset_statistics.values():
            self._apply_statistics_overrides(statistics, overrides)

    def _get_item_from_global_idx(self, global_idx):
        """Get item from global index, handling single, list,
            and grouped cases.

        Args:
            global_idx (int): The global index of the item to get.

        Returns:
            The item from the dataset.
        """
        if self.is_grouped:
            # Case 3: Grouped datasets
            # Find which group the global index belongs to
            current_idx = 0
            for group_name, group_cumulative_lens in \
                    self.grouped_cumulative_lens.items():
                group_total_len = group_cumulative_lens[-1]
                if global_idx < current_idx + group_total_len:
                    # Index belongs to this group
                    local_idx = global_idx - current_idx

                    # Find which dataset in the group
                    dataset_idx = np.searchsorted(
                        group_cumulative_lens[1:], local_idx, side='right')
                    dataset_local_idx = local_idx - group_cumulative_lens[
                        dataset_idx]

                    # Get the dataset and its statistics
                    dataset = self.grouped_datasets[group_name][dataset_idx]
                    group_statistics = self.grouped_dataset_statistics[
                        group_name]

                    return dataset.__getitem__(dataset_local_idx,
                                               group_statistics)

                current_idx += group_total_len

            raise IndexError(f'Global index {global_idx} is out of range')

        elif self.is_list:
            # Case 2: List of datasets
            # Binary search to find which dataset the index belongs to
            dataset_idx = np.searchsorted(
                self.cumulative_lens[1:], global_idx, side='right')
            local_idx = global_idx - self.cumulative_lens[dataset_idx]
            return self.datasets[dataset_idx].__getitem__(
                local_idx, self.dataset_statistics)
        else:
            # Case 1: Single dataset
            return self.dataset.__getitem__(global_idx,
                                            self.dataset_statistics)

    def get_dataset_statistics(self,
                               stats,
                               static_keys: List[str],
                               name_mappings: Optional[Dict] = None) -> Dict:
        """Collect and combine statistics from multiple datasets.

        Args:
            stats (list[dict]): List of statistics from each dataset.
            static_keys (list[str]): Keys for which to collect statistics.
            name_mappings (dict, optional): Mappings for statistic names.
                Defaults to None.
        Returns:
            dict: Combined statistics for the specified keys.
        """
        dataset_statistics = defaultdict(lambda: defaultdict(list))

        # Collect statistics from each dataset
        for stat in stats:
            for key in static_keys:
                if key not in stat['stats']:
                    raise KeyError(f"Missing dataset statistic key: '{key}'.")

                stat_data = stat['stats'][key]

                # Collect basic statistics
                dataset_statistics[key]['min'].append(stat_data['min'])
                dataset_statistics[key]['max'].append(stat_data['max'])
                dataset_statistics[key]['mean'].append(stat_data['mean'])
                dataset_statistics[key]['std'].append(stat_data['std'])

                # Collect count if available
                if 'count' in stat_data:
                    dataset_statistics[key]['count'].append(stat_data['count'])

                # Collect quantiles if available
                if 'q01' in stat_data:
                    dataset_statistics[key]['q01'].append(stat_data['q01'])
                if 'q99' in stat_data:
                    dataset_statistics[key]['q99'].append(stat_data['q99'])

                # Collect other quantiles if available
                for q in ['q25', 'q50', 'q75']:
                    if q in stat_data:
                        dataset_statistics[key][q].append(stat_data[q])

        # Combine collected statistics
        metadata = {}

        for key in static_keys:
            # Handle name mappings
            if name_mappings and key in name_mappings:
                if isinstance(name_mappings[key], str):
                    mapped_keys = [name_mappings[key]]
                else:
                    assert isinstance(name_mappings[key], list), \
                        f"Expected list for key '{key}' in name_mappings, got {type(name_mappings[key])}"  # noqa: E501
                    mapped_keys = name_mappings[key]
            else:
                mapped_keys = [key]

            # Combine statistics for this key
            key_stats = dataset_statistics[key]
            # Apply dimension padding to individual
            # dataset statistics if dim is specified
            if self.dim is not None:
                for stat_type in [
                        'min', 'max', 'mean', 'std', 'count', 'q01', 'q99',
                        'q25', 'q50', 'q75'
                ]:
                    if stat_type in key_stats and len(
                            key_stats[stat_type]) > 0:
                        padded_stats = []
                        for stat in key_stats[stat_type]:
                            padded_stat = self._pad_statistics_to_dim(
                                {stat_type: stat})[stat_type]
                            padded_stats.append(padded_stat)
                        key_stats[stat_type] = padded_stats

            # Global min and max
            global_min = np.asarray(
                key_stats['min'], dtype=np.float64).min(axis=0).tolist()
            global_max = np.asarray(
                key_stats['max'], dtype=np.float64).max(axis=0).tolist()

            means = np.asarray(key_stats['mean'], dtype=np.float64)
            stds = np.asarray(key_stats['std'], dtype=np.float64)
            counts = self._counts_to_weights(key_stats, means, key)

            weighted_mean, combined_std = self._combine_mean_and_std(
                means, stds, counts)
            weighted_mean = weighted_mean.tolist()
            combined_std = combined_std.tolist()

            # Quantiles can only be approximated without raw samples. Use
            # count-weighted averaging when every dataset provides a compatible
            # count; otherwise median is the least surprising fallback.
            q01_value = self._combine_optional_statistic(
                key_stats, 'q01', counts, len(means))
            q99_value = self._combine_optional_statistic(
                key_stats, 'q99', counts, len(means))

            # Set combined statistics for all mapped keys
            for mapped_key in mapped_keys:
                metadata[mapped_key] = {
                    'mean': weighted_mean,
                    'std': combined_std,
                    'min': global_min,
                    'max': global_max,
                    'q01': q01_value,
                    'q99': q99_value
                }

                # Add other quantiles if available
                for q in ['q25', 'q50', 'q75']:
                    q_value = self._combine_optional_statistic(
                        key_stats, q, counts, len(means))
                    if q_value is not None:
                        metadata[mapped_key][q] = q_value

        return {self.statistic_name: metadata}

    def _combine_mean_and_std(self, means, stds, weights=None):
        """Merge per-dataset mean/std into global population statistics."""
        if weights is None:
            mean = means.mean(axis=0)
            variance = np.mean(stds**2 + (means - mean)**2, axis=0)
        else:
            mean = self._weighted_average(means, weights)
            variance = self._weighted_average(stds**2 + (means - mean)**2,
                                              weights)

        return mean, np.sqrt(np.maximum(variance, 0.0))

    def _combine_optional_statistic(self, key_stats, stat_type, weights,
                                    expected_num_stats):
        if stat_type not in key_stats or len(key_stats[stat_type]) == 0:
            return None

        values = np.asarray(key_stats[stat_type], dtype=np.float64)
        if weights is not None and len(
                key_stats[stat_type]) == expected_num_stats:
            try:
                return self._weighted_average(values, weights).tolist()
            except ValueError:
                pass

        return np.median(values, axis=0).tolist()

    def _counts_to_weights(self, key_stats, values, stat_key):
        counts = key_stats.get('count')
        if counts is None or len(counts) != len(values):
            return None

        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape[0] != values.shape[0]:
            return None
        if np.any(counts < 0):
            raise ValueError(
                f"Negative dataset count in statistic '{stat_key}'")

        if values.ndim == 1:
            if counts.ndim > 1 and np.prod(counts.shape[1:]) != 1:
                raise ValueError(
                    f'Count shape {counts.shape} is incompatible with '
                    f"statistic '{stat_key}' shape {values.shape}")
            weights = counts.reshape(counts.shape[0])
        else:
            value_shape = values.shape[1:]
            if counts.ndim == 1:
                weights = counts.reshape((-1, ) + (1, ) * len(value_shape))
            elif counts.shape[1:] == value_shape:
                weights = counts
            elif np.prod(counts.shape[1:]) == 1:
                weights = counts.reshape((-1, ) + (1, ) * len(value_shape))
            else:
                raise ValueError(
                    f'Count shape {counts.shape} is incompatible with '
                    f"statistic '{stat_key}' shape {values.shape}")

        try:
            broadcast_weights = self._broadcast_weights_for_values(
                weights, values)
        except ValueError as exc:
            raise ValueError(
                f'Count shape {counts.shape} is incompatible with '
                f"statistic '{stat_key}' shape {values.shape}") from exc

        if np.any(broadcast_weights.sum(axis=0) <= 0):
            return None

        return weights

    def _weighted_average(self, values, weights):
        values = np.asarray(values, dtype=np.float64)
        weights = self._broadcast_weights_for_values(weights, values)
        total_weight = weights.sum(axis=0)
        if np.any(total_weight <= 0):
            raise ValueError('Weights must have positive sum.')
        return np.sum(values * weights, axis=0) / total_weight

    def _broadcast_weights_for_values(self, weights, values):
        weights = np.asarray(weights, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        return np.broadcast_to(weights, values.shape)

    def _pad_statistics_to_dim(self, stats_dict):
        """Pad statistics to be an integer multiple of self.dim.

        Args:
            stats_dict: Dictionary containing statistics
                (mean, std, min, max, count, q01, q99, q25, q50, q75)

        Returns:
            Dictionary with padded statistics
        """
        if self.dim is None:
            return stats_dict

        padded_stats = {}
        for key, value in stats_dict.items():
            if isinstance(value, (list, np.ndarray)):
                array_value = np.asarray(value)
                if array_value.ndim == 0 or array_value.size == 0:
                    padded_stats[key] = array_value.item(
                    ) if array_value.ndim == 0 else value
                    continue

                # Get the first dimension (sequence length)
                orig_len = array_value.shape[0]
                # Calculate target length as integer multiple of dim
                target_len = ((orig_len + self.dim - 1) // self.dim) * self.dim

                if target_len == orig_len:
                    # No padding needed
                    padded_stats[key] = array_value.tolist()
                else:
                    # Pad by copying the original data
                    repeat_times = (target_len + orig_len - 1) // orig_len
                    padded_value = np.tile(array_value,
                                           (repeat_times, ) + (1, ) *
                                           (array_value.ndim - 1))[:target_len]
                    padded_stats[key] = padded_value.tolist()
            else:
                # Non-array data or scalar, keep as is
                padded_stats[key] = value

        return padded_stats

    # If we had access to more raw data,
    # we could implement a more precise merging of quantiles.
    def combine_quantiles_precisely(self,
                                    values_list,
                                    counts_list,
                                    quantiles=[0.01, 0.25, 0.5, 0.75, 0.99]):
        # This requires more raw data information;
        # here is just a sample framework
        pass

    def __iter__(self):
        # Incorporate DataLoader worker info so data is split across both
        # distributed processes and per-process DataLoader workers.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        total_world = self.world_size * num_workers
        total_rank = self.rank * num_workers + worker_id

        while True:
            epoch = self._epoch
            if self.reshuffle_each_epoch:
                self._epoch += 1

            # Create indices for the entire virtual concatenated dataset
            indices = np.arange(self.total_len)
            if self.shuffle:
                epoch_offset = epoch if self.reshuffle_each_epoch else 0
                rng = np.random.default_rng(self.seed + epoch_offset)
                rng.shuffle(indices)

            # Distribute indices across distributed ranks and workers.
            shard = indices[total_rank::total_world].tolist()

            for idx in shard:
                yield self._get_item_from_global_idx(idx)

    def __len__(self):
        """Return the total length of all datasets."""
        return self.total_len
