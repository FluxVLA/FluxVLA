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

import hashlib
import math
import unittest

import numpy as np

from fluxvla.datasets.balanced_dataset_wrapper import \
    DistributedBalancedRepeatingDataset
from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset


def _make_balanced_wrapper(*,
                           source_lengths,
                           sampling_probabilities,
                           total_len,
                           seed,
                           shuffle=True,
                           reshuffle_each_epoch=True):
    wrapper = DistributedBalancedRepeatingDataset.__new__(
        DistributedBalancedRepeatingDataset)
    wrapper.source_lengths = list(source_lengths)
    wrapper.sampling_probabilities = sampling_probabilities
    wrapper.total_len = total_len
    wrapper.seed = seed
    wrapper.shuffle = shuffle
    wrapper.reshuffle_each_epoch = reshuffle_each_epoch
    return wrapper


class TestDistributedBalancedRepeatingDataset(unittest.TestCase):

    def test_weighted_mapping_matches_pre_rebase_dit4dit_algorithm(self):
        source_lengths = [3, 7, 11]
        probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
        wrapper = _make_balanced_wrapper(
            source_lengths=source_lengths,
            sampling_probabilities=probabilities,
            total_len=97,
            seed=123,
            shuffle=True)

        for epoch in range(3):
            for virtual_index in range(wrapper.total_len):
                value = repr(
                    (epoch, virtual_index, wrapper.seed)).encode('utf-8')
                digest = hashlib.sha256(value).hexdigest()
                mapping_seed = int(digest, 16) & ((1 << 128) - 1)
                rng = np.random.default_rng(mapping_seed)
                source_index = int(
                    rng.choice(len(source_lengths), p=probabilities))
                sample_index = int(rng.choice(source_lengths[source_index]))

                self.assertEqual(
                    wrapper._sample_dataset_and_index(epoch, virtual_index),
                    (source_index, sample_index))

            indices = np.arange(wrapper.total_len, dtype=np.int64)
            rng = np.random.default_rng(wrapper.seed + epoch)
            rng.shuffle(indices)
            np.testing.assert_array_equal(
                wrapper._ordered_virtual_indices(epoch), indices)
            for rank in range(4):
                np.testing.assert_array_equal(
                    wrapper._shard_virtual_indices(epoch, rank, 4),
                    indices[rank::4])

    def test_unweighted_mapping_matches_main_balanced_cycle(self):
        source_lengths = [3, 7, 11]
        wrapper = _make_balanced_wrapper(
            source_lengths=source_lengths,
            sampling_probabilities=None,
            total_len=33,
            seed=456,
            shuffle=True)

        for epoch in range(3):
            rng = np.random.default_rng(wrapper.seed + 104729 * epoch)
            source_order = rng.permutation(len(source_lengths))
            source_offsets = np.asarray(
                [rng.integers(length) for length in source_lengths],
                dtype=np.int64)
            for virtual_index in range(wrapper.total_len):
                source_slot = virtual_index % len(source_lengths)
                source_index = int(source_order[source_slot])
                cycle = virtual_index // len(source_lengths)
                sample_index = int((source_offsets[source_index] + cycle) %
                                   source_lengths[source_index])
                self.assertEqual(
                    wrapper._sample_dataset_and_index(epoch, virtual_index),
                    (source_index, sample_index))

            permutation_rng = np.random.default_rng(wrapper.seed +
                                                    130363 * epoch + 17)
            multiplier = int(permutation_rng.integers(1, wrapper.total_len))
            while math.gcd(multiplier, wrapper.total_len) != 1:
                multiplier = (multiplier + 1) % wrapper.total_len
                if multiplier == 0:
                    multiplier = 1
            offset = int(permutation_rng.integers(wrapper.total_len))
            for rank in range(4):
                positions = np.arange(
                    rank, wrapper.total_len, 4, dtype=np.int64)
                expected = (multiplier * positions +
                            offset) % wrapper.total_len
                np.testing.assert_array_equal(
                    wrapper._shard_virtual_indices(epoch, rank, 4), expected)


class TestDistributedRepeatingDatasetStatistics(unittest.TestCase):

    def _make_wrapper(self, dim=None):
        wrapper = DistributedRepeatingDataset.__new__(
            DistributedRepeatingDataset)
        wrapper.statistic_name = 'private'
        wrapper.dim = dim
        return wrapper

    def test_combines_weighted_mean_and_std_with_scalar_counts(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 8.0],
                        'max': [2.0, 12.0],
                        'mean': [1.0, 10.0],
                        'std': [0.5, 1.0],
                        'count': 2,
                        'q01': [0.0, 8.0],
                        'q99': [2.0, 12.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [3.0, 11.0],
                        'max': [7.0, 17.0],
                        'mean': [5.0, 14.0],
                        'std': [1.5, 2.0],
                        'count': 6,
                        'q01': [3.0, 11.0],
                        'q99': [7.0, 17.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [4.0, 13.0])
        np.testing.assert_allclose(combined['std'], [np.sqrt(4.75), 2.5])
        np.testing.assert_allclose(combined['q01'], [2.25, 10.25])
        np.testing.assert_allclose(combined['q99'], [5.75, 15.75])

    def test_unweighted_std_includes_between_dataset_variance(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0],
                        'max': [0.0],
                        'mean': [0.0],
                        'std': [0.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0],
                        'max': [10.0],
                        'mean': [10.0],
                        'std': [0.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [5.0])
        np.testing.assert_allclose(combined['std'], [5.0])

    def test_combines_weighted_mean_and_std_with_vector_counts(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 10.0],
                        'max': [0.0, 10.0],
                        'mean': [0.0, 10.0],
                        'std': [0.0, 0.0],
                        'count': [1, 9],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0, 20.0],
                        'max': [10.0, 20.0],
                        'mean': [10.0, 20.0],
                        'std': [0.0, 0.0],
                        'count': [9, 1],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [9.0, 11.0])
        np.testing.assert_allclose(combined['std'], [3.0, 3.0])

    def test_incomplete_counts_fall_back_to_unweighted_merge(self):
        wrapper = self._make_wrapper()
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0],
                        'max': [0.0],
                        'mean': [0.0],
                        'std': [0.0],
                        'count': 100,
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [10.0],
                        'max': [10.0],
                        'mean': [10.0],
                        'std': [0.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [5.0])
        np.testing.assert_allclose(combined['std'], [5.0])

    def test_padding_applies_to_quantiles_and_vector_counts(self):
        wrapper = self._make_wrapper(dim=4)
        stats = [
            {
                'stats': {
                    'action': {
                        'min': [0.0, 1.0, 2.0],
                        'max': [0.0, 1.0, 2.0],
                        'mean': [0.0, 1.0, 2.0],
                        'std': [0.0, 0.0, 0.0],
                        'count': [1, 1, 1],
                        'q25': [0.0, 1.0, 2.0],
                    }
                }
            },
            {
                'stats': {
                    'action': {
                        'min': [4.0, 5.0, 6.0],
                        'max': [4.0, 5.0, 6.0],
                        'mean': [4.0, 5.0, 6.0],
                        'std': [0.0, 0.0, 0.0],
                        'count': [3, 3, 3],
                        'q25': [4.0, 5.0, 6.0],
                    }
                }
            },
        ]

        combined = wrapper.get_dataset_statistics(
            stats, ['action'])['private']['action']

        np.testing.assert_allclose(combined['mean'], [3.0, 4.0, 5.0, 3.0])
        np.testing.assert_allclose(combined['q25'], [3.0, 4.0, 5.0, 3.0])


if __name__ == '__main__':
    unittest.main()
