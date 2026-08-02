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

import unittest

import numpy as np

from fluxvla.datasets.parquet_dataset import ParquetDataset


class TestParquetDatasetSampling(unittest.TestCase):

    @staticmethod
    def _make_dataset():
        dataset = ParquetDataset.__new__(ParquetDataset)
        dataset.dataset = {
            'episode_index': [0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        }
        dataset.dataset_cumulative_sizes = np.array([0, 4, 10])
        dataset.full_length = 10
        return dataset

    def test_default_sampling_preserves_frame_proportions(self):
        dataset = self._make_dataset()

        indices = dataset._build_sample_indices(1.0)

        np.testing.assert_array_equal(indices, np.arange(10))

    def test_balanced_sampling_equalizes_data_roots(self):
        dataset = self._make_dataset()

        indices = dataset._build_sample_indices(1.0, balance_data_roots=True)

        first_root = indices[indices < 4]
        second_root = indices[indices >= 4]
        self.assertEqual(len(first_root), 6)
        self.assertEqual(len(second_root), 6)
        np.testing.assert_array_equal(first_root, [0, 1, 1, 2, 3, 3])
        np.testing.assert_array_equal(second_root, [4, 5, 6, 7, 8, 9])

    def test_episode_fraction_is_applied_before_balancing(self):
        dataset = self._make_dataset()

        indices = dataset._build_sample_indices(0.5, balance_data_roots=True)

        np.testing.assert_array_equal(indices, [0, 1, 1, 4, 5, 6])


if __name__ == '__main__':
    unittest.main()
