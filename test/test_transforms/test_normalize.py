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

from fluxvla.transforms.normalize import (DenormalizeLiberoAction,
                                          PostprocessLiberoAction)


class TestPostprocessLiberoAction(unittest.TestCase):

    def test_matches_stats_free_legacy_gripper_processing(self):
        action = np.array(
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.75], dtype=np.float32)
        postprocess = PostprocessLiberoAction()

        actual = postprocess(dict(action=action.copy()))

        np.testing.assert_array_equal(
            actual,
            np.array(
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -1.0],
                dtype=np.float32))

    def test_requires_action(self):
        with self.assertRaisesRegex(AssertionError, 'Action is not found'):
            PostprocessLiberoAction()({})

    def test_legacy_quantile_denormalization_remains_available(self):
        transform = DenormalizeLiberoAction(
            norm_stats={
                'libero_object_no_noops': {
                    'action': {
                        'q01': [0.0, 10.0],
                        'q99': [2.0, 20.0],
                    }
                }
            },
            norm_type='quantile',
            normalize_gripper_action=False,
            invert_gripper_action=False)

        action = transform(
            dict(
                action=np.array([-1.0, 1.0], dtype=np.float32),
                norm_stats_key='libero_object_no_noops'))

        np.testing.assert_array_equal(action, np.array([0.0, 20.0]))


if __name__ == '__main__':
    unittest.main()