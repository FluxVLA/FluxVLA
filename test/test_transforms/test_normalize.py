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

from fluxvla.transforms.normalize import DenormalizeLiberoAction


def test_libero_denormalize_accepts_transformed_actions_stats_key():
    transform = DenormalizeLiberoAction(
        norm_stats={
            'libero_native': {
                'actions': {
                    'q01': [-2.0, 10.0],
                    'q99': [2.0, 20.0],
                },
            },
        },
        action_dim=2,
        norm_type='quantile',
        normalize_gripper_action=False,
        invert_gripper_action=False,
    )

    output = transform({
        'action': np.array([-1.0, 1.0], dtype=np.float32),
        'norm_stats_key': 'libero_native',
    })

    np.testing.assert_allclose(output, [-2.0, 20.0])


def test_libero_denormalize_preserves_raw_action_stats_key():
    transform = DenormalizeLiberoAction(
        norm_stats={
            'libero': {
                'action': {
                    'mean': [1.0, 2.0],
                    'std': [2.0, 4.0],
                },
            },
        },
        action_dim=2,
        norm_type='mean_std',
        normalize_gripper_action=False,
        invert_gripper_action=False,
    )

    output = transform({
        'action': np.array([0.5, -0.5], dtype=np.float32),
        'norm_stats_key': 'libero',
    })

    np.testing.assert_allclose(output, [2.0, 0.0])
