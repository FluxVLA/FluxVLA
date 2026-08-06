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

from fluxvla.transforms.modality_state_action import (
    resolve_groot_n17_embodiment_key, resolve_groot_n17_flat_slices,
    select_groot_n17_metadata)


class TestGrootN17MetadataResolver(unittest.TestCase):

    @staticmethod
    def _metadata(embodiment_key='libero_sim'):
        modality_config = {
            'state': {
                'modality_keys': ['x', 'gripper']
            },
            'action': {
                'modality_keys': ['x', 'gripper']
            },
        }
        statistics = {
            'state': {
                'x': {
                    'mean': [0.0]
                },
                'gripper': {
                    'mean': [0.0, 0.0]
                },
            },
            'action': {
                'x': {
                    'mean': [0.0]
                },
                'gripper': {
                    'mean': [0.0]
                },
            },
        }
        return ({'modality_configs': {embodiment_key: modality_config}},
                {embodiment_key: statistics})

    def test_resolves_public_libero_name(self):
        self.assertEqual(
            resolve_groot_n17_embodiment_key('LIBERO_PANDA'), 'libero_sim')
        self.assertEqual(
            resolve_groot_n17_embodiment_key(env_name='libero_sim/task0'),
            'libero_sim')

    def test_checkpoint_embodiment_id_takes_precedence(self):
        processor_kwargs, statistics = self._metadata()

        selected = select_groot_n17_metadata(
            processor_kwargs,
            statistics,
            {'libero_sim': 17},
            embodiment_tag='LIBERO_PANDA')

        self.assertEqual(selected['embodiment_key'], 'libero_sim')
        self.assertEqual(selected['embodiment_id'], 17)
        self.assertEqual(selected['embodiment_id_source'], 'checkpoint')
        self.assertEqual(selected['modality_source'], 'checkpoint')

    def test_libero_has_a_validated_id_fallback(self):
        processor_kwargs, statistics = self._metadata()

        selected = select_groot_n17_metadata(
            processor_kwargs,
            statistics,
            None,
            embodiment_tag='libero_sim')

        self.assertEqual(selected['embodiment_id'], 2)
        self.assertEqual(selected['embodiment_id_source'],
                         'validated_default')

    def test_missing_checkpoint_metadata_fails(self):
        processor_kwargs, statistics = self._metadata()
        with self.assertRaisesRegex(KeyError, 'No checkpoint modality config'):
            select_groot_n17_metadata(
                processor_kwargs,
                statistics,
                None,
                embodiment_tag='unsupported_robot')

        other_kwargs, other_statistics = self._metadata('other_robot')
        with self.assertRaisesRegex(KeyError, 'No checkpoint embodiment id'):
            select_groot_n17_metadata(
                other_kwargs,
                other_statistics,
                None,
                embodiment_tag='other_robot')

    def test_flat_layout_comes_from_metadata_for_libero_only(self):
        processor_kwargs, statistics = self._metadata()
        modality_config = processor_kwargs['modality_configs']['libero_sim']

        slices = resolve_groot_n17_flat_slices(
            modality_config,
            statistics['libero_sim'],
            'libero_sim',
            'state')

        self.assertEqual(slices, {'x': (0, 1), 'gripper': (1, 3)})
        with self.assertRaisesRegex(ValueError, 'validated only'):
            resolve_groot_n17_flat_slices(
                modality_config,
                statistics['libero_sim'],
                'other_robot',
                'state')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            resolve_groot_n17_flat_slices(
                modality_config,
                statistics['libero_sim'],
                'libero_sim',
                'state',
                flat_layout='metadata')


if __name__ == '__main__':
    unittest.main()