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
from unittest.mock import patch

import torch

from fluxvla.datasets.transformed_eval_dataset import TransformedEvalDataset


class TestTransformedEvalDataset(unittest.TestCase):

    @staticmethod
    def _build_dataset(transform):
        with patch(
                'fluxvla.datasets.transformed_eval_dataset.'
                'build_transform_from_cfg', return_value=transform):
            return TransformedEvalDataset(
                transforms=[dict(type='FakeTransform')],
                batch_keys=['input_ids', 'state'])

    def test_returns_declared_batch_without_changing_values(self):
        input_ids = torch.tensor([1, 2, 3])
        state = torch.ones(1, 29)
        replay_img = object()

        def transform(data):
            data.update(
                input_ids=input_ids,
                state=state,
                replay_img=replay_img,
                intermediate='not a model input')
            return data

        dataset = self._build_dataset(transform)
        batch, replay = dataset(dict(is_new_episode=True))

        self.assertEqual(set(batch), {'input_ids', 'state', 'reset_history'})
        self.assertIs(batch['input_ids'], input_ids)
        self.assertIs(batch['state'], state)
        self.assertTrue(batch['reset_history'])
        self.assertIs(replay, replay_img)

    def test_missing_declared_batch_key_fails(self):
        dataset = self._build_dataset(
            lambda data: {**data, 'input_ids': torch.tensor([1])})

        with self.assertRaisesRegex(KeyError, r"\['state'\]"):
            dataset({})


if __name__ == '__main__':
    unittest.main()