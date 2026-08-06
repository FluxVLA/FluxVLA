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

import torch

from fluxvla.collators.qwen_vl_action_prediction_collator import \
    QwenVLSplitActionPredictionCollator


class TestQwenVLSplitActionPredictionCollator(unittest.TestCase):

    def test_pads_tokens_concatenates_vision_and_stacks_actions(self):
        collator = QwenVLSplitActionPredictionCollator(
            pad_token_id=99, padding_side='left')
        features = [
            dict(
                input_ids=torch.tensor([10, 11]),
                attention_mask=torch.tensor([1, 1]),
                mm_token_type_ids=torch.tensor([1, 1]),
                pixel_values=torch.full((2, 3), 1.0),
                image_grid_thw=torch.tensor([[1, 2, 2], [1, 1, 2]]),
                state=torch.full((1, 4), 1.0),
                action=torch.full((2, 4), 1.0),
                action_mask=torch.ones(2, 4),
                embodiment_id=torch.tensor(2),
                expanded_text='raw sample 0'),
            dict(
                input_ids=torch.tensor([20, 21, 22, 23]),
                attention_mask=torch.tensor([1, 1, 1, 1]),
                mm_token_type_ids=torch.tensor([0, 0, 1, 1]),
                pixel_values=torch.full((1, 3), 2.0),
                image_grid_thw=torch.tensor([[1, 1, 1]]),
                state=torch.full((1, 4), 2.0),
                action=torch.full((2, 4), 2.0),
                action_mask=torch.ones(2, 4),
                embodiment_id=torch.tensor(2),
                expanded_text='raw sample 1'),
        ]

        inputs = collator(features)['inputs']

        torch.testing.assert_close(
            inputs['input_ids'],
            torch.tensor([[99, 99, 10, 11], [20, 21, 22, 23]]))
        torch.testing.assert_close(
            inputs['attention_mask'],
            torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]))
        torch.testing.assert_close(
            inputs['mm_token_type_ids'],
            torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]))
        self.assertEqual(inputs['pixel_values'].shape, (3, 3))
        self.assertEqual(inputs['image_grid_thw'].shape, (3, 3))
        self.assertEqual(inputs['state'].shape, (2, 1, 4))
        self.assertEqual(inputs['action'].shape, (2, 2, 4))
        self.assertEqual(inputs['action_mask'].shape, (2, 2, 4))
        self.assertEqual(inputs['embodiment_id'].shape, (2, ))
        self.assertNotIn('expanded_text', inputs)

    def test_requires_explicit_pad_token(self):
        collator = QwenVLSplitActionPredictionCollator()

        with self.assertRaisesRegex(ValueError, 'pad_token_id is required'):
            collator([dict(input_ids=torch.tensor([1]))])


if __name__ == '__main__':
    unittest.main()