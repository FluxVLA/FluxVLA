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

from unittest.mock import patch

import numpy as np
import torch

from fluxvla.transforms.normalize import NormalizeStatesAndActions
from fluxvla.transforms.transform_images import (PrepareVideo, ResizeImages,
                                                 SimpleNormalizeImages)
from fluxvla.transforms.transform_inputs import ProcessLiberoEvalInputs
from fluxvla.transforms.transform_prompts import ProcessCosmos25Prompt


def test_dit4dit_prompt_is_tokenized_in_transform():

    class FakeCosmosTokenizer:
        pad_token_id = 0
        model_max_length = 3

        def apply_chat_template(self, conversations, **kwargs):
            assert conversations[1]['content'][0]['text'] == (
                'put the bowl on the plate')
            assert kwargs['max_length'] == self.model_max_length
            return [11, 12, 0]

    with patch(
            'fluxvla.engines.build_tokenizer_from_cfg',
            return_value=FakeCosmosTokenizer()):
        transform = ProcessCosmos25Prompt(
            tokenizer=dict(type='PretrainedTokenizer'),
            input_key='task_description',
            remove_input_key=True,
        )

    output = transform({'task_description': 'put the bowl on the plate'})

    np.testing.assert_array_equal(output['lang_tokens'], [11, 12, 0])
    np.testing.assert_array_equal(output['lang_masks'], [True, True, False])
    assert 'task_description' not in output


def test_dit4dit_libero_eval_images_match_training_range():
    primary = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.full((8, 8, 3), 255, dtype=np.uint8)
    output = ProcessLiberoEvalInputs(
        img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
        use_pil=False)({
            'agentview_image': primary,
            'robot0_eye_in_hand_image': wrist,
        })
    output = ResizeImages(
        key='pixel_values',
        height=4,
        width=4,
        backend='cv2',
        interpolation='area',
        output_layout='flattened_chw')(
            output)
    output = SimpleNormalizeImages(
        key='pixel_values', preserve_leading_dims=True, output_type='torch')(
            output)
    output = PrepareVideo(
        num_views=2,
        frame_window_size=1,
        tile_direction='horizontal',
        combine_view_masks=True)(
            output)

    pixel_values = output['pixel_values']
    assert isinstance(pixel_values, torch.Tensor)
    assert tuple(pixel_values.shape) == (3, 1, 4, 8)
    assert output['img_masks'] == [True]
    assert torch.isclose(pixel_values[..., :4].max(), torch.tensor(-1.0))
    assert torch.isclose(pixel_values[..., 4:].min(), torch.tensor(1.0))
    assert pixel_values.min() >= -1.0
    assert pixel_values.max() <= 1.0


def test_dit4dit_normalization_keeps_float32_arithmetic_before_fp16_cast():
    transform = NormalizeStatesAndActions(
        state_key='proprio',
        action_key='action',
        state_norm_type='none',
        action_norm_type='min_max',
        normalization_epsilon=0.0,
        preserve_input_dtype=True,
        output_dtype='float16')
    actions = np.asarray([[-2.0, 0.1234567]], dtype=np.float32)
    stats = {
        'action': {
            'min': [-0.987654, -0.25],
            'max': [1.234567, 0.75],
        }
    }

    output = transform({
        'states': np.asarray([0.5], dtype=np.float32),
        'actions': actions,
        'stats': stats,
    })
    expected = ((actions - np.asarray(stats['action']['min'], np.float32)) /
                (np.asarray(stats['action']['max'], np.float32) -
                 np.asarray(stats['action']['min'], np.float32)) * 2.0 -
                1.0).astype(np.float16)

    assert output['states'].dtype == np.float16
    assert output['actions'].dtype == np.float16
    np.testing.assert_array_equal(output['actions'], expected)
    assert output['actions'][0, 0] != np.float16(
        (np.float16(actions[0, 0]) - np.float16(stats['action']['min'][0])) /
        (np.float16(stats['action']['max'][0]) -
         np.float16(stats['action']['min'][0])) * np.float16(2.0) -
        np.float16(1.0))


def test_main_float32_normalization_output_contract_is_preserved():
    transform = NormalizeStatesAndActions(
        state_key='proprio',
        action_key='action',
        state_norm_type='none',
        action_norm_type='quantile',
        output_dtype='float32')
    output = transform({
        'states': np.asarray([1.0], dtype=np.float64),
        'actions': np.asarray([[0.25]], dtype=np.float64),
        'stats': {
            'action': {
                'q01': [0.0],
                'q99': [1.0],
            }
        },
    })

    assert output['states'].dtype == np.float32
    assert output['actions'].dtype == np.float32
    np.testing.assert_allclose(output['actions'], [[-0.5000005]])
