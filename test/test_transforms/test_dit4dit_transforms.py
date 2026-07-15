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

from fluxvla.transforms.transform_inputs import ProcessDiT4DiTLiberoEvalInputs
from fluxvla.transforms.transform_prompts import (CanonicalizePrompt,
                                                  ProcessCosmos25Prompt)


def test_dit4dit_prompt_aliases_are_canonicalized_in_transform():
    transform = CanonicalizePrompt(remove_source_keys=True)

    output = transform({
        'task_description': 'put the bowl on the plate',
        'lang': 'lower-priority alias',
    })

    assert output['prompt'] == 'put the bowl on the plate'
    assert 'task_description' not in output
    assert 'lang' not in output


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
            remove_input_key=True,
        )

    output = transform({'prompt': 'put the bowl on the plate'})

    np.testing.assert_array_equal(output['lang_tokens'], [11, 12, 0])
    np.testing.assert_array_equal(output['lang_masks'], [True, True, False])
    assert 'prompt' not in output


def test_dit4dit_libero_eval_images_match_training_range():
    primary = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.full((8, 8, 3), 255, dtype=np.uint8)
    transform = ProcessDiT4DiTLiberoEvalInputs(image_size=4)

    output = transform({
        'agentview_image': primary,
        'robot0_eye_in_hand_image': wrist,
    })

    pixel_values = output['pixel_values']
    assert isinstance(pixel_values, torch.Tensor)
    assert tuple(pixel_values.shape) == (1, 3, 4, 8)
    assert torch.isclose(pixel_values[..., :4].max(), torch.tensor(-1.0))
    assert torch.isclose(pixel_values[..., 4:].min(), torch.tensor(1.0))
    assert pixel_values.min() >= -1.0
    assert pixel_values.max() <= 1.0
