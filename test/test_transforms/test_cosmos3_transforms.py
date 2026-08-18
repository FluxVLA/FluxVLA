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

import numpy as np

from fluxvla.transforms.transform_cosmos3 import (BuildCosmos3Sequence,
                                                  ProcessCosmos3Prompt,
                                                  SetCosmos3ActionMetadata)


class _FakeCosmosTokenizer:

    def __init__(self):
        self.conversations = None
        self.kwargs = None

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations = conversations
        self.kwargs = kwargs
        return [11, 12, 13, 14]


def test_cosmos3_eval_metadata_reaches_sequence_builder():
    metadata = SetCosmos3ActionMetadata(
        conditioning_fps=20.0, prepend_state_to_action=False)
    sequence = BuildCosmos3Sequence(
        mode='wam',
        frame_window_size=17,
        raw_action_dim=2,
        conditioning_fps=20.0)
    data = metadata({
        'actions': np.zeros((16, 4), dtype=np.float32),
        'embodiment_ids': np.array(5),
    })

    output = sequence(data)

    assert output['conditioning_fps'].item() == 20.0
    assert output['action_fps'].item() == 20.0
    assert output['raw_action_dim'].item() == 2
    assert output['sequence_plan'].has_action is True
    assert output['sequence_plan'].condition_frame_indexes_vision == [0]
    assert output['sequence_plan'].condition_frame_indexes_action == []
    assert output['sequence_plan'].action_start_frame_offset == 1


def test_cosmos3_json_prompt_produces_token_ids_and_mask():
    transform = ProcessCosmos3Prompt(
        tokenizer={'type': 'PretrainedTokenizer'},
        max_len=3,
        format_prompt_as_json=True,
        action_metadata={
            'append_viewpoint': False,
            'frame_window_size': 17,
            'conditioning_fps': 20.0,
            'video_height': 192,
            'video_width': 320,
        },
        output_attention_mask_key='lang_masks',
    )
    tokenizer = _FakeCosmosTokenizer()
    transform._tokenizer = tokenizer

    output = transform({
        'task_description': 'move the mug',
        'action_horizon': np.array(16),
    })
    prompt = json.loads(output['task_description'])

    assert prompt['fps'] == 20.0
    assert prompt['resolution'] == {'H': 192, 'W': 320}
    assert prompt['actions'][0]['description'] == 'move the mug.'
    assert prompt['actions'][0]['idle_frame'] == '0 out of 16.'
    assert output['text_token_ids'].tolist() == [11, 12, 13]
    assert output['lang_masks'].tolist() == [True, True, True]
    assert tokenizer.conversations == [{
        'role': 'user',
        'content': output['task_description'],
    }]
    assert tokenizer.kwargs['add_vision_id'] is False
