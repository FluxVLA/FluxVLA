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

from fluxvla.engines.utils import TOKENIZERS
from fluxvla.transforms.prompters import PreparePromptWithState
from fluxvla.transforms.transform_prompts import ProcessPrompts


@TOKENIZERS.register_module(name='CharacterTokenizerForPromptTest', force=True)
class _CharacterTokenizer:
    """Minimal tokenizer that makes truncation behavior directly observable."""

    def __call__(self, prompt, add_special_tokens=True, **kwargs):
        tokens = [1] if add_special_tokens else []
        tokens.extend(ord(character) for character in prompt)
        return {'input_ids': tokens}


def _decode_character_tokens(tokens, mask):
    valid_tokens = np.asarray(tokens)[np.asarray(mask, dtype=bool)]
    return ''.join(chr(token) for token in valid_tokens[1:])


def test_pi05_prompt_tokenizes_only_the_configured_state_dimensions():
    states = np.zeros(32, dtype=np.float32)
    states[0] = -1.0
    states[13] = 1.0
    original_states = states.copy()
    transform = PreparePromptWithState(max_state_dim=32, token_state_dim=14)

    output = transform({
        'states': states,
        'task_description': 'Move_the_block',
    })

    assert output['prompt'] == (
        'Task: Move the block, State: '
        '0 128 128 128 128 128 128 128 128 128 128 128 128 255;\n'
        'Action: ')
    np.testing.assert_array_equal(output['states'], original_states)


def test_pi05_long_prompt_truncates_task_and_preserves_state_suffix():
    prepare = PreparePromptWithState(max_state_dim=32, token_state_dim=3)
    process = ProcessPrompts(
        tokenizer={'type': 'CharacterTokenizerForPromptTest'},
        max_len=52,
        preserve_suffix_after=', State: ',
    )
    task = 'move every block to the matching container ' * 8
    data = prepare({
        'states': np.array([-1.0, 0.0, 1.0] + [0.0] * 29),
        'task_description': task,
    })

    output = process(data)
    decoded = _decode_character_tokens(output['lang_tokens'],
                                       output['lang_masks'])

    assert output['lang_masks'].sum() <= 52
    assert decoded.startswith('Task: move every')
    assert decoded.endswith(', State: 0 128 255;\nAction: ')
    assert decoded == output['prompt']
