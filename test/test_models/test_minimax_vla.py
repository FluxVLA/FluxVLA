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

import base64
import copy
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.vlas.minimax_vla import (MiniMaxLiberoVLA,
                                             MiniMaxRobocasaVLA)
from fluxvla.models.vlas.openai_vla import OpenAILiberoVLA, OpenAIRobocasaVLA

ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ('anthropic', 'chat_completions', 'responses')
MOVES = [
    {
        'targets': {
            'x': 0.05,
            'z': 0.9,
            'gripper': 1
        },
        'note': 'approach the bowl'
    },
    {
        'targets': {
            'rotation_delta': [0, 0, 0.1],
            'gripper': 0
        },
        'note': 'align and grasp'
    },
]
GR1_COMMANDS = [
    {
        'left_arm_delta': [0.05, -0.02, 0, 0.3, 0, 0, 0.01],
        'right_hand': 'close',
        'note': 'reach with the left arm'
    },
    {
        'waist_delta': [0.01, 0, -0.2],
        'left_hand': 'close',
        'note': 'turn the waist'
    },
]


def libero_inputs(step):
    image = np.full((8, 8, 3), 40 * step, dtype=np.uint8)
    return dict(
        images=[image, image.copy()],
        image_names=['agentview_image', 'robot0_eye_in_hand_image'],
        task_description='put the bowl on the plate',
        eef_position=np.array([0.0, 0.1, 1.0]),
        eef_quaternion=np.array([0, 0, 0, 1.0]),
        gripper_position=np.zeros(2),
        reset_history=step == 0)


def robocasa_inputs(step):
    return dict(
        images=[np.full((8, 8, 3), 40 * step, dtype=np.uint8)],
        task_description='move the cup to the drawer',
        left_arm=np.zeros(7),
        left_hand=np.zeros(6),
        right_arm=np.zeros(7),
        right_hand=np.zeros(6),
        waist=np.zeros(3),
        reset_history=step == 0)


def native_reply(protocol, call_id, name, arguments):
    """A MiniMax reply in the wire format of ``protocol``."""
    if protocol == 'anthropic':
        return {
            'id':
            f'msg_{call_id}',
            'type':
            'message',
            'role':
            'assistant',
            'model':
            'MiniMax-M3.1-Flash',
            'content': [
                {
                    'type': 'thinking',
                    'thinking': f'plan for {call_id}',
                    'signature': f'sig-{call_id}'
                },
                {
                    'type': 'text',
                    'text': 'Moving now.'
                },
                {
                    'type': 'tool_use',
                    'id': call_id,
                    'name': name,
                    'input': arguments
                },
            ],
            'stop_reason':
            'tool_use',
            'usage': {
                'input_tokens': 11,
                'output_tokens': 7
            },
            'base_resp': {
                'status_code': 0,
                'status_msg': ''
            },
        }
    if protocol == 'chat_completions':
        return {
            'id':
            f'chat_{call_id}',
            'model':
            'MiniMax-M3.1-Flash',
            'choices': [{
                'index': 0,
                'finish_reason': 'tool_calls',
                'message': {
                    'role':
                    'assistant',
                    'name':
                    'MiniMax AI',
                    'content':
                    'Moving now.',
                    'reasoning_content':
                    f'plan for {call_id}',
                    'tool_calls': [{
                        'id': call_id,
                        'type': 'function',
                        'function': {
                            'name': name,
                            'arguments': json.dumps(arguments)
                        },
                        'index': 0,
                    }],
                    'audio_content':
                    '',
                },
            }],
            'usage': {
                'prompt_tokens': 11,
                'completion_tokens': 7
            },
            'base_resp': {
                'status_code': 0,
                'status_msg': ''
            },
        }
    return {
        'id':
        f'resp_{call_id}',
        'model':
        'MiniMax-M3.1-Flash',
        'status':
        'completed',
        'output': [
            {
                'type':
                'reasoning',
                'id':
                f'{call_id}_rs',
                'status':
                'completed',
                'summary': [],
                'content': [{
                    'type': 'reasoning_text',
                    'text': f'plan for {call_id}'
                }],
            },
            {
                'type': 'function_call',
                'id': f'{call_id}_fc_0',
                'status': 'completed',
                'call_id': call_id,
                'name': name,
                'arguments': json.dumps(arguments),
            },
        ],
        'usage': {
            'input_tokens': 11,
            'output_tokens': 7
        },
    }


class FakeTransport:
    """Replace ``_http_post``; record requests and replay canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, url, headers, body, api_name=None):
        self.requests.append(
            dict(url=url, headers=dict(headers), body=copy.deepcopy(body)))
        return copy.deepcopy(self.replies.pop(0))


def run_minimax(monkeypatch, model, name, commands, inputs_fn):
    monkeypatch.setenv(model.api_key_env, 'test-key')
    transport = FakeTransport([
        native_reply(model.api_protocol, f'call_{index}', name, command)
        for index, command in enumerate(commands)
    ])
    monkeypatch.setattr(model, '_http_post', transport)
    actions = [
        model.predict_action(**inputs_fn(step))
        for step in range(len(commands))
    ]
    return transport, actions


def run_reference(monkeypatch, model, name, commands, inputs_fn):
    replies = [
        native_reply('responses', f'call_{index}', name, command)
        for index, command in enumerate(commands)
    ]
    monkeypatch.setattr(model, '_post_json', lambda _: replies.pop(0))
    return [
        model.predict_action(**inputs_fn(step))
        for step in range(len(commands))
    ]


def image_blocks(content):
    return [block for block in content if block['type'] == 'image']


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_libero_actions_match_openai_responses_policy(monkeypatch, protocol):
    model = MiniMaxLiberoVLA(api_protocol=protocol)
    _, actions = run_minimax(monkeypatch, model, 'move_to', MOVES,
                             libero_inputs)
    expected = run_reference(monkeypatch, OpenAILiberoVLA(), 'move_to', MOVES,
                             libero_inputs)

    for got, want in zip(actions, expected):
        assert got.dtype == torch.float32
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert model.last_note == 'align and grasp'
    assert model.last_response_metadata['input_tokens'] == 11
    assert model.last_response_metadata['output_tokens'] == 7


def test_anthropic_request_merges_turns_and_replays_thinking(monkeypatch):
    model = MiniMaxLiberoVLA(image_horizon=1)
    transport, _ = run_minimax(monkeypatch, model, 'move_to', MOVES,
                               libero_inputs)
    first, second = transport.requests

    assert first['url'] == 'https://api.minimax.cn/anthropic/v1/messages'
    assert first['headers'] == {
        'Authorization': 'Bearer test-key',
        'Content-Type': 'application/json',
        'User-Agent': 'FluxVLA/MiniMaxLiberoVLA',
    }
    body = first['body']
    assert body['model'] == 'MiniMax-M3.1-Flash'
    assert body['max_tokens'] == 16384
    assert body['output_config'] == {'effort': 'low'}
    assert body['system'] == model.system_prompt
    # The Messages API only accepts tool_choice {'type': 'auto'}.
    assert body['tool_choice'] == {'type': 'auto'}
    assert body['tools'] == [{
        'name': 'move_to',
        'description': model.tools[0]['description'],
        'input_schema': model.tools[0]['parameters'],
    }]
    assert 'thinking' not in body and 'temperature' not in body
    # Goal and first observation share one user turn.
    assert [message['role'] for message in body['messages']] == ['user']
    content = body['messages'][0]['content']
    assert content[0] == {
        'type': 'text',
        'text': 'Goal: put the bowl on the plate'
    }
    images = image_blocks(content)
    assert len(images) == 2
    for image in images:
        assert image['source']['type'] == 'base64'
        assert image['source']['media_type'] == 'image/jpeg'
        assert image['source']['detail'] == 'default'
        assert base64.b64decode(image['source']['data'])[:2] == b'\xff\xd8'

    messages = second['body']['messages']
    assert [message['role']
            for message in messages] == ['user', 'assistant', 'user']
    # The raw assistant turn, including the signed thinking block, is
    # replayed verbatim.
    reply = native_reply('anthropic', 'call_0', 'move_to', MOVES[0])
    assert messages[1]['content'] == reply['content']
    assert messages[2]['content'][0] == {
        'type': 'tool_result',
        'tool_use_id': 'call_0',
        'content': model._history[4]['output'],
    }
    assert messages[2]['content'][1]['text'].startswith('Current observation.')
    assert len(image_blocks(messages[2]['content'])) == 2
    # image_horizon=1 strips the earlier frames from the first turn.
    assert not image_blocks(messages[0]['content'])
    assert {
        'type': 'text',
        'text': '[2 prior camera frame(s) omitted]'
    } in messages[0]['content']


def test_chat_completions_request_replays_reasoning_content(monkeypatch):
    model = MiniMaxLiberoVLA(api_protocol='chat_completions')
    transport, _ = run_minimax(monkeypatch, model, 'move_to', MOVES,
                               libero_inputs)
    first, second = transport.requests

    assert first['url'] == 'https://api.minimax.cn/v1/chat/completions'
    assert first['headers'] == {
        'Authorization': 'Bearer test-key',
        'Content-Type': 'application/json',
        'User-Agent': 'FluxVLA/MiniMaxLiberoVLA',
    }
    body = first['body']
    assert body['tools'] == [{
        'type': 'function',
        'function': {
            'name': 'move_to',
            'description': model.tools[0]['description'],
            'parameters': model.tools[0]['parameters'],
        },
    }]
    # Neither field is part of MiniMax's Chat Completions reference.
    assert 'tool_choice' not in body and 'parallel_tool_calls' not in body
    assert 'max_tokens' not in body and 'thinking' not in body
    assert body['reasoning_effort'] == 'low'
    assert body['max_completion_tokens'] == 16384
    assert [message['role']
            for message in body['messages']] == ['system', 'user', 'user']
    assert body['messages'][1]['content'] == (
        'Goal: put the bowl on the plate')
    parts = body['messages'][2]['content']
    images = [part for part in parts if part['type'] == 'image_url']
    assert len(images) == 2
    for image in images:
        assert image['image_url']['url'].startswith('data:image/jpeg;base64,')
        assert image['image_url']['detail'] == 'default'

    messages = second['body']['messages']
    assert [message['role'] for message in messages
            ] == ['system', 'user', 'user', 'assistant', 'tool', 'user']
    reply = native_reply('chat_completions', 'call_0', 'move_to', MOVES[0])
    assert messages[3] == reply['choices'][0]['message']
    assert messages[4] == {
        'role': 'tool',
        'tool_call_id': 'call_0',
        'content': model._history[4]['output'],
    }


def test_responses_request_replays_reasoning_items(monkeypatch):
    model = MiniMaxLiberoVLA(api_protocol='responses')
    transport, _ = run_minimax(monkeypatch, model, 'move_to', MOVES,
                               libero_inputs)
    first, second = transport.requests

    assert first['url'] == 'https://api.minimax.cn/v1/responses'
    body = first['body']
    assert body['tools'] == model.tools
    assert body['tool_choice'] == 'auto'
    assert 'parallel_tool_calls' not in body
    assert body['reasoning'] == {'effort': 'low'}
    assert body['max_output_tokens'] == 16384
    images = [
        item for item in body['input'][2]['content']
        if item['type'] == 'input_image'
    ]
    assert len(images) == 2
    for image in images:
        assert set(image) == {'type', 'image_url'}
        assert image['image_url']['url'].startswith('data:image/jpeg;base64,')
        assert image['image_url']['detail'] == 'default'

    reply = native_reply('responses', 'call_0', 'move_to', MOVES[0])
    items = second['body']['input']
    assert items[3:5] == reply['output']
    assert items[5]['type'] == 'function_call_output'
    assert items[5]['call_id'] == 'call_0'


def test_required_tool_choice_is_opt_in_for_chat_completions(monkeypatch):
    model = MiniMaxLiberoVLA(
        api_protocol='chat_completions', tool_choice='required')
    transport, _ = run_minimax(monkeypatch, model, 'move_to', MOVES[:1],
                               libero_inputs)
    assert transport.requests[0]['body']['tool_choice'] == 'required'


@pytest.mark.parametrize('protocol', ['anthropic', 'responses'])
def test_required_tool_choice_is_rejected_where_undocumented(protocol):
    with pytest.raises(ValueError, match="only accepts tool_choice='auto'"):
        MiniMaxLiberoVLA(api_protocol=protocol, tool_choice='required')


def test_assistant_replay_can_be_disabled(monkeypatch):
    model = MiniMaxLiberoVLA(replay_assistant_turns=False)
    transport, _ = run_minimax(monkeypatch, model, 'move_to', MOVES,
                               libero_inputs)
    assert transport.requests[1]['body']['messages'][1]['content'] == [{
        'type':
        'tool_use',
        'id':
        'call_0',
        'name':
        'move_to',
        'input':
        MOVES[0],
    }]


def test_new_episode_drops_stored_assistant_turns(monkeypatch):
    model = MiniMaxLiberoVLA()
    run_minimax(monkeypatch, model, 'move_to', MOVES[:1], libero_inputs)
    assert model._assistant_turns
    model._reset_episode('another task')
    assert model._assistant_turns == {}


@pytest.mark.parametrize('content', [[],
                                     [
                                         {
                                             'type': 'thinking',
                                             'thinking': 'planning',
                                             'signature': 's'
                                         },
                                         {
                                             'type': 'text',
                                             'text': 'Moving now.'
                                         },
                                     ]])
def test_reply_without_action_stops_generation_after_one_retry(
        monkeypatch, content):
    monkeypatch.setenv('MINIMAX_CN_API_KEY', 'test-key')
    model = MiniMaxLiberoVLA()
    reply = {'content': content, 'stop_reason': 'end_turn'}
    transport = FakeTransport([reply, reply])
    monkeypatch.setattr(model, '_http_post', transport)
    actions = model.predict_action(**libero_inputs(0))
    torch.testing.assert_close(actions, model._hold_actions())
    assert len(transport.requests) == 2
    assert model._llm_calls == 2
    assert model._action_generation_stopped
    assert model.last_response_metadata['stop_reason'] == 'missing_tool_call'
    assert model._assistant_turns == {}
    assert all(item.get('type') != 'function_call' for item in model._history)
    history = copy.deepcopy(model._history)
    for step in range(1, 10):
        torch.testing.assert_close(
            model.predict_action(**libero_inputs(step)), actions)
    assert len(transport.requests) == 2
    assert model._llm_calls == 2  # Do not pretend the budget was exhausted.
    assert model._history == history


@pytest.mark.parametrize('cls, tool, commands, inputs_fn', [
    (MiniMaxLiberoVLA, 'move_to', MOVES, libero_inputs),
    (MiniMaxRobocasaVLA, 'control_gr1', GR1_COMMANDS, robocasa_inputs),
])
def test_end_turn_recovery_preserves_actions_and_history(
        monkeypatch, cls, tool, commands, inputs_fn):
    model = cls(max_llm_calls=3)
    monkeypatch.setenv(model.api_key_env, 'test-key')
    replies = [
        native_reply('anthropic', f'call_{index}', tool, command)
        for index, command in enumerate(commands)
    ]
    transport = FakeTransport([{
        'content': [{
            'type': 'text',
            'text': 'Moving now.'
        }],
        'stop_reason':
        'end_turn',
    }, *replies])
    monkeypatch.setattr(model, '_http_post', transport)
    first = model.predict_action(**inputs_fn(0))
    assert model._llm_calls == 2
    second = model.predict_action(**inputs_fn(1))
    assert model._llm_calls == 3
    assert len(transport.requests) == 3
    reference = cls(max_llm_calls=3)
    _, expected = run_minimax(monkeypatch, reference, tool, commands,
                              inputs_fn)
    for actual, want in zip((first, second), expected):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    original = transport.requests[0]['body']
    repair = transport.requests[1]['body']
    assert repair['tools'] == original['tools']
    assert repair['tool_choice'] == {'type': 'auto'}
    assert repair['messages'][-1]['content'][:-1] == \
        original['messages'][-1]['content']
    reminder = repair['messages'][-1]['content'][-1]['text']
    assert f'Use exactly one {tool} tool call' in reminder
    assert 'API calls used: 1/3.' in reminder
    actual_tools = [
        item for item in model._history
        if item.get('type') in ('function_call', 'function_call_output')
    ]
    reference_tools = [
        item for item in reference._history
        if item.get('type') in ('function_call', 'function_call_output')
    ]
    assert actual_tools == reference_tools
    assert 'No tool call was returned' not in json.dumps(model._history)
    assert 'Action call: 3/3.' in json.dumps(transport.requests[-1]['body'])
    assert model._assistant_turns == reference._assistant_turns
    model.predict_action(**inputs_fn(2))
    assert len(transport.requests) == 3  # No request beyond the call budget.


def test_end_turn_at_call_budget_does_not_retry(monkeypatch):
    model = MiniMaxLiberoVLA(max_llm_calls=1)
    monkeypatch.setenv(model.api_key_env, 'test-key')
    transport = FakeTransport([{
        'content': [{
            'type': 'text',
            'text': 'Moving now.'
        }],
        'stop_reason':
        'end_turn',
    }])
    monkeypatch.setattr(model, '_http_post', transport)
    actions = model.predict_action(**libero_inputs(0))
    torch.testing.assert_close(actions, model._hold_actions())
    assert model._action_generation_stopped
    assert len(transport.requests) == model._llm_calls == 1


@pytest.mark.parametrize('cls, tool, commands, inputs_fn', [
    (MiniMaxLiberoVLA, 'move_to', MOVES, libero_inputs),
    (MiniMaxRobocasaVLA, 'control_gr1', GR1_COMMANDS, robocasa_inputs),
])
def test_stopped_policy_preserves_control_state_and_resets_next_episode(
        monkeypatch, cls, tool, commands, inputs_fn):
    model = cls(max_llm_calls=5)
    monkeypatch.setenv(model.api_key_env, 'test-key')
    missing = {
        'id': 'text-only',
        'content': [{
            'type': 'text',
            'text': 'I am done.'
        }],
        'stop_reason': 'end_turn',
    }
    transport = FakeTransport([
        native_reply('anthropic', 'before', tool, commands[-1]),
        missing,
        missing,
        native_reply('anthropic', 'next-episode', tool, commands[0]),
    ])
    monkeypatch.setattr(model, '_http_post', transport)
    model.predict_action(**inputs_fn(0))
    inputs = inputs_fn(1)
    if tool == 'control_gr1':
        inputs['left_arm'] = np.arange(7) * 0.01
        inputs['right_arm'] = -np.arange(7) * 0.02
        inputs['waist'] = np.array([0.02, 0.03, -0.01])
        expected = model._hold_robocasa_actions(inputs['left_arm'],
                                                inputs['right_arm'],
                                                inputs['waist'])
    else:
        expected = model._hold_actions()
        assert torch.all(expected[..., :6] == 0)
        assert torch.all(expected[..., 6] == 1)  # Keep the closed gripper.
    torch.testing.assert_close(model.predict_action(**inputs), expected)
    assert model._action_generation_stopped
    assert model._llm_calls == 3
    assert model.last_response_metadata['id'] == 'text-only'
    assert model.last_response_metadata['response_text'] == 'I am done.'
    torch.testing.assert_close(model.predict_action(**inputs), expected)
    assert len(transport.requests) == 3
    model.predict_action(**inputs_fn(0))  # Same task, a new trial.
    assert not model._action_generation_stopped
    assert len(transport.requests) == 4
    assert model._llm_calls == 1
    assert model.last_response_metadata['id'] == 'msg_next-episode'
    assert 'before' not in model._assistant_turns


@pytest.mark.parametrize('cls, inputs_fn', [
    (OpenAILiberoVLA, libero_inputs),
    (OpenAIRobocasaVLA, robocasa_inputs),
])
def test_openai_empty_responses_remain_errors(monkeypatch, cls, inputs_fn):
    model = cls()
    monkeypatch.setattr(model, '_post_json', lambda _: {'output': []})
    with pytest.raises(RuntimeError, match='Expected exactly one'):
        model.predict_action(**inputs_fn(0))
    assert not model._action_generation_stopped


def test_libero_runner_finishes_failed_episode_and_runs_next(
        monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fluxvla.engines import build_runner_from_cfg
    from fluxvla.engines.runners import base_eval_runner
    from fluxvla.engines.runners import libero_eval_runner as module

    obs = {
        'agentview_image': np.zeros((16, 16, 3), dtype=np.uint8),
        'robot0_eye_in_hand_image': np.zeros((16, 16, 3), dtype=np.uint8),
        'robot0_eef_pos': np.zeros(3),
        'robot0_eef_quat': np.array([0, 0, 0, 1]),
        'robot0_gripper_qpos': np.zeros(2),
        'robot0_joint_pos': np.zeros(7),
    }

    class FakeEnv:
        episodes = 0
        closed = False

        def reset(self):
            self.episodes += 1

        def set_init_state(self, _):
            return obs.copy()

        def step(self, action):
            return obs.copy(), 0, self.episodes == 2, {}

        def close(self):
            self.closed = True

    env = FakeEnv()
    task = SimpleNamespace(language='put the bowl on the plate')
    suite = SimpleNamespace(
        n_tasks=1,
        get_task=lambda _: task,
        get_task_init_states=lambda _: [None, None])
    monkeypatch.setattr(
        module, '_get_libero_benchmark', lambda: SimpleNamespace(
            get_benchmark_dict=lambda: {'libero_object': lambda: suite}))
    monkeypatch.setattr(module, 'get_libero_env', lambda *args, **kwargs:
                        (env, task.language))
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'local_rank', lambda: 0, raising=False)
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'distributed_state', None, raising=False)
    monkeypatch.setattr(module.overwatch, 'rank', lambda: 0)
    monkeypatch.setattr(module.overwatch, 'world_size', lambda: 1)
    monkeypatch.setattr(module.torch.cuda, 'current_device', lambda: 'cpu')
    for name in ('broadcast_object_list', 'all_reduce', 'barrier'):
        monkeypatch.setattr(module.dist, name, lambda *args, **kwargs: None)
    cfg = load_config(monkeypatch,
                      'minimax_m31_flash_libero_object_inference.py')
    cfg.eval.update(
        task_ids=[0],
        num_trials_per_task=2,
        num_steps_wait=0,
        max_steps=10,
        resize_size=16,
        output_dir=str(tmp_path),
        save_rollout_videos=False,
        save_multi_view_rollout_videos=False)
    cfg.eval.dataset.resize_size = 16
    runner_cfg = cfg.eval.copy()
    runner_cfg.cfg, runner_cfg.ckpt_path = cfg, None
    runner = build_runner_from_cfg(runner_cfg)
    monkeypatch.setenv(runner.vla.api_key_env, 'test-key')
    missing = {'content': [], 'stop_reason': 'end_turn'}
    transport = FakeTransport([
        missing,
        missing,
        native_reply('anthropic', 'second-episode', 'move_to', MOVES[0]),
    ])
    monkeypatch.setattr(runner.vla, '_http_post', transport)
    runner.run()
    assert env.closed and env.episodes == 2
    assert len(transport.requests) == 3
    assert runner.vla._llm_calls == 1
    assert not runner.vla._action_generation_stopped
    summary = json.loads((Path(runner.run_dir) / 'summary.json').read_text())
    stats = summary['suite_stats']['libero_object']
    assert stats['total_trials'] == 2
    assert stats['total_successes'] == 1
    task_result = json.loads((Path(runner.run_dir) / 'libero_object' /
                              'task0_results.json').read_text())
    assert task_result['failure_episodes'] == [0]
    assert task_result['success_episodes'] == [1]


@pytest.mark.parametrize('protocol, reply', [
    ('anthropic', {
        'content': [{
            'type': 'thinking',
            'thinking': '...',
            'signature': 's'
        }],
        'stop_reason': 'max_tokens'
    }),
    ('chat_completions', {
        'choices': [{
            'finish_reason': 'length',
            'message': {
                'role': 'assistant',
                'content': ''
            }
        }]
    }),
    ('responses', {
        'status': 'incomplete',
        'incomplete_details': {
            'reason': 'max_output_tokens'
        },
        'output': []
    }),
])
def test_thinking_truncation_points_to_max_output_tokens(
        monkeypatch, protocol, reply):
    monkeypatch.setenv('MINIMAX_CN_API_KEY', 'test-key')
    model = MiniMaxLiberoVLA(api_protocol=protocol)
    monkeypatch.setattr(model, '_http_post', FakeTransport([reply]))
    with pytest.raises(RuntimeError, match='max_output_tokens'):
        model.predict_action(**libero_inputs(0))


@pytest.mark.parametrize('protocol, reply', [
    ('chat_completions', {
        'choices': [{
            'finish_reason': 'content_filter',
            'message': {
                'role': 'assistant',
                'content': ''
            }
        }]
    }),
    ('responses', {
        'status': 'incomplete',
        'incomplete_details': {
            'reason': 'content_filter'
        },
        'output': []
    }),
])
def test_content_filter_is_reported(monkeypatch, protocol, reply):
    monkeypatch.setenv('MINIMAX_CN_API_KEY', 'test-key')
    model = MiniMaxLiberoVLA(api_protocol=protocol)
    monkeypatch.setattr(model, '_http_post', FakeTransport([reply]))
    with pytest.raises(RuntimeError, match='content_filter'):
        model.predict_action(**libero_inputs(0))


def test_minimax_error_payload_is_not_treated_as_empty_reply(monkeypatch):
    monkeypatch.setenv('MINIMAX_CN_API_KEY', 'test-key')
    model = MiniMaxLiberoVLA()
    monkeypatch.setattr(
        model, '_http_post',
        FakeTransport([{
            'base_resp': {
                'status_code': 1008,
                'status_msg': 'insufficient balance'
            }
        }]))
    with pytest.raises(RuntimeError, match='1008'):
        model.predict_action(**libero_inputs(0))


class FakeHTTPResponse(io.BytesIO):

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def fake_urlopen(monkeypatch, outcomes):
    """Serve ``outcomes`` in order: an int raises that HTTP error, a dict is
    returned as the JSON body."""
    from fluxvla.models.vlas import minimax_vla
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        outcome = outcomes[len(requests) - 1]
        if isinstance(outcome, int):
            raise urllib.error.HTTPError(request.full_url, outcome, 'error',
                                         {}, io.BytesIO(b'{"error": "busy"}'))
        return FakeHTTPResponse(json.dumps(outcome).encode('utf-8'))

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(minimax_vla.time, 'sleep', lambda _: None)
    return requests


def test_transport_retries_http_errors_and_sends_documented_auth(monkeypatch):
    monkeypatch.setenv('MINIMAX_CN_API_KEY', 'test-key')
    reply = native_reply('anthropic', 'call_0', 'move_to', MOVES[0])
    requests = fake_urlopen(monkeypatch, [429, 529, reply])
    model = MiniMaxLiberoVLA(max_retries=2, request_timeout=7.0)

    actions = model.predict_action(**libero_inputs(0))
    assert actions.shape[0] == 1 and len(requests) == 3
    request, timeout = requests[-1]
    assert request.full_url == ('https://api.minimax.cn/anthropic/v1/'
                                'messages')
    assert request.get_method() == 'POST'
    assert request.get_header('Authorization') == 'Bearer test-key'
    assert request.get_header('X-api-key') is None
    assert request.get_header('Anthropic-version') is None
    assert json.loads(request.data)['model'] == 'MiniMax-M3.1-Flash'
    assert timeout == 7.0

    fake_urlopen(monkeypatch, [401])
    with pytest.raises(
            RuntimeError, match='MiniMax Messages API returned HTTP 401'):
        model.predict_action(**libero_inputs(1))


def test_transport_retries_retryable_base_resp_codes(monkeypatch):
    monkeypatch.setenv('MINIMAX_CN_API_KEY', 'test-key')
    reply = native_reply('chat_completions', 'call_0', 'move_to', MOVES[0])
    rate_limited = {
        'base_resp': {
            'status_code': 1002,
            'status_msg': 'rate limit'
        }
    }
    requests = fake_urlopen(monkeypatch, [rate_limited, reply])
    model = MiniMaxLiberoVLA(api_protocol='chat_completions')
    model.predict_action(**libero_inputs(0))
    assert len(requests) == 2
    assert requests[-1][0].full_url == (
        'https://api.minimax.cn/v1/chat/completions')

    # Insufficient balance (1008) is final and must not be retried.
    requests = fake_urlopen(monkeypatch, [{
        'base_resp': {
            'status_code': 1008,
            'status_msg': 'insufficient balance'
        }
    }])
    with pytest.raises(RuntimeError, match='1008'):
        model.predict_action(**libero_inputs(1))
    assert len(requests) == 1

    requests = fake_urlopen(monkeypatch, [rate_limited] * 3)
    with pytest.raises(RuntimeError, match='1002'):
        model.predict_action(**libero_inputs(2))
    assert len(requests) == model.max_retries + 1


@pytest.mark.parametrize('region, env', [('cn', 'MINIMAX_CN_API_KEY'),
                                         ('intl', 'MINIMAX_INTL_API_KEY')])
def test_missing_api_key_names_region_and_variable(monkeypatch, region, env):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(RuntimeError, match=f"'{region}' API key.*'{env}'"):
        MiniMaxLiberoVLA(region=region).predict_action(**libero_inputs(0))


@pytest.mark.parametrize('region, other_env', [
    ('cn', 'MINIMAX_INTL_API_KEY'),
    ('intl', 'MINIMAX_CN_API_KEY'),
])
def test_no_cross_region_or_generic_key_fallback(monkeypatch, region,
                                                 other_env):
    for env in ('MINIMAX_CN_API_KEY', 'MINIMAX_INTL_API_KEY'):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(other_env, 'other-region-key')
    monkeypatch.setenv('MINIMAX_API_KEY', 'generic-key')
    model = MiniMaxLiberoVLA(region=region)
    transport = FakeTransport([])
    monkeypatch.setattr(model, '_http_post', transport)
    with pytest.raises(RuntimeError, match=f"'{region}' API key"):
        model.predict_action(**libero_inputs(0))
    assert transport.requests == []


REGION_DEFAULTS = {
    'cn': ('https://api.minimax.cn/v1', 'https://api.minimax.cn/anthropic',
           'MINIMAX_CN_API_KEY'),
    'intl': ('https://api.minimax.io/v1', 'https://api.minimax.io/anthropic',
             'MINIMAX_INTL_API_KEY'),
}


def resolved_endpoints(model):
    return model.base_url, model.anthropic_base_url, model.api_key_env


@pytest.mark.parametrize('cls', [MiniMaxLiberoVLA, MiniMaxRobocasaVLA])
@pytest.mark.parametrize('region', sorted(REGION_DEFAULTS))
def test_region_selects_endpoints_and_key_variable(cls, region):
    model = cls(region=region)
    assert model.region == region
    assert resolved_endpoints(model) == REGION_DEFAULTS[region]


@pytest.mark.parametrize('protocol, url', [
    ('anthropic', 'https://api.minimax.io/anthropic/v1/messages'),
    ('chat_completions', 'https://api.minimax.io/v1/chat/completions'),
    ('responses', 'https://api.minimax.io/v1/responses'),
])
def test_intl_requests_use_intl_endpoint_and_key(monkeypatch, protocol, url):
    monkeypatch.setenv('MINIMAX_CN_API_KEY', 'cn-key')
    model = MiniMaxLiberoVLA(region='intl', api_protocol=protocol)
    monkeypatch.setenv('MINIMAX_INTL_API_KEY', 'intl-key')
    transport = FakeTransport(
        [native_reply(protocol, 'call_0', 'move_to', MOVES[0])])
    monkeypatch.setattr(model, '_http_post', transport)
    model.predict_action(**libero_inputs(0))
    assert transport.requests[0]['url'] == url
    assert transport.requests[0]['headers']['Authorization'] == (
        'Bearer intl-key')


def test_explicit_endpoints_and_key_variable_override_region():
    model = MiniMaxLiberoVLA(
        region='intl',
        base_url='https://gateway.example/v1/',
        anthropic_base_url='https://gateway.example/anthropic/',
        api_key_env='GATEWAY_KEY')
    assert resolved_endpoints(model) == ('https://gateway.example/v1',
                                         'https://gateway.example/anthropic',
                                         'GATEWAY_KEY')


@pytest.mark.parametrize('kwargs', [
    {
        'region': 'intl',
        'base_url': 'https://api.minimax.cn/v1'
    },
    {
        'region': 'cn',
        'anthropic_base_url': 'https://api.minimax.io/anthropic'
    },
    {
        'region': 'intl',
        'api_key_env': 'MINIMAX_CN_API_KEY'
    },
    {
        'region': 'cn',
        'api_key_env': 'MINIMAX_INTL_API_KEY'
    },
])
def test_other_region_endpoint_or_key_is_rejected(kwargs):
    with pytest.raises(ValueError, match='set region='):
        MiniMaxLiberoVLA(**kwargs)


@pytest.mark.parametrize('region', ['CN', 'global', 'us', ''])
def test_unknown_region_is_rejected(region):
    with pytest.raises(ValueError, match='region must be one of'):
        MiniMaxLiberoVLA(region=region)


@pytest.mark.parametrize('kwargs', [
    {
        'api_protocol': 'gemini'
    },
    {
        'tool_choice': 'any'
    },
    {
        'tool_choice': 'required'
    },
    {
        'max_output_tokens': 0
    },
    {
        'api_protocol': 'anthropic',
        'max_output_tokens': None
    },
])
def test_invalid_protocol_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MiniMaxLiberoVLA(**kwargs)


@pytest.mark.parametrize('cls', [MiniMaxLiberoVLA, MiniMaxRobocasaVLA])
def test_constructor_defaults_and_base_passthrough(cls):
    model = cls(action_horizon=3, device='cpu', torch_dtype=None)
    assert model.model == 'MiniMax-M3.1-Flash'
    assert model.region == 'cn'
    assert resolved_endpoints(model) == REGION_DEFAULTS['cn']
    assert model.api_protocol == 'anthropic'
    assert model.tool_choice == 'auto'
    assert model.reasoning_effort == 'low'
    assert model.image_detail == 'default'
    assert model.max_output_tokens == 16384
    assert model.action_horizon == 3
    with pytest.raises(RuntimeError, match='inference-only'):
        model()


def test_robocasa_constructor_keeps_robocasa_signature():
    model = MiniMaxRobocasaVLA(max_arm_joint_delta=0.2)
    assert model.max_arm_joint_delta == 0.2
    assert model.max_llm_calls == 90
    assert model.tools[0]['name'] == 'control_gr1'
    with pytest.raises(TypeError):
        MiniMaxRobocasaVLA(workspace_bounds=[[0, 1], [0, 1], [0, 1]])


@pytest.mark.parametrize('protocol', PROTOCOLS)
def test_robocasa_actions_match_openai_responses_policy(monkeypatch, protocol):
    model = MiniMaxRobocasaVLA(api_protocol=protocol)
    transport, actions = run_minimax(monkeypatch, model, 'control_gr1',
                                     GR1_COMMANDS, robocasa_inputs)
    expected = run_reference(monkeypatch, OpenAIRobocasaVLA(), 'control_gr1',
                             GR1_COMMANDS, robocasa_inputs)

    for got, want in zip(actions, expected):
        assert got.shape == (1, 8, 29)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    tools = transport.requests[0]['body']['tools']
    names = [
        tool['function']['name'] if 'function' in tool else tool['name']
        for tool in tools
    ]
    assert names == ['control_gr1']
    if protocol == 'anthropic':
        assert tools[0]['input_schema'] == model.tools[0]['parameters']
        reply = native_reply(protocol, 'call_0', 'control_gr1',
                             GR1_COMMANDS[0])
        messages = transport.requests[1]['body']['messages']
        assert messages[1]['content'] == reply['content']
        assert messages[2]['content'][0]['tool_use_id'] == 'call_0'


LIBERO_CONFIGS = {
    'minimax_m31_flash_libero_inference.py': ('libero_object', True),
    'minimax_m31_flash_libero_10_inference.py': ('libero_10', False),
    'minimax_m31_flash_libero_goal_inference.py': ('libero_goal', False),
    'minimax_m31_flash_libero_object_inference.py': ('libero_object', True),
    'minimax_m31_flash_libero_spatial_inference.py': ('libero_spatial', False),
}


def load_config(monkeypatch, name, region=None):
    for key in ('MINIMAX_REGION', 'MINIMAX_BASE_URL',
                'MINIMAX_ANTHROPIC_BASE_URL', 'MINIMAX_API_KEY_ENV'):
        monkeypatch.delenv(key, raising=False)
    if region is not None:
        monkeypatch.setenv('MINIMAX_REGION', region)
    # Generic SDK settings must not redirect the MiniMax policy.
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
    return Config.fromfile(ROOT / 'configs/minimax' / name)


def build_eval_model(cfg):
    model_cfg = copy.deepcopy(cfg.inference_model)
    # Mirrors BaseEvalRunner.prepare_eval_model_cfg for model_build_device.
    model_cfg['device'] = cfg.eval.model_build_device
    return build_vla_from_cfg(model_cfg)


def assert_minimax_defaults(model_cfg):
    assert model_cfg.model == 'MiniMax-M3.1-Flash'
    assert model_cfg.region == 'cn'
    assert model_cfg.api_protocol == 'anthropic'
    # Left to the policy so that a region switch also switches these.
    assert model_cfg.base_url is None
    assert model_cfg.anthropic_base_url is None
    assert model_cfg.api_key_env is None
    assert model_cfg.tool_choice == 'auto'
    assert model_cfg.reasoning_effort == 'low'
    assert model_cfg.image_detail == 'default'
    assert model_cfg.max_output_tokens >= 8192


@pytest.mark.parametrize('name', sorted(LIBERO_CONFIGS))
def test_libero_configs_build_without_network(monkeypatch, name):
    suite, keeps_hints = LIBERO_CONFIGS[name]
    cfg = load_config(monkeypatch, name)

    assert cfg.inference_model.type == 'MiniMaxLiberoVLA'
    assert_minimax_defaults(cfg.inference_model)
    assert cfg.eval.type == 'LiberoEvalRunner'
    assert cfg.eval.task_suite_name == suite
    assert cfg.eval.task_ids is None
    assert cfg.eval.model_family == 'minimax-m3.1-flash'
    assert cfg.eval.requires_dataset_stats is False
    assert cfg.eval.dataset.type == 'OpenAILiberoEvalDataset'
    assert cfg.eval.denormalize_action.type == 'IdentityLiberoAction'
    assert cfg.inference_model.action_horizon == cfg.eval.eval_chunk_size
    assert bool(cfg.inference_model.task_visual_hints) == keeps_hints

    model = build_eval_model(cfg)
    assert isinstance(model, MiniMaxLiberoVLA)
    assert resolved_endpoints(model) == REGION_DEFAULTS['cn']
    assert model.tools[0]['name'] == 'move_to'


def test_robocasa_config_builds_without_network(monkeypatch):
    cfg = load_config(monkeypatch, 'minimax_m31_flash_robocasa_inference.py')
    # The reference config's top-level ``from os import ...`` would otherwise
    # switch mmengine into lazy-import mode, which cannot evaluate it.
    reference = Config.fromfile(
        ROOT / 'configs/openai/gpt6_astra_robocasa_inference.py',
        lazy_import=False)

    assert cfg.inference_model.type == 'MiniMaxRobocasaVLA'
    assert_minimax_defaults(cfg.inference_model)
    assert cfg.eval.type == 'RobocasaEvalRunner'
    assert cfg.eval.model_family == 'minimax-m3.1-flash'
    assert cfg.eval.dataset.type == 'OpenAIRobocasaEvalDataset'
    assert cfg.eval.task_list == reference.eval.task_list
    assert cfg.eval.action_keys == reference.eval.action_keys
    assert (cfg.inference_model.action_horizon == cfg.eval.eval_chunk_size ==
            reference.eval.eval_chunk_size)

    model = build_eval_model(cfg)
    assert isinstance(model, MiniMaxRobocasaVLA)
    assert resolved_endpoints(model) == REGION_DEFAULTS['cn']
    assert model.tools[0]['name'] == 'control_gr1'


REGION_CONFIGS = [
    'minimax_m31_flash_libero_inference.py',
    'minimax_m31_flash_libero_10_inference.py',
    'minimax_m31_flash_robocasa_inference.py',
]


@pytest.mark.parametrize('name', REGION_CONFIGS)
def test_minimax_region_env_selects_intl(monkeypatch, name):
    cfg = load_config(monkeypatch, name, region='intl')
    assert cfg.inference_model.region == 'intl'
    model = build_eval_model(cfg)
    assert model.region == 'intl'
    assert resolved_endpoints(model) == REGION_DEFAULTS['intl']


@pytest.mark.parametrize('name', REGION_CONFIGS)
def test_cfg_options_region_switch_changes_endpoints_and_key(
        monkeypatch, name):
    cfg = load_config(monkeypatch, name)
    # Equivalent to --cfg-options inference_model.region=intl.
    cfg.merge_from_dict({'inference_model.region': 'intl'})
    assert resolved_endpoints(build_eval_model(cfg)) == (
        REGION_DEFAULTS['intl'])

    cfg = load_config(monkeypatch, name, region='intl')
    cfg.merge_from_dict({'inference_model.region': 'cn'})
    assert resolved_endpoints(build_eval_model(cfg)) == REGION_DEFAULTS['cn']


def test_invalid_minimax_region_env_fails_at_build(monkeypatch):
    cfg = load_config(
        monkeypatch, 'minimax_m31_flash_libero_inference.py', region='mars')
    with pytest.raises(ValueError, match='region must be one of'):
        build_eval_model(cfg)


def test_config_reads_gateway_settings_from_environment(monkeypatch):
    monkeypatch.setenv('MINIMAX_BASE_URL', 'https://gateway.example/v1/')
    monkeypatch.setenv('MINIMAX_ANTHROPIC_BASE_URL',
                       'https://gateway.example/anthropic/')
    monkeypatch.setenv('MINIMAX_API_KEY_ENV', 'GATEWAY_KEY')
    for region in REGION_DEFAULTS:
        monkeypatch.setenv('MINIMAX_REGION', region)
        cfg = Config.fromfile(
            ROOT / 'configs/minimax/minimax_m31_flash_libero_inference.py')
        model = build_eval_model(cfg)
        assert model.region == region
        assert resolved_endpoints(model) == (
            'https://gateway.example/v1', 'https://gateway.example/anthropic',
            'GATEWAY_KEY')
