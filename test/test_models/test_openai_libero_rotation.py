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
from pathlib import Path

import numpy as np
import pytest
from mmengine import Config

from fluxvla.models.vlas.openai_responses_vla import OpenAIResponsesVLA
from fluxvla.transforms.normalize import IdentityLiberoAction


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('sign', [-1, 1])
def test_rotation_uses_native_channels_and_is_total_per_call(axis, sign):
    model = OpenAIResponsesVLA()
    rotation = np.zeros(3)
    rotation[axis] = sign * 0.1
    actions = model._actions_from_targets(
        {'rotation_delta': rotation.tolist()}, np.zeros(3))[0].numpy()

    assert actions.shape == (4, 7)
    np.testing.assert_allclose(actions[:, :3], 0)
    np.testing.assert_allclose(actions[:, 6], -1)
    np.testing.assert_allclose(
        actions[:, 3:6].sum(0) * model.rotation_action_scale, rotation)
    # The identity transform must not discard or invert the rotation vector.
    np.testing.assert_allclose(IdentityLiberoAction()({
        'action': actions
    }), actions)


def test_large_combined_rotation_keeps_axis_and_angular_speed_limit():
    model = OpenAIResponsesVLA(action_horizon=3)
    rotation = np.array([1., -2., 3.])
    actions = model._actions_from_targets(
        {'rotation_delta': rotation.tolist()}, np.zeros(3))[0].numpy()
    expected = rotation / np.linalg.norm(rotation) * 0.25

    assert actions.shape == (3, 7)
    np.testing.assert_allclose(
        actions[:, 3:6], np.tile(expected, (3, 1)), rtol=1e-6)
    assert np.all(np.linalg.norm(actions[:, 3:6], axis=1) <= 0.25 + 1e-6)


def test_rotation_shares_chunk_with_translation_and_gripper_settling():
    model = OpenAIResponsesVLA(gripper_settle_steps=8)
    actions = model._actions_from_targets(
        {
            'x': 0.01,
            'rotation_delta': [0, 0, 0.1],
            'gripper': 0
        }, np.zeros(3))[0].numpy()

    assert actions.shape == (8, 7)
    np.testing.assert_allclose(actions[:, :3].sum(0) * 0.01, [0.01, 0, 0])
    np.testing.assert_allclose(actions[:, 3:6].sum(0) * 0.1, [0, 0, 0.1])
    np.testing.assert_allclose(actions[:, 6], 1)
    hold = model._actions_from_targets({}, np.zeros(3))[0].numpy()
    np.testing.assert_allclose(hold[:, :6], 0)
    np.testing.assert_allclose(hold[:, 6], 1)
    model._reset_episode('next trial')
    np.testing.assert_allclose(model._hold_actions()[0, :, 6], -1)


@pytest.mark.parametrize('targets', [
    {
        'yaw': 0.1
    },
    {
        'rotation_delta': [0, 0]
    },
    {
        'rotation_delta': [[0], [0], [0.1]]
    },
    {
        'rotation_delta': '0, 0, 0.1'
    },
    {
        'rotation_delta': [0, 0, float('nan')]
    },
    {
        'rotation_delta': [0, float('inf'), 0]
    },
    {
        'rotation_delta': [False, 0, 0]
    },
    {
        'x': float('nan')
    },
    {
        'y': float('inf')
    },
    {
        'z': '0.1'
    },
    {
        'gripper': None
    },
    {
        'gripper': 2
    },
    {
        'gripper': float('nan')
    },
])
def test_invalid_commands_are_rejected_before_simulation(targets):
    with pytest.raises(RuntimeError):
        OpenAIResponsesVLA()._actions_from_targets(targets, np.zeros(3))


@pytest.mark.parametrize('position', [
    [0, 0],
    [0, 0, 0, 0],
    [0, float('nan'), 0],
    [[0, 0, 0], [0, 0, 0]],
])
def test_invalid_robot_position_is_rejected(position):
    with pytest.raises(RuntimeError, match='eef_position'):
        OpenAIResponsesVLA()._actions_from_targets({}, position)


@pytest.mark.parametrize('kwargs', [
    {
        'rotation_action_scale': 0
    },
    {
        'rotation_action_scale': float('nan')
    },
    {
        'position_action_scale': float('inf')
    },
    {
        'max_rotation_speed_fraction': 0
    },
    {
        'max_rotation_speed_fraction': 1.1
    },
    {
        'workspace_bounds': [[0, 1], [1, 0], [0, 1]]
    },
])
def test_invalid_control_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        OpenAIResponsesVLA(**kwargs)


def test_predict_action_exposes_rotation_and_uses_runtime_budget(monkeypatch):
    model = OpenAIResponsesVLA(max_llm_calls=10)
    captured = {}

    def respond(body):
        captured.update(body)
        return {
            'output': [{
                'type':
                'function_call',
                'name':
                'move_to',
                'call_id':
                'rotate',
                'arguments':
                json.dumps({
                    'targets': {
                        'rotation_delta': [0, 0, 0.1]
                    },
                    'note': 'align',
                }),
            }]
        }

    monkeypatch.setattr(model, '_post_json', respond)
    actions = model.predict_action(
        images=[np.zeros((16, 16, 3), dtype=np.uint8)],
        task_description='align the gripper',
        eef_position=np.zeros(3),
        eef_quaternion=np.array([1, 0, 0, 0]),  # Downward, not identity.
        gripper_position=np.zeros(2),
        reset_history=True)

    fields = captured['tools'][0]['parameters']['properties']['targets']
    assert fields['properties']['rotation_delta']['minItems'] == 3
    text = captured['input'][-1]['content'][0]['text']
    assert 'by call 2' in text and 'by call 5' in text and 'by call 9' in text
    assert 'held fixed' not in captured['input'][0]['content']
    np.testing.assert_allclose(actions[0, :, 5], 0.25)
    # Rotation is world-relative; the flipped starting quaternion must not
    # flip the sign of the commanded world-z rotation.
    np.testing.assert_allclose(actions[0, :, :5], 0)


@pytest.mark.parametrize('arguments', ['[]', 'null', '{}'])
def test_malformed_tool_arguments_do_not_silently_hold(monkeypatch, arguments):
    model = OpenAIResponsesVLA()
    monkeypatch.setattr(
        model, '_post_json', lambda _: {
            'output': [{
                'type': 'function_call',
                'name': 'move_to',
                'call_id': 'bad',
                'arguments': arguments,
            }]
        })
    with pytest.raises(RuntimeError, match='must be an object'):
        model.predict_action(
            images=[],
            task_description='test',
            eef_position=np.zeros(3),
            eef_quaternion=np.array([0, 0, 0, 1]),
            gripper_position=np.zeros(2))


def test_rotation_outside_targets_is_not_silently_ignored(monkeypatch):
    model = OpenAIResponsesVLA()
    monkeypatch.setattr(
        model, '_post_json', lambda _: {
            'output': [{
                'type':
                'function_call',
                'name':
                'move_to',
                'call_id':
                'bad',
                'arguments':
                json.dumps({
                    'targets': {},
                    'rotation_delta': [0, 0, 0.2]
                }),
            }]
        })
    with pytest.raises(RuntimeError, match='Unknown move_to arguments'):
        model.predict_action(
            images=[],
            task_description='test',
            eef_position=np.zeros(3),
            eef_quaternion=np.array([0, 0, 0, 1]),
            gripper_position=np.zeros(2))


@pytest.mark.parametrize('suite', ['10', 'spatial', 'object', 'goal'])
def test_suite_configs_inherit_rotation_and_keep_all_tasks(suite):
    cfg = Config.fromfile(
        Path(__file__).resolve().parents[2] /
        f'configs/openai/gpt6_astra_libero_{suite}_inference.py')
    assert cfg.inference_model.rotation_action_scale == 0.1
    assert cfg.inference_model.max_rotation_speed_fraction == 0.25
    assert cfg.inference_model.action_horizon == cfg.eval.eval_chunk_size
    assert cfg.eval.task_ids is None
