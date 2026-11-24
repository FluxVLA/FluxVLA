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
"""All native robot/RTC policy entries, with fake sensors and senders."""

import copy
import json
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.engines.runners.aloha_inference_runner import AlohaInferenceRunner
from fluxvla.engines.runners.aloha_rtc_inference_runner import \
    AlohaRTCInferenceRunner
from fluxvla.engines.runners.franka_inference_runner import \
    FrankaInferenceRunner
from fluxvla.engines.runners.oli_inference_runner import OliInferenceRunner
from fluxvla.engines.runners.oli_rtc_inference_runner import \
    OliRTCInferenceRunner
from fluxvla.engines.runners.rebot_dual_inference_runner import \
    RebotDualInferenceRunner
from fluxvla.engines.runners.rebot_dual_rtc_inference_runner import \
    RebotDualRTCInferenceRunner
from fluxvla.engines.runners.tron2_inference_runner import Tron2InferenceRunner
from fluxvla.engines.runners.tron2_rtc_inference_runner import \
    Tron2RTCInferenceRunner
from fluxvla.engines.runners.ur_inference_runner import URInferenceRunner
from fluxvla.engines.runners.ur_rtc_inference_runner import \
    URRTCInferenceRunner

RUNNERS = [(AlohaInferenceRunner, 14), (AlohaRTCInferenceRunner, 14),
           (FrankaInferenceRunner, 16), (RebotDualInferenceRunner, 14),
           (RebotDualRTCInferenceRunner, 14), (Tron2InferenceRunner, 18),
           (Tron2RTCInferenceRunner, 18), (URInferenceRunner, 7),
           (URRTCInferenceRunner, 7), (URInferenceRunner, 8),
           (URRTCInferenceRunner, 8), (FrankaInferenceRunner, 8),
           (OliInferenceRunner, 33), (OliRTCInferenceRunner, 33)]
GOAL = 'Put the cup on the tray.'
LABELS = ['Grasp the cup.', 'Place the cup on the tray.']


class Model(torch.nn.Module):

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.calls = []

    def predict_action(self, **inputs):
        self.calls.append(copy.deepcopy(inputs))
        return torch.arange(5 * self.width).float().reshape(1, 5, self.width)


class Operator:

    def __init__(self):
        self.sent = 0
        self.delay = 0
        self.event = threading.Event()
        self.dispatch_threads = []

    def stop_trajectory(self):
        self.event.set()

    def execute_trajectory(self, *args, **kwargs):
        assert not kwargs['async_exec']  # The policy owns the single actor.
        self.event = kwargs['stop_event']
        trajectories = (
            args or list(kwargs.get('arm_trajectories', {}).values())
            or [kwargs['left_arm_trajectory']])
        count = len(trajectories[0])
        for index in range(count):
            if self.event.is_set():
                return index
            self.send_action(None)
        return count

    def send_action(self, action):
        self.sent += 1
        self.dispatch_threads.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)

    def servoj(self, action):
        pass

    servol = servoj
    movegrip = send_action


def make_runner(monkeypatch, tmp_path, runner_cls, width, async_execution):
    import fluxvla.engines as engines
    from fluxvla.engines.runners import (base_inference_runner,
                                         oli_inference_runner)

    operator, model = Operator(), Model(width)
    inputs = []

    def preprocess(observation):
        inputs.append(copy.deepcopy(observation))
        assert runner._policy_actor is None  # Planning snapshots are barriers.
        return dict(states=torch.zeros(1, width))

    monkeypatch.setattr(engines, 'build_vla_from_cfg', lambda _: model)
    monkeypatch.setattr(engines, 'build_dataset_from_cfg',
                        lambda _: preprocess)
    monkeypatch.setattr(engines, 'build_transform_from_cfg',
                        lambda _: lambda data: data['action'][0])
    monkeypatch.setattr(base_inference_runner, 'build_operator_from_cfg',
                        lambda _: operator)
    monkeypatch.setattr(oli_inference_runner.signal, 'signal', lambda *_: None)
    ros = ModuleType('rospy')
    ros.is_shutdown = lambda: False
    ros.Rate = lambda _: SimpleNamespace(sleep=lambda: None)
    monkeypatch.setitem(sys.modules, 'rospy', ros)
    path = tmp_path / 'meta/episodes.jsonl'
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            dict(
                episode_index=0,
                length=10,
                tasks=[GOAL],
                dense_subtask_names=LABELS,
                dense_subtask_start_frames=[0, 5],
                dense_subtask_end_frames=[5, 10])))
    checkpoint = tmp_path / 'checkpoints/fake.pt'
    checkpoint.parent.mkdir()
    torch.save({}, checkpoint)
    (tmp_path / 'dataset_statistics.json').write_text('{}')
    options = dict(async_execution=async_execution, execute_horizon=2)
    if issubclass(runner_cls, FrankaInferenceRunner):
        options['action_mode'] = 'joint'
        if width == 8:
            options['active_arms'] = ('left', )
    if issubclass(runner_cls, URInferenceRunner) and width == 8:
        options['action_mode'] = 'cartesian'
    if 'RTC' in runner_cls.__name__:
        options.update(rtc_config=dict(enabled=True, prefix_len=2))
        if issubclass(runner_cls, OliRTCInferenceRunner):
            options['async_remaining_actions_threshold'] = 2
    runner = runner_cls(
        cfg=Config(dict(inference_model=dict(type='Fake'))),
        ckpt_path=str(checkpoint),
        dataset={},
        denormalize_action={},
        action_chunk=5,
        max_publish_step=8,
        publish_rate=1000,
        task_descriptions={'1': GOAL},
        enable_mixed_precision=False,
        policy=dict(
            type='HierarchicalPolicy',
            subtask_source=dict(data_root=str(tmp_path)),
            completion_mode='disabled',
            replan_every_chunks=2,
            planner_image_keys=['cam_high'],
            planner=dict(
                type='ScriptedSubtaskPlanner',
                choices=['skill_0000', 'skill_0001'])),
        **options)
    runner._get_user_task_instruction = lambda _: [GOAL] * 4
    runner._poll_keyboard_pause = lambda: False
    runner.update_observation_window = lambda: dict(
        qpos=np.zeros(width), cam_high=np.zeros((8, 8, 3), np.uint8))
    return runner, model, operator, inputs


@pytest.mark.parametrize('async_execution', [False, True])
@pytest.mark.parametrize(
    'runner_cls,width', RUNNERS, ids=[cls.__name__ for cls, _ in RUNNERS])
def test_all_robot_and_rtc_entries_share_policy_lifecycle(
        monkeypatch, tmp_path, runner_cls, width, async_execution):
    runner, model, operator, inputs = make_runner(monkeypatch, tmp_path,
                                                  runner_cls, width,
                                                  async_execution)
    runner._run_episode(GOAL)
    assert operator.sent == 8
    assert runner.policy.controller.total_chunks == 4
    assert runner.policy._pending_horizon is None
    assert runner._policy_actor is None
    assert set(operator.dispatch_threads) == {'fluxvla-policy-actor'}
    assert [data['task_description'] for data in inputs
            ] == [LABELS[0], LABELS[0], LABELS[1], LABELS[1]]
    if 'RTC' in runner_cls.__name__:
        assert ['prev_actions' in data
                for data in model.calls] == [False, True, False, True]
        torch.testing.assert_close(
            model.calls[1]['prev_actions'],
            torch.arange(5 * width).float().reshape(1, 5, width)[:, 2:])
    # A new episode may not carry the previous subtask or RTC prefix.
    runner._run_episode(GOAL)
    assert 'prev_actions' not in model.calls[4]
    assert operator.sent == 16


@pytest.mark.parametrize(
    'runner_cls,width', RUNNERS, ids=[cls.__name__ for cls, _ in RUNNERS])
def test_pause_cancels_actor_and_acknowledges_only_published_steps(
        monkeypatch, tmp_path, runner_cls, width):
    runner, model, operator, _ = make_runner(monkeypatch, tmp_path, runner_cls,
                                             width, True)
    operator.delay = 0.04
    runner._poll_keyboard_pause = lambda: operator.sent >= 1
    runner._run_episode(GOAL)
    assert operator.sent == 1 and len(model.calls) == 1
    assert runner.policy.controller.total_chunks == 1
    assert runner.policy._pending_horizon is None
    assert runner._policy_actor is None
    count = operator.sent
    time.sleep(0.05)
    assert operator.sent == count


def test_oli_continuous_policy_does_not_use_rospy(monkeypatch, tmp_path):
    runner, _, operator, _ = make_runner(monkeypatch, tmp_path,
                                         OliRTCInferenceRunner, 33, True)
    monkeypatch.delitem(sys.modules, 'rospy')
    runner._run_continuous()
    assert operator.sent == 8


def test_old_actor_is_joined_before_a_new_snapshot(monkeypatch, tmp_path):
    runner, _, operator, _ = make_runner(monkeypatch, tmp_path,
                                         AlohaRTCInferenceRunner, 14, True)
    old_actor = threading.Thread(target=operator.event.wait)
    operator._traj_thread = old_actor
    old_actor.start()
    capture = runner.update_observation_window

    def snapshot():
        assert not old_actor.is_alive()
        return capture()

    runner.update_observation_window = snapshot
    runner._run_episode(GOAL)
    assert operator.sent == 8


def test_non_stopping_actor_blocks_planning(monkeypatch, tmp_path):
    runner, model, operator, _ = make_runner(monkeypatch, tmp_path,
                                             AlohaInferenceRunner, 14, True)
    release = threading.Event()
    old_actor = threading.Thread(target=release.wait)
    operator._traj_thread = old_actor
    old_actor.start()
    runner.policy_stop_timeout_s = 0.02
    try:
        with pytest.raises(TimeoutError, match='refusing'):
            runner._run_episode(GOAL)
        assert not model.calls and operator.sent == 0
    finally:
        release.set()
        old_actor.join()


@pytest.mark.parametrize('runner_cls,width', [(AlohaInferenceRunner, 14),
                                              (FrankaInferenceRunner, 16),
                                              (RebotDualInferenceRunner, 14),
                                              (Tron2InferenceRunner, 18)])
def test_plain_vla_keeps_native_async_dispatch(monkeypatch, tmp_path,
                                               runner_cls, width):
    runner, _, operator, _ = make_runner(monkeypatch, tmp_path, runner_cls,
                                         width, True)
    runner.release_policy()
    operator.execute_trajectory = Mock(return_value=None)
    runner._execute_actions(
        np.zeros((3, width)), runner._policy_publish_rate())
    options = operator.execute_trajectory.call_args.kwargs
    assert options['async_exec'] is True
    assert 'stop_event' not in options


def test_plain_oli_rtc_keeps_its_original_scheduler(monkeypatch, tmp_path):
    runner, _, _, _ = make_runner(monkeypatch, tmp_path, OliRTCInferenceRunner,
                                  33, True)
    runner.release_policy()
    runner._run_rtc_instruction = Mock()
    runner._run_episode(GOAL)
    runner._run_rtc_instruction.assert_called_once_with(
        instruction=GOAL, chunk_count=4)


@pytest.mark.parametrize(
    'runner_cls,width', RUNNERS, ids=[cls.__name__ for cls, _ in RUNNERS])
def test_disabled_robot_never_fabricates_execution(monkeypatch, tmp_path,
                                                   runner_cls, width):
    runner, _, operator, _ = make_runner(monkeypatch, tmp_path, runner_cls,
                                         width, True)
    runner.disable_puppet_arm = True
    runner._run_episode(GOAL)
    assert operator.sent == runner.policy.controller.total_chunks == 0
    assert runner.policy._pending_horizon is None
    assert runner._policy_actor is None


@pytest.mark.parametrize('operator_kind', ['base', 'aloha', 'tron2'])
@pytest.mark.parametrize('stop_after', [0, 2, 3])
def test_native_operators_preserve_cancellation_and_publication_count(
        monkeypatch, operator_kind, stop_after):
    from fluxvla.engines.operators.aloha_operator import AlohaOperator
    from fluxvla.engines.operators.base_operator import BaseOperator
    from fluxvla.engines.operators.tron2_operator import Tron2Operator

    ros = ModuleType('rospy')
    ros.is_shutdown = lambda: False
    ros.Rate = lambda _: SimpleNamespace(sleep=lambda: None)
    monkeypatch.setitem(sys.modules, 'rospy', ros)
    event, sent = threading.Event(), []
    if stop_after == 0:
        event.set()

    def send(*args, **kwargs):
        sent.append(args)
        if len(sent) == stop_after:
            event.set()

    cls = {
        'base': BaseOperator,
        'aloha': AlohaOperator,
        'tron2': Tron2Operator
    }[operator_kind]
    operator = object.__new__(cls)
    operator._traj_stop_event = threading.Event()
    operator._traj_thread = None
    operator.command_mode = 'joint'
    operator.trajectory_exec_mode = 'movej'
    operator.send_joints = send
    operator._send_joints = send
    operator._send_gripper = lambda **_: None
    operator._send_head = lambda **_: None
    actions = np.zeros((3, 7))
    options = dict(stop_event=event, dt=0.00001)
    if operator_kind == 'base':
        count = operator.execute_trajectory({'left': actions}, **options)
    elif operator_kind == 'aloha':
        count = operator.execute_trajectory(actions, actions, **options)
    else:
        count = operator.execute_trajectory(actions, actions, np.zeros(3),
                                            np.zeros(3), **options)
    assert count == len(sent) == stop_after
    assert operator._traj_stop_event is event


@pytest.mark.parametrize('length', [1, 3])
@pytest.mark.parametrize('stop_after', [0, 1, 2, 3])
def test_tron2_servoj_reports_waypoints_not_interpolation_samples(
        length, stop_after):
    from fluxvla.engines.operators.tron2_operator import Tron2Operator

    operator = object.__new__(Tron2Operator)
    operator._traj_stop_event = threading.Event()
    operator.trajectory_exec_mode = 'servoj'
    operator.servoj_frequency = 1000
    event, grips, joints = threading.Event(), [], []
    if stop_after == 0:
        event.set()

    def grip(**kwargs):
        grips.append(kwargs)
        if len(grips) == stop_after:
            event.set()

    operator._send_gripper = grip
    operator._servo_step = lambda target, *args: joints.append(target)
    operator._resolve_head_positions = lambda target: list(target)
    arms = np.repeat(np.arange(length)[:, None], 7, axis=1)
    count = operator.execute_trajectory(
        arms,
        arms,
        np.zeros(length),
        np.zeros(length),
        head_trajectory=np.zeros((length, 2)),
        dt=0.001,
        stop_event=event)
    assert count == len(grips) == len(joints) == min(length, stop_after)
    assert [target[0] for target in joints] == list(range(count))


def test_tron2_interrupted_interpolation_is_not_a_completed_waypoint():
    from fluxvla.engines.operators.tron2_operator import Tron2Operator

    operator = object.__new__(Tron2Operator)
    operator._traj_stop_event = threading.Event()
    operator.trajectory_exec_mode = 'servoj'
    operator.servoj_frequency = 4
    event, joints = threading.Event(), []

    def servo(target, *args):
        joints.append(target)
        if len(joints) == 2:
            event.set()

    operator._send_gripper = lambda **_: None
    operator._servo_step = servo
    operator._resolve_head_positions = lambda target: list(target)
    arms = np.repeat(np.arange(3)[:, None], 7, axis=1)
    count = operator.execute_trajectory(
        arms,
        arms,
        np.zeros(3),
        np.zeros(3),
        head_trajectory=np.zeros((3, 2)),
        dt=1,
        stop_event=event)
    assert count == 1 and len(joints) == 2
