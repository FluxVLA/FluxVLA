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
"""Memory, real ACKs, interchangeable planners and no success-oracle leaks."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.engines.runners.serving.serving_runner import (
    FluxVLAServingRunner, PolicyStop)
from fluxvla.engines.utils import build_policy_from_cfg
from fluxvla.harness import AgentHarness, EpisodeMemory, VisualChangeObserver
from fluxvla.planning import OpenAISubtaskPlanner
from fluxvla.policies import AgentHarnessPolicy

ROOT = Path(__file__).resolve().parents[2]
GOAL = 'Put the cup on the tray.'
LABELS = ['Grasp the cup.', 'Place the cup on the tray.']
CAMERAS = ['cam_high']


class Planner:
    completion_mode = 'disabled'

    def __init__(self, choices):
        self.choices = choices
        self.calls = []
        self.reset()

    def reset(self):
        self.index = 0

    def choose(self, **request):
        self.calls.append(copy.deepcopy(request))
        choice = self.choices[min(self.index, len(self.choices) - 1)]
        self.index += 1
        if isinstance(choice, Exception):
            raise choice
        return dict(choice=choice)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = []
        self.resets = 0

    def predict_action(self, **batch):
        self.calls.append(copy.deepcopy(batch))
        return torch.ones(1, 4, 2)

    def reset_inference_history(self):
        self.resets += 1


class Inputs:

    def __init__(self):
        self.calls = []

    def __call__(self, observation):
        self.calls.append(copy.deepcopy(observation))
        return dict(states=torch.zeros(1, 2))


@pytest.fixture
def source(tmp_path):
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
                dense_subtask_end_frames=[5, 10])) + '\n')
    return tmp_path


def settings(source, planner, **harness):
    return dict(
        type='AgentHarnessPolicy',
        subtask_source=dict(data_root=str(source)),
        planner=planner,
        completion_mode='disabled',
        planner_image_keys=CAMERAS,
        harness=harness)


def make_policy(source, choices=('skill_0000', 'skill_0001'), **harness):
    planner, model, inputs = Planner(choices), Model(), Inputs()
    policy = build_policy_from_cfg(
        settings(source, planner, **harness),
        default_args=dict(
            vla=model,
            input_processor=inputs,
            output_processor=lambda actions, **_: (actions[0].numpy(), []),
            enable_mixed_precision=False))
    policy.reset(GOAL)
    return policy, planner, model, inputs


def observation(value=0):
    return dict(
        cam_high=np.full((8, 8, 3), value, np.uint8),
        states=np.zeros(2, np.float32),
        task_description=GOAL,
        predicates={'success': True},
        reward=100,
        info=dict(future_subtask='secret annotation'))


def test_registered_harness_debounces_and_counts_partial_execution(source):
    policy, planner, model, inputs = make_policy(
        source, choices=['skill_0000', 'skill_0001', 'skill_0001'])
    assert isinstance(policy, AgentHarnessPolicy)
    assert not policy.supports_per_request_inference
    policy.step(observation(), 0)
    assert policy.harness.memory.executed_steps == 0
    with pytest.raises(RuntimeError, match='Acknowledge'):
        policy.step(observation(), 1)
    policy.acknowledge(2)
    policy.step(observation(10), 2)  # Minimum two executed chunks.
    assert len(planner.calls) == 1
    policy.acknowledge(1)
    policy.step(observation(20), 3)  # First different candidate vote.
    assert policy.controller.active.choice == 'skill_0000'
    policy.acknowledge(1)
    output = policy.step(observation(30), 4)  # Confirm on a fresh execution.
    assert policy.controller.active.choice == 'skill_0001'
    assert output.metadata['harness']['selection']['reason'] == \
        'confirmed_switch'
    assert [call['task_description'] for call in inputs.calls] == \
        [LABELS[0]] * 3 + [LABELS[1]]
    assert model.resets == 2
    policy.acknowledge(2)
    memory = policy.harness.memory.snapshot()
    assert memory['executed_chunks'] == 4 and memory['executed_steps'] == 6
    assert memory['completion'] == 'unknown'
    assert all(s['completion'] == 'unverified'
               for s in memory['subtask_summaries'])


def test_invalid_and_zero_ack_never_claim_execution(source):
    policy, _, _, _ = make_policy(source)
    policy.step(observation(), 0)
    for count in (-1, 5, True):
        with pytest.raises(ValueError):
            policy.acknowledge(count)
    assert policy.harness.memory.executed_steps == 0
    assert policy.harness.memory.pending is not None
    policy.acknowledge(0)
    assert policy.harness.memory.pending is None
    assert not policy.harness.memory.outcomes
    with pytest.raises(ValueError):
        policy.acknowledge(0)
    policy.step(observation(), 1)
    assert policy.harness.no_change_chunks == 0
    assert policy.harness.memory.last_observation['state'] == 'unknown'


def test_memory_is_per_episode_not_per_goal_and_pending_is_unknown(
        source, tmp_path):
    policy, _, _, _ = make_policy(
        source, memory=dict(trace_dir=str(tmp_path / 'trace')))
    policy.set_episode_context('run-a:1')
    policy.step(observation(), 0)
    old_path = policy.harness.memory.trace_path
    policy.reset(GOAL)
    policy.set_episode_context('run-a:2')
    assert policy.harness.memory.executed_steps == 0
    assert not policy.harness.memory.recent
    assert policy.harness.active is None
    rows = [json.loads(line) for line in old_path.read_text().splitlines()]
    assert rows[-1]['event'] == 'episode_interrupted'
    assert rows[-1]['execution_status'] == 'unknown'
    policy.step(observation(), 0)
    assert policy.harness.memory.trace_path != old_path
    with pytest.raises(RuntimeError):
        policy.set_episode_context('run-a:3')


def test_memory_windows_and_saved_keyframes_are_bounded(tmp_path):
    memory = EpisodeMemory(
        recent_events=3,
        summary_subtasks=2,
        recent_keyframes=2,
        saved_keyframes=2,
        trace_dir=tmp_path)
    images = {'camera': np.zeros((8, 8, 3), np.uint8)}
    for index in range(10):
        memory.issue(
            dict(choice=str(index), instruction='original label'), index, 8,
            index)
        memory.acknowledge(3)
        memory.capture(images, index)
    snapshot = memory.snapshot()
    assert len(snapshot['recent_events']) == 3
    assert len(snapshot['subtask_summaries']) == 2
    assert snapshot['evicted_subtasks'] == 8
    assert snapshot['executed_steps'] == 30
    assert len(snapshot['keyframes']) == 2
    assert len(list(memory.trace_path.parent.glob('*.png'))) == 2
    assert len(memory.trace_path.read_text().splitlines()) == 31
    saved = json.loads(memory.trace_path.with_name('memory.json').read_text())
    assert saved['pending'] is None and saved['executed_steps'] == 30


def test_initial_choice_does_not_save_duplicate_keyframes(source, tmp_path):
    policy, _, _, _ = make_policy(
        source, memory=dict(trace_dir=str(tmp_path / 'trace')))
    policy.step(observation(), 0)
    memory = policy.harness.memory
    assert len(memory.keyframes) == memory.saved_keyframes == 1
    assert len(list(memory.trace_path.parent.glob('*.png'))) == 1


def test_observer_is_not_a_completion_detector_and_does_not_read_truth():
    observer = VisualChangeObserver(CAMERAS)
    first = observer.observe(observation(), 0, executed=False)
    assert first['state'] == 'unknown'
    unchanged = observer.observe(observation(), 8, executed=True)
    assert unchanged['state'] == 'no_observed_change'
    changed = observer.observe(observation(255), 16, executed=True)
    assert changed['state'] == 'changed'
    assert changed['semantic_progress'] == changed['completion'] == 'unknown'
    assert 'secret annotation' not in repr(changed)
    cancelled = observer.observe(observation(0), 17, executed=False)
    assert cancelled['state'] == 'unknown'


@pytest.mark.parametrize('options', [
    dict(min_subtask_chunks=0),
    dict(switch_confirmations=True),
    dict(min_subtask_chunks=4, max_hold_chunks=2),
    dict(no_change_patience=0),
    dict(max_stall_replans=-1),
    dict(stop_on_stall=1),
    dict(memory=dict(recent_events=0)),
    dict(observer=dict(threshold=float('nan'))),
])
def test_invalid_settings_are_rejected(options):
    with pytest.raises(ValueError):
        AgentHarness(CAMERAS, **options)


@pytest.mark.parametrize('stop', [False, True])
def test_stall_replanning_is_bounded_and_never_fabricates_actions(
        source, stop):
    policy, planner, model, _ = make_policy(
        source,
        choices=['skill_0000'],
        no_change_patience=1,
        max_stall_replans=2,
        stop_on_stall=stop)
    for index in range(3):
        policy.step(observation(), index * 2)
        policy.acknowledge(2)
    result = policy.step(observation(), 6)
    assert policy.harness.stall_replans == 2
    assert all(call['facts'] == {} for call in planner.calls)
    if stop:
        assert result.status == 'failed' and result.actions is None
        assert result.metadata['planning']['reason'] == \
            'harness_no_observed_change'
        assert len(model.calls) == 3
    else:
        assert result.status == 'ready' and len(model.calls) == 4
    policy.reset(GOAL)
    assert policy.harness.no_change_chunks == policy.harness.stall_replans == 0


def test_invalid_planner_choice_is_not_hidden_by_switch_debouncing(source):
    policy, _, model, _ = make_policy(
        source,
        choices=['skill_0000', 'not_a_training_candidate'],
        min_subtask_chunks=1)
    policy.step(observation(), 0)
    policy.acknowledge(2)
    result = policy.step(observation(255), 2)
    assert result.status == 'failed' and result.actions is None
    assert len(model.calls) == 1
    assert policy.harness.memory.pending is None


def test_stall_budget_does_not_bypass_debouncing_forever(source):
    policy, planner, _, _ = make_policy(
        source,
        choices=['skill_0000', 'skill_0001', 'skill_0000'],
        no_change_patience=1,
        max_stall_replans=1,
        switch_confirmations=99)
    policy.step(observation(), 0)
    policy.acknowledge(2)
    policy.step(observation(), 2)
    assert policy.controller.active.choice == 'skill_0001'
    policy.acknowledge(2)
    policy.step(observation(), 4)
    assert len(planner.calls) == 2  # Back to the minimum execution hold.
    policy.acknowledge(2)
    result = policy.step(observation(), 6)
    assert len(planner.calls) == 3 and policy.harness.stall_replans == 1
    assert result.metadata['subtask']['choice'] == 'skill_0001'
    assert result.metadata['harness']['selection']['reason'] == \
        'awaiting_switch_confirmation'


def test_api_planner_receives_memory_but_vla_inputs_do_not(source):
    policy, planner, _, inputs = make_policy(source, min_subtask_chunks=1)
    policy.step(observation(), 0)
    policy.acknowledge(2)
    policy.step(observation(255), 2)
    memory = planner.calls[-1]['feedback']['agent_memory']
    assert memory['memory']['executed_steps'] == 2
    assert memory['memory']['completion'] == 'unknown'
    assert memory['memory']['subtask_summaries'][0]['instruction'] == LABELS[0]
    assert 'secret annotation' not in repr(memory)
    assert 'predicates' not in repr(memory) and 'reward' not in repr(memory)
    assert 'agent_memory' not in inputs.calls[-1]
    assert 'predicates' not in inputs.calls[-1]
    api = OpenAISubtaskPlanner(
        completion_mode='disabled',
        allow_request_finish=False,
        image_keys=CAMERAS)
    requests = []
    api.client.generate = lambda *args: requests.append(args) or dict(
        choice='skill_0000')
    api.choose(**planner.calls[-1])
    assert requests[-1][1]['feedback']['agent_memory'] == memory
    assert 'NOT verified task completion' in requests[-1][0]


def test_maximum_hold_allows_a_valid_switch_without_extra_votes(source):
    policy, _, _, _ = make_policy(
        source,
        min_subtask_chunks=1,
        max_hold_chunks=2,
        switch_confirmations=99,
        choices=['skill_0000', 'skill_0001'])
    policy.step(observation(), 0)
    policy.acknowledge(2)
    policy.step(observation(10), 2)
    policy.acknowledge(2)
    result = policy.step(observation(20), 4)
    assert result.metadata['subtask']['choice'] == 'skill_0001'


def test_serving_uses_real_episode_context_and_acknowledgments(source):
    runner = FluxVLAServingRunner(
        vla=Model(),
        dataset=Inputs(),
        denormalize_action=lambda data: data['action'],
        device='cpu',
        enable_mixed_precision=False,
        policy=settings(source, Planner(['skill_0000'])))
    request = observation()
    request['policy_context'] = dict(
        episode_id='run:episode', sequence=1, observation_id=0)
    actions, _ = runner.predict_action(request, '', 7)
    assert actions.shape == (4, 2)
    assert runner.policy.harness.memory.episode_id == 'run:episode'
    assert runner.policy.harness.memory.executed_steps == 0
    request['policy_context'] = dict(
        episode_id='run:episode', sequence=2, observation_id=2)
    runner.predict_action(request, '', 7)
    assert runner.policy.harness.memory.executed_steps == 2
    assert runner.last_prediction_metadata['harness']['active_candidate'][
        'instruction'] == LABELS[0]
    with pytest.raises(ValueError):
        runner.predict_action(request, '', 7)  # Duplicate execution ACK.
    assert runner.policy.harness.memory.executed_steps == 2


def test_serving_refuses_per_request_harness_before_model_use(source):
    with pytest.raises(ValueError, match='per_request'):
        FluxVLAServingRunner(
            vla=Model(),
            dataset=Inputs(),
            denormalize_action=lambda data: data['action'],
            device='cpu',
            enable_mixed_precision=False,
            policy_execution='per_request',
            policy=settings(source, Planner(['skill_0000'])))


def test_serving_does_not_invent_terminal_ack_on_new_episode(source, tmp_path):
    runner = FluxVLAServingRunner(
        vla=Model(),
        dataset=Inputs(),
        denormalize_action=lambda data: data['action'],
        device='cpu',
        enable_mixed_precision=False,
        policy=settings(
            source,
            Planner(['skill_0000']),
            memory=dict(trace_dir=str(tmp_path / 'trace'))))
    data = observation()
    data['policy_context'] = dict(
        episode_id='first', sequence=1, observation_id=0)
    runner.predict_action(data, '', 7)
    memory = runner.policy.harness.memory
    trace_path = memory.trace_path
    assert memory.pending is not None and memory.executed_steps == 0
    data['policy_context'] = dict(
        episode_id='second', sequence=1, observation_id=0)
    runner.predict_action(data, '', 7)
    saved = json.loads(trace_path.with_name('memory.json').read_text())
    assert saved['pending']['horizon'] == 4
    assert saved['executed_steps'] == 0
    assert saved['recent_events'][-1]['event'] == 'episode_interrupted'
    assert saved['recent_events'][-1]['execution_status'] == 'unknown'
    assert memory.episode_id == 'second' and memory.executed_steps == 0


def test_serving_optional_stall_stop_is_an_explicit_nonexecuting_result(
        source):
    runner = FluxVLAServingRunner(
        vla=Model(),
        dataset=Inputs(),
        denormalize_action=lambda data: data['action'],
        device='cpu',
        enable_mixed_precision=False,
        policy=settings(
            source,
            Planner(['skill_0000']),
            no_change_patience=1,
            max_stall_replans=1,
            stop_on_stall=True))
    for index in range(2):
        data = observation()
        data['policy_context'] = dict(
            episode_id='episode', sequence=index + 1, observation_id=index * 2)
        runner.predict_action(data, '', 7)
    data['policy_context'] = dict(
        episode_id='episode', sequence=3, observation_id=4)
    stopped, _ = runner.predict_action(data, '', 7)
    assert stopped == PolicyStop('failed', 'harness_no_observed_change')
    assert len(runner.vla.calls) == 2


@pytest.mark.parametrize('backend', ['qwen35', 'gpt6'])
def test_harness_configs_preserve_models_and_full_schedule(backend):
    base = Config.fromfile(ROOT /
                           f'configs/hierarchical/robodojo_pi05_{backend}.py')
    cfg = Config.fromfile(
        ROOT /
        f'configs/hierarchical/robodojo_pi05_{backend}_agent_harness.py')
    assert cfg.inference_model == base.inference_model
    assert cfg.policy.planner == base.policy.planner
    assert cfg.eval.dataset == base.eval.dataset
    assert cfg.policy.subtask_source == base.policy.subtask_source
    assert cfg.policy.type == 'AgentHarnessPolicy'
    assert cfg.policy.completion_mode == 'disabled'
    assert not cfg.policy.harness.stop_on_stall
    assert cfg.policy.harness.memory.trace_dir is None
    assert cfg.themis.server.mode == 'stateful'
    assert cfg.themis.server.runner.policy_execution == 'episode'
    assert cfg.themis.evaluation.policy_client.send_policy_context
    assert cfg.themis.evaluation.episodes_per_task == \
        base.themis.evaluation.episodes_per_task
    assert cfg.themis.evaluation.environment.task_name == 'all'
