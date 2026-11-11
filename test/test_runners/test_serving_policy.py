# Copyright 2026 Limx Dynamics
"""In-process planner -> VLA serving; no GPU, API or simulator required."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fluxvla.engines.runners.serving.distributed_server import (
    FluxVLAInferenceServer, _encode_prediction_payload)
from fluxvla.engines.runners.serving.serving_runner import (
    FluxVLAServingRunner, PolicyStop)
from fluxvla.policies import HierarchicalPolicy

GOAL = 'Stack the three blocks with different textures.'
LABELS = ['The left arm placed a block.', 'The right arm grasped a block.']
CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = []

    def predict_action(self, **batch):
        self.calls.append((os.getpid(), copy.deepcopy(batch)))
        return torch.rand(1, 50, 14)


class Inputs:

    def __init__(self):
        self.calls = []
        self.last_raw_state = None

    def __call__(self, observation):
        self.calls.append(copy.deepcopy(observation))
        self.last_raw_state = observation['states'].copy()
        return dict(states=torch.as_tensor(self.last_raw_state)[None])


@pytest.fixture
def policy_config(tmp_path):
    path = tmp_path / 'meta/episodes/chunk-000/file-000.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([
            dict(
                episode_index=0,
                length=20,
                tasks=[GOAL],
                dense_subtask_names=LABELS,
                dense_subtask_start_frames=[0, 10],
                dense_subtask_end_frames=[10, 20])
        ]), path)
    return dict(
        type='HierarchicalPolicy',
        subtask_source=dict(data_root=str(tmp_path)),
        planner=dict(
            type='ScriptedSubtaskPlanner',
            choices=['skill_0000', 'skill_0001', 'skill_0001']),
        monitor=dict(type='ObservationPredicateMonitor'),
        goal_conditions={'environment_success': True})


def make_runner(policy):
    calls = []

    def denormalize(data):
        calls.append(copy.deepcopy(data))
        return data['action'] * 2 + data['state']

    result = FluxVLAServingRunner(
        vla=Model(),
        dataset=Inputs(),
        denormalize_action=denormalize,
        denormalize_per_action=True,
        device='cpu',
        enable_mixed_precision=False,
        policy=policy)
    result.denormalize_calls = calls
    return result


def configure_agent(policy, choices):
    policy.update(
        completion_mode='agent',
        planner_image_keys=CAMERAS,
        monitor=None,
        goal_conditions=None)
    policy['planner'] = dict(
        type='ScriptedSubtaskPlanner',
        choices=[
            dict(choice=c, evidence='Fixture visual judgment.')
            for c in choices
        ])


def observation(sequence=1, step=0, episode='episode-1', **updates):
    data = dict(
        task_description=GOAL,
        states=np.arange(14, dtype=np.float32),
        cam_high=np.zeros((8, 8, 3), dtype=np.uint8),
        cam_left_wrist=np.zeros((8, 8, 3), dtype=np.uint8),
        cam_right_wrist=np.zeros((8, 8, 3), dtype=np.uint8),
        predicates={'environment_success': False},
        policy_context=dict(
            episode_id=episode, sequence=sequence, observation_id=step))
    data.update(updates)
    return data


def test_same_process_policy_then_original_vla_contract(
        policy_config, monkeypatch):
    runner = make_runner(policy_config)
    assert isinstance(runner.policy, HierarchicalPolicy)
    steps = []
    original = HierarchicalPolicy.step

    def traced_step(self, *args, **kwargs):
        steps.append(os.getpid())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HierarchicalPolicy, 'step', traced_step)
    request = observation()
    actions, duration = runner.predict_action(request, '', policy_seed=17)
    assert duration >= 0
    assert steps == [os.getpid()]
    assert runner.vla.calls[0][0] == os.getpid()
    assert actions.shape == (50, 14)
    assert np.isfinite(actions).all()
    assert runner.dataset.calls[0]['task_description'] == LABELS[0]
    assert request['task_description'] == GOAL
    assert 'policy_context' not in runner.dataset.calls[0]
    assert 'predicates' not in runner.dataset.calls[0]
    assert runner.policy.controller.total_chunks == 0
    assert runner.last_prediction_metadata['planning'][
        'instruction'] == LABELS[0]
    assert runner.last_prediction_metadata['subtask'][
        'completion_verified'] is False
    # Same model seed/device/reset flag and denormalization as plain serving.
    plain = make_runner(None)
    expected, _ = plain.predict_action(observation(), '', policy_seed=17)
    np.testing.assert_array_equal(actions, expected)
    assert len(runner.denormalize_calls) == 50
    assert runner.vla.calls[0][1]['reset_history'] is True


def test_actual_execution_then_replan_and_new_episode(policy_config, tmp_path):
    runner = make_runner(policy_config)
    runner.predict_action(observation(), '', policy_seed=17)
    # Reset/next episode use cached annotations, not another metadata read.
    for path in tmp_path.rglob('*.parquet'):
        path.unlink()
    runner.predict_action(observation(2, 8), '', policy_seed=17)
    assert runner.policy.controller.total_chunks == 1
    assert runner.policy.controller.history[0]['instruction'] == LABELS[0]
    assert runner.dataset.calls[-1]['task_description'] == LABELS[1]
    runner.predict_action(observation(episode='episode-2'), '', policy_seed=19)
    assert runner.policy.controller.history == []
    assert runner.policy.controller.total_chunks == 0
    assert runner.dataset.calls[-1]['task_description'] == LABELS[0]
    with pytest.raises(ValueError, match='interleaving'):
        runner.predict_action(observation(3, 16), '', policy_seed=17)
    assert len(runner.vla.calls) == 3


@pytest.mark.parametrize('sequence, step', [(1, 0), (3, 8), (2, 0), (2, 51),
                                            (True, 8), (2, -1)])
def test_invalid_ack_never_runs_vla(policy_config, sequence, step):
    runner = make_runner(policy_config)
    runner.predict_action(observation(), '', policy_seed=17)
    with pytest.raises(ValueError):
        runner.predict_action(observation(sequence, step), '', policy_seed=17)
    assert len(runner.vla.calls) == 1
    assert runner.policy.controller.total_chunks == 0


@pytest.mark.parametrize('change', ['goal', 'seed', 'normalization'])
def test_episode_identity_cannot_change(policy_config, change):
    runner = make_runner(policy_config)
    runner.predict_action(observation(), 'robot', policy_seed=17)
    request = observation(2, 8)
    if change == 'goal':
        request['task_description'] = 'Different goal'
    with pytest.raises(ValueError):
        runner.predict_action(
            request,
            'other' if change == 'normalization' else 'robot',
            policy_seed=18 if change == 'seed' else 17)
    assert len(runner.vla.calls) == 1


@pytest.mark.parametrize('choice', ['blocked', 'request_finish', 'skill_9999'])
def test_planner_failure_never_sends_dummy_actions(policy_config, choice):
    policy_config['planner']['choices'] = [choice]
    runner = make_runner(policy_config)
    result, _ = runner.predict_action(observation(), '', policy_seed=17)
    assert isinstance(result, PolicyStop)
    assert not runner.vla.calls
    assert not runner.dataset.calls
    with pytest.raises(RuntimeError, match='episode failed'):
        runner.predict_action(observation(2, 8), '', policy_seed=17)


def test_unverified_finish_at_step160_reports_why_it_stopped(policy_config):
    policy_config['planner']['choices'] = ['skill_0000'] * 20 + [
        'request_finish'
    ]
    runner = make_runner(policy_config)
    for index in range(20):
        runner.predict_action(observation(index + 1, index * 8), '', 17)
    result, _ = runner.predict_action(observation(21, 160), '', 17)
    assert result == PolicyStop('waiting', 'goal_not_verified')
    assert len(runner.vla.calls) == 20
    assert runner.last_prediction_metadata['planning']['decision'] == {
        'decision': 'request_finish'
    }


def test_robodojo_server_agent_finish_stops_without_env_verification(
        policy_config, monkeypatch):
    from mmengine import Config

    from fluxvla.planning.openai import OpenAIResponsesClient

    cfg = Config.fromfile(
        Path(__file__).resolve().parents[2] /
        'configs/hierarchical/robodojo_pi05_gpt6.py')
    configured = cfg.eval.policy.to_dict()
    configured['subtask_source'] = policy_config['subtask_source']
    requests = []

    def post_json(self, body):
        requests.append(copy.deepcopy(body))
        choices = body['text']['format']['schema']['properties']['choice'][
            'enum']
        # Finish is a final agent decision, NOT another verifier request.
        choice = ('request_finish' if len(requests) == 21
                  and 'request_finish' in choices else choices[0])
        return dict(
            status='completed',
            output=[
                dict(
                    type='message',
                    role='assistant',
                    status='completed',
                    content=[
                        dict(
                            type='output_text',
                            text=json.dumps(
                                dict(
                                    choice=choice,
                                    evidence='Observed outcome.')))
                    ])
            ])

    monkeypatch.setattr(OpenAIResponsesClient, '_post_json', post_json)
    runner = make_runner(configured)
    for index in range(20):
        actions, _ = runner.predict_action(
            observation(index + 1, index * 8), '', 17)
        assert actions.shape == (50, 14)
    stopped, _ = runner.predict_action(observation(21, 160), '', 17)
    assert stopped == PolicyStop('completed', 'agent_visual_completion')
    assert len(requests) == 21
    assert len(runner.vla.calls) == 20
    assert runner._episode_failed
    assert runner.policy._pending_version is None
    for body in requests:
        choices = body['text']['format']['schema']['properties']['choice'][
            'enum']
        assert choices == [
            'skill_0000', 'skill_0001', 'request_finish', 'blocked'
        ]
        content = body['input'][0]['content']
        payload = json.loads(content[0]['text'])
        assert 'facts' not in payload
        assert 'completion_verification' not in payload['feedback']
        assert len([c for c in content if c['type'] == 'input_image']) == 6
        assert [c['text'] for c in content[1:]
                if c['type'] == 'input_text'] == [
                    f'{phase} camera: {key}'
                    for phase in ['CURRENT', 'INITIAL'] for key in CAMERAS
                ]
    event = runner.last_prediction_metadata['planning']
    assert event['planning_attempts'] == [
        dict(attempt=0, choice='request_finish', evidence='Observed outcome.')
    ]
    assert event['completion'] == dict(
        source='agent_visual',
        agent_success=True,
        evidence='Observed outcome.')
    assert event['episode_control_replans'] == 0
    assert runner.policy.controller.total_chunks == 20
    assert runner.dataset.calls[-1]['task_description'] == LABELS[0]


def test_stop_error_does_not_echo_provider_details(policy_config, monkeypatch):
    from fluxvla.policies.vla_policy import PolicyOutput

    runner = make_runner(policy_config)
    result = PolicyOutput(
        status='failed',
        metadata=dict(planning=dict(reason='private provider details')))
    monkeypatch.setattr(runner, 'step_policy', lambda *a, **k: result)
    result, _ = runner.predict_action(observation(), '', 17)
    assert result == PolicyStop('failed', 'planner_error')
    payload = _encode_prediction_payload(result,
                                         runner.last_prediction_metadata)
    assert b'private provider details' not in payload
    # Original event remains inspectable locally, but is not sent over RPC.
    assert runner.last_prediction_metadata['planning'][
        'reason'] == 'private provider details'
    assert not runner.vla.calls


def test_native_success_not_gpt_certifies_completion(policy_config):
    runner = make_runner(policy_config)
    result, _ = runner.predict_action(
        observation(predicates={'environment_success': True}), '', 17)
    assert result == PolicyStop('succeeded', 'goal_verified')
    assert not runner.vla.calls


def test_context_is_required_and_unknown_is_not_false(policy_config):
    runner = make_runner(policy_config)
    with pytest.raises(ValueError, match='send_policy_context'):
        runner.predict_action(observation(policy_context=None), '', 17)
    result, _ = runner.predict_action(observation(predicates={}), '', 17)
    assert result == PolicyStop('waiting', 'goal_state_unknown')
    assert not runner.vla.calls


def test_replan_cadence_and_budget_preserved(policy_config):
    policy_config.update(replan_every_chunks=2, max_episode_chunks=3)
    runner = make_runner(policy_config)
    for sequence, step in [(1, 0), (2, 8), (3, 16)]:
        runner.predict_action(observation(sequence, step), '', 17)
    assert [d['task_description'] for d in runner.dataset.calls
            ] == [LABELS[0], LABELS[0], LABELS[1]]
    result, _ = runner.predict_action(observation(4, 24), '', 17)
    assert result == PolicyStop('failed', 'episode_chunk_budget_exhausted')
    assert len(runner.vla.calls) == 3


def test_unexpected_model_exception_is_not_a_policy_stop(
        policy_config, monkeypatch):
    runner = make_runner(policy_config)

    def crash(**batch):
        raise RuntimeError('model execution failed')

    monkeypatch.setattr(runner.vla, 'predict_action', crash)
    with pytest.raises(RuntimeError, match='model execution failed'):
        runner.predict_action(observation(), '', 17)
    assert runner._episode_failed


@pytest.mark.parametrize('status, actions', [('unexpected', None),
                                             ('blocked', np.ones((1, 14)))])
def test_invalid_stop_is_still_a_protocol_error(policy_config, monkeypatch,
                                                status, actions):
    from fluxvla.policies.vla_policy import PolicyOutput

    runner = make_runner(policy_config)
    monkeypatch.setattr(
        runner, 'step_policy',
        lambda *a, **k: PolicyOutput(status=status, actions=actions))
    with pytest.raises(ValueError, match='Invalid non-executing'):
        runner.predict_action(observation(), '', 17)
    assert not runner.vla.calls


@pytest.mark.parametrize('agent_completion', [False, True])
def test_stop_at256_saves_artifacts_and_runs_all_ten_episodes(
        policy_config, monkeypatch, tmp_path, agent_completion):
    pytest.importorskip('fluxthemis')
    from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                    execute_episode)
    from fluxthemis.environments.sim import robodojo
    from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient
    from fluxthemis.structures import TaskSpec
    from fluxthemis.transports.zmq_protocol import decode_frame, decode_header

    status = 'completed' if agent_completion else 'blocked'
    reason = ('agent_visual_completion'
              if agent_completion else 'planner_cannot_progress')
    tag = 'agent_done' if agent_completion else 'fail'
    if agent_completion:
        configure_agent(policy_config,
                        ['skill_0000'] * 32 + ['request_finish'])
    else:
        policy_config['planner']['choices'] = ['skill_0000'] * 32 + ['blocked']
    runner = make_runner(policy_config)
    requests, responses, reset_calls, video_calls = [], [], [], []
    client = FluxVLAPolicyClient(
        transport=dict(image_keys=CAMERAS, state_keys=['states']),
        send_policy_context=True,
        send_environment_feedback=not agent_completion)

    def send(header, payload):
        h, data = decode_header(header), decode_frame(payload)
        requests.append(copy.deepcopy(data))
        prediction, elapsed = runner.predict_action(data, h['unnorm_key'],
                                                    h['policy_seed'])
        payload = _encode_prediction_payload(prediction,
                                             runner.last_prediction_metadata)
        responses.append(decode_frame(payload))
        return dict(type='predict_response', ok=True,
                    request_id=h['request_id'], inference_time_s=elapsed), \
            payload

    client.transport = SimpleNamespace(
        server_deployment_id='test',
        request_timeout_s=5,
        request=send,
        validate_response=lambda *args: None)
    native = SimpleNamespace(
        close=lambda: None, save_dir=str(tmp_path / 'native'))
    Path(native.save_dir).mkdir()

    def reset(seed):
        reset_calls.append(seed)
        native.success = [True]
        native.end_flag = [False]
        native.take_action_cnt = [0]

    def take_action(action):
        native.take_action_cnt[0] += 1

    def save_video(index, path, tag):
        video_calls.append((index, native.take_action_cnt[0], tag))
        for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            # Fixture files test finalization/moving, not video encoding.
            p = Path(path).with_name(f'{Path(path).stem}_{camera}_{tag}.mp4')
            p.write_bytes(b'video fixture')

    native.reset, native.take_action = reset, take_action
    native.get_obs, native.save_video = lambda: observation(), save_video
    task = TaskSpec('stack_blocks', GOAL)
    env = object.__new__(robodojo.RoboDojoEnvironment)
    env.completion_diagnostics = False
    env._env, env._task = native, task
    env._task_ids = {task.task_id}
    env._episode_active = False
    env._physx_monitor = None
    env.save_videos = True
    env.worker_id = 'test-worker'
    env._video_dir = tmp_path / 'videos'
    env._video_dir.mkdir()
    env._action_to_dict = lambda action: action
    env._current_score = lambda success: 0.25
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)

    jobs = build_episode_jobs([task], 10, 0, 'ten-episodes')
    results = []
    for job in jobs:
        result = execute_episode(
            env, client, job, 550, 8, True, save_videos=True)
        job.validate_result(result)
        results.append(result)
        assert result.success is False
        assert result.steps == native.take_action_cnt[0] == 256
        assert result.model_calls == 33
        assert result.termination_reason == f'policy_{status}'
        assert result.score == 0.25
        expected_stop = dict(status=status, reason=reason)
        assert result.info['policy_stop'] == expected_stop
        if agent_completion:
            assert result.info['agent_success'] is True
            assert result.info['completion_source'] == 'agent_visual'
            event = result.prediction_metadata[-1]['policy']['planning']
            assert event['completion']['agent_success'] is True
            assert event['completion'][
                'evidence'] == 'Fixture visual judgment.'
            assert 'completion_verification' not in event
            assert 'predicates' not in requests[-1]
            assert runner.policy._pending_version is None
        else:
            assert 'agent_success' not in result.info
        assert len(result.info['video_paths']) == 3
        assert all(Path(p).is_file() for p in result.info['video_paths'])
        assert env._video_finalized
        assert runner.policy.controller.total_chunks == 32
        assert runner.policy.controller.history[0]['executed_chunks'] == 32
        # No native success flags, action counts or outcomes are fabricated.
        assert native.success == [True] and native.end_flag == [False]

    assert len(results) == 10
    assert reset_calls == [[i] for i in range(10)]
    assert video_calls == [(0, 256, tag)] * 10
    assert len(list(env._video_dir.glob('*.mp4'))) == 30
    assert len(runner.vla.calls) == 320
    assert len(requests) == len(responses) == 330
    for index in range(10):
        reset_context = requests[index * 33]['policy_context']
        assert reset_context == dict(
            episode_id=f'ten-episodes:{index + 1}',
            sequence=1,
            observation_id=0)
        stopped = responses[index * 33 + 32]
        assert 'actions' not in stopped
        assert stopped['policy_stop']['status'] == status
    if agent_completion:
        assert all('predicates' not in r for r in requests)


@pytest.mark.parametrize('recover', [True, False])
def test_ten_episode_completion_loop_through_ordinary_themis_client(
        policy_config, monkeypatch, tmp_path, recover):
    pytest.importorskip('fluxthemis')
    from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                    execute_episode)
    from fluxthemis.environments.sim import robodojo
    from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient
    from fluxthemis.structures import TaskSpec
    from fluxthemis.transports.zmq_protocol import decode_frame, decode_header

    policy_config.update(
        max_control_replans=2,
        max_episode_control_replans=12,
        completion_requirements=dict(
            blocks_stacked='Stack the blocks.',
            arms_at_origin='Return both arms.'))
    policy_config['planner']['choices'] = [
        'skill_0000', 'request_finish', 'blocked',
        'skill_0000' if recover else 'blocked'
    ]
    runner = make_runner(policy_config)
    requests = []
    responses = []

    def send(header, payload):
        header, data = decode_header(header), decode_frame(payload)
        requests.append(copy.deepcopy(data))
        prediction, elapsed = runner.predict_action(data, header['unnorm_key'],
                                                    header['policy_seed'])
        payload = _encode_prediction_payload(prediction,
                                             runner.last_prediction_metadata)
        responses.append(decode_frame(payload))
        response_header = dict(
            type='predict_response',
            ok=True,
            request_id=header['request_id'],
            inference_time_s=elapsed)
        return response_header, payload

    client = FluxVLAPolicyClient(
        transport=dict(image_keys=['cam_high'], state_keys=['states']),
        send_policy_context=True)
    client.transport = SimpleNamespace(
        server_deployment_id='test',
        request_timeout_s=5,
        request=send,
        validate_response=lambda *args: None)
    native = SimpleNamespace(
        close=lambda: None, save_dir=str(tmp_path / 'native'))
    Path(native.save_dir).mkdir()

    def reset(seed):
        native.success, native.end_flag = [True], [False]
        native.take_action_cnt = [0]

    def take_action(action):
        native.take_action_cnt[0] += 1
        native.end_flag[0] = native.take_action_cnt[0] == 16

    def save_video(index, path, tag):
        for camera in ['head', 'left', 'right']:
            video = Path(path).with_name(
                f'{Path(path).stem}_{camera}_{tag}.mp4')
            video.write_bytes(b'video fixture')

    native.reset, native.take_action = reset, take_action
    native.get_obs, native.save_video = lambda: observation(), save_video
    native._stacked_xy_thresholds = lambda: [0.015]
    native.reward_manager = SimpleNamespace(
        is_stacked=lambda **kwargs: 'stacked',
        all_robot_back_to_origin=lambda: 'home',
        call_func_parser=lambda check, _: float(native.take_action_cnt[0] >= (
            8 if check == 'stacked' else 16)))
    task = TaskSpec('stack_blocks', GOAL)
    env = object.__new__(robodojo.RoboDojoEnvironment)
    env.completion_diagnostics = True
    env._env, env._task = native, task
    env._task_ids, env._episode_active = {task.task_id}, False
    env._physx_monitor, env.save_videos = None, True
    env.worker_id, env._video_dir = 'test', tmp_path / 'videos'
    env._video_dir.mkdir()
    env._action_to_dict = lambda action: action
    env._current_score = lambda success: 1.0 if success else 0.15
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)

    jobs = build_episode_jobs([task], 10, 0, 'completion-test')
    for job in jobs:
        result = execute_episode(
            env, client, job, 550, 8, True, save_videos=True)
        job.validate_result(result)
        assert result.success is recover
        assert result.steps == native.take_action_cnt[0]
        assert result.steps == (16 if recover else 8)
        assert result.model_calls == 2
        assert result.termination_reason == ('success'
                                             if recover else 'policy_failed')
        assert len(result.info['video_paths']) == 3
        assert all(Path(p).is_file() for p in result.info['video_paths'])
        assert runner.policy.controller.planner.index == 4
        assert runner.policy.controller.episode_control_replans == 2
        event = result.prediction_metadata[-1]['policy']['planning']
        assert event['completion_verification']['unmet'] == ['arms_at_origin']
        assert [a['choice'] for a in event['planning_attempts']] == [
            'request_finish', 'blocked', 'skill_0000' if recover else 'blocked'
        ]
        if not recover:
            assert event['reason'] == 'planner_recovery_exhausted'
            assert 'actions' not in responses[-1]
            assert responses[-1]['policy_stop']['reason'] == event['reason']
        # Same selected literal goes to tokenize; no home instruction added.
        assert runner.dataset.calls[-1]['task_description'] == LABELS[0]
        assert 'predicates' not in runner.dataset.calls[-1]
    assert len(requests) == 20
    assert len(runner.vla.calls) == (20 if recover else 10)
    assert len(list(env._video_dir.glob('*.mp4'))) == 30
    for i in range(10):
        assert requests[i * 2]['policy_context'] == dict(
            episode_id=f'completion-test:{i + 1}',
            sequence=1,
            observation_id=0)
        assert requests[i * 2 + 1]['predicates'] == dict(
            environment_success=False,
            blocks_stacked=True,
            arms_at_origin=False)


@pytest.mark.parametrize('agent_completion', [False, True])
def test_worker_sends_stop_result_and_accepts_next_episode(
        policy_config, monkeypatch, tmp_path, agent_completion):
    import zmq

    from fluxvla.engines.runners.serving import distributed_server as server
    from fluxvla.engines.utils import torch_utils

    if agent_completion:
        configure_agent(policy_config, ['skill_0000', 'request_finish'])
    else:
        policy_config['planner']['choices'] = ['skill_0000', 'blocked']
    runner = make_runner(policy_config)
    config = tmp_path / 'worker.py'
    config.write_text('test_config = True\n')
    common = dict(deployment_id='test', worker_id='model-worker-0')
    requests = [
        observation(),
        observation(2, 8),
        observation(3, 16),
        observation(episode='episode-2')
    ]
    incoming = [[
        server.encode_header(
            'predict', **common, request_id=str(i), policy_seed=17),
        server.encode_frame(data)
    ] for i, data in enumerate(requests)]
    incoming.append([server.encode_header('shutdown_worker', **common)])
    sent = []
    backend = SimpleNamespace(
        setsockopt=lambda *args: None,
        connect=lambda endpoint: None,
        send_multipart=sent.append,
        recv_multipart=lambda: incoming.pop(0))
    monkeypatch.setattr(zmq, 'Context',
                        lambda: SimpleNamespace(socket=lambda _: backend))
    monkeypatch.setattr(server, 'build_serving_runner_from_config',
                        lambda *a, **k: runner)
    monkeypatch.setattr(torch_utils, 'configure_inference_attention_defaults',
                        lambda: None)

    server.run_model_worker(
        str(config), '', 'inproc://worker', common['worker_id'], 'node',
        common['deployment_id'], 'cpu')

    headers = [server.decode_header(frames[0]) for frames in sent]
    assert [h['type'] for h in headers] == [
        'register_worker', 'ready', 'predict_result', 'predict_result',
        'predict_error', 'predict_result'
    ]
    assert [h['request_id'] for h in headers[2:]] == ['0', '1', '2', '3']
    stop = server.decode_frame(sent[3][1])
    assert 'actions' not in stop
    assert stop['policy_stop'] == dict(
        status='completed' if agent_completion else 'blocked',
        reason='agent_visual_completion'
        if agent_completion else 'planner_cannot_progress')
    assert 'episode failed' in headers[4]['error']
    for i in (2, 5):
        actions = server.decode_frame(sent[i][1])['actions']
        assert actions.shape == (50, 14)
        assert actions.dtype == np.float32
    assert len(runner.vla.calls) == 2
    assert runner.policy.controller.total_chunks == 0
    assert not runner._episode_failed


def test_stateful_server_cannot_round_robin_controller_state():
    kwargs = dict(frontend_bind='inproc://front', backend_bind='inproc://back')
    with pytest.raises(ValueError, match='one model worker'):
        FluxVLAInferenceServer(
            **kwargs, expected_workers=2, server_mode='stateful')
    plain = FluxVLAInferenceServer(**kwargs, expected_workers=2)
    assert plain.server_mode == 'stateless'


def test_ordinary_themis_client_round_trip(policy_config):
    pytest.importorskip('fluxthemis')
    from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient
    from fluxthemis.structures import TaskSpec
    from fluxthemis.transports.zmq_protocol import (decode_frame,
                                                    decode_header,
                                                    encode_frame)

    runner = make_runner(policy_config)
    client = FluxVLAPolicyClient(
        transport=dict(image_keys=['cam_high'], state_keys=['states']),
        send_policy_context=True)
    requests = []

    def send(header, payload):
        h, data = decode_header(header), decode_frame(payload)
        requests.append(data)
        result, elapsed = runner.predict_action(data, h['unnorm_key'],
                                                h['policy_seed'])
        return dict(type='predict_response', ok=True,
                    request_id=h['request_id'], inference_time_s=elapsed), \
            encode_frame(dict(actions=result,
                              metadata=runner.last_prediction_metadata))

    client.transport = SimpleNamespace(
        server_deployment_id='test',
        request_timeout_s=5,
        request=send,
        validate_response=lambda *args: None)
    client.reset(TaskSpec('stack_blocks'), 0, 17)
    client.set_episode_id('episode-1')
    for step in (0, 8):
        result = client.predict_action(
            observation(),
            dict(
                episode_id='episode-1',
                task_description=GOAL,
                info=dict(step=step)))
    assert len(requests) == 2
    assert requests[1]['policy_context']['observation_id'] == 8
    assert requests[0]['task_description'] == GOAL
    assert result.metadata['policy']['planning']['instruction'] == LABELS[1]
    assert result.actions.shape == (50, 14)
    assert not hasattr(client, 'planner_process')


@pytest.mark.parametrize('mode, enabled', [('stateless', True),
                                           ('stateful', False)])
def test_themis_rejects_mismatched_server_instead_of_ignoring_planner(
        mode, enabled):
    pytest.importorskip('fluxthemis')
    from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient

    closed = []
    client = FluxVLAPolicyClient(
        transport=dict(image_keys=['cam_high'], state_keys=['states']),
        send_policy_context=enabled)
    client.transport = SimpleNamespace(
        server_mode=mode,
        setup=lambda: None,
        close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match='server mode'):
        client.setup()
    assert closed == [True]


@pytest.mark.parametrize('terminated, success', [(False, True), (False, False),
                                                 (True, True), (True, False)])
def test_native_robodojo_forwards_terminal_success_not_score(
        monkeypatch, terminated, success):
    pytest.importorskip('fluxthemis')
    from fluxthemis.environments.sim import robodojo

    env = object.__new__(robodojo.RoboDojoEnvironment)
    env.completion_diagnostics = False
    env._env = SimpleNamespace(
        get_obs=lambda: {},
        take_action=lambda _: None,
        end_flag=[terminated],
        success=[success],
        take_action_cnt=[8])
    env._task = SimpleNamespace(task_id='stack_blocks')
    env._episode_layout_index = 0
    env._require_episode = lambda: None
    env._action_to_dict = lambda _: {}
    env._current_score = lambda _: 1.0
    env._episode_task = env._task
    env._episode_index = 0
    env._finalize_video = lambda *args: []
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda _: {})
    result = env.step(np.zeros(14))
    assert result.observation['predicates']['environment_success'] is (
        terminated and success)
    assert result.success is (success if terminated else None)


@pytest.mark.parametrize('terminal_success', [True, False])
def test_robodojo_reset_through_policy_until_native_episode_end(
        policy_config, monkeypatch, terminal_success):
    pytest.importorskip('fluxthemis')
    from fluxthemis.engines.runners.episode import (build_episode_jobs,
                                                    execute_episode)
    from fluxthemis.environments.sim import robodojo
    from fluxthemis.structures import ActionChunk, TaskSpec

    runner = make_runner(policy_config)
    native = SimpleNamespace()

    def reset(seed):
        native.success = [True]
        native.end_flag = [False]
        native.take_action_cnt = [0]

    def take_action(action):
        native.take_action_cnt[0] += 1
        if native.take_action_cnt[0] == 9:
            native.end_flag[0] = True
            native.success[0] = terminal_success

    native.reset = reset
    native.take_action = take_action
    native.get_obs = lambda: observation()
    task = TaskSpec('stack_blocks', GOAL)
    env = object.__new__(robodojo.RoboDojoEnvironment)
    env.completion_diagnostics = False
    env._env = native
    env._task = task
    env._task_ids = {task.task_id}
    env._episode_active = False
    env._physx_monitor = None
    env.save_videos = False
    env._action_to_dict = lambda action: action
    monkeypatch.setattr(robodojo, 'robodojo_obs_to_flat', lambda obs: obs)
    requests = []

    def predict(data, context):
        requests.append(copy.deepcopy(data))
        data = dict(
            data,
            policy_context=dict(
                episode_id=context['episode_id'],
                sequence=len(requests),
                observation_id=context['info'].get('step', 0)))
        actions, _ = runner.predict_action(data, '', context['policy_seed'])
        return ActionChunk(actions=actions)

    client = SimpleNamespace(
        reset=lambda *args: None,
        set_episode_id=lambda value: None,
        predict_action=predict)
    job = build_episode_jobs([task], 1, 17, 'native-regression')[0]
    result = execute_episode(
        env,
        client,
        job,
        max_episode_steps=20,
        execute_horizon=8,
        stop_on_success=True)

    assert result.success is terminal_success
    assert result.steps == 9
    assert result.model_calls == 2
    assert result.termination_reason == ('success'
                                         if terminal_success else 'terminated')
    assert len(requests) == 2
    assert all(data['predicates']['environment_success'] is False
               for data in requests)
    assert [data['task_description']
            for data in runner.dataset.calls] == LABELS
    assert not runner._episode_failed


@pytest.mark.parametrize('section', ['eval', 'inference'])
def test_builder_installs_policy_from_selected_config_section(
        policy_config, tmp_path, monkeypatch, section):
    from mmengine import Config

    import fluxvla.engines as engines
    from fluxvla.engines.runners.serving import serving_runner

    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/hierarchical/robodojo_pi05_gpt6.py')
    cfg.eval.policy = policy_config
    if section == 'inference':
        cfg.inference = cfg.eval.copy()
    cfg.themis.server.dataset_section = section
    ckpt = tmp_path / 'checkpoints/model.pt'
    ckpt.parent.mkdir()
    torch.save({}, ckpt)
    stats = tmp_path / 'dataset_statistics.json'
    stats.write_text('{}')
    cfg.themis.server.ckpt_path = str(ckpt)
    cfg.themis.server.norm_stats_path = str(stats)
    cfg.themis.server.enable_mixed_precision = False
    monkeypatch.setattr(engines, 'build_vla_from_cfg', lambda _: Model())
    monkeypatch.setattr(engines, 'build_dataset_from_cfg', lambda _: Inputs())
    monkeypatch.setattr(engines, 'build_transform_from_cfg',
                        lambda _: lambda data: data['action'])
    monkeypatch.setattr(serving_runner, 'load_checkpoint',
                        lambda *a, **k: None)
    monkeypatch.setattr(serving_runner, 'validate_denormalization_stats',
                        lambda *a: None)
    runner = serving_runner.build_serving_runner_from_config(cfg, device='cpu')
    assert isinstance(runner.policy, HierarchicalPolicy)
    actions, _ = runner.predict_action(observation(), '', 17)
    assert actions.shape == (50, 14)
    assert runner.dataset.calls[0]['task_description'] == LABELS[0]
