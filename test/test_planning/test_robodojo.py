# Copyright 2026 Limx Dynamics
"""Optional FluxThemis bridge tests; no Isaac runtime or API calls needed."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mmengine import Config

pytest.importorskip('fluxthemis')

from fluxthemis.model_clients.fluxvla_policy import \
    FluxVLAPolicyClient  # noqa: E402
from fluxthemis.structures import ActionChunk, TaskSpec  # noqa: E402

from tools.robodojo_hierarchical import (  # noqa: E402
    HierarchicalFluxVLAPolicyClient, RoboDojoStackBlocksEnvironment)

CONFIG = (
    Path(__file__).resolve().parents[2] /
    'configs/hierarchical/robodojo_pi05_gpt6.py')


class FakePlanner:

    def __init__(self, status='ready'):
        self.requests = []
        self.status = status

    def request(self, **request):
        self.requests.append(copy.deepcopy(request))
        if request['command'] == 'reset':
            return None
        return dict(
            status=self.status,
            instruction='Stack two blocks.',
            decision={'decision': 'execute'},
            subtask_version=1)

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    cfg = Config.fromfile(str(CONFIG))
    settings = dict(
        planner_python=sys.executable, policy=cfg.eval.policy.to_dict())
    client = HierarchicalFluxVLAPolicyClient(
        transport=cfg.themis.transport.to_dict(), **settings)
    client.planner_process = FakePlanner()
    client.vla_prompts = []

    def predict(self, observation, context):
        self.vla_prompts.append(copy.deepcopy(context))
        return ActionChunk(
            np.ones((50, 14), dtype=np.float32),
            metadata={'denormalized': True})

    monkeypatch.setattr(FluxVLAPolicyClient, 'predict_action', predict)
    client.reset(TaskSpec('stack_blocks'), 0, 123)
    client.set_episode_id('episode-0')
    yield client
    client.close()


def context(step=0, episode='episode-0'):
    return dict(
        episode_id=episode,
        task_description='Stack three blocks.',
        info={'step': step})


def test_config_checkpoint_and_contract(monkeypatch):
    monkeypatch.delenv('ROBODOJO_SUBTASK_DATA_ROOT', raising=False)
    cfg = Config.fromfile(str(CONFIG))
    assert cfg.inference_model.name_mapping == {}
    assert cfg.inference_model.ori_action_dim == 14
    assert cfg.themis.evaluation.execute_horizon == 8
    assert cfg.themis.evaluation.episodes_per_task == 10
    assert cfg.themis.evaluation.episodes_per_task_overrides == {}
    assert cfg.eval.num_trials_per_task == 10
    assert cfg.eval.num_trials_per_task_overrides == {}
    assert cfg.themis.server.ckpt_path.endswith(
        'step-060000-epoch-04-loss=0.0010.safetensors')
    assert cfg.eval.denormalize_action.delta_action_mask == ([True] * 6 +
                                                             [False] +
                                                             [True] * 6 +
                                                             [False])
    assert cfg.eval.dataset.transforms[3].tokenizer.model_path == str(
        Path(cfg.themis.server.norm_stats_path).parent / 'tokenizer')
    assert cfg.themis.evaluation.policy_client.type == 'FluxVLAPolicyClient'
    assert cfg.themis.evaluation.policy_client.send_policy_context is True
    assert cfg.themis.evaluation.environment.type == 'RoboDojoEnvironment'
    assert cfg.themis.server.mode == 'stateful'
    assert 'custom_imports' not in cfg.themis
    policy = cfg.eval.policy
    assert policy.type == 'HierarchicalPolicy'
    assert 'skill_catalog' not in policy
    assert policy.subtask_source.data_root == (
        '/mnt/data/oss/datasets/RoboDojo_lerobot_v3')
    assert policy.subtask_source.subtask_prefix == 'dense'
    assert policy.replan_every_chunks == 1
    assert policy.planner.allow_request_finish is True
    assert policy.max_control_replans == 2
    assert policy.max_episode_control_replans == 12
    assert policy.completion_mode == policy.planner.completion_mode == 'agent'
    assert policy.monitor is policy.goal_conditions is None
    assert 'completion_requirements' not in policy
    assert policy.planner_image_keys == policy.planner.image_keys
    assert len(policy.planner_image_keys) == 3
    assert cfg.themis.evaluation.environment.completion_diagnostics is False
    assert not cfg.themis.evaluation.policy_client.send_environment_feedback


def test_candidate_data_path_override(monkeypatch):
    monkeypatch.setenv('ROBODOJO_SUBTASK_DATA_ROOT', '/tmp/training-data')
    cfg = Config.fromfile(str(CONFIG))
    assert cfg.eval.policy.subtask_source['data_root'] == '/tmp/training-data'


@pytest.mark.parametrize('override', [None, '', 'test-planner'])
def test_planner_model_default_and_override(monkeypatch, override):
    if override is None:
        monkeypatch.delenv('OPENAI_PLANNER_MODEL', raising=False)
    else:
        monkeypatch.setenv('OPENAI_PLANNER_MODEL', override)
    cfg = Config.fromfile(str(CONFIG))
    assert cfg.eval.policy.planner.model == (override or 'gpt-5.6-luna')


def test_config_schedules_ten_stack_blocks_episodes():
    from fluxthemis.engines.runners.episode import build_episode_jobs

    evaluation = Config.fromfile(str(CONFIG)).themis.evaluation
    assert evaluation.environment.task_name == 'stack_blocks'
    jobs = build_episode_jobs(
        tasks=[TaskSpec(evaluation.environment.task_name)],
        episodes_per_task=evaluation.episodes_per_task,
        base_seed=evaluation.base_seed,
        run_session_id='ten-episodes',
        episodes_per_task_overrides=evaluation.episodes_per_task_overrides)
    assert len(jobs) == 10
    assert len({job.episode_id for job in jobs}) == 10
    assert {job.task.task_id for job in jobs} == {'stack_blocks'}
    assert [job.episode_index for job in jobs] == list(range(10))
    assert [job.environment_case_index for job in jobs] == list(range(10))
    assert [job.policy_seed for job in jobs
            ] == [evaluation.base_seed + index for index in range(10)]


def test_subtask_replaces_prompt_not_goal_and_acknowledges_execution(client):
    observation = {'task_description': 'Stack three blocks.'}
    original = copy.deepcopy(observation)
    result = client.predict_action(observation, context())
    assert observation == original
    assert result.metadata['goal'] == 'Stack three blocks.'
    assert result.metadata['denormalized'] is True
    assert result.actions.shape == (50, 14)
    assert client.vla_prompts[-1]['task_description'] == 'Stack two blocks.'
    assert client.planner_process.requests[-1][
        'executed_previous_chunk'] is False
    client.predict_action(observation, context(step=8))
    assert client.planner_process.requests[-1][
        'executed_previous_chunk'] is True
    assert client.planner_process.requests[-1]['observation_id'] == 8
    assert [r['command'] for r in client.planner_process.requests
            ] == ['reset', 'step', 'step']


@pytest.mark.parametrize('step', [0, -1, 51, True])
def test_rejects_invalid_execution_acknowledgement(client, step):
    client.predict_action({}, context())
    with pytest.raises(ValueError):
        client.predict_action({}, context(step=step))
    assert len(client.vla_prompts) == 1


def test_reset_does_not_leak_pending_chunk_or_goal(client):
    client.predict_action({}, context())
    client.reset(TaskSpec('stack_blocks'), 1, 456)
    client.set_episode_id('episode-1')
    new_context = context(episode='episode-1')
    new_context['task_description'] = 'Stack the textured blocks.'
    client.predict_action({}, new_context)
    requests = client.planner_process.requests
    assert requests[-2]['command'] == 'reset'
    assert requests[-2]['goal'] == new_context['task_description']
    assert requests[-1]['executed_previous_chunk'] is False


def test_rejects_cross_episode_requests(client):
    with pytest.raises(ValueError, match='episode id'):
        client.predict_action({}, context(episode='other'))
    assert not client.vla_prompts


@pytest.mark.parametrize('status',
                         ['blocked', 'failed', 'waiting', 'succeeded'])
def test_nonready_never_dispatches_dummy_actions(client, status):
    client.planner_process.status = status
    with pytest.raises(RuntimeError, match='without issuing actions'):
        client.predict_action({}, context())
    assert not client.vla_prompts


def test_predicates_are_live_not_historical_scores():

    class RewardManager:
        released = True

        def is_stacked(self, labels, xy_threshold):
            assert xy_threshold == [0.0175]
            return tuple(labels)

        def is_all_gripper_open(self, threshold):
            assert threshold == 0.8
            return 'grippers'

        def check_once(self, check, env_idx):
            assert env_idx == 0
            if check == 'grippers':
                return self.released
            return len(check) == 2

    adapter = object.__new__(RoboDojoStackBlocksEnvironment)
    rm = RewardManager()
    adapter._env = SimpleNamespace(
        reward_manager=rm,
        _stacked_xy_thresholds=lambda: [0.0175],
        success=[False])
    observation = {'score': 1.0}
    assert adapter._with_predicates(observation)['predicates'] == dict(
        two_stacked=True, three_stacked=False, environment_success=False)
    rm.released = False
    assert adapter._with_predicates(
        observation)['predicates']['two_stacked'] is False
    assert observation == {'score': 1.0}


def test_adapter_rejects_other_tasks_before_loading_simulator():
    with pytest.raises(ValueError, match='stack_blocks only'):
        RoboDojoStackBlocksEnvironment(task_name='all')


def test_worker_controller_progression_and_episode_reset(monkeypatch):
    from scripts.planner_worker import serve

    monkeypatch.delenv('FLUXVLA_PLANNER_DEBUG_PORT', raising=False)
    policy = Config.fromfile(str(CONFIG)).eval.policy.to_dict()
    # Keep the original predicate-driven controller path covered too.
    policy.pop('subtask_source')
    policy.update(
        completion_mode='environment',
        monitor=dict(type='ObservationPredicateMonitor'),
        goal_conditions={'environment_success': True})
    policy['skill_catalog'] = dict(
        entities={},
        skills=[
            dict(
                name='stack_two',
                instruction='Stack two blocks.',
                preconditions={'two_stacked': False},
                completion={'two_stacked': True}),
            dict(
                name='stack_third',
                instruction='Stack a third block.',
                preconditions={'two_stacked': True},
                completion={'three_stacked': True}),
            dict(
                name='return_home',
                instruction='Return home.',
                preconditions={'three_stacked': True},
                completion={'environment_success': True}),
        ])
    policy['planner'] = dict(
        type='ScriptedSubtaskPlanner',
        choices=['skill_0000', 'skill_0001', 'skill_0002'])

    def step(index, two=False, three=False, success=False, executed=True):
        return dict(
            command='step',
            observation_id=index,
            executed_previous_chunk=executed,
            observation={
                'predicates':
                dict(
                    two_stacked=two,
                    three_stacked=three,
                    environment_success=success)
            })

    requests = iter([
        dict(command='configure', policy=policy),
        dict(command='reset', goal='Stack three blocks.'),
        step(0, executed=False),
        step(8),
        step(16, two=True),
        step(24, two=True, three=True),
        step(32, two=True, three=True, success=True),
        dict(command='reset', goal='New episode'),
        step(0, executed=False),
        dict(command='close')
    ])

    class Connection:
        responses = []

        def recv(self):
            return next(requests)

        def send(self, response):
            self.responses.append(response)

        def close(self):
            pass

    connection = Connection()
    serve(connection)
    assert all(r['ok'] for r in connection.responses)
    events = [r['result'] for r in connection.responses if r['result']]
    assert [e['decision']['skill'] for e in events[:4]
            ] == ['stack_two', 'stack_two', 'stack_third', 'return_home']
    assert events[1]['decision']['decision'] == 'continue'
    assert events[4]['status'] == 'succeeded'
    assert events[5]['subtask_version'] == 1


def test_planner_debugger_is_optional(monkeypatch):
    from scripts.planner_worker import start_debugger

    monkeypatch.delenv('FLUXVLA_PLANNER_DEBUG_PORT', raising=False)
    assert start_debugger() is None


def test_planner_debugger_listens_without_waiting(monkeypatch):
    from scripts.planner_worker import start_debugger

    calls = []
    debugger = SimpleNamespace(
        listen=lambda address: calls.append(address),
        wait_for_client=lambda: calls.append('wait'))
    monkeypatch.setitem(sys.modules, 'debugpy', debugger)
    monkeypatch.setenv('FLUXVLA_PLANNER_DEBUG_PORT', '5679')
    monkeypatch.setenv('FLUXVLA_PLANNER_DEBUG_WAIT', '1')
    wait = start_debugger()
    assert calls == [('127.0.0.1', 5679)]
    wait()
    assert calls[-1] == 'wait'


@pytest.mark.parametrize('value', ['-1', '0', '65536', 'abc'])
def test_planner_debugger_rejects_invalid_ports(monkeypatch, value):
    from scripts.planner_worker import start_debugger

    monkeypatch.setenv('FLUXVLA_PLANNER_DEBUG_PORT', value)
    with pytest.raises(ValueError):
        start_debugger()
