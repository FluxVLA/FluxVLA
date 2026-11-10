# Copyright 2026 Limx Dynamics
"""Shared eval/inference policy contracts; no API, GPU or robot connection."""

import copy
import inspect
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.engines.runners.base_eval_runner import BaseEvalRunner
from fluxvla.engines.runners.base_inference_runner import BaseInferenceRunner
from fluxvla.engines.runners.libero_eval_runner import LiberoEvalRunner
from fluxvla.engines.runners.robocasa_eval_runner import RobocasaEvalRunner
from fluxvla.engines.runners.robotwin_eval_runner import RobotwinEvalRunner

ROOT = Path(__file__).resolve().parents[2]
TASK = 'put the cup in the drawer and close it'
CAMERA = 'video.ego_view_bg_crop_pad_res256_freq20'


def policy_config(max_chunks=1, choices=('skill_0000', )):
    cfg = copy.deepcopy(
        Config.fromfile(
            ROOT / 'configs/hierarchical/robocasa_pi05_gpt6.py').eval.policy)
    cfg.planner = dict(type='ScriptedSubtaskPlanner', choices=list(choices))
    for skill in cfg.skill_catalog.skills:
        skill.max_chunks = max_chunks
    return cfg


def observation(grasped=False, finished=False):
    return dict(
        qpos=np.array([10., 20.], dtype=np.float32),
        **{CAMERA: np.zeros((16, 16, 3), dtype=np.uint8)},
        predicates={
            'grasped:cup': grasped,
            'inside:cup:drawer': finished,
            'closed:drawer': finished,
        })


class Inputs:

    def __init__(self):
        self.calls = []

    def __call__(self, data):
        self.calls.append(copy.deepcopy(data))
        self.last_raw_state = data.get('qpos')
        return dict(states=torch.ones(1, 2))


class Model(torch.nn.Module):

    def __init__(self, batched=False, width=2):
        super().__init__()
        self.calls = []
        self.resets = 0
        self.batched = batched
        self.width = width

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        value = torch.arange(
            3 * self.width, dtype=torch.float32).reshape(3, self.width)
        return value[None] if self.batched else value

    def reset_inference_history(self):
        self.resets += 1


@pytest.fixture
def fake_ros(monkeypatch):
    module = ModuleType('rospy')
    module.is_shutdown = lambda: False
    module.Rate = lambda _: SimpleNamespace(sleep=lambda: None)
    monkeypatch.setitem(sys.modules, 'rospy', module)
    return module


class InferenceRunner(BaseInferenceRunner):
    policy_execution_ack = True

    def update_observation_window(self):
        self.captures += 1
        return self.next_observation()

    def _execute_actions(self, actions, rate):
        count = min(len(actions), self.ack_limit)
        self.published.extend(actions[:count].copy())
        return count

    def _init_zmq_client(self, cfg):
        pass  # Fake transport only; never create a socket.

    def _predict_action_remote(self, inputs):
        self.remote_requests.append(copy.deepcopy(inputs))
        return torch.full((3, 2), 42.)  # Already native.


def make_inference(monkeypatch,
                   tmp_path,
                   *,
                   remote=False,
                   policy=True,
                   max_chunks=1,
                   max_steps=4,
                   repeats=3,
                   ack_limit=2):
    import fluxvla.engines as engines
    from fluxvla.engines.runners import base_inference_runner

    model, inputs = Model(), Inputs()
    denorm_calls = []

    def denormalize(data):
        denorm_calls.append(copy.deepcopy(data))
        return data['action'] * 2 + data['state']

    monkeypatch.setattr(engines, 'build_vla_from_cfg', lambda _: model)
    monkeypatch.setattr(engines, 'build_dataset_from_cfg', lambda _: inputs)
    monkeypatch.setattr(engines, 'build_transform_from_cfg',
                        lambda _: denormalize)
    operator = SimpleNamespace(stop_trajectory=Mock())
    monkeypatch.setattr(base_inference_runner, 'build_operator_from_cfg',
                        lambda _: operator)
    ckpt = tmp_path / 'checkpoints' / 'fake.pt'
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({}, ckpt)
    (tmp_path / 'dataset_statistics.json').write_text('{}')
    runner = InferenceRunner(
        cfg=Config(dict(inference_model=dict(type='Fake'))),
        ckpt_path=None if remote else str(ckpt),
        dataset={},
        denormalize_action={},
        action_chunk=2,
        max_publish_step=max_steps,
        remote_inference={} if remote else None,
        enable_mixed_precision=False,
        policy_poll_interval_s=0,
        policy=policy_config(max_chunks) if policy else None)
    runner.next_observation = observation
    runner.captures = 0
    runner.published = []
    runner.remote_requests = []
    runner.ack_limit = ack_limit
    runner._get_user_task_instruction = lambda _: [TASK] * repeats
    runner._poll_keyboard_pause = lambda: False
    return runner, model, inputs, denorm_calls


def test_policy_configuration_lives_in_bases():
    for base in (BaseEvalRunner, BaseInferenceRunner):
        assert 'policy' in inspect.signature(base).parameters
        assert 'observation_adapter' in inspect.signature(base).parameters
    for runner in (RobocasaEvalRunner, LiberoEvalRunner, RobotwinEvalRunner):
        assert 'policy' not in inspect.signature(runner).parameters
        assert 'observation_adapter' not in inspect.signature(
            runner).parameters
        assert runner.step_policy is BaseEvalRunner.step_policy
    assert BaseInferenceRunner.step_policy is BaseEvalRunner.step_policy


def test_base_config_does_not_copy_initialized_backend():
    backend = Model()
    config = dict(type='HierarchicalPolicy', planner=backend, options=[1])
    runner = BaseEvalRunner(policy=config)
    assert runner.policy_config['planner'] is backend
    config['options'].append(2)
    assert runner.policy_config['options'] == [1]


@pytest.mark.parametrize('remote', [False, True])
def test_inference_uses_subtask_and_stops_on_budget(monkeypatch, tmp_path,
                                                    fake_ros, remote):
    runner, model, inputs, denorm_calls = make_inference(
        monkeypatch, tmp_path, remote=remote)
    runner._run_episode(TASK)
    assert len(runner.published) == 2
    assert runner.captures == 2
    assert runner.last_policy_output.status == 'failed'
    assert runner.policy.controller.chunks == 1
    assert runner.policy._pending_horizon is None
    assert runner.ros_operator.stop_trajectory.call_count >= 2
    if remote:
        assert not model.calls and not inputs.calls and not denorm_calls
        request = runner.remote_requests[0]
        assert request['task_description'] == 'unlocked_waist: pick up the cup'
        assert request['is_new_episode'] is True
        assert 'predicates' not in request
        np.testing.assert_array_equal(runner.published, np.full((2, 2), 42.))
    else:
        assert len(model.calls) == len(inputs.calls) == len(denorm_calls) == 1
        assert inputs.calls[0]['task_description'] == \
            'unlocked_waist: pick up the cup'
        np.testing.assert_array_equal(runner.published, [[10, 22], [14, 26]])
        np.testing.assert_array_equal(denorm_calls[0]['state'], [10, 20])
    # A new episode reuses the loaded backend but not the failed session.
    policy = runner.policy
    runner._run_episode(TASK)
    assert runner.policy is policy
    assert len(runner.published) == 4
    runner.release_policy()
    assert runner.policy is None


def test_inference_partial_execution_and_limit(monkeypatch, tmp_path,
                                               fake_ros):
    runner, _, _, _ = make_inference(
        monkeypatch,
        tmp_path,
        max_chunks=9,
        max_steps=3,
        repeats=6,
        ack_limit=1)
    runner._run_episode(TASK)
    assert len(runner.published) == 3
    assert runner.policy.controller.chunks == 3
    assert runner.captures == 3


def test_inference_waits_for_fresh_observations(monkeypatch, tmp_path,
                                                fake_ros):
    runner, model, _, _ = make_inference(monkeypatch, tmp_path)
    runner.policy.controller.stable_observations = 2
    runner.next_observation = lambda: observation(finished=True)
    runner._run_episode(TASK)
    assert runner.captures == 2
    assert not runner.published and not model.calls
    assert runner.last_policy_output.status == 'succeeded'


def test_inference_discards_stale_prediction(monkeypatch, tmp_path, fake_ros):
    from fluxvla.engines.runners import base_inference_runner
    runner, model, _, _ = make_inference(monkeypatch, tmp_path)
    clock = [0.]
    monkeypatch.setattr(base_inference_runner.time, 'monotonic',
                        lambda: clock[0])
    original = model.predict_action

    def slow(**kwargs):
        clock[0] += 100.
        return original(**kwargs)

    model.predict_action = slow
    with pytest.raises(TimeoutError, match='expired'):
        runner._run_episode(TASK)
    assert not runner.published
    assert runner.policy.controller.chunks == 0
    assert runner.policy._pending_horizon is None
    assert runner.ros_operator.stop_trajectory.called


def test_inference_legacy_path_is_unchanged(monkeypatch, tmp_path, fake_ros):
    runner, model, inputs, _ = make_inference(
        monkeypatch, tmp_path, policy=False, repeats=2)
    runner._run_episode(TASK)
    assert runner.policy is None
    assert len(runner.published) == 4
    assert len(model.calls) == 2
    assert all(c['task_description'] == TASK for c in inputs.calls)


@pytest.mark.parametrize(
    'mode', ['rtc', 'async', 'mobile', 'custom_loop', 'missing_ack'])
def test_unsupported_policy_execution_fails_before_operator_build(
        monkeypatch, mode):
    from fluxvla.engines.runners import base_inference_runner
    construct = Mock(side_effect=AssertionError('must not connect'))
    monkeypatch.setattr(base_inference_runner, 'build_operator_from_cfg',
                        construct)
    attributes = {}
    name = 'FakeRTC' if mode == 'rtc' else 'FakeExecutor'
    if mode == 'async':
        attributes['async_execution'] = True
    if mode == 'custom_loop':
        attributes['_run_episode'] = lambda *_: None
    if mode == 'missing_ack':
        attributes['policy_execution_ack'] = False
    cls = type(name, (InferenceRunner, ), attributes)
    with pytest.raises(ValueError, match='synchronous|acknowledgments'):
        cls(policy=policy_config(),
            remote_inference={},
            use_robot_base=mode == 'mobile')
    construct.assert_not_called()


def patch_eval_runtime(monkeypatch, module):
    from fluxvla.engines.runners import base_eval_runner
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'local_rank', lambda: 0, raising=False)
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'distributed_state', None, raising=False)
    monkeypatch.setattr(module.overwatch, 'rank', lambda: 0)
    monkeypatch.setattr(module.overwatch, 'world_size', lambda: 1)
    monkeypatch.setattr(module.torch.cuda, 'current_device',
                        lambda: torch.device('cpu'))
    for name in ('barrier', 'all_reduce', 'broadcast_object_list'):
        monkeypatch.setattr(module.dist, name, lambda *args, **kwargs: None)


@pytest.mark.parametrize('hierarchical', [False, True])
def test_libero_full_loop_preserves_hooks_and_normalization(
        monkeypatch, tmp_path, hierarchical):
    import fluxvla.engines as engines
    from fluxvla.engines.runners import libero_eval_runner as module
    patch_eval_runtime(monkeypatch, module)
    model, inputs = Model(batched=True), Inputs()
    feedback = []
    model.configure_eval_env = Mock()
    model.build_eval_predict_action_kwargs = Mock(return_value={'extra': 123})
    model.after_eval_env_step = feedback.append
    monkeypatch.setattr(engines, 'build_vla_from_cfg', lambda _: model)
    monkeypatch.setattr(engines, 'build_dataset_from_cfg',
                        lambda _: lambda data: (inputs(data), None))
    denorm_calls = []

    def denormalize(data):
        denorm_calls.append(copy.deepcopy(data))
        return data['action'] + 50

    monkeypatch.setattr(engines, 'build_transform_from_cfg',
                        lambda _: denormalize)

    class Env:

        def __init__(self):
            self.actions = []
            self.closed = False

        def reset(self):
            pass

        def set_init_state(self, state):
            return observation()

        def step(self, action):
            self.actions.append(action)
            # First 2 commands are warmup; then complete the grasp skill.
            return observation(grasped=len(self.actions) >= 4), 0., False, {}

        def close(self):
            self.closed = True

    env = Env()
    suite = SimpleNamespace(
        n_tasks=1, get_task=lambda _: None, get_task_init_states=lambda _: [0])
    monkeypatch.setattr(
        module, '_get_libero_benchmark', lambda: SimpleNamespace(
            get_benchmark_dict=lambda: {'libero_spatial': lambda: suite}))
    monkeypatch.setattr(module, 'get_libero_env', lambda *a, **k: (env, TASK))
    runner = LiberoEvalRunner(
        cfg=Config(dict(model=dict(type='Fake'))),
        seed=7,
        ckpt_path=None,
        model_family='fake',
        task_suite_name='libero_spatial',
        dataset={},
        denormalize_action={},
        requires_dataset_stats=False,
        eval_chunk_size=2,
        num_steps_wait=2,
        max_steps=8,
        num_trials_per_task=1,
        num_inference_steps=7,
        inference_seed=13,
        norm_stats_key='custom-stats',
        save_rollout_videos=False,
        output_dir=str(tmp_path),
        model_build_device='cpu',
        enable_mixed_precision_training=False,
        policy=policy_config(1, ('skill_0000',
                                 'skill_0001')) if hierarchical else None)
    runner.run()
    assert env.closed
    assert (Path(runner.run_dir) / 'summary.txt').is_file()
    np.testing.assert_array_equal(env.actions[2], [50, 51])
    np.testing.assert_array_equal(feedback[0], [0, 1])
    assert all(c['task_suite_name'] == 'libero_spatial'
               and c['norm_stats_key'] == 'custom-stats' for c in denorm_calls)
    assert all(c['unnorm_key'] == 'libero_spatial' and c['seed'] == 13
               and c['num_inference_steps'] == 7 and c['extra'] == 123
               for c in model.calls)
    if hierarchical:
        assert len(env.actions) == 6
        assert len(inputs.calls) == 2  # No stale pre-tokenized overall prompt.
        assert inputs.calls[0]['task_description'].endswith('pick up the cup')
        assert inputs.calls[1]['task_description'].endswith(
            'place the cup into the drawer')
        assert runner.last_policy_output.status == 'failed'
        assert runner.policy._pending_horizon is None
    else:
        assert runner.policy is None
        assert len(env.actions) == 10
        assert all(c['task_description'] == TASK for c in inputs.calls)
    runner.cleanup()
    assert runner.policy is None


def test_robotwin_hooks_execute_acknowledge_and_stop(monkeypatch, tmp_path):
    import fluxvla.engines as engines
    from fluxvla.engines.runners import robotwin_eval_runner as module
    model, inputs = Model(batched=True, width=14), Inputs()
    monkeypatch.setattr(engines, 'build_vla_from_cfg', lambda _: model)
    monkeypatch.setattr(engines, 'build_dataset_from_cfg', lambda _: inputs)
    monkeypatch.setattr(engines, 'build_transform_from_cfg',
                        lambda _: lambda data: data['action'][0] + 10)
    monkeypatch.setattr(module, '_load_state_dict', lambda _: {})
    monkeypatch.setattr(module, '_ACTIVE_POLICY', None)
    monkeypatch.setattr(module, '_ACTIVE_KEY', None)
    checkpoint = tmp_path / 'checkpoints' / 'fake.pt'
    checkpoint.parent.mkdir()
    checkpoint.touch()
    (tmp_path / 'dataset_statistics.json').write_text('{}')

    class Adapter:

        def __call__(self, obs, env):
            assert env is environment
            return dict(obs, predicates=observation()['predicates'])

    cfg = Config(
        dict(
            model=dict(type='Fake'),
            eval=dict(
                policy=policy_config(),
                observation_adapter=dict(type=Adapter))))
    # Standalone get_model also reads policy settings from cfg.eval.
    runtime = module.get_model(
        dict(
            _fluxvla_cfg=cfg,
            ckpt_path=str(checkpoint),
            dataset={},
            denormalize_action={},
            eval_chunk_size=2,
            device='cpu'))
    camera = dict(rgb=np.zeros((16, 16, 3), dtype=np.uint8))
    obs = dict(
        observation=dict(
            head_camera=camera, left_camera=camera, right_camera=camera),
        joint_action=dict(vector=np.zeros(14)))
    taken = []

    def take(action, **kwargs):
        taken.append(action.copy())
        environment.take_action_cnt += 1

    environment = SimpleNamespace(
        step_lim=10,
        take_action_cnt=0,
        eval_success=False,
        get_instruction=lambda: TASK,
        take_action=take)
    module.reset_model(runtime)
    module.eval(environment, runtime, obs)
    assert len(taken) == 2
    assert runtime.policy.controller.chunks == 1
    np.testing.assert_array_equal(taken[0], np.arange(14) + 10)
    module.eval(environment, runtime, obs)
    assert runtime.last_policy_output.status == 'failed'
    assert environment.step_lim == environment.take_action_cnt == 2
    assert len(model.calls) == 1
    module.reset_model(runtime)
    environment.step_lim = 10
    module.eval(environment, runtime, obs)
    assert len(model.calls) == 2
    module.close_model()
    assert runtime.policy is None and module._ACTIVE_POLICY is None


@pytest.mark.parametrize('operator_kind', ['base', 'aloha'])
@pytest.mark.parametrize('stop_after', [0, 2, 3])
def test_synchronous_operators_report_partial_publication(
        monkeypatch, fake_ros, operator_kind, stop_after):
    from fluxvla.engines.operators.aloha_operator import AlohaOperator
    from fluxvla.engines.operators.base_operator import BaseOperator
    cls = BaseOperator if operator_kind == 'base' else AlohaOperator
    operator = object.__new__(cls)
    operator._traj_stop_event = threading.Event()
    operator.command_mode = 'joint'
    published = []
    fake_ros.is_shutdown = lambda: len(published) >= stop_after
    if operator_kind == 'base':
        operator.send_joints = published.append
        count = operator.execute_trajectory({'arm': np.ones((3, 7))})
    else:
        operator._send_joints = lambda left, right: published.append(left)
        count = operator.execute_trajectory(np.ones((3, 7)), np.ones((3, 7)))
    assert count == len(published) == stop_after


def test_robotwin_resume_does_not_mix_baseline_and_policy(tmp_path):
    runner = object.__new__(RobotwinEvalRunner)
    runner.configure_policy()
    runner.ckpt_path = str(tmp_path / 'step-10.pt')
    runner.result_output_dir = str(tmp_path)
    runner.resume = True
    runner.model_family = 'fake'
    runner.run_id_suffix = None
    runner.task_suite_name = 'clean'
    runner.num_trials_per_task = 1
    runner.max_episode_steps = 5
    runner.seed = 7
    runner.instruction_type = 'unseen'
    runner.eval_chunk_size = 2
    runner.task_list = ['task']
    previous = tmp_path / 'eval_runs/step-10/EVAL-robotwin-fake-old'
    previous.mkdir(parents=True)
    payload = dict(
        ckpt=runner.ckpt_path,
        task_suite_name='clean',
        trials_per_task=1,
        max_episode_steps=5,
        seed=7,
        instruction_type='unseen',
        eval_chunk_size=2,
        task_list=['task'],
        overall=dict(completed_tasks=0, total_tasks=1))
    summary = previous / 'summary.json'
    summary.write_text(json.dumps(payload))
    assert runner._resolve_run_dir() == str(previous)
    runner.configure_policy(policy_config())
    assert runner._resolve_run_dir() != str(previous)
    payload['policy_fingerprint'] = runner._policy_fingerprint()
    summary.write_text(json.dumps(payload))
    assert runner._resolve_run_dir() == str(previous)
    runner.configure_policy(policy_config(max_chunks=2))
    assert runner._resolve_run_dir() != str(previous)


def test_adapter_only_uses_default_policy_without_leaking_environment():
    model, inputs = Model(), Inputs()
    env = object()

    class Adapter:

        def __call__(self, obs, context):
            assert context is env
            return dict(obs, predicates={'observed': True})

    runner = BaseEvalRunner(observation_adapter=dict(type=Adapter))
    runner.setup_policy(
        vla=model,
        input_processor=inputs,
        output_processor=lambda value, **_: (value.numpy(), []),
        enable_mixed_precision=False)
    runner.reset_policy(TASK)
    result = runner.step_policy(observation(), 0, environment=env)
    assert result.status == 'ready'
    assert 'predicates' not in inputs.calls[0]
    assert len(model.calls) == 1
    runner.acknowledge_policy(1)
