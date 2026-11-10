# Copyright 2026 Limx Dynamics
"""Offline policy, simulator-adapter and runner-contract regressions."""

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine import Config

from fluxvla.engines.observation_adapters import \
    RobocasaDrawerObservationAdapter
from fluxvla.engines.utils import build_policy_from_cfg
from fluxvla.policies import ActionChunkProcessor, VLAPolicy

CONFIG = (
    Path(__file__).resolve().parents[2] /
    'configs/hierarchical/robocasa_pi05_gpt6.py')
TASK = 'unlocked_waist: put the cup in the drawer and close it'
CAMERA = 'video.ego_view_bg_crop_pad_res256_freq20'


class FakeInputs:

    def __init__(self):
        self.calls = []
        self.recorded = []
        self.last_raw_state = np.array([10., 20.])

    def __call__(self, observation):
        self.calls.append(copy.deepcopy(observation))
        self.last_debug = {'text': observation['task_description']}
        return {'states': torch.ones(1, 2)}, None

    def record_observation(self, observation, step):
        self.recorded.append(step)


class FakeVLA:

    def __init__(self):
        self.calls = []
        self.resets = 0

    def eval(self):
        return self

    def reset_inference_history(self):
        self.resets += 1

    def predict_action(self, **batch):
        self.calls.append(batch)
        return torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)


class FakeDenormalizer:

    def __init__(self, offset=0):
        self.calls = []
        self.offset = offset

    def __call__(self, inputs):
        self.calls.append(copy.deepcopy(inputs))
        result = (
            inputs['action'] + inputs.get('state', 0) +
            inputs['action_horizon_index'] + self.offset)
        # Even a mutating transform cannot corrupt the next temporal row.
        if 'state' in inputs:
            inputs['state'][:] = -999
        return result

    def get_action_stats_horizon(self):
        return 4


def policy_config(choices=('skill_0000', ), max_chunks=60):
    cfg = copy.deepcopy(Config.fromfile(CONFIG).eval.policy)
    cfg.planner = dict(type='ScriptedSubtaskPlanner', choices=list(choices))
    for skill in cfg.skill_catalog.skills:
        skill.max_chunks = max_chunks
    return cfg


def make_policy(
        choices=('skill_0000', ),
        *,
        hierarchical=True,
        max_chunks=60,
        planner=None):
    cfg = policy_config(choices, max_chunks) if hierarchical else dict(
        type='VLAPolicy')
    if planner is not None:
        cfg['planner'] = planner
    result = build_policy_from_cfg(
        cfg,
        default_args=dict(
            vla=FakeVLA(),
            input_processor=FakeInputs(),
            output_processor=ActionChunkProcessor(FakeDenormalizer(), 2, 'rc'),
            unnorm_key='rc',
            enable_mixed_precision=False))
    result.reset(TASK)
    return result


def observation(grasped=False, inside=False, closed=False):
    return {
        'annotation.human.coarse_action': TASK,
        CAMERA: np.zeros((16, 16, 3), dtype=np.uint8),
        'predicates': {
            'grasped:cup': grasped,
            'inside:cup:drawer': inside,
            'closed:drawer': closed,
        },
    }


def test_config_uses_pi05_checkpoint_assets(monkeypatch):
    monkeypatch.setenv('PI05_ROBOCASA_CHECKPOINT_DIR', '/tmp/pi05-run')
    monkeypatch.setenv('OPENAI_PLANNER_MODEL', 'test-gpt')
    cfg = Config.fromfile(CONFIG)
    assert cfg.model.type == 'PI05FlowMatching'
    assert cfg.model.ori_action_dim == 29
    assert cfg.model.n_action_steps == 16
    assert cfg.eval.type == 'RobocasaEvalRunner'
    assert cfg.eval.task_ids == [2]
    assert 'PnPCupToDrawerClose' in cfg.eval.task_list[2]
    assert cfg.eval.num_trials_per_task == 1
    assert cfg.eval.norm_stats_path == '/tmp/pi05-run/dataset_statistics.json'
    tokenizer = next(t.tokenizer for t in cfg.eval.dataset.transforms
                     if t.type == 'ProcessPrompts')
    assert tokenizer.model_path == '/tmp/pi05-run/tokenizer'
    assert 'planning' not in cfg.eval
    assert cfg.eval.policy.type == 'HierarchicalPolicy'
    assert cfg.eval.policy.planner.model == 'test-gpt'
    assert cfg.eval.policy.monitor.type == 'ObservationPredicateMonitor'
    assert cfg.eval.observation_adapter.type == \
        'RobocasaDrawerObservationAdapter'
    assert cfg.eval.action_keys['action.left_hand'] == (7, 13)
    assert cfg.eval.denormalize_action.action_dim == 29
    assert cfg.eval.eval_chunk_size == 8
    assert cfg.eval.max_episode_steps == 720
    assert [s.max_chunks for s in cfg.eval.policy.skill_catalog.skills] == \
        [60, 60, 60]


def test_plain_policy_preserves_prompt_and_native_action_pipeline():
    policy = make_policy(hierarchical=False)
    obs = observation()
    result = policy.step(obs, 0)
    assert result.status == 'ready'
    assert result.metadata == {}
    assert policy.input_processor.calls[0]['task_description'] == TASK
    assert policy.input_processor.calls[0]['is_new_episode'] is True
    assert 'predicates' not in policy.input_processor.calls[0]
    assert 'task_description' not in obs  # Never mutate the env observation.
    assert policy.vla.calls[0]['unnorm_key'] == 'rc'
    np.testing.assert_array_equal(result.actions, [[10, 21], [13, 24]])
    denorm = policy.output_processor.denormalize_action
    assert [c['action_horizon_index'] for c in denorm.calls] == [0, 1]
    for call in denorm.calls:
        np.testing.assert_array_equal(call['state'], [10, 20])
        assert call['task_suite_name'] == 'rc'
    np.testing.assert_array_equal(policy.input_processor.last_raw_state,
                                  [10, 20])
    assert policy.vla.resets == 0
    policy.acknowledge(2)
    policy.reset('')  # Baseline permits environments without a description.
    assert policy.step(obs, 0).status == 'ready'


def test_subtasks_switch_only_after_observed_completion():
    policy = make_policy(('skill_0000', 'skill_0001', 'skill_0002'))
    steps = [
        (0, observation(), 'pick up the cup'),
        (2, observation(), 'pick up the cup'),
        (4, observation(grasped=True), 'place the cup into the drawer'),
        (6, observation(inside=True), 'close the drawer'),
    ]
    for step, obs, instruction in steps:
        result = policy.step(obs, step)
        assert result.status == 'ready'
        assert result.metadata['subtask']['instruction'] == instruction
        assert policy.input_processor.calls[-1]['task_description'] == \
            'unlocked_waist: ' + instruction
        policy.acknowledge(2)
    assert policy.controller.planner.index == 3
    assert policy.vla.resets == 3
    result = policy.step(observation(inside=True, closed=True), 8)
    assert result.status == 'succeeded'
    assert result.actions is None
    assert len(policy.vla.calls) == 4
    assert result.event()['status'] == 'succeeded'


def test_budget_counts_acknowledged_execution_and_returns_failure():
    policy = make_policy(max_chunks=1)
    policy.step(observation(), 0)
    assert policy.controller.chunks == 0
    policy.acknowledge(0)
    assert policy.controller.chunks == 0
    policy.step(observation(), 1)
    policy.acknowledge(1)  # A partially executed chunk consumes one dispatch.
    assert policy.controller.chunks == 1
    result = policy.step(observation(), 2)
    assert result.status == 'failed'
    assert result.actions is None
    assert result.metadata['planning']['reason'] == \
        'subtask_chunk_budget_exhausted'
    assert len(policy.vla.calls) == 2


@pytest.mark.parametrize('count', [-1, 3, True, 1.0, None])
def test_invalid_acknowledgment_does_not_consume_budget(count):
    policy = make_policy()
    policy.step(observation(), 0)
    with pytest.raises(ValueError, match='acknowledgment'):
        policy.acknowledge(count)
    assert policy.controller.chunks == 0
    policy.acknowledge(2)
    with pytest.raises(ValueError, match='acknowledgment'):
        policy.acknowledge(2)


def test_reset_and_missing_acknowledgment():
    policy = make_policy()
    policy.step(observation(), 0)
    with pytest.raises(RuntimeError, match='Acknowledge'):
        policy.step(observation(), 2)
    policy.reset(TASK)
    assert policy.controller.active is None
    assert policy.controller.chunks == 0
    assert policy.controller.history == []
    assert policy.controller.planner.index == 0
    assert policy.step(observation(), 0).status == 'ready'


@pytest.mark.parametrize(('choice', 'status'), [
    ('blocked', 'blocked'),
    ('request_finish', 'waiting'),
    ('invented_skill', 'failed'),
])
def test_non_execution_decisions_never_invoke_vla(choice, status):
    policy = make_policy((choice, ))
    result = policy.step(observation(), 0)
    assert result.status == status
    assert result.actions is None
    assert not policy.vla.calls
    assert not policy.input_processor.calls
    assert policy.vla.resets == 0
    assert policy.controller.chunks == 0


def test_initialized_local_planner_sees_raw_snapshot_before_tokenization():

    class LocalPlanner:

        def reset(self):
            self.calls = []

        def choose(self, **kwargs):
            self.calls.append(kwargs)
            assert not policy.input_processor.calls
            # A local VLM can consume random numbers before the VLA is seeded.
            torch.rand(97)
            return {'choice': 'skill_0000'}

    local = LocalPlanner()
    policy = make_policy(planner=local)
    draws = []
    original_predict = policy.vla.predict_action

    def predict(**batch):
        draws.append(torch.rand(3))
        return original_predict(**batch)

    policy.vla.predict_action = predict
    obs = observation()
    policy.step(obs, 0, policy_seed=123)
    torch.manual_seed(123)
    torch.testing.assert_close(draws[0], torch.rand(3))
    assert policy.controller.planner is local
    assert local.calls[0]['goal'] == TASK
    assert local.calls[0]['observation'][CAMERA].dtype == np.uint8
    assert local.calls[0]['observation'][CAMERA].shape == (16, 16, 3)
    assert '_robocasa_env' not in local.calls[0]['observation']
    assert 'task_description' not in local.calls[0]['observation']
    obs[CAMERA][:] = 255
    assert not local.calls[0]['observation'][CAMERA].any()
    assert not hasattr(policy.input_processor, 'planner')


def test_provider_error_is_failure_without_actions():

    class BrokenPlanner:

        def reset(self):
            pass

        def choose(self, **kwargs):
            raise TimeoutError('provider timed out')

    policy = make_policy(planner=BrokenPlanner())
    result = policy.step(observation(), 0)
    assert result.status == 'failed'
    assert 'provider timed out' in result.metadata['planning']['reason']
    assert not policy.vla.calls


def test_episode_output_processor_can_switch_group_statistics():
    policy = make_policy(hierarchical=False)
    policy.reset(
        TASK,
        output_processor=ActionChunkProcessor(
            FakeDenormalizer(offset=100), 2, 'other-group'))
    result = policy.step(observation(), 0)
    np.testing.assert_array_equal(result.actions, [[110, 121], [113, 124]])
    assert policy.output_processor.denormalize_action.calls[0][
        'task_suite_name'] == 'other-group'


@pytest.mark.parametrize('shape', [(2, 4, 2), (4, 2), (2, ), (1, 2, 2, 2)])
def test_action_processor_rejects_wrong_batch_shapes(shape):
    with pytest.raises(ValueError, match='Expected actions'):
        ActionChunkProcessor(FakeDenormalizer(), 2, 'rc')(torch.zeros(shape))


def test_action_processor_horizons_finite_checks_and_single_action():
    processor = ActionChunkProcessor(FakeDenormalizer(), 1, 'rc')
    actions, _ = processor(torch.tensor([[1., 2.]]))
    np.testing.assert_array_equal(actions, [[1, 2]])
    with pytest.raises(ValueError, match='model output horizon'):
        ActionChunkProcessor(FakeDenormalizer(), 5, 'rc')(torch.zeros(1, 4, 2))
    with pytest.raises(ValueError, match='statistics horizon'):
        ActionChunkProcessor(FakeDenormalizer(), 5, 'rc')(torch.zeros(1, 6, 2))
    with pytest.raises(ValueError, match='non-finite'):
        processor(torch.tensor([[float('nan'), 0.]]))
    with pytest.raises(TypeError, match='Tensor'):
        processor(np.zeros((1, 2)))
    with pytest.raises(ValueError, match='finite'):
        ActionChunkProcessor(lambda _: np.array([float('inf')]), 1, 'rc')(
            torch.ones(1, 2))


def test_drawer_adapter_uses_geometry_not_fixture_contact(monkeypatch):
    module = ModuleType('robocasa.utils.object_utils')
    calls = []

    def inside(**kwargs):
        calls.append(kwargs)
        return False

    module.obj_inside_of = inside
    monkeypatch.setitem(sys.modules, module.__name__, module)
    simulator = SimpleNamespace(
        behavior='close',
        get_subtask_term_signals=lambda: dict(grasp_object=1, obj_in_drawer=1),
        objects={'obj': SimpleNamespace(name='test-cup')},
        drawer=SimpleNamespace(get_door_state=lambda **_: {'door': 0.005}))
    env = SimpleNamespace(unwrapped=SimpleNamespace(env=simulator))
    adapter = RobocasaDrawerObservationAdapter()
    original = {'camera': np.zeros((4, 4, 3), dtype=np.uint8)}
    result = adapter(original, env)
    assert result['predicates'] == {
        'grasped:cup': True,
        'inside:cup:drawer': False,
        'closed:drawer': True,
    }
    assert 'predicates' not in original
    assert set(result) == {'camera', 'predicates'}
    assert calls[0]['partial_check'] is True
    assert calls[0]['obj_name'] == 'test-cup'
    simulator.behavior = 'open'
    with pytest.raises(ValueError, match='drawer-close'):
        adapter(original, env)


@pytest.mark.parametrize(
    'mode',
    ['baseline', 'budget_failure', 'early_success', 'unverified_success'])
def test_runner_execution_and_artifacts(monkeypatch, tmp_path, mode):
    import fluxvla.engines as engines
    from fluxvla.engines.runners import base_eval_runner, robocasa_eval_runner
    from fluxvla.engines.runners.robocasa_eval_runner import RobocasaEvalRunner
    from fluxvla.engines.utils import eval_utils

    inputs, vla, denorm = FakeInputs(), FakeVLA(), FakeDenormalizer()
    monkeypatch.setattr(engines, 'build_dataset_from_cfg', lambda _: inputs)
    monkeypatch.setattr(engines, 'build_vla_from_cfg', lambda _: vla)
    monkeypatch.setattr(engines, 'build_transform_from_cfg', lambda _: denorm)
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'local_rank', lambda: 0, raising=False)
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'distributed_state', None, raising=False)
    monkeypatch.setattr(robocasa_eval_runner.overwatch, 'rank', lambda: 0)
    monkeypatch.setattr(robocasa_eval_runner.overwatch, 'world_size',
                        lambda: 1)
    monkeypatch.setattr(robocasa_eval_runner.dist, 'barrier', lambda: None)
    monkeypatch.setattr(robocasa_eval_runner.dist, 'all_reduce',
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_utils, 'check_robosuite_runtime', lambda _: None)
    for name in ('gymnasium', 'robocasa', 'robocasa.utils',
                 'robocasa.utils.gym_utils'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['robocasa.utils.gym_utils'].GrootRoboCasaEnv = object

    class Env:

        def __init__(self):
            self.actions = []
            self.closed = False

        def reset(self, **kwargs):
            return observation(), {}

        def step(self, action):
            self.actions.append(copy.deepcopy(action))
            return observation(), 0., False, False, {
                'success': mode == 'early_success'
            }

        def close(self):
            self.closed = True

    env = Env()
    sys.modules['gymnasium'].make = lambda *args, **kwargs: env
    frames = []

    def writer(path, **kwargs):
        Path(path).touch()
        return SimpleNamespace(append_data=frames.append, close=lambda: None)

    monkeypatch.setattr(robocasa_eval_runner.imageio, 'get_writer', writer)

    class Adapter:

        def __call__(self, obs, environment):
            assert environment is env
            return dict(
                obs,
                predicates=observation(
                    inside=mode == 'unverified_success',
                    closed=mode == 'unverified_success')['predicates'])

    runner = RobocasaEvalRunner(
        cfg=Config(dict(model=dict(type='Fake'))),
        seed=7,
        ckpt_path=None,
        model_family='fake',
        task_list=['Fake/PnPCupToDrawerClose'],
        dataset={},
        denormalize_action={},
        eval_chunk_size=2,
        max_episode_steps=3,
        num_trials_per_task=1,
        model_build_device='cpu',
        action_keys={'arm': [0, 2]},
        requires_dataset_stats=False,
        output_dir=str(tmp_path),
        rollout_video_key=CAMERA,
        policy=(policy_config(
            max_chunks=1) if mode in ('budget_failure',
                                      'unverified_success') else None),
        observation_adapter=dict(type=Adapter))
    runner.run()
    expected_steps = {
        'baseline': 3,
        'budget_failure': 2,
        'early_success': 1,
        'unverified_success': 0
    }[mode]
    assert len(env.actions) == expected_steps
    assert env.closed
    assert inputs.recorded == list(range(expected_steps + 1))
    assert len(frames) == expected_steps + 1
    if expected_steps:
        np.testing.assert_array_equal(env.actions[0]['arm'], [10, 21])
        assert runner.policy._pending_horizon is None
    if mode == 'baseline':
        assert type(runner.policy) is VLAPolicy
    root = Path(runner.run_dir)
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['overall']['total_episodes'] == 1
    assert summary['overall']['successes'] == int(mode == 'early_success')
    assert (root / 'robocasa/task0_results.json').is_file()
    assert len(list((root / 'rollouts').glob('*.mp4'))) == 1
    if mode == 'budget_failure':
        assert 'subtask_chunk_budget_exhausted' in (root /
                                                    'rank0.txt').read_text()
