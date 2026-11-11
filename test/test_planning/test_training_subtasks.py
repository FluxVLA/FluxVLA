# Copyright 2026 Limx Dynamics
"""Metadata-backed candidates: literal text, task isolation and reselection."""

import copy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from mmengine import Config

from fluxvla.planning import (ObservationPredicateMonitor,
                              OpenAISubtaskPlanner, ScriptedSubtaskPlanner,
                              TrainingSubtaskCatalog,
                              TrainingSubtaskController)

TASK = 'Stack the three blocks with different textures.'
OTHER_TASK = 'Stack the three bowls together.'
LEFT = 'The left arm placed the wood block on the purple block.'
RIGHT = 'The right arm grasped the purple block as the {base}.'
BOWL = 'The right arm placed the blue bowl in the white bowl.'
CONFIG = (
    Path(__file__).resolve().parents[2] /
    'configs/hierarchical/robodojo_pi05_gpt6.py')


def record(index, task=TASK, names=None):
    names = [RIGHT, LEFT] if names is None else names
    return dict(
        episode_index=index,
        length=len(names) * 10,
        tasks=[task],
        dense_subtask_names=names,
        dense_subtask_start_frames=list(range(0,
                                              len(names) * 10, 10)),
        dense_subtask_end_frames=list(range(10,
                                            len(names) * 10 + 10, 10)))


def write_records(root, records, name='file-000.parquet'):
    path = root / 'meta/episodes/chunk-000' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


@pytest.fixture
def training_root(tmp_path):
    write_records(tmp_path, [record(0), record(1, OTHER_TASK, [BOWL])])
    write_records(tmp_path, [record(2, names=['  ' + LEFT + '  '])],
                  'file-001.parquet')
    return tmp_path


def controller(root,
               choices=('skill_0000', 'skill_0000', 'skill_0001'),
               **kwargs):
    result = TrainingSubtaskController(
        catalog=TrainingSubtaskCatalog(root),
        planner=ScriptedSubtaskPlanner(choices),
        monitor=ObservationPredicateMonitor(),
        goal_conditions={'environment_success': True},
        **kwargs)
    result.reset(TASK)
    return result


def observation(success=False):
    return dict(predicates={'environment_success': success})


CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']


def camera_observation(value=0, **extras):
    return dict({key: np.full((8, 8, 3), value, np.uint8)
                 for key in CAMERAS}, **extras)


def visual_choice(choice='skill_0000', evidence='The block is still held.'):
    return dict(choice=choice, evidence=evidence)


def test_candidates_equal_training_strings_with_provenance(training_root):
    catalog = TrainingSubtaskCatalog(training_root)
    assert not catalog.candidates
    catalog.select_task(TASK)
    assert [c.instruction for c in catalog.available({})] == [LEFT, RIGHT]
    assert catalog.source_for('skill_0000') == dict(
        data_root=str(training_root),
        annotation_field='dense_subtask_names',
        task=TASK,
        occurrences=2,
        example_metadata_file='meta/episodes/chunk-000/file-000.parquet',
        example_episode_index=0,
        example_subtask_index=1)
    # Braces in literal labels must never be treated as template fields.
    assert '{base}' in catalog.resolve('skill_0001').instruction
    assert catalog.resolve('not_a_candidate') is None
    catalog.select_task(OTHER_TASK)
    assert [c.instruction for c in catalog.available({})] == [BOWL]
    with pytest.raises(ValueError, match='No training subtask candidates'):
        catalog.select_task('an unknown task')
    assert not catalog.candidates
    assert catalog.task is None


def test_excess_candidates_raise_instead_of_truncating(training_root):
    catalog = TrainingSubtaskCatalog(training_root, max_candidates=1)
    with pytest.raises(ValueError, match='refusing to truncate'):
        catalog.select_task(TASK)
    assert not catalog.candidates
    catalog.select_task(OTHER_TASK)
    assert len(catalog.candidates) == 1


def test_missing_dataset_and_columns_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match='No episode parquet'):
        TrainingSubtaskCatalog(tmp_path)
    write_records(tmp_path, [{'episode_index': 0}])
    with pytest.raises(ValueError, match='Missing annotations'):
        TrainingSubtaskCatalog(tmp_path)


@pytest.mark.parametrize('updates', [
    {
        'dense_subtask_names': ['', LEFT]
    },
    {
        'dense_subtask_names': None
    },
    {
        'dense_subtask_start_frames': [0, 9]
    },
    {
        'dense_subtask_end_frames': [10, 19]
    },
    {
        'dense_subtask_end_frames': [10]
    },
    {
        'tasks': [TASK, OTHER_TASK]
    },
    {
        'tasks': []
    },
])
def test_invalid_annotations_use_training_validation(tmp_path, updates):
    row = record(0)
    row.update(updates)
    write_records(tmp_path, [row])
    with pytest.raises(ValueError):
        TrainingSubtaskCatalog(tmp_path)


def test_duplicate_episodes_across_metadata_files_fail(training_root):
    write_records(training_root, [record(0)], 'file-002.parquet')
    with pytest.raises(ValueError, match='Duplicate episode_index'):
        TrainingSubtaskCatalog(training_root)


def test_reselection_and_continuation_do_not_claim_completion(training_root):
    state = controller(training_root)
    first = state.step(observation(), 0)
    assert first['instruction'] == LEFT
    assert first['candidate_count'] == 2
    assert first['annotation_source']['task'] == TASK
    assert state.planner.index == 1
    # Predicting without an execution acknowledgment does not spend budget
    # or trigger another API call.
    state.step(observation(), 1)
    assert state.planner.index == 1
    assert state.total_chunks == 0
    state.record_dispatch(state.version)
    second = state.step(observation(), 8)
    assert second['decision']['decision'] == 'continue'
    assert second['subtask_version'] == first['subtask_version']
    state.record_dispatch(state.version)
    third = state.step(observation(), 16)
    assert third['instruction'] == RIGHT
    assert third['subtask_version'] == first['subtask_version'] + 1
    assert state.history[-1]['result'] == 'executed_not_completion_verified'
    assert state.history[-1]['executed_chunks'] == 2
    assert state.history[-1]['instruction'] == LEFT
    # Even all old stack-stage predicates cannot certify episode success.
    obs = observation()
    obs['predicates'].update(two_stacked=True, three_stacked=True)
    assert state.step(obs, 17)['status'] == 'ready'
    assert state.step(observation(True), 18)['status'] == 'succeeded'
    assert state.planner.index == 3


def test_replan_interval_and_episode_budget_count_executed_chunks(
        training_root):
    state = controller(
        training_root, replan_every_chunks=2, max_episode_chunks=3)
    state.step(observation(), 0)
    state.record_dispatch(state.version)
    state.step(observation(), 8)
    assert state.planner.index == 1
    state.record_dispatch(state.version)
    state.step(observation(), 16)
    assert state.planner.index == 2
    state.record_dispatch(state.version)
    event = state.step(observation(), 24)
    assert event['status'] == 'failed'
    assert event['reason'] == 'episode_chunk_budget_exhausted'
    assert event['instruction'] is None


@pytest.mark.parametrize('choice, status', [
    ('skill_9999', 'failed'),
    ('rewritten subtask', 'failed'),
    ('blocked', 'blocked'),
    ('request_finish', 'waiting'),
])
def test_invalid_or_nonexecuting_choice_never_emits_instruction(
        training_root, choice, status):
    state = controller(training_root, choices=[choice])
    event = state.step(observation(), 0)
    assert event['status'] == status
    assert event['instruction'] is None
    assert state.total_chunks == 0


def test_reset_clears_task_history_budget_and_active_label(training_root):
    state = controller(training_root)
    state.step(observation(), 0)
    state.record_dispatch(state.version)
    state.reset(OTHER_TASK)
    assert state.history == []
    assert state.total_chunks == state.version == 0
    assert state.active is None
    assert state.step(observation(), 0)['instruction'] == BOWL
    with pytest.raises(ValueError, match='No training subtask candidates'):
        state.reset('unrecognized')
    assert state.step(observation(), 0)['status'] == 'failed'


def recovery_controller(root, choices, **kwargs):
    return controller(
        root,
        choices=choices,
        max_control_replans=2,
        completion_requirements=dict(
            blocks_stacked='Stack all three blocks.',
            arms_at_origin='Return both arms.'),
        **kwargs)


def completion_observation(stacked=True, home=False, success=False):
    return dict(
        predicates=dict(
            environment_success=success,
            blocks_stacked=stacked,
            arms_at_origin=home))


@pytest.mark.parametrize('control', ['request_finish', 'blocked'])
def test_control_reconsideration_uses_native_feedback_same_snapshot(
        training_root, monkeypatch, control):
    state = recovery_controller(training_root,
                                ['skill_0000', control, 'skill_0000'])
    first = state.step(completion_observation(False), 0)
    state.record_dispatch(state.version)
    before = copy.deepcopy(state.history)
    requests = []
    choose = state.planner.choose

    def capture(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        result = choose(**kwargs)
        # A backend cannot fabricate verification by mutating its inputs.
        kwargs['facts']['environment_success'] = True
        kwargs['feedback'].clear()
        return result

    monkeypatch.setattr(state.planner, 'choose', capture)
    event = state.step(completion_observation(), 8)
    assert event['status'] == 'ready'
    assert event['subtask_version'] == first['subtask_version']
    assert event['instruction'] == LEFT
    assert state.total_chunks == 1 and state.history == before
    assert state.last_observation_id == 8
    assert event['planning_attempts'] == [
        dict(attempt=0, choice=control),
        dict(attempt=1, choice='skill_0000')
    ]
    assert event['episode_control_replans'] == 1
    assert len(requests) == 2
    assert requests[0]['observation'] == requests[1]['observation']
    feedback = requests[1]['feedback']
    assert feedback['same_observation'] is True
    assert feedback['previous_choice'] == control
    assert feedback['completion_verification']['verified'] is False
    assert feedback['completion_verification']['unmet'] == ['arms_at_origin']
    assert feedback['completion_verification']['unknown'] == []
    assert feedback['active_candidate']['instruction'] == LEFT
    assert state.terminal is None


@pytest.mark.parametrize('choices, reason', [
    (['request_finish'] * 3, 'finish_verification_exhausted'),
    (['blocked'] * 3, 'planner_recovery_exhausted'),
    (['request_finish', 'blocked', 'request_finish'
      ], 'finish_verification_exhausted'),
])
def test_repeated_controls_are_bounded_and_do_not_fake_progress(
        training_root, choices, reason):
    state = recovery_controller(training_root, choices)
    event = state.step(completion_observation(), 0)
    assert event['status'] == 'failed' and event['reason'] == reason
    assert state.planner.index == 3
    assert state.episode_control_replans == 2
    assert state.total_chunks == 0 and state.history == []
    assert state.active is None and event['instruction'] is None
    assert state.step(completion_observation(), 0) == event
    assert state.planner.index == 3
    state.reset(TASK)
    assert state.episode_control_replans == 0
    assert state.planning_attempts == []
    assert state.completion_verification == {}


def test_episode_replan_budget_is_shared_across_observations(training_root):
    state = recovery_controller(
        training_root,
        ['blocked', 'skill_0000', 'request_finish', 'skill_0000', 'blocked'],
        max_episode_control_replans=2)
    for step in [0, 8]:
        assert state.step(completion_observation(), step)['status'] == 'ready'
        state.record_dispatch(state.version)
    event = state.step(completion_observation(), 16)
    assert event['reason'] == 'control_replan_budget_exhausted'
    assert state.episode_control_replans == 2
    assert state.total_chunks == 2 and state.planner.index == 5


def test_success_requires_native_terminal_evidence_not_diagnostics(
        training_root):
    state = recovery_controller(training_root, ['request_finish'] * 3)
    # Even all diagnostic predicates true cannot certify native success.
    event = state.step(completion_observation(home=True), 0)
    assert event['status'] == 'failed'
    assert event['completion_verification']['verified'] is False
    state.reset(TASK)
    event = state.step(completion_observation(home=True, success=True), 0)
    assert event['status'] == 'succeeded'
    assert state.planner.index == 0


def test_missing_diagnostic_is_unknown_not_a_failed_goal(training_root):
    state = recovery_controller(training_root,
                                ['request_finish', 'skill_0000'])
    event = state.step(observation(), 0)
    assert event['status'] == 'ready'
    assert set(event['completion_verification']['unknown']) == {
        'blocks_stacked', 'arms_at_origin'
    }
    assert event['completion_verification']['verified'] is False
    state.reset(TASK)
    event = state.step(observation(None), 0)
    assert event['status'] == 'waiting'
    assert event['reason'] == 'goal_state_unknown'
    assert state.planner.index == 0


@pytest.mark.parametrize('options', [
    dict(max_control_replans=-1),
    dict(max_control_replans=True),
    dict(max_episode_control_replans=1.5),
    dict(max_control_replans=3, max_episode_control_replans=2),
    dict(completion_requirements={'arms': False}),
    dict(completion_requirements=[]),
    dict(completion_guidance=None),
])
def test_recovery_config_is_validated(training_root, options):
    with pytest.raises(ValueError):
        controller(training_root, **options)


@pytest.mark.parametrize('allow_request_finish', [True, False])
def test_gpt_receives_only_current_task_candidates_and_active_context(
        training_root, monkeypatch, allow_request_finish):
    state = controller(training_root)
    planner = OpenAISubtaskPlanner(allow_request_finish=allow_request_finish)
    requests = []
    controls = ['request_finish', 'blocked'
                ] if allow_request_finish else ['blocked']

    def generate(system, payload, obs, schema):
        requests.append(copy.deepcopy(payload))
        assert 'past-tense training annotations' in system
        assert schema['properties']['choice']['enum'] == [
            'skill_0000', 'skill_0001'
        ] + controls
        assert payload['controls'] == controls
        if not allow_request_finish:
            assert 'request_finish' not in system
            assert 'Only the environment can end' in system
        return {'choice': 'skill_0000'}

    monkeypatch.setattr(planner.client, 'generate', generate)
    state.planner = planner
    state.step(observation(), 0)
    state.record_dispatch(state.version)
    state.step(observation(), 8)
    assert [c['instruction']
            for c in requests[0]['candidates']] == [LEFT, RIGHT]
    assert requests[1]['feedback']['active_candidate']['instruction'] == LEFT
    assert requests[1]['feedback']['active_executed_chunks'] == 1
    assert requests[1]['recent_history'][0]['instruction'] == LEFT


@pytest.mark.parametrize('choice, status', [
    ('request_finish', 'failed'),
    ('blocked', 'blocked'),
])
def test_disabled_finish_rejects_finish_but_preserves_blocked(
        training_root, monkeypatch, choice, status):
    state = controller(training_root)
    state.planner = OpenAISubtaskPlanner(allow_request_finish=False)
    monkeypatch.setattr(state.planner.client, 'generate',
                        lambda *args: {'choice': choice})
    event = state.step(observation(), 0)
    assert event['status'] == status
    assert event['instruction'] is None
    assert state.total_chunks == 0


@pytest.mark.parametrize('value', ['false', 0, None])
def test_finish_control_requires_a_boolean(value):
    with pytest.raises(ValueError, match='allow_request_finish'):
        OpenAISubtaskPlanner(allow_request_finish=value)


def test_worker_loads_source_once_and_filters_on_episode_reset(
        training_root, monkeypatch):
    from scripts.planner_worker import serve

    monkeypatch.delenv('FLUXVLA_PLANNER_DEBUG_PORT', raising=False)
    cfg = Config.fromfile(CONFIG)
    policy = cfg.eval.policy.to_dict()
    policy['subtask_source']['data_root'] = str(training_root)
    policy['planner'] = dict(
        type='ScriptedSubtaskPlanner',
        choices=[visual_choice('skill_0000'),
                 visual_choice('skill_0001')])
    requests = iter([
        dict(command='configure', policy=policy),
        dict(command='reset', goal=TASK),
        dict(
            command='step',
            observation=camera_observation(),
            observation_id=0,
            executed_previous_chunk=False),
        dict(
            command='step',
            observation=camera_observation(),
            observation_id=8,
            executed_previous_chunk=True),
        dict(command='reset', goal=OTHER_TASK),
        dict(
            command='step',
            observation=camera_observation(),
            observation_id=0,
            executed_previous_chunk=False),
        dict(command='close'),
    ])

    class Connection:
        responses = []

        def recv(self):
            request = next(requests)
            if request['command'] == 'reset':
                # Reset must use the cached metadata, not read frame labels.
                for path in training_root.rglob('*.parquet'):
                    path.unlink()
            return request

        def send(self, response):
            self.responses.append(response)

        def close(self):
            pass

    connection = Connection()
    serve(connection)
    assert all(r['ok'] for r in connection.responses)
    events = [r['result'] for r in connection.responses if r['result']]
    assert [e['instruction'] for e in events] == [LEFT, RIGHT, BOWL]
    assert [e['executed_chunks'] for e in events] == [0, 1, 0]


def agent_controller(root, choices, **options):
    settings = dict(
        catalog=TrainingSubtaskCatalog(root),
        planner=ScriptedSubtaskPlanner(choices),
        completion_mode='agent',
        planner_image_keys=CAMERAS,
        max_control_replans=2)
    settings.update(options)
    state = TrainingSubtaskController(**settings)
    state.reset(TASK)
    return state


@pytest.mark.parametrize('success', [False, True, None, 'invalid'])
@pytest.mark.parametrize('choice', ['skill_0000', 'request_finish'])
def test_agent_ignores_environment_truth_and_uses_only_cameras(
        training_root, monkeypatch, success, choice):
    state = agent_controller(training_root, [visual_choice(choice)])
    requests = []
    choose = state.planner.choose

    def capture(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return choose(**kwargs)

    monkeypatch.setattr(state.planner, 'choose', capture)
    obs = camera_observation(
        predicates=dict(environment_success=success, arms_at_origin=False),
        info=dict(success=True),
        reward=1.0,
        world_state={'stacked': True},
        _initial_images=camera_observation(255))
    event = state.step(obs, 0)
    assert event['status'] == ('completed'
                               if choice == 'request_finish' else 'ready')
    assert len(requests) == 1
    assert set(requests[0]['observation']) == set(CAMERAS)
    assert requests[0]['facts'] == {}
    assert 'completion_verification' not in requests[0]['feedback']
    assert 'completion_verification' not in event
    assert state.total_chunks == 0
    if choice == 'request_finish':
        assert event['completion']['agent_success'] is True
        assert event['completion']['source'] == 'agent_visual'
        assert event['reason'] == 'agent_visual_completion'
        assert event['instruction'] is None
        assert state.step({}, 0) == event
        assert len(requests) == 1
    else:
        assert event['completion']['agent_success'] is None
    assert obs['predicates']['environment_success'] == success


@pytest.mark.parametrize('options', [
    dict(monitor=ObservationPredicateMonitor()),
    dict(goal_conditions={'environment_success': True}),
    dict(goal_conditions={}),
    dict(completion_requirements={'home': 'Arms at origin.'}),
    dict(planner_image_keys=[]),
    dict(completion_mode='other'),
    dict(planner=OpenAISubtaskPlanner(completion_mode='environment')),
])
def test_agent_rejects_environment_monitor_configuration(
        training_root, options):
    with pytest.raises(ValueError):
        agent_controller(training_root, [], **options)


@pytest.mark.parametrize('options', [
    dict(image_keys=[]),
    dict(image_keys=CAMERAS, allow_request_finish=False),
])
def test_agent_backend_requires_images_and_finish_control(options):
    with pytest.raises(ValueError):
        OpenAISubtaskPlanner(completion_mode='agent', **options)


@pytest.mark.parametrize('image', [
    None,
    np.zeros((2, 2, 3), np.float32),
    np.zeros((2, 2), np.uint8),
    np.zeros((0, 0, 3), np.uint8)
])
def test_agent_missing_or_invalid_camera_never_calls_planner(
        training_root, image):
    state = agent_controller(training_root, [visual_choice('request_finish')])
    obs = camera_observation()
    obs['cam_high'] = image
    event = state.step(obs, 0)
    assert event['status'] == 'failed' and event['instruction'] is None
    assert state.planner.index == state.total_chunks == 0


@pytest.mark.parametrize('output', [
    {
        'choice': 'request_finish'
    },
    visual_choice('request_finish', ''),
    visual_choice('request_finish', '   '),
    visual_choice('request_finish', None),
    visual_choice('request_finish', 'x' * 1025),
    visual_choice('skill_9999'),
    dict(visual_choice('request_finish'), environment_success=True),
])
def test_agent_invalid_output_fails_closed(training_root, output):
    state = agent_controller(training_root, [output])
    event = state.step(camera_observation(), 0)
    assert event['status'] == 'failed'
    assert event['instruction'] is None
    assert event['completion']['agent_success'] is None
    assert state.planner.index == 1 and state.total_chunks == 0


@pytest.mark.parametrize('next_choice', ['request_finish', 'skill_0000'])
def test_agent_blocked_reconsiders_same_snapshot_without_env_feedback(
        training_root, monkeypatch, next_choice):
    state = agent_controller(training_root, [
        visual_choice(),
        visual_choice('blocked'),
        visual_choice(next_choice)
    ])
    state.step(camera_observation(), 0)
    state.record_dispatch(state.version)
    history = copy.deepcopy(state.history)
    requests = []
    choose = state.planner.choose

    def capture(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return choose(**kwargs)

    monkeypatch.setattr(state.planner, 'choose', capture)
    event = state.step(camera_observation(30), 8)
    assert event['status'] == ('completed'
                               if next_choice == 'request_finish' else 'ready')
    assert len(requests) == 2 and state.total_chunks == 1
    assert state.history == history
    assert event['episode_control_replans'] == 1
    for key in CAMERAS:
        np.testing.assert_array_equal(requests[0]['observation'][key],
                                      requests[1]['observation'][key])
    assert requests[1]['feedback']['same_observation'] is True
    assert requests[1]['feedback']['result'] == 'cannot_progress_reconsider'
    assert requests[1]['facts'] == {}
    state.reset(TASK)
    assert state.history == [] and state.terminal is None
    assert state.completion_decision == {}
    assert state.episode_control_replans == state.total_chunks == 0


def test_agent_persistent_blocked_is_bounded(training_root):
    state = agent_controller(training_root, [visual_choice('blocked')] * 3)
    event = state.step(camera_observation(), 0)
    assert event['status'] == 'failed'
    assert event['reason'] == 'planner_recovery_exhausted'
    assert state.planner.index == 3 and state.total_chunks == 0
    assert state.episode_control_replans == 2


def test_agent_backend_owns_initial_reference_and_resets_it(
        training_root, monkeypatch):
    planner = OpenAISubtaskPlanner(completion_mode='agent', image_keys=CAMERAS)
    state = agent_controller(training_root, [], planner=planner)
    requests = []

    def generate(system, payload, obs, schema):
        requests.append(copy.deepcopy((payload, obs)))
        assert 'You decide task completion yourself' in system
        assert 'no environment verifier' in system
        assert schema['required'] == ['choice', 'evidence']
        assert schema['additionalProperties'] is False
        # Even a backend that mutates inputs cannot corrupt reference cache.
        obs['_initial_images']['cam_high'][:] = 250
        return visual_choice()

    monkeypatch.setattr(planner.client, 'generate', generate)
    obs = camera_observation(10, _initial_images=camera_observation(255))
    assert state.step(obs, 0)['status'] == 'ready'
    obs['cam_high'][:] = 200
    state.record_dispatch(state.version)
    assert state.step(camera_observation(20), 8)['status'] == 'ready'
    first_payload, first = requests[0]
    second_payload, second = requests[1]
    assert first_payload['recent_history'] == []
    assert second_payload['recent_history'][0]['instruction'] == LEFT
    for key in CAMERAS:
        assert (first[key] == 10).all()
        assert (second[key] == 20).all()
        assert (second['_initial_images'][key] == 10).all()
    assert 'facts' not in first_payload
    assert 'completion_verification' not in second_payload['feedback']
    state.reset(OTHER_TASK)
    assert planner._initial_images is None
    assert state.step(camera_observation(30), 0)['status'] == 'ready'
    payload, images = requests[-1]
    assert payload['goal'] == OTHER_TASK and payload['recent_history'] == []
    assert (images['_initial_images']['cam_high'] == 30).all()

    # A direct caller cannot smuggle facts or measured feedback past the
    # controller boundary. Only the explicit camera/feedback keys survive.
    kwargs = dict(
        goal=TASK,
        observation=obs,
        facts={},
        candidates=[dict(choice='skill_0000')],
        history=[],
        feedback={
            'completion_verification': {
                'verified': True
            },
            'reward': 1.0
        })
    planner.choose(**kwargs)
    assert requests[-1][0]['feedback'] == {}
    assert set(requests[-1][1]) == set(CAMERAS) | {'_initial_images'}
    kwargs['facts'] = {'environment_success': True}
    with pytest.raises(ValueError, match='must not receive predicate'):
        planner.choose(**kwargs)
