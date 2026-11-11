# Copyright 2026 Limx Dynamics
"""Literal training-label candidates and observation-driven reselection.

Episode annotations define a vocabulary, NOT an online trajectory oracle.
Only the goal selects a task's vocabulary; rollout frame/episode ids never
index the training segments. Without per-label completion predicates we
reselect after executed chunks, and never report a subtask as verified done.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .backends import _planner_camera_observation, _validate_image_keys
from .catalog import has_unknown, matches, validate_choice
from .controller import SubtaskController


@dataclass(frozen=True)
class TrainingSubtaskCandidate:
    choice: str
    instruction: str

    def decision(self, kind='execute'):
        return dict(decision=kind, skill=self.choice, arguments={})

    def describe(self):
        return dict(
            choice=self.choice,
            **self.decision(),
            instruction=self.instruction)


class TrainingSubtaskCatalog:
    """Deduplicate validated LeRobot v3 labels per whole-task description.

    The training parser supplies exactly the same whitespace normalization
    and segment validation. No rewriting, template expansion, semantic
    merging, candidate truncation, or fallback to another task is allowed.
    Metadata is loaded once; selecting/resetting a task requires no I/O.
    """

    predicate_keys = frozenset()

    def __init__(self, data_root, subtask_prefix='dense', max_candidates=1024):
        import pyarrow.parquet as pq

        from fluxvla.datasets.utils.subtask_annotations import \
            build_subtask_index

        if not isinstance(subtask_prefix, str) or not subtask_prefix:
            raise ValueError('subtask_prefix must be a nonempty string')
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError('max_candidates must be a positive integer')
        self.data_root = Path(data_root).expanduser().resolve()
        self.subtask_prefix = subtask_prefix
        self.max_candidates = max_candidates
        self.tasks = {}
        self.candidates = {}
        self.task = None
        paths = sorted((self.data_root / 'meta/episodes').rglob('*.parquet'))
        if not paths:
            raise FileNotFoundError(
                f'No episode parquet metadata under {self.data_root}')
        columns = ['episode_index', 'length', 'tasks'] + [
            f'{subtask_prefix}_subtask_{suffix}'
            for suffix in ('names', 'start_frames', 'end_frames')
        ]
        seen_episodes = set()
        for path in paths:
            parquet = pq.ParquetFile(path)
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f'Missing annotations in {path}: '
                                 f'{sorted(missing)}')
            records = parquet.read(columns=columns).to_pylist()
            if not records:
                continue
            segments = build_subtask_index(records, subtask_prefix)
            for record in records:
                episode = record['episode_index']
                if episode in seen_episodes:
                    raise ValueError(f'Duplicate episode_index: {episode}')
                seen_episodes.add(episode)
                tasks = record['tasks']
                if (not isinstance(tasks, list) or len(tasks) != 1
                        or not isinstance(tasks[0], str)
                        or not tasks[0].strip()):
                    raise ValueError(
                        f'Episode {episode} needs exactly one task '
                        'description for unambiguous candidate filtering')
                labels = self.tasks.setdefault(tasks[0].strip(), {})
                for index, segment in enumerate(segments[episode]):
                    source = labels.setdefault(
                        segment.name,
                        dict(
                            occurrences=0,
                            example_metadata_file=str(
                                path.relative_to(self.data_root)),
                            example_episode_index=episode,
                            example_subtask_index=index))
                    source['occurrences'] += 1
        if not self.tasks:
            raise ValueError('No training subtask annotations found')

    def select_task(self, goal):
        # Clear first, so an invalid reset cannot reuse the previous task.
        self.task = None
        self.candidates = {}
        if not isinstance(goal, str) or goal.strip() not in self.tasks:
            raise ValueError(
                'No training subtask candidates for this task description')
        task = goal.strip()
        labels = sorted(self.tasks[task])
        if len(labels) > self.max_candidates:
            raise ValueError(
                f'Task has {len(labels)} training candidates, exceeding '
                f'max_candidates={self.max_candidates}; refusing to truncate')
        self.task = task
        self.candidates = {
            f'skill_{index:04d}':
            TrainingSubtaskCandidate(f'skill_{index:04d}', label)
            for index, label in enumerate(labels)
        }

    def available(self, facts):
        # Feasibility depends on current camera evidence, not training order.
        return list(self.candidates.values())

    def resolve(self, choice):
        return self.candidates.get(choice)

    def source_for(self, choice):
        candidate = self.candidates[choice]
        return dict(
            data_root=str(self.data_root),
            annotation_field=f'{self.subtask_prefix}_subtask_names',
            task=self.task,
            **copy.deepcopy(self.tasks[self.task][candidate.instruction]))


class TrainingSubtaskController(SubtaskController):
    """Select literal labels at a bounded, executed-chunk cadence.

    Unlike SubtaskController, switching labels is NOT certified completion.
    The planner sees the active instruction and executed history, and may
    continue it. Agent mode declares visual completion without any environment
    predicates; it is kept distinct from externally verified success.
    """

    def __init__(self,
                 catalog,
                 planner,
                 monitor=None,
                 goal_conditions=None,
                 replan_every_chunks=1,
                 max_episode_chunks=70,
                 stable_observations=1,
                 max_rejections=1,
                 max_control_replans=0,
                 max_episode_control_replans=12,
                 completion_requirements=None,
                 completion_guidance='',
                 completion_mode='environment',
                 planner_image_keys=()):
        if completion_mode not in {'environment', 'agent'}:
            raise ValueError('completion_mode must be environment or agent')
        self.completion_mode = completion_mode
        self.planner_image_keys = _validate_image_keys(planner_image_keys)
        if completion_mode == 'agent':
            if (monitor is not None or goal_conditions is not None
                    or completion_requirements):
                raise ValueError('Agent mode forbids environment monitors, '
                                 'goal predicates and measured requirements')
            if not self.planner_image_keys:
                raise ValueError('Agent mode requires planner_image_keys')
        elif not goal_conditions:
            raise ValueError('Environment mode requires goal_conditions')
        planner_mode = getattr(planner, 'completion_mode', completion_mode)
        if planner_mode != completion_mode:
            raise ValueError('Controller and planner completion_mode differ')
        for name, value in [('replan_every_chunks', replan_every_chunks),
                            ('max_episode_chunks', max_episode_chunks)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.replan_every_chunks = replan_every_chunks
        self.max_episode_chunks = max_episode_chunks
        for name, value in [('max_control_replans', max_control_replans),
                            ('max_episode_control_replans',
                             max_episode_control_replans)]:
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if max_control_replans > max_episode_control_replans:
            raise ValueError('Episode replan budget must cover one request')
        if completion_requirements is None:
            completion_requirements = {}
        if (not isinstance(completion_requirements, Mapping)
                or any(not isinstance(k, str) or not k.strip()
                       or not isinstance(v, str) or not v.strip()
                       for k, v in completion_requirements.items())):
            raise ValueError('completion_requirements must map names to text')
        if not isinstance(completion_guidance, str):
            raise ValueError('completion_guidance must be a string')
        self.max_control_replans = max_control_replans
        self.max_episode_control_replans = max_episode_control_replans
        self.completion_requirements = dict(completion_requirements)
        self.completion_guidance = completion_guidance
        super().__init__(catalog, planner, monitor, goal_conditions,
                         stable_observations, max_rejections)
        self.keys |= set(self.completion_requirements)
        descriptions = getattr(monitor, 'descriptions', None)
        if descriptions is not None and self.keys - descriptions.keys():
            raise ValueError('Monitor must describe completion diagnostics')

    def reset(self, goal):
        super().reset(goal)
        self.total_chunks = 0
        self.chunks_since_plan = 0
        self.episode_control_replans = 0
        self.planning_attempts = []
        self.completion_verification = {}
        self.completion_decision = {}
        self.catalog.task = None
        self.catalog.candidates = {}
        # The base constructor initializes with an empty goal; the client
        # supplies the actual environment instruction on the first reset.
        if goal:
            self.catalog.select_task(goal)

    def _event(self, status, decision=None, reason=None):
        event = super()._event(status, decision, reason)
        event.update(
            candidate_count=len(self.catalog.candidates),
            candidate_task=self.catalog.task,
            executed_chunks=self.total_chunks)
        if (self.max_control_replans or self.completion_requirements
                or self.completion_mode == 'agent'):
            event.update(
                planning_attempts=copy.deepcopy(self.planning_attempts),
                episode_control_replans=self.episode_control_replans)
            if self.completion_mode == 'agent':
                event['completion'] = copy.deepcopy(self.completion_decision)
            else:
                event['completion_verification'] = copy.deepcopy(
                    self.completion_verification)
        if status == 'ready':
            event['candidate_id'] = self.active.choice
            event['annotation_source'] = self.catalog.source_for(
                self.active.choice)
        if self.terminal is not None:
            self.terminal = copy.deepcopy(event)
        return event

    def _step_with_facts(self, observation, facts):
        self.planning_attempts = []
        agent = self.completion_mode == 'agent'
        if agent:
            observation = _planner_camera_observation(self.planner_image_keys,
                                                      observation)
            facts = {}
            self.completion_decision = dict(
                source='agent_visual', agent_success=None)
        # Diagnostics explain missing requirements; they never replace the
        # native terminal predicate. Replans below reuse this SAME snapshot.
        self.completion_verification = {} if agent else dict(
            verified=matches(self.goal_conditions, facts),
            goal_conditions=copy.deepcopy(self.goal_conditions),
            requirements={
                key: dict(description=text, satisfied=facts[key])
                for key, text in self.completion_requirements.items()
            },
            unmet=[
                k for k in self.completion_requirements if facts[k] is False
            ],
            unknown=[
                k for k in self.completion_requirements if facts[k] is None
            ])
        if not self.catalog.candidates:
            raise ValueError(
                'Reset with a known training task before planning')
        if not agent and matches(self.goal_conditions, facts):
            self.goal_streak += 1
            if self.goal_streak >= self.stable_observations:
                return self._event('succeeded', reason='goal_verified')
            return self._event('waiting', reason='confirming_goal')
        self.goal_streak = 0
        if not agent and has_unknown(self.goal_conditions, facts):
            return self._event('waiting', reason='goal_state_unknown')
        if self.total_chunks >= self.max_episode_chunks:
            return self._event(
                'failed', reason='episode_chunk_budget_exhausted')
        if self.active and self.chunks_since_plan < self.replan_every_chunks:
            return self._event('ready', self.active.decision('continue'))

        available = self.catalog.available(facts)
        choices = [c.choice for c in available] + ['request_finish', 'blocked']
        feedback = dict(
            self.feedback,
            selection_mode='literal_training_annotations',
            active_candidate=(self.active.describe() if self.active else None),
            active_executed_chunks=self.chunks,
            subtask_completion_verified=False)
        if agent:
            feedback['completion_guidance'] = self.completion_guidance
        elif self.max_control_replans or self.completion_requirements:
            feedback.update(
                completion_verification=copy.deepcopy(
                    self.completion_verification),
                completion_guidance=self.completion_guidance)
        for attempt in range(self.max_control_replans + 1):
            output = self.planner.choose(
                goal=self.goal,
                observation=copy.deepcopy(observation),
                facts=copy.deepcopy(facts),
                candidates=[c.describe() for c in available],
                history=copy.deepcopy(self.history),
                feedback=copy.deepcopy(feedback))
            choice = validate_choice(output, choices, with_evidence=agent)
            self.planning_attempts.append(dict(attempt=attempt, **output))
            if agent and choice == 'request_finish':
                self.completion_decision.update(
                    agent_success=True, evidence=output['evidence'])
                print(
                    '[FluxVLA][policy] agent visually judged task complete: '
                    f'{output["evidence"]}',
                    flush=True)
                return self._event('completed', dict(decision=choice),
                                   'agent_visual_completion')
            if choice not in {'blocked', 'request_finish'}:
                break
            if not self.max_control_replans:
                break  # Legacy callers may explicitly handle waiting.
            result = ('goal_not_verified' if choice == 'request_finish' else
                      'cannot_progress_reconsider')
            if agent:
                print(
                    '[FluxVLA][policy] agent cannot progress; reconsidering '
                    f'camera evidence: {output["evidence"]}',
                    flush=True)
            else:
                print(
                    f'[FluxVLA][policy] {choice}: {result}; '
                    f'unmet={self.completion_verification["unmet"]}, '
                    f'unknown={self.completion_verification["unknown"]}',
                    flush=True)
            if attempt == self.max_control_replans:
                reason = ('finish_verification_exhausted'
                          if choice == 'request_finish' else
                          'planner_recovery_exhausted')
                return self._event('failed', dict(decision=choice), reason)
            if (self.episode_control_replans >=
                    self.max_episode_control_replans):
                return self._event('failed', dict(decision=choice),
                                   'control_replan_budget_exhausted')
            self.episode_control_replans += 1
            feedback.update(
                result=result,
                previous_choice=choice,
                same_observation=True,
                control_replans_remaining=min(
                    self.max_control_replans - attempt - 1,
                    self.max_episode_control_replans -
                    self.episode_control_replans),
                planning_attempts=copy.deepcopy(self.planning_attempts))
        self.feedback = {}
        if choice == 'blocked':
            return self._event('blocked', dict(decision='blocked'),
                               'planner_cannot_progress')
        if choice == 'request_finish':
            self.feedback = dict(result='goal_not_verified')
            return self._event('waiting', dict(decision='request_finish'),
                               'goal_not_verified')
        candidate = self.catalog.resolve(choice)
        if candidate is None:
            raise ValueError('Unknown training candidate')
        continuing = self.active is not None and self.active.choice == choice
        if not continuing:
            self.version += 1
            self.chunks = 0
        self.active = candidate
        self.chunks_since_plan = 0
        return self._event(
            'ready',
            self.active.decision('continue' if continuing else 'execute'))

    def record_dispatch(self, version):
        super().record_dispatch(version)
        self.total_chunks += 1
        self.chunks_since_plan += 1
        entry = dict(
            result='executed_not_completion_verified',
            subtask_version=self.version,
            executed_chunks=self.chunks,
            **self.active.describe())
        if (self.history
                and self.history[-1]['subtask_version'] == self.version):
            self.history[-1] = entry
        else:
            self.history.append(entry)
        self.history = self.history[-64:]
