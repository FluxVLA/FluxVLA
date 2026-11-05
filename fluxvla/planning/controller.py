# Copyright 2026 Limx Dynamics
"""Stateful subtask lifecycle, independent of datasets and robot control."""

import copy

from .catalog import has_unknown, matches, validate_choice, validate_conditions


class SubtaskController:
    """Commit one skill until observed completion; never trust model 'done'.

    Planning happens at skill boundaries. An active skill continues locally,
    avoiding instruction jitter and a network call for every action chunk.
    Completion requires fresh observations. The first positive observation
    holds execution while subsequent observations confirm the condition.
    """

    def __init__(self,
                 catalog,
                 planner,
                 monitor,
                 goal_conditions,
                 stable_observations=2,
                 max_rejections=3):
        if type(stable_observations) is not int or stable_observations < 1:
            raise ValueError('stable_observations must be a positive integer')
        if type(max_rejections) is not int or max_rejections < 1:
            raise ValueError('max_rejections must be a positive integer')
        self.catalog = catalog
        self.planner = planner
        self.monitor = monitor
        self.goal_conditions = validate_conditions(goal_conditions,
                                                   'goal_conditions')
        self.keys = catalog.predicate_keys | set(self.goal_conditions)
        descriptions = getattr(monitor, 'descriptions', None)
        if descriptions is not None and self.keys - descriptions.keys():
            raise ValueError(
                'Visual monitor must describe every required predicate')
        self.stable_observations = stable_observations
        self.max_rejections = max_rejections
        self.reset('')

    def reset(self, goal):
        self.goal = goal
        self.active = None
        self.version = 0
        self.chunks = 0
        self.goal_streak = 0
        self.skill_streak = 0
        self.rejections = 0
        self.last_observation_id = -1
        self.history = []
        self.feedback = {}
        self.terminal = None
        self.planner.reset()

    def _event(self, status, decision=None, reason=None):
        event = dict(
            status=status,
            observation_id=self.last_observation_id,
            subtask_version=self.version,
            decision=decision,
            instruction=self.active.instruction if status == 'ready' else None,
            reason=reason)
        if status in {
                'succeeded', 'blocked', 'failed', 'timeout', 'cancelled'
        }:
            self.terminal = event
        return copy.deepcopy(event)

    def finish(self, status, reason):
        if status not in {'failed', 'timeout', 'cancelled'}:
            raise ValueError('Only verified observations can declare success')
        return self._event(status, reason=reason)

    def _reject(self, error):
        self.goal_streak = self.skill_streak = 0
        self.rejections += 1
        self.feedback = dict(error=type(error).__name__, detail=str(error))
        if self.rejections >= self.max_rejections:
            status = 'failed'
        else:
            status = 'rejected'
        return self._event(status, reason=str(error))

    def step(self, observation, observation_id):
        if self.terminal:
            return copy.deepcopy(self.terminal)
        if (type(observation_id) is not int
                or observation_id <= self.last_observation_id):
            return self._reject(
                ValueError('Observation id must strictly increase'))
        self.last_observation_id = observation_id
        try:
            facts = self.monitor.observe(observation, self.keys)
            if (not isinstance(facts, dict) or set(facts) != self.keys or any(
                    v is not None and type(v) is not bool
                    for v in facts.values())):
                raise ValueError(
                    'Monitor must return bool/unknown for every predicate')
            event = self._step_with_facts(observation, facts)
        except Exception as exc:
            # Fail closed on provider errors too; no output is sent to a VLA.
            return self._reject(exc)
        self.rejections = 0
        return event

    def _step_with_facts(self, observation, facts):
        if matches(self.goal_conditions, facts):
            self.goal_streak += 1
            if self.goal_streak >= self.stable_observations:
                return self._event('succeeded', reason='goal_verified')
            return self._event('waiting', reason='confirming_goal')
        self.goal_streak = 0

        if self.active:
            if matches(self.active.completion, facts):
                self.skill_streak += 1
                if self.skill_streak < self.stable_observations:
                    return self._event('waiting', reason='confirming_subtask')
                self.feedback = dict(
                    result='subtask_succeeded', **self.active.decision())
                self.history.append(copy.deepcopy(self.feedback))
                self.history = self.history[-64:]
                self.active = None
                self.chunks = self.skill_streak = 0
            else:
                self.skill_streak = 0
                if self.chunks >= self.active.max_chunks:
                    return self._event(
                        'failed', reason='subtask_chunk_budget_exhausted')
                if has_unknown(self.active.completion, facts):
                    return self._event(
                        'waiting', reason='subtask_state_unknown')
                return self._event('ready', self.active.decision('continue'))

        available = self.catalog.available(facts)
        choices = [c.choice for c in available] + ['request_finish', 'blocked']
        output = self.planner.choose(
            goal=self.goal,
            observation=observation,
            facts=copy.deepcopy(facts),
            candidates=[c.describe() for c in available],
            history=copy.deepcopy(self.history),
            feedback=copy.deepcopy(self.feedback))
        choice = validate_choice(output, choices)
        if choice == 'blocked':
            return self._event('blocked', dict(decision='blocked'),
                               'planner_cannot_progress')
        if choice == 'request_finish':
            self.feedback = dict(result='goal_not_verified')
            return self._event('waiting', dict(decision='request_finish'),
                               'goal_not_verified')
        self.active = self.catalog.resolve(choice)
        # Defense in depth; do not trust a backend to enforce the enum.
        if self.active is None or not matches(self.active.preconditions,
                                              facts):
            raise ValueError('Unknown skill or unmet preconditions')
        self.version += 1
        self.chunks = 0
        return self._event('ready', self.active.decision())

    def record_dispatch(self, version):
        """Count an executed action chunk or a preview chunk."""
        if self.terminal or self.active is None or version != self.version:
            raise ValueError('Cannot dispatch a stale or inactive subtask')
        self.chunks += 1
