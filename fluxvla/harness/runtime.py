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
"""Bounded scheduling around an interchangeable subtask planner."""

import copy

from .memory import EpisodeMemory, positive_integer
from .observer import VisualChangeObserver


class AgentHarness:
    """Remember execution; debounce instruction changes on fresh snapshots.

    The observer is a visual-change heuristic, not semantic progress. Stalls
    force a bounded reconsideration of valid training candidates. Automatic
    stop-on-stall is opt-in because camera-motion thresholds need calibration.
    """

    def __init__(self,
                 image_keys,
                 min_subtask_chunks=2,
                 switch_confirmations=2,
                 max_hold_chunks=8,
                 no_change_patience=4,
                 max_stall_replans=2,
                 stop_on_stall=False,
                 keyframe_interval_chunks=8,
                 observer=None,
                 memory=None):
        for name, value in (
            ('min_subtask_chunks', min_subtask_chunks),
            ('switch_confirmations', switch_confirmations),
            ('max_hold_chunks', max_hold_chunks),
            ('no_change_patience', no_change_patience),
            ('max_stall_replans', max_stall_replans),
            ('keyframe_interval_chunks', keyframe_interval_chunks),
        ):
            setattr(self, name, positive_integer(name, value))
        if min_subtask_chunks > max_hold_chunks:
            raise ValueError('max_hold_chunks must cover min_subtask_chunks')
        if type(stop_on_stall) is not bool:
            raise ValueError('stop_on_stall must be a bool')
        self.stop_on_stall = stop_on_stall
        self.observer = VisualChangeObserver(image_keys, **(observer or {}))
        self.memory = EpisodeMemory(**(memory or {}))
        self.reset('')

    def reset(self, goal):
        self.memory.reset(goal)
        self.observer.reset()
        self.active = None
        self.active_chunks = 0
        self.proposed = None
        self.proposal_count = 0
        self.last_vote_execution = -1
        self.no_change_chunks = self.stall_replans = self.planner_calls = 0
        self.last_observed_execution = self.last_keyframe_chunk = 0
        self.last_selection = None

    @property
    def stalled(self):
        return self.no_change_chunks >= self.no_change_patience

    @property
    def should_stop(self):
        return (self.stop_on_stall and self.stalled
                and self.stall_replans >= self.max_stall_replans)

    def observe(self, observation, observation_id):
        if (type(observation_id) is not int
                or observation_id <= self.memory.last_observation_id):
            raise ValueError('Harness observations must strictly increase')
        executed = self.memory.executed_chunks > self.last_observed_execution
        evidence = self.observer.observe(
            observation, observation_id, executed=executed)
        if evidence['state'] == 'no_observed_change':
            self.no_change_chunks += 1
        elif evidence['state'] == 'changed':
            self.no_change_chunks = self.stall_replans = 0
        self.last_observed_execution = self.memory.executed_chunks
        self.memory.last_observation_id = observation_id
        self.memory.last_observation = evidence
        self.memory.record(
            'observation', observation_id=observation_id, **evidence)
        if (not self.memory.keyframes
                or self.memory.executed_chunks - self.last_keyframe_chunk >=
                self.keyframe_interval_chunks):
            self.capture()

    def capture(self):
        if (self.memory.keyframes
                and self.memory.keyframes[-1]['observation_id']
                == self.memory.last_observation_id):
            return
        self.memory.capture(self.observer.images,
                            self.memory.last_observation_id)
        self.last_keyframe_chunk = self.memory.executed_chunks

    def context(self):
        return dict(
            memory=self.memory.context(),
            active_candidate=copy.deepcopy(self.active),
            active_executed_chunks=self.active_chunks,
            no_change_chunks=self.no_change_chunks,
            stall_replans=self.stall_replans,
            stall_replan_budget=self.max_stall_replans,
            semantic_progress='unknown')

    def choose(self, planner, **request):
        from fluxvla.planning.catalog import validate_choice

        choices = [c['choice'] for c in request['candidates']]
        active = self.active['choice'] if self.active else None
        stall_replan = (
            self.stalled and self.stall_replans < self.max_stall_replans)
        if (active in choices and self.active_chunks < self.min_subtask_chunks
                and not stall_replan):
            return self._select(active, active, 'minimum_execution_hold')
        feedback = copy.deepcopy(request['feedback'])
        # This is constructed here, not accepted from observation['info'].
        feedback['agent_memory'] = self.context()
        request = dict(request, feedback=feedback)
        output = planner.choose(**request)
        choice = validate_choice(output, choices)
        self.planner_calls += 1
        if stall_replan:
            self.stall_replans += 1
        if choice == active:
            self.proposed, self.proposal_count = None, 0
            return self._select(choice, choice, 'continue_candidate')
        if (active not in choices or stall_replan
                or self.active_chunks >= self.max_hold_chunks):
            self.proposed, self.proposal_count = None, 0
            return self._select(choice, choice, 'accepted_replan')
        # Confirm a switch only after another real action chunk executed.
        if self.proposed != choice:
            self.proposed, self.proposal_count = choice, 0
            self.last_vote_execution = -1
        if self.memory.executed_chunks > self.last_vote_execution:
            self.proposal_count += 1
            self.last_vote_execution = self.memory.executed_chunks
        if self.proposal_count >= self.switch_confirmations:
            self.proposed, self.proposal_count = None, 0
            return self._select(choice, choice, 'confirmed_switch')
        return self._select(active, choice, 'awaiting_switch_confirmation')

    def _select(self, choice, proposal, reason):
        self.last_selection = dict(
            choice=choice,
            proposed_choice=proposal,
            reason=reason,
            confirmation_count=self.proposal_count)
        self.memory.record('candidate_selection', **self.last_selection)
        return dict(choice=choice)

    def issue(self, candidate, version, horizon, observation_id):
        if self.active is None or self.active['choice'] != candidate['choice']:
            self.active = dict(
                choice=candidate['choice'],
                instruction=candidate['instruction'])
            self.active_chunks = 0
            self.capture()
        self.memory.issue(candidate, version, horizon, observation_id)

    def acknowledge(self, executed_steps):
        self.memory.acknowledge(executed_steps)
        if executed_steps:
            self.active_chunks += 1

    def snapshot(self):
        return dict(
            **self.context(),
            planner_calls=self.planner_calls,
            selection=copy.deepcopy(self.last_selection),
            automatic_stall_stop=self.stop_on_stall,
            trace_path=(str(self.memory.trace_path)
                        if self.memory.trace_path else None))
