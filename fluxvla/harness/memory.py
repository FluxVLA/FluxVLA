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
"""Bounded, evidence-linked memory; issued actions are not executed actions."""

import copy
import hashlib
import json
import os
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class EpisodeMemory:
    """One episode, bounded RAM, optional append-only diagnostic artifacts.

    Disk traces are evidence for inspection, NOT executable checkpoints.
    Neither reset nor a process restart replays saved robot commands.
    Outcomes always remain unverified: sensor motion cannot prove completion.
    """

    def __init__(self,
                 recent_events=8,
                 summary_subtasks=32,
                 recent_keyframes=8,
                 saved_keyframes=64,
                 trace_dir=None,
                 trace_fsync=True):
        self.recent_limit = positive_integer('recent_events', recent_events)
        self.summary_limit = positive_integer('summary_subtasks',
                                              summary_subtasks)
        self.keyframe_limit = positive_integer('recent_keyframes',
                                               recent_keyframes)
        self.saved_limit = positive_integer('saved_keyframes', saved_keyframes)
        if type(trace_fsync) is not bool:
            raise ValueError('trace_fsync must be a bool')
        self.trace_dir = (
            Path(trace_dir).expanduser() if trace_dir is not None else None)
        self.trace_fsync = trace_fsync
        self.runtime_id = uuid.uuid4().hex
        self.reset('')

    def reset(self, goal):
        if getattr(self, 'pending', None) is not None:
            self.record('episode_interrupted', execution_status='unknown')
        self.goal = goal
        self.episode_id = uuid.uuid4().hex
        self.recent = deque(maxlen=self.recent_limit)
        self.outcomes = OrderedDict()
        self.keyframes = deque(maxlen=self.keyframe_limit)
        self.pending = None
        self.issued_chunks = self.executed_chunks = self.executed_steps = 0
        self.evicted_subtasks = self.saved_keyframes = 0
        self.sequence = 0
        self.trace_path = None
        self.last_observation_id = -1
        self.last_observation = None

    def bind_episode(self, episode_id):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError('episode_id must be a nonempty string')
        if self.sequence or self.pending is not None:
            raise RuntimeError('Bind episode context before its first step')
        self.episode_id = episode_id

    def snapshot(self):
        return copy.deepcopy(
            dict(
                schema_version=1,
                episode_id=self.episode_id,
                goal=self.goal,
                issued_chunks=self.issued_chunks,
                executed_chunks=self.executed_chunks,
                executed_steps=self.executed_steps,
                pending=self.pending,
                completion='unknown',
                last_observation_id=self.last_observation_id,
                observation_evidence=self.last_observation,
                recent_events=list(self.recent),
                subtask_summaries=list(self.outcomes.values()),
                evicted_subtasks=self.evicted_subtasks,
                keyframes=list(self.keyframes),
                trace_path=str(self.trace_path) if self.trace_path else None))

    def context(self):
        """Compact trusted data, never an arbitrary observation mapping."""
        result = self.snapshot()
        result.pop('trace_path')
        result['subtask_summaries'] = result['subtask_summaries'][-8:]
        result['keyframes'] = [
            dict(observation_id=frame['observation_id'])
            for frame in result['keyframes']
        ]
        # Filesystem paths are not model-accessible image evidence.
        return result

    def record(self, event, **data):
        self.sequence += 1
        item = dict(sequence=self.sequence, event=event, **copy.deepcopy(data))
        self.recent.append(item)
        if self.trace_dir is not None:
            self._persist(item)
        return item

    def issue(self, candidate, version, horizon, observation_id):
        if self.pending is not None:
            raise RuntimeError('Acknowledge pending actions before issuing')
        positive_integer('action horizon', horizon)
        self.pending = dict(
            choice=candidate['choice'],
            instruction=candidate['instruction'],
            version=version,
            horizon=horizon,
            observation_id=observation_id)
        self.issued_chunks += 1
        self.record('actions_issued', **self.pending)

    def acknowledge(self, executed_steps):
        if (self.pending is None or type(executed_steps) is not int
                or not 0 <= executed_steps <= self.pending['horizon']):
            raise ValueError('Invalid harness execution acknowledgment')
        pending = self.pending
        self.pending = None
        if executed_steps:
            self.executed_chunks += 1
            self.executed_steps += executed_steps
            choice = pending['choice']
            summary = self.outcomes.pop(choice, None) or dict(
                choice=choice,
                instruction=pending['instruction'],
                executed_chunks=0,
                executed_steps=0,
                completion='unverified')
            summary['executed_chunks'] += 1
            summary['executed_steps'] += executed_steps
            summary['last_observation_id'] = pending['observation_id']
            self.outcomes[choice] = summary
            if len(self.outcomes) > self.summary_limit:
                self.outcomes.popitem(last=False)
                self.evicted_subtasks += 1
        self.record(
            'actions_acknowledged' if executed_steps else 'actions_cancelled',
            **pending,
            executed_steps=executed_steps,
            completion='unverified')

    def capture(self, images, observation_id):
        frame = dict(observation_id=observation_id, images=[])
        if (self.trace_path is not None
                and self.saved_keyframes < self.saved_limit):
            for index, image in enumerate(images.values()):
                filename = (f'frame-{observation_id:08d}-'
                            f'camera-{index:02d}.png')
                Image.fromarray(image).save(self.trace_path.parent / filename)
                frame['images'].append(filename)
            self.saved_keyframes += 1
        self.keyframes.append(frame)
        self.record('keyframe', **frame)

    def _persist(self, item):
        if self.trace_path is None:
            key = hashlib.sha256(self.episode_id.encode()).hexdigest()
            directory = self.trace_dir / self.runtime_id / key
            directory.mkdir(parents=True, exist_ok=True)
            self.trace_path = directory / 'events.jsonl'
            # Never overwrite another episode's journal.
            with self.trace_path.open('x', encoding='utf-8') as stream:
                json.dump(
                    dict(
                        event='episode_start',
                        episode_id=self.episode_id,
                        goal=self.goal),
                    stream,
                    ensure_ascii=False)
                stream.write('\n')
        with self.trace_path.open('a', encoding='utf-8') as stream:
            json.dump(
                dict(
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    episode_id=self.episode_id,
                    **item),
                stream,
                ensure_ascii=False,
                allow_nan=False)
            stream.write('\n')
            stream.flush()
            if self.trace_fsync:
                os.fsync(stream.fileno())
        if item['event'] in {
                'actions_issued', 'actions_acknowledged', 'actions_cancelled',
                'episode_interrupted'
        }:
            path = self.trace_path.with_name('memory.json')
            temporary = path.with_suffix('.tmp')
            try:
                with temporary.open('w', encoding='utf-8') as stream:
                    json.dump(
                        self.snapshot(),
                        stream,
                        ensure_ascii=False,
                        allow_nan=False)
                    stream.flush()
                    if self.trace_fsync:
                        os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
