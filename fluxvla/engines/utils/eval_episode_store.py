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
"""Durable LIBERO episode outcomes, including recovery of older rank logs."""

import json
import math
import os
import re
import tempfile
from pathlib import Path


class EvalEpisodeStore:
    """Store completed trials only; an API error is not a failed trial.

    Each trial is written atomically, independently of the final distributed
    reduction. Legacy logs can recover success/failure but not rollout timing.
    """

    def __init__(self, run_dir, metadata):
        self.run_dir = Path(run_dir)
        # Normalize ConfigDict / tuple values to the persisted representation.
        self.metadata = json.loads(json.dumps(metadata, default=str))
        self.episode_dir = self.run_dir / 'episode_results'

    @staticmethod
    def _atomic_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f'.{path.name}.', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _validate_record(self, record):
        task_id, trial_id = record['task_id'], record['trial_id']
        if (type(task_id) is not int or type(trial_id) is not int
                or task_id not in self.metadata['task_ids']
                or not 0 <= trial_id < self.metadata['num_trials_per_task']):
            raise ValueError(f'Invalid resumed episode: {(task_id, trial_id)}')
        if (record.get('task_suite') != self.metadata['task_suite_name']
                or record.get('status') != 'completed'
                or type(record.get('success')) is not bool):
            raise ValueError(f'Invalid episode outcome: {(task_id, trial_id)}')
        for key in ('duration_seconds', 'start_time'):
            value = record.get(key)
            if value is not None and (not math.isfinite(float(value))
                                      or float(value) < 0):
                raise ValueError(f'Invalid episode {key}: {value}')
        return task_id, trial_id

    def save(self, record):
        task_id, trial_id = self._validate_record(record)
        path = self.episode_dir / f'task{task_id}_trial{trial_id}.json'
        self._atomic_json(path, record)

    def _legacy_records(self):
        """Only an explicit final Success line proves that a trial finished."""
        records = {}
        for path in sorted(self.run_dir.glob('rank*.txt')):
            current = None
            for line in path.read_text(encoding='utf-8').splitlines():
                match = re.fullmatch(r'Evaluating Task (\d+), Trial (\d+)',
                                     line)
                if match:
                    current = tuple(map(int, match.groups()))
                elif current is not None and line in ('Success: True',
                                                      'Success: False'):
                    record = dict(
                        task_suite=self.metadata['task_suite_name'],
                        task_id=current[0],
                        trial_id=current[1],
                        status='completed',
                        success=line == 'Success: True',
                        duration_seconds=None,
                        start_time=None,
                        recovered_from=path.name)
                    self._validate_record(record)
                    if (current in records and
                            records[current]['success'] != record['success']):
                        raise ValueError(
                            f'Conflicting legacy outcome: {current}')
                    records[current] = record
                    current = None
        return records

    def prepare(self, resume=False):
        """Called by rank zero before other ranks begin writing results."""
        metadata_path = self.run_dir / 'eval_metadata.json'
        if metadata_path.exists():
            stored = json.loads(metadata_path.read_text(encoding='utf-8'))
            if stored['evaluation'] != self.metadata:
                changed = sorted(
                    k for k in set(stored['evaluation']) | set(self.metadata)
                    if stored['evaluation'].get(k) != self.metadata.get(k))
                raise ValueError('Resume evaluation settings differ: ' +
                                 ', '.join(changed))
        elif resume:
            expected_prefix = (f'EVAL-{self.metadata["task_suite_name"]}-'
                               f'{self.metadata["model_family"]}-')
            if (not self.run_dir.name.startswith(expected_prefix)
                    or not list(self.run_dir.glob('rank*.txt'))):
                raise ValueError('Resume directory has no matching LIBERO run')

        records = self._legacy_records() if resume else {}
        for path in sorted(self.episode_dir.glob('*.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            pair = self._validate_record(record)
            if (pair in records
                    and records[pair]['success'] != record['success']):
                raise ValueError(f'Conflicting stored outcome: {pair}')
            records[pair] = record

        if not resume and records:
            raise ValueError('Run already contains episodes; use resume_from')
        # Preserve every recovered legacy outcome before launching new work.
        for record in records.values():
            self.save(record)
        if not metadata_path.exists():
            self._atomic_json(
                metadata_path,
                dict(
                    schema_version=1,
                    evaluation=self.metadata,
                    legacy_metadata_unverified=bool(resume)))
        return records
