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
"""Durable ROS evaluation events and native FluxVLA evaluation artifacts.

``FluxVLAROSEvaluationReporter`` is the small adapter boundary used by ROS 1
and ROS 2 servers.  A server passes versioned ``run_start``,
``episode_start``, ``episode_end`` and ``run_end`` events to
``process_event``.  The reporter validates ordering and idempotency, journals
accepted events, maintains native live progress, and writes the native LIBERO
or RoboCasa result schema selected by the evaluation config.

The class deliberately has no ROS dependency.  Server adapters may construct
it before ROS initialization and later replace the default Overwatch logger
with :meth:`set_logger`.
"""

from __future__ import annotations
import copy
import csv
import json
import math
import os
import threading
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.feishu_reporter import \
    maybe_report_summary_to_feishu

overwatch = initialize_overwatch(__name__)

EVENT_VERSION = 1
RUN_SCHEMA_VERSION = '1.0'
EVENT_TYPES = frozenset(
    {'run_start', 'episode_start', 'episode_end', 'run_end'})
LIBERO_SUITE_TASK_COUNTS = {
    'libero_spatial': 10,
    'libero_object': 10,
    'libero_goal': 10,
    'libero_10': 10,
    'libero_90': 90,
}
REPORT_KINDS = frozenset({'libero', 'robocasa', 'robodojo'})
ROBOCASA_GROUP_ORDER = ('Cabinet', 'Drawer', 'Microwave', 'Generalization')


class EvaluationEventError(ValueError):
    """A rejected ROS evaluation event."""


@dataclass(frozen=True)
class _TaskManifestEntry:
    task_id: str
    task_index: int
    description: str
    metadata: dict


@dataclass
class _RunState:
    session_id: str
    run_id: str
    run_dir: Path
    start: dict
    tasks_by_id: dict[str, _TaskManifestEntry]
    tasks_by_index: dict[int, _TaskManifestEntry]
    next_sequence: int = 2
    active_episodes: dict[tuple[int, int], dict] = field(default_factory=dict)
    episodes: list[dict] = field(default_factory=list)
    completed_keys: set[tuple[int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class _CachedRequest:
    fingerprint: str
    response: dict


def _cfg_get(config, key: str, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _runner_eval_config(config):
    runner = _cfg_get(config, 'runner')
    if runner is None:
        return config
    if isinstance(config, Mapping) and isinstance(runner, Mapping):
        # Some configs wrap runner-native fields in ``eval.runner`` while the
        # ROS server adds authoritative task overrides at ``eval`` level.
        # Preserve both, with the outer values taking precedence.
        merged = dict(runner)
        merged.update(
            {key: value
             for key, value in config.items() if key != 'runner'})
        return merged
    return runner


def _resolve_report_kind(config, explicit=None) -> str:
    value = explicit
    if value is None:
        value = _cfg_get(config, 'report_kind')
    if value is None:
        value = _cfg_get(config, 'benchmark')
    if value is None:
        runner_type = str(_cfg_get(config, 'type', '')).lower()
        value = 'robocasa' if 'robocasa' in runner_type else 'libero'
    value = _require_string(value, 'report_kind').lower()
    if value not in REPORT_KINDS:
        raise EvaluationEventError(
            f'report_kind must be one of {sorted(REPORT_KINDS)}, got '
            f'{value!r}')
    return value


def _require_mapping(value, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise EvaluationEventError(f'{name} must be a mapping')
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        message = f'{name} must be JSON serializable: {exc}'
        raise EvaluationEventError(message) from exc


def _require_string(value, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise EvaluationEventError(f'{name} must be a string')
    value = value.strip()
    if not value and not allow_empty:
        raise EvaluationEventError(f'{name} cannot be empty')
    return value


def _require_int(value, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationEventError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise EvaluationEventError(f'{name} must be >= {minimum}')
    return value


def _require_number(value,
                    name: str,
                    minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationEventError(f'{name} must be a number')
    value = float(value)
    if not math.isfinite(value):
        raise EvaluationEventError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise EvaluationEventError(f'{name} must be >= {minimum}')
    return value


def _require_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationEventError(f'{name} must be a boolean')
    return value


def _require_episode_count_overrides(value, name: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvaluationEventError(f'{name} must be a mapping')
    normalized = {}
    for task_id, count in value.items():
        task_id = _require_string(task_id, f'{name} key')
        if task_id in normalized:
            raise EvaluationEventError(
                f'{name} contains duplicate task id {task_id!r}')
        normalized[task_id] = _require_int(count, f'{name}[{task_id!r}]', 1)
    return normalized


def _require_datetime(value, name: str) -> tuple[str, float]:
    value = _require_string(value, name)
    normalized = value[:-1] + '+00:00' if value.endswith('Z') else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise EvaluationEventError(f'{name} must be an ISO-8601 timestamp') \
            from exc
    return value, parsed.timestamp()


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds:02d}s'
    if seconds < 3600:
        return f'{seconds // 60:02d}m{seconds % 60:02d}s'
    hours, remainder = divmod(seconds, 3600)
    return f'{hours:02d}h{remainder // 60:02d}m{remainder % 60:02d}s'


def _robocasa_group_name(env_name: str) -> str:
    if 'ToCabinet' in env_name:
        return 'Cabinet'
    if 'ToDrawer' in env_name:
        return 'Drawer'
    if 'ToMicrowave' in env_name:
        return 'Microwave'
    return 'Generalization'


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=4, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class FluxVLAROSEvaluationReporter:
    """Consume ROS evaluation lifecycle events and write native artifacts.

    Args:
        result_root: Result root. It is resolved to an absolute path;
            individual runs are placed below ``eval_runs/<checkpoint stem>``.
        config_path: Authoritative FluxVLA MMEngine config path.
        ckpt_path: Authoritative model checkpoint path.
        eval_config: The selected FluxVLA eval runner config. It must define
            ``model_family`` and ``num_trials_per_task``. Optional
            ``num_trials_per_task_overrides`` entries replace that default
            for exact task ids. LIBERO additionally requires
            ``task_suite_name``; RoboCasa requires ``task_list``.
        report_kind: Optional explicit ``libero`` or ``robocasa`` selector.
        logger: Optional ``Callable[[str], None]``. Defaults to Overwatch.
        feishu: Optional mapping with ``sheet_url``, ``app_id``, ``app_secret``
            and ``timeout``. Missing credentials retain the native environment
            variable fallback.

    ``process_event`` returns a response dict containing ``accepted``,
    ``error``, ``run_dir``, ``duplicate``, ``next_sequence`` and ``status``.
    Protocol validation failures are returned in-band rather than raised. The
    first event of every run has sequence ``1``.
    """

    def __init__(self,
                 result_root,
                 config_path,
                 ckpt_path,
                 eval_config,
                 logger=None,
                 feishu=None,
                 report_kind=None):
        self.result_root = Path(result_root).expanduser().resolve()
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        self.eval_config = _runner_eval_config(eval_config)
        self.report_kind = _resolve_report_kind(self.eval_config, report_kind)
        suite_default = 'robocasa' if self.report_kind == 'robocasa' else None
        self.task_suite_name = _require_string(
            _cfg_get(self.eval_config, 'task_suite_name', suite_default),
            'eval_config.task_suite_name')
        self.model_family = _require_string(
            _cfg_get(self.eval_config, 'model_family'),
            'eval_config.model_family')
        self.num_trials_per_task = _require_int(
            _cfg_get(self.eval_config, 'num_trials_per_task'),
            'eval_config.num_trials_per_task', 1)
        self.num_trials_per_task_overrides = \
            _require_episode_count_overrides(
                _cfg_get(self.eval_config,
                         'num_trials_per_task_overrides'),
                'eval_config.num_trials_per_task_overrides')
        self.action_order = _require_string(
            _cfg_get(self.eval_config, 'action_order',
                     'n15' if self.model_family == 'groot' else 'fluxvla'),
            'eval_config.action_order')
        self.task_list = self._configured_task_list()
        self.configured_task_ids = _cfg_get(self.eval_config, 'task_ids')
        self.result_gpu_id = _require_int(
            _cfg_get(self.eval_config, 'result_gpu_id', 0),
            'eval_config.result_gpu_id', 0)
        self.expected_task_indexes = self._expected_task_indexes()
        self._logger = logger or overwatch.info
        self._feishu = _require_mapping(feishu or {}, 'feishu')
        self._active: Optional[_RunState] = None
        self._requests: dict[str, _CachedRequest] = {}
        self._last_run_dir: Optional[Path] = None
        self._lock = threading.RLock()

    def set_logger(self, logger=None) -> None:
        """Bind a runtime logger; ``None`` restores Overwatch."""
        if logger is not None and not callable(logger):
            raise TypeError('logger must be callable or None')
        self._logger = logger or overwatch.info

    @property
    def run_dir(self) -> Optional[str]:
        run_dir = self._active.run_dir if self._active else self._last_run_dir
        return str(run_dir) if run_dir is not None else None

    def process_event(self,
                      event_type,
                      request_id,
                      run_session_id,
                      sequence,
                      payload,
                      *,
                      version=EVENT_VERSION) -> dict:
        """Validate and persist one evaluation lifecycle event."""
        with self._lock:
            try:
                return self._process_event(event_type, request_id,
                                           run_session_id, sequence, payload,
                                           version)
            # Protocol and artifact errors are returned in-band.
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                self._log(f'[ros-eval] rejected event: {error}')
                active = self._active
                return {
                    'accepted': False,
                    'duplicate': False,
                    'error': error,
                    'run_dir': str(active.run_dir) if active else self.run_dir,
                    'next_sequence': active.next_sequence if active else 1,
                    'status': 'running' if active else 'idle',
                }

    def _process_event(self, event_type, request_id, run_session_id, sequence,
                       payload, version) -> dict:
        event_type = _require_string(event_type, 'event_type').lower()
        if event_type not in EVENT_TYPES:
            raise EvaluationEventError(f'unsupported event_type: {event_type}')
        request_id = _require_string(request_id, 'request_id')
        run_session_id = _require_string(run_session_id, 'run_session_id')
        sequence = _require_int(sequence, 'sequence', 1)
        version = _require_int(version, 'version', 1)
        if version != EVENT_VERSION:
            raise EvaluationEventError(f'unsupported event version {version}; '
                                       f'expected {EVENT_VERSION}')
        payload = _require_mapping(payload, 'payload')
        fingerprint = json.dumps(
            [version, event_type, run_session_id, sequence, payload],
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'))

        cached = self._requests.get(request_id)
        if cached is not None:
            if cached.fingerprint != fingerprint:
                raise EvaluationEventError(
                    f'request_id {request_id!r} was reused with different data'
                )
            response = copy.deepcopy(cached.response)
            response['duplicate'] = True
            return response

        if event_type == 'run_start':
            if self._active is not None:
                raise EvaluationEventError(
                    f'run {self._active.session_id!r} is already active')
            if sequence != 1:
                raise EvaluationEventError(
                    f'run_start sequence must be 1, got {sequence}')
            state = self._start_run(run_session_id, payload)
            self._append_event(state, version, event_type, request_id,
                               sequence, payload)
            self._active = state
            response = self._response(state, status='running')
        else:
            state = self._active
            if state is None:
                raise EvaluationEventError('no evaluation run is active')
            if run_session_id != state.session_id:
                raise EvaluationEventError(
                    f'event session {run_session_id!r} does not match active '
                    f'run {state.session_id!r}')
            if sequence != state.next_sequence:
                raise EvaluationEventError(
                    f'expected sequence {state.next_sequence}, got {sequence}')
            snapshot = self._snapshot_mutable_state(state)
            try:
                if event_type == 'episode_start':
                    self._episode_start(state, payload)
                    status = 'running'
                elif event_type == 'episode_end':
                    self._episode_end(state, payload)
                    status = 'running'
                else:
                    status = self._run_end(state, payload)
                    # Local finalization is intentionally completed before
                    # the run_end event is committed. All local writes are
                    # idempotent, so a transient filesystem failure can safely
                    # retry the same request and sequence without duplicating
                    # a Feishu row.
                    summary_path = self._write_summary_artifacts(state)
                self._append_event(state, version, event_type, request_id,
                                   sequence, payload)
            except Exception:
                # Event handlers update in-memory progress before the durable
                # journal append. Restore it when either an artifact write or
                # the journal commit fails, so the same request/sequence can
                # be retried without observing a half-accepted episode.
                self._restore_mutable_state(state, snapshot)
                raise
            state.next_sequence += 1
            response = self._response(state, status=status)
            if event_type == 'run_end':
                response.update(
                    self._report_finalized_run(state, payload, summary_path))
                self._last_run_dir = state.run_dir
                self._active = None

        response['error'] = ''
        response['accepted'] = True
        response['duplicate'] = False
        self._requests[request_id] = _CachedRequest(
            fingerprint=fingerprint, response=copy.deepcopy(response))
        return response

    @staticmethod
    def _snapshot_mutable_state(state: _RunState) -> tuple[dict, int, set]:
        return (
            copy.deepcopy(state.active_episodes),
            len(state.episodes),
            set(state.completed_keys),
        )

    @staticmethod
    def _restore_mutable_state(state: _RunState, snapshot: tuple[dict, int,
                                                                 set]) -> None:
        active_episodes, episode_count, completed_keys = snapshot
        state.active_episodes = active_episodes
        del state.episodes[episode_count:]
        state.completed_keys = completed_keys

    def _start_run(self, session_id: str, payload: dict) -> _RunState:
        schema_version = payload.get('schema_version')
        if str(schema_version) not in {'1', '1.0'}:
            raise EvaluationEventError(
                f'unsupported run schema_version {schema_version!r}')
        run_name = _require_string(
            payload.get('run_name', ''), 'payload.run_name', allow_empty=True)
        if run_name and (Path(run_name).name != run_name
                         or run_name in {'.', '..'}):
            raise EvaluationEventError(
                'payload.run_name must be a path component')
        seed = _require_int(payload.get('seed'), 'payload.seed')
        episodes_per_task = _require_int(
            payload.get('episodes_per_task'), 'payload.episodes_per_task', 1)
        max_episode_steps = _require_int(
            payload.get('max_episode_steps'), 'payload.max_episode_steps', 1)
        execute_horizon = payload.get('execute_horizon')
        if execute_horizon is not None:
            execute_horizon = _require_int(execute_horizon,
                                           'payload.execute_horizon', 1)
        total_tasks = _require_int(
            payload.get('total_tasks'), 'payload.total_tasks', 1)
        total_episodes = _require_int(
            payload.get('total_episodes'), 'payload.total_episodes', 1)
        full_suite = _require_bool(
            payload.get('full_suite'), 'payload.full_suite')
        tasks_value = payload.get('tasks')
        if (not isinstance(tasks_value, Sequence)
                or isinstance(tasks_value, (str, bytes))):
            raise EvaluationEventError('payload.tasks must be a sequence')
        if len(tasks_value) != total_tasks:
            raise EvaluationEventError(
                'payload.tasks length must equal payload.total_tasks')
        tasks_by_id = {}
        tasks_by_index = {}
        for position, value in enumerate(tasks_value):
            task = _require_mapping(value, f'payload.tasks[{position}]')
            task_id = _require_string(
                task.get('task_id'), f'payload.tasks[{position}].task_id')
            task_index = _require_int(
                task.get('task_index'),
                f'payload.tasks[{position}].task_index', 0)
            description = _require_string(
                task.get('description', ''),
                f'payload.tasks[{position}].description',
                allow_empty=True)
            metadata = _require_mapping(
                task.get('metadata', {}),
                f'payload.tasks[{position}].metadata')
            metadata_benchmark = metadata.get('benchmark')
            if (metadata_benchmark is not None
                    and str(metadata_benchmark).lower() != self.report_kind):
                raise EvaluationEventError(
                    f'task {task_id!r} metadata benchmark='
                    f'{metadata_benchmark!r} does not match authoritative '
                    f'report kind {self.report_kind!r}')
            for suite_key in ('task_suite_name', 'suite'):
                metadata_suite = metadata.get(suite_key)
                if (metadata_suite is not None
                        and str(metadata_suite) != self.task_suite_name):
                    raise EvaluationEventError(
                        f'task {task_id!r} metadata {suite_key}='
                        f'{metadata_suite!r} does not match authoritative '
                        'suite '
                        f'{self.task_suite_name!r}')
            if self.report_kind == 'robocasa':
                self._validate_robocasa_manifest_entry(
                    task_id=task_id,
                    task_index=task_index,
                    metadata=metadata,
                )
            if task_id in tasks_by_id:
                raise EvaluationEventError(f'duplicate task_id {task_id!r}')
            if task_index in tasks_by_index:
                raise EvaluationEventError(
                    f'duplicate task_index {task_index}')
            entry = _TaskManifestEntry(task_id, task_index, description,
                                       metadata)
            tasks_by_id[task_id] = entry
            tasks_by_index[task_index] = entry

        episode_counts_value = payload.get('episodes_per_task_by_task')
        if episode_counts_value is None:
            episodes_per_task_by_task = {
                task_id: episodes_per_task
                for task_id in tasks_by_id
            }
        else:
            episodes_per_task_by_task = _require_episode_count_overrides(
                episode_counts_value, 'payload.episodes_per_task_by_task')
            expected_task_ids = set(tasks_by_id)
            actual_task_ids = set(episodes_per_task_by_task)
            if actual_task_ids != expected_task_ids:
                missing = sorted(expected_task_ids - actual_task_ids)
                extra = sorted(actual_task_ids - expected_task_ids)
                raise EvaluationEventError(
                    'payload.episodes_per_task_by_task must exactly match '
                    f'the task manifest; missing={missing}, extra={extra}')
        expected_total_episodes = sum(episodes_per_task_by_task.values())
        if total_episodes != expected_total_episodes:
            raise EvaluationEventError(
                'payload.total_episodes must equal the sum of '
                'episodes_per_task_by_task')

        run_id, run_dir = self._allocate_run_dir(run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / 'rank_progress').mkdir()
        _write_json_atomic(run_dir / 'rank_progress' / 'rank0.json', {
            'rank': 0,
            'episodes': 0,
            'successes': 0,
        })
        state = _RunState(
            session_id=session_id,
            run_id=run_id,
            run_dir=run_dir,
            start={
                'schema_version': RUN_SCHEMA_VERSION,
                'run_name': run_name,
                'seed': seed,
                'episodes_per_task': episodes_per_task,
                'episodes_per_task_by_task': episodes_per_task_by_task,
                'max_episode_steps': max_episode_steps,
                'execute_horizon': execute_horizon,
                'total_tasks': total_tasks,
                'total_episodes': total_episodes,
                'full_suite': full_suite,
            },
            tasks_by_id=tasks_by_id,
            tasks_by_index=tasks_by_index,
        )
        self._append_log(
            state, f'ROS evaluation session: {session_id}\n'
            f'task_suite: {self.task_suite_name}\n'
            f'model_family: {self.model_family}\n'
            f'config: {self.config_path}\n'
            f'ckpt: {self.ckpt_path}\n')
        self._log(f'[ros-eval] run_start session={session_id} '
                  f'episodes={total_episodes} run_dir={run_dir}')
        return state

    def _episode_start(self, state: _RunState, payload: dict) -> None:
        task, episode_index, seed, description = self._episode_identity(
            state, payload)
        started_at, started_epoch = _require_datetime(
            payload.get('started_at'), 'payload.started_at')
        key = (task.task_index, episode_index)
        if key in state.completed_keys:
            raise EvaluationEventError(f'episode {key} already completed')
        if key in state.active_episodes:
            raise EvaluationEventError(f'episode {key} is already active')
        episode_id = payload.get('episode_id')
        if episode_id is not None:
            episode_id = _require_string(episode_id, 'payload.episode_id')
            if any(
                    active.get('episode_id') == episode_id
                    for active in state.active_episodes.values()):
                raise EvaluationEventError(
                    f'episode_id {episode_id!r} is already active')
        current = {
            'task_id': task.task_id,
            'task_index': task.task_index,
            'description': description,
            'episode_index': episode_index,
            'seed': seed,
            'started_at': started_at,
            'started_epoch': started_epoch,
        }
        if episode_id is not None:
            current['episode_id'] = episode_id
        state.active_episodes[key] = current
        self._append_log(
            state,
            f'Evaluating Task {task.task_index}, Trial {episode_index}\n'
            f'\nTask: {description}\n'
            f'Starting episode {episode_index + 1}...\n')
        self._log(f'Evaluating Task {task.task_index}, Trial {episode_index}')
        self._log(f'\nTask: {description}')
        self._log(f'Starting episode {episode_index + 1}...')

    def _episode_end(self, state: _RunState, payload: dict) -> None:
        task, episode_index, seed, description = self._episode_identity(
            state, payload)
        episode_key = (task.task_index, episode_index)
        current = state.active_episodes.get(episode_key)
        if current is None:
            raise EvaluationEventError(
                f'episode_end has no matching episode_start for '
                f'episode {episode_key}')
        episode_id = payload.get('episode_id')
        current_episode_id = current.get('episode_id')
        if current_episode_id is not None:
            episode_id = _require_string(episode_id, 'payload.episode_id')
            if episode_id != current_episode_id:
                raise EvaluationEventError(
                    'episode_end episode_id does not match episode_start')
        elif episode_id is not None:
            episode_id = _require_string(episode_id, 'payload.episode_id')
            current['episode_id'] = episode_id
        for key, value in (
            ('task_index', task.task_index),
            ('episode_index', episode_index),
            ('seed', seed),
            ('description', description),
        ):
            if current[key] != value:
                raise EvaluationEventError(
                    f'episode_end {key} does not match episode_start')
        success = _require_bool(payload.get('success'), 'payload.success')
        episode_return = _require_number(
            payload.get('episode_return'), 'payload.episode_return')
        steps = _require_int(payload.get('steps'), 'payload.steps', 0)
        duration = _require_number(
            payload.get('duration_s'), 'payload.duration_s', 0.0)
        termination_reason = _require_string(
            payload.get('termination_reason'), 'payload.termination_reason')
        model_calls = _require_int(
            payload.get('model_calls'), 'payload.model_calls', 0)
        score = payload.get('score')
        if score is not None:
            score = _require_number(score, 'payload.score', 0.0)
            if score > 1.0:
                raise EvaluationEventError(
                    f'payload.score must be <= 1.0, got {score!r}')
        info = _require_mapping(payload.get('info', {}), 'payload.info')
        prediction_metadata = payload.get('prediction_metadata', [])
        if (not isinstance(prediction_metadata, Sequence)
                or isinstance(prediction_metadata, (str, bytes))):
            raise EvaluationEventError(
                'payload.prediction_metadata must be a sequence')
        prediction_metadata = json.loads(
            json.dumps(list(prediction_metadata), ensure_ascii=False))
        completed = _require_int(
            payload.get('completed_episodes'), 'payload.completed_episodes', 1)
        total = _require_int(
            payload.get('total_episodes'), 'payload.total_episodes', 1)
        if total != state.start['total_episodes']:
            raise EvaluationEventError(
                'episode_end total_episodes changed during the run')
        if completed != len(state.episodes) + 1:
            raise EvaluationEventError(
                f'episode_end completed_episodes must be '
                f'{len(state.episodes) + 1}')
        result = {
            **current,
            'success': success,
            'episode_return': episode_return,
            'steps': steps,
            'duration_s': duration,
            'termination_reason': termination_reason,
            'model_calls': model_calls,
            'info': info,
            'prediction_metadata': prediction_metadata,
        }
        if score is not None:
            result['score'] = score
        state.episodes.append(result)
        state.completed_keys.add(episode_key)
        del state.active_episodes[episode_key]
        successes = sum(bool(item['success']) for item in state.episodes)
        success_rate = successes / len(state.episodes) * 100
        scores = [
            float(item['score']) for item in state.episodes
            if item.get('score') is not None
        ]
        mean_score = sum(scores) / len(scores) if scores else None
        rank_progress = {
            'rank': 0,
            'episodes': len(state.episodes),
            'successes': successes,
        }
        if score is not None:
            rank_progress['score'] = score
        if mean_score is not None:
            rank_progress['scored_episodes'] = len(scores)
            rank_progress['mean_score'] = mean_score
        _write_json_atomic(state.run_dir / 'rank_progress' / 'rank0.json',
                           rank_progress)
        score_log = ''
        if score is not None:
            score_log += f'# local score: {score * 100:.1f}%\n'
        if mean_score is not None:
            score_log += f'# local mean score: {mean_score * 100:.1f}%\n'
        self._append_log(
            state, f'Success: {success}\n'
            f'# local episodes completed so far: {len(state.episodes)}\n'
            f'# local successes: {successes} ({success_rate:.1f}%)\n'
            f'{score_log}')
        score_progress = ''
        if score is not None:
            score_progress += f' score={score * 100:.2f}%'
        if mean_score is not None:
            score_progress += f' mean_score={mean_score * 100:.2f}%'
        self._log('[eval-progress] '
                  f'episodes={len(state.episodes)}/'
                  f"{state.start['total_episodes']} "
                  f'successes={successes} '
                  f'success_rate={success_rate:.2f}%'
                  f'{score_progress}')

    def _episode_identity(
            self, state: _RunState,
            payload: dict) -> tuple[_TaskManifestEntry, int, int, str]:
        task_id = _require_string(payload.get('task_id'), 'payload.task_id')
        task_index = _require_int(
            payload.get('task_index'), 'payload.task_index', 0)
        task = state.tasks_by_id.get(task_id)
        if task is None or task.task_index != task_index:
            raise EvaluationEventError(
                f'task {task_id!r}/{task_index} is not in the run manifest')
        description = _require_string(
            payload.get('description', ''),
            'payload.description',
            allow_empty=True)
        if description != task.description:
            raise EvaluationEventError(
                f'task {task_id!r} description changed during the run')
        episode_index = _require_int(
            payload.get('episode_index'), 'payload.episode_index', 0)
        episodes_per_task = self._episodes_for_task(state, task)
        if episode_index >= episodes_per_task:
            raise EvaluationEventError(
                f'payload.episode_index exceeds episodes_per_task for '
                f'task {task.task_id!r}')
        seed = _require_int(payload.get('seed'), 'payload.seed')
        return task, episode_index, seed, description

    def _run_end(self, state: _RunState, payload: dict) -> str:
        status = _require_string(payload.get('status'),
                                 'payload.status').lower()
        if status not in {'finished', 'failed', 'interrupted'}:
            raise EvaluationEventError(
                'payload.status must be finished, failed, or interrupted')
        if status == 'finished' and state.active_episodes:
            raise EvaluationEventError(
                'finished run still has active episodes: '
                f'{sorted(state.active_episodes)}')
        full_suite = _require_bool(
            payload.get('full_suite'), 'payload.full_suite')
        completed = _require_int(
            payload.get('completed_episodes'), 'payload.completed_episodes', 0)
        total = _require_int(
            payload.get('total_episodes'), 'payload.total_episodes', 1)
        _require_datetime(payload.get('finished_at'), 'payload.finished_at')
        _require_number(payload.get('duration_s'), 'payload.duration_s', 0.0)
        if completed != len(state.episodes):
            raise EvaluationEventError(
                'run_end completed_episodes does not match accepted episodes')
        if total != state.start['total_episodes']:
            raise EvaluationEventError('run_end total_episodes changed')
        if full_suite != state.start['full_suite']:
            raise EvaluationEventError('run_end full_suite changed during run')
        if 'error' in payload:
            json.dumps(payload['error'], ensure_ascii=False)
        state.active_episodes.clear()
        self._append_log(
            state, f'Run status: {status}\n'
            f'# episodes completed: {completed}\n'
            f'# successes: '
            f"{sum(bool(item['success']) for item in state.episodes)}\n")
        self._log(f'[ros-eval] run_end status={status} '
                  f'episodes={completed}/{total} run_dir={state.run_dir}')
        return status

    def _report_finalized_run(self, state: _RunState, end: dict,
                              summary_path: Path) -> dict:
        eligible, reason = self._feishu_eligibility(state, end)
        if not eligible:
            self._log(f'[ros-eval] Feishu skipped: {reason}')
            return {
                'reported_to_feishu': False,
                'report_reason': reason,
                'summary_path': str(summary_path),
            }
        try:
            result = maybe_report_summary_to_feishu(
                str(summary_path),
                self.report_kind,
                sheet_url=self._feishu.get('sheet_url'),
                app_id=self._feishu.get('app_id'),
                app_secret=self._feishu.get('app_secret'),
                config=self.config_path,
                timeout=float(self._feishu.get('timeout', 10.0)),
                logger=self._log,
                log_unconfigured=True)
        except Exception as exc:  # Feishu is best effort after local commit.
            reason = f'{type(exc).__name__}: {exc}'
            self._log(f'[ros-eval] Feishu skipped: {reason}')
            return {
                'reported_to_feishu': False,
                'report_reason': reason,
                'summary_path': str(summary_path),
            }
        return {
            'reported_to_feishu': bool(result.wrote),
            'report_reason': result.reason,
            'summary_path': str(summary_path),
        }

    def _write_summary_artifacts(self, state: _RunState) -> Path:
        if self.report_kind == 'robocasa':
            return self._write_robocasa_summary_artifacts(state)
        return self._write_libero_summary_artifacts(state)

    def _write_libero_summary_artifacts(self, state: _RunState) -> Path:
        grouped = defaultdict(list)
        for episode in state.episodes:
            grouped[episode['task_index']].append(episode)
        suite_dir = state.run_dir / self.task_suite_name
        status_dir = state.run_dir / 'task_status'
        suite_dir.mkdir(exist_ok=True)
        status_dir.mkdir(exist_ok=True)
        task_results = {}
        total_successes = 0
        total_trials = 0
        total_time = 0.0
        max_time = 0.0
        all_scores = []
        for task_index in sorted(grouped):
            episodes = sorted(
                grouped[task_index], key=lambda item: item['episode_index'])
            task = state.tasks_by_index[task_index]
            successes = sum(bool(item['success']) for item in episodes)
            duration = sum(float(item['duration_s']) for item in episodes)
            success_episodes = [
                item['episode_index'] for item in episodes if item['success']
            ]
            failure_episodes = [
                item['episode_index'] for item in episodes
                if not item['success']
            ]
            start_epoch = min(item['started_epoch'] for item in episodes)
            start_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(start_epoch))
            count = len(episodes)
            scores = [
                float(item['score']) for item in episodes
                if item.get('score') is not None
            ]
            score_metrics = ({
                'scored_episodes': len(scores),
                'mean_score': sum(scores) / len(scores),
            } if scores else {})
            per_task = {
                'task_suite': self.task_suite_name,
                'task_id': task_index,
                'task_description': task.description,
                'successes': successes,
                'total_episodes': count,
                'success_episodes': success_episodes,
                'failure_episodes': failure_episodes,
                'start_time': start_time,
                'duration': duration,
                'gpu_id': self.result_gpu_id,
                **score_metrics,
            }
            _write_json_atomic(suite_dir / f'task{task_index}_results.json',
                               per_task)
            manager_suite_dir = self.result_root / self.task_suite_name
            manager_suite_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                manager_suite_dir /
                f'gpu{self.result_gpu_id}_task{task_index}_results.json',
                per_task)
            complete = count == self._episodes_for_task(state, task)
            task_state = 'SUCCESS' if complete else 'PARTIAL'
            (status_dir /
             f'{self.task_suite_name}_task{task_index}.status').write_text(
                 f'{task_state}|{successes}|{count}|{int(start_epoch)}',
                 encoding='utf-8')
            rate = successes / count * 100
            task_results[f'{self.task_suite_name}_{task_index}'] = {
                'success_rate': rate,
                'duration': duration,
                'total_episodes': count,
                'successes': successes,
                'task_description': task.description,
                **score_metrics,
            }
            all_scores.extend(scores)
            total_successes += successes
            total_trials += count
            total_time += duration
            max_time = max(max_time, duration)
            self._log(f'Task {task_index} completed: '
                      f'{successes}/{count} successes')
            self._log(f'Time taken: {duration:.2f} seconds')

        completed_tasks = len(task_results)
        success_rate = total_successes / max(total_trials, 1) * 100
        average_time = total_time / max(completed_tasks, 1)
        (state.run_dir / 'failed_tasks.txt').write_text('', encoding='utf-8')
        self._write_summary_text(state, completed_tasks, total_trials,
                                 total_successes, success_rate, total_time,
                                 average_time, max_time)
        self._write_summary_csv(state, success_rate, average_time, max_time)
        self._write_task_csv(state, task_results)
        summary = {
            'run_id': state.run_id,
            'ckpt': self.ckpt_path,
            'config': Path(self.config_path).stem,
            'suite_stats': {
                self.task_suite_name: {
                    'total_tasks': completed_tasks,
                    'total_trials': total_trials,
                    'total_successes': total_successes,
                    'total_time': total_time,
                    'max_time': max_time,
                }
            },
            'task_results': task_results,
            'overall': {
                'average_success_rate': success_rate,
                'total_time': total_time,
                'average_task_time': average_time,
            },
        }
        if all_scores:
            score_metrics = {
                'scored_episodes': len(all_scores),
                'mean_score': sum(all_scores) / len(all_scores),
            }
            summary['suite_stats'][self.task_suite_name].update(score_metrics)
            summary['overall'].update(score_metrics)
        summary_path = state.run_dir / 'summary.json'
        _write_json_atomic(summary_path, summary)
        self._log(f'# episodes completed: {total_trials}')
        self._log(f'# successes: {total_successes} ({success_rate:.1f}%)')
        if all_scores:
            self._log(f'# mean score: '
                      f'{sum(all_scores) / len(all_scores) * 100:.1f}%')
        self._log(f'[ros-eval] wrote {self.report_kind.upper()} summary '
                  f'artifacts to {state.run_dir}')
        return summary_path

    def _write_robocasa_summary_artifacts(self, state: _RunState) -> Path:
        grouped = defaultdict(list)
        for episode in state.episodes:
            grouped[episode['task_index']].append(episode)

        task_result_dir = state.run_dir / 'robocasa'
        status_dir = state.run_dir / 'task_status'
        manager_result_dir = self.result_root / 'robocasa'
        task_result_dir.mkdir(exist_ok=True)
        status_dir.mkdir(exist_ok=True)
        manager_result_dir.mkdir(parents=True, exist_ok=True)
        group_stats = {
            group: {
                'total_tasks': 0,
                'total_trials': 0,
                'total_successes': 0,
                'total_time': 0.0,
                'max_time': 0.0,
            }
            for group in ROBOCASA_GROUP_ORDER
        }
        task_stats = {}
        total_trials = 0
        total_successes = 0
        total_time = 0.0
        overall_max_time = 0.0
        incomplete_tasks = []

        for task_index in sorted(state.tasks_by_index):
            episodes = sorted(
                grouped.get(task_index, ()),
                key=lambda item: item['episode_index'])
            count = len(episodes)
            if count == 0:
                incomplete_tasks.append(task_index)
                continue
            env_name = self.task_list[task_index]
            group = _robocasa_group_name(env_name)
            successes = sum(bool(item['success']) for item in episodes)
            duration = sum(float(item['duration_s']) for item in episodes)
            rate = successes / count * 100
            stats = group_stats[group]
            stats['total_tasks'] += 1
            stats['total_trials'] += count
            stats['total_successes'] += successes
            stats['total_time'] += duration
            stats['max_time'] = max(stats['max_time'], duration)
            task_stats[env_name] = {
                'task_id': task_index,
                'group': group,
                'total_episodes': count,
                'successes': successes,
                'success_rate': rate,
                'duration': duration,
            }
            per_task = {
                'task_id': task_index,
                'env_name': env_name,
                'group': group,
                'successes': successes,
                'total_episodes': count,
                'success_rate': rate,
                'duration': duration,
                'gpu_id': self.result_gpu_id,
            }
            _write_json_atomic(
                task_result_dir / f'task{task_index}_results.json', per_task)
            _write_json_atomic(
                manager_result_dir /
                f'gpu{self.result_gpu_id}_task{task_index}_results.json',
                per_task)
            complete = count == self._episodes_for_task(state, task)
            if not complete:
                incomplete_tasks.append(task_index)
            start_epoch = min(item['started_epoch'] for item in episodes)
            task_state = 'SUCCESS' if complete else 'PARTIAL'
            (status_dir / f'robocasa_task{task_index}.status').write_text(
                f'{task_state}|{successes}|{count}|{int(start_epoch)}',
                encoding='utf-8')
            total_trials += count
            total_successes += successes
            total_time += duration
            overall_max_time = max(overall_max_time, duration)
            self._log(f'Task {task_index} ({env_name}) completed: '
                      f'{successes}/{count} successes')
            self._log(f'Time taken: {duration:.2f} seconds')

        overall_rate = total_successes / max(total_trials, 1) * 100
        with (state.run_dir / 'summary.csv').open(
                'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['', *ROBOCASA_GROUP_ORDER, 'all'])
            group_rates = []
            for group in ROBOCASA_GROUP_ORDER:
                stats = group_stats[group]
                if stats['total_trials']:
                    rate = (
                        stats['total_successes'] / stats['total_trials'] * 100)
                    group_rates.append(f'{rate:.2f}')
                else:
                    group_rates.append('')
            writer.writerow(
                ['Success Rate (%)', *group_rates, f'{overall_rate:.2f}'])
            writer.writerow([
                'Episodes', *[
                    group_stats[group]['total_trials']
                    for group in ROBOCASA_GROUP_ORDER
                ], total_trials
            ])
            writer.writerow([
                'Successes', *[
                    group_stats[group]['total_successes']
                    for group in ROBOCASA_GROUP_ORDER
                ], total_successes
            ])

        with (state.run_dir / 'task_success_rates.csv').open(
                'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ['Task', 'Group', 'Successes', 'Episodes', 'Success Rate (%)'])
            for env_name, stats in task_stats.items():
                writer.writerow([
                    env_name, stats['group'], stats['successes'],
                    stats['total_episodes'], f"{stats['success_rate']:.2f}"
                ])

        text_lines = ['=== RoboCasa Evaluation Results Summary ===']
        for group in ROBOCASA_GROUP_ORDER:
            stats = group_stats[group]
            rate = (
                stats['total_successes'] / max(stats['total_trials'], 1) *
                100 if stats['total_trials'] else 0.0)
            text_lines.extend([
                '',
                f'{group}:',
                f"- Tasks completed: {stats['total_tasks']}",
                f"- Total attempts: {stats['total_trials']}",
                f"- Successful attempts: {stats['total_successes']}",
                f'- Success rate: {rate:.2f}%',
                f"- Total time: {_format_duration(stats['total_time'])}",
            ])
        text_lines.extend([
            '',
            'Overall statistics:',
            f'- Success rate: {overall_rate:.2f}%',
            f'- Total attempts: {total_trials}',
            f'- Successful attempts: {total_successes}',
            f'- Total time: {_format_duration(total_time)}',
        ])
        (state.run_dir / 'summary.txt').write_text(
            '\n'.join(text_lines) + '\n', encoding='utf-8')
        (state.run_dir / 'failed_tasks.txt').write_text(
            ''.join(f'{task_index}\n' for task_index in incomplete_tasks),
            encoding='utf-8')

        summary = {
            'run_id': state.run_id,
            'ckpt': self.ckpt_path,
            'config': self.config_path,
            'model_family': self.model_family,
            'action_order': self.action_order,
            'task_ids': sorted(state.tasks_by_index),
            'group_stats': group_stats,
            'task_stats': task_stats,
            'overall': {
                'success_rate': overall_rate,
                'total_episodes': total_trials,
                'successes': total_successes,
                'total_time': total_time,
                'max_time': overall_max_time,
            },
        }
        summary_path = state.run_dir / 'summary.json'
        _write_json_atomic(summary_path, summary)
        self._log(f'# episodes completed: {total_trials}')
        self._log(f'# successes: {total_successes} ({overall_rate:.1f}%)')
        self._log(f'[ros-eval] wrote RoboCasa summary artifacts to '
                  f'{state.run_dir}')
        return summary_path

    def _allocate_run_dir(self, run_name: str) -> tuple[str, Path]:
        """Allocate a timestamp run id without overwriting prior runs."""
        checkpoint_root = (
            self.result_root / 'eval_runs' / Path(self.ckpt_path).stem)
        now = time.time()
        for second_offset in range(86400):
            timestamp = time.strftime('%Y_%m_%d-%H_%M_%S',
                                      time.localtime(now + second_offset))
            run_id = (f'EVAL-{self.task_suite_name}-{self.model_family}-'
                      f'{timestamp}')
            if run_name:
                run_id = f'{run_id}-{run_name}'
            run_dir = checkpoint_root / run_id
            if not run_dir.exists():
                return run_id, run_dir
        raise FileExistsError(
            f'Could not allocate a unique evaluation run under '
            f'{checkpoint_root}')

    def _write_summary_text(self, state, completed_tasks, total_trials,
                            total_successes, success_rate, total_time,
                            average_time, max_time) -> None:
        cfg = self.eval_config
        task_ids = sorted(state.tasks_by_index)
        lines = [
            f'task_suite: {self.task_suite_name}',
            f'model_family: {self.model_family}',
            f'task_ids: {task_ids}',
            f"num_trials_per_task: {state.start['episodes_per_task']}",
            'episodes_per_task_by_task: ' + json.dumps(
                state.start['episodes_per_task_by_task'],
                ensure_ascii=False,
                sort_keys=True),
            f"eval_chunk_size: {state.start['execute_horizon']}",
            f"num_steps_wait: {_cfg_get(cfg, 'num_steps_wait', None)}",
            f'num_inference_steps: '
            f"{_cfg_get(cfg, 'num_inference_steps', None)}",
            f"max_steps: {state.start['max_episode_steps']}",
            'eval_shard_strategy: ros',
            f'preprocess_every_step: '
            f"{_cfg_get(cfg, 'preprocess_every_step', None)}",
            f'save_rollout_videos: '
            f"{_cfg_get(cfg, 'save_rollout_videos', False)}",
            f'save_failed_rollout_videos: '
            f"{_cfg_get(cfg, 'save_failed_rollout_videos', False)}",
            f'save_multi_view_rollout_videos: '
            f"{_cfg_get(cfg, 'save_multi_view_rollout_videos', False)}",
            f"rollout_dir: {_cfg_get(cfg, 'rollout_dir', None)}",
            f"seed: {state.start['seed']}",
            f'# successes: {total_successes} / {total_trials} '
            f'({success_rate:.1f}%)',
            '',
            '=== Evaluation Results Summary ===',
            '',
            f'{self.task_suite_name}:',
            f'- Tasks completed: {completed_tasks}',
            f'- Total attempts: {total_trials}',
            f'- Successful attempts: {total_successes}',
            f'- Success rate: {success_rate:.2f}%',
            f'- Total time: {_format_duration(total_time)}',
            f'- Average time per task: {_format_duration(average_time)}',
            f'- Longest task time: {_format_duration(max_time)}',
        ]
        (state.run_dir / 'summary.txt').write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')

    def _write_summary_csv(self, state, success_rate, average_time,
                           max_time) -> None:
        with (state.run_dir / 'summary.csv').open(
                'w', newline='', encoding='utf-8') as stream:
            stream.write(f'{Path(self.ckpt_path).name}\n')
            writer = csv.writer(stream)
            writer.writerow(['', self.task_suite_name, 'Overall'])
            writer.writerow([
                'Success Rate (%)', f'{success_rate:.2f}',
                f'{success_rate:.2f}'
            ])
            writer.writerow([
                'Average Time (s)', f'{average_time:.2f}',
                f'{average_time:.2f}'
            ])
            writer.writerow(
                ['Max Time (s)', f'{max_time:.2f}', f'{max_time:.2f}'])

    def _write_task_csv(self, state, task_results) -> None:
        with (state.run_dir / 'task_success_rates.csv').open(
                'w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Task', 'Description', 'Success Rate (%)'])
            for task_index in sorted(state.tasks_by_index):
                key = f'{self.task_suite_name}_{task_index}'
                if key not in task_results:
                    continue
                result = task_results[key]
                writer.writerow([
                    key, result['task_description'],
                    f"{result['success_rate']:.2f}"
                ])

    def _feishu_eligibility(self, state: _RunState,
                            end: dict) -> tuple[bool, str]:
        if end['status'].lower() != 'finished':
            return False, 'run status is not finished'
        if not state.start['full_suite'] or not end['full_suite']:
            return False, 'run is not marked as a full suite'
        if self.configured_task_ids is not None:
            return False, 'authoritative eval config contains a task filter'
        if self.expected_task_indexes is None:
            return False, 'complete authoritative task manifest is unknown'
        selected = set(state.tasks_by_index)
        if selected != self.expected_task_indexes:
            return (False,
                    'selected task indexes do not match the full manifest')
        if state.start['total_tasks'] != len(self.expected_task_indexes):
            return False, 'total task count does not match eval metadata'
        if state.start['episodes_per_task'] != self.num_trials_per_task:
            return False, 'episodes_per_task does not match eval metadata'
        expected_counts = {
            task.task_id:
            self.num_trials_per_task_overrides.get(task.task_id,
                                                   self.num_trials_per_task)
            for task in state.tasks_by_index.values()
        }
        if state.start['episodes_per_task_by_task'] != expected_counts:
            return (False,
                    'episodes_per_task_by_task does not match eval metadata')
        expected_total = sum(expected_counts.values())
        if (end['completed_episodes'] != end['total_episodes']
                or end['total_episodes'] != expected_total):
            return False, 'completed episode total is not the full suite total'
        counts = defaultdict(int)
        for episode in state.episodes:
            counts[episode['task_index']] += 1
        if any(counts[index] != expected_counts[
                state.tasks_by_index[index].task_id]
               for index in self.expected_task_indexes):
            return False, 'one or more tasks has an incomplete episode count'
        return True, 'eligible full-suite evaluation'

    @staticmethod
    def _episodes_for_task(state: _RunState, task: _TaskManifestEntry) -> int:
        return state.start['episodes_per_task_by_task'][task.task_id]

    def _expected_task_indexes(self) -> Optional[set[int]]:
        if self.task_list is not None:
            return set(range(len(self.task_list)))
        manifest = _cfg_get(self.eval_config, 'task_manifest')
        if manifest is not None:
            indexes = set()
            for position, task in enumerate(manifest):
                task = _require_mapping(
                    task, f'eval_config.task_manifest[{position}]')
                indexes.add(
                    _require_int(
                        task.get('task_index'),
                        f'eval_config.task_manifest[{position}].task_index',
                        0))
            return indexes
        total_tasks = _cfg_get(self.eval_config, 'total_tasks')
        if total_tasks is None:
            total_tasks = _cfg_get(self.eval_config, 'num_tasks')
        if total_tasks is None:
            total_tasks = LIBERO_SUITE_TASK_COUNTS.get(self.task_suite_name)
        if total_tasks is None:
            return None
        total_tasks = _require_int(total_tasks, 'eval_config.total_tasks', 1)
        return set(range(total_tasks))

    def _configured_task_list(self) -> Optional[tuple[str, ...]]:
        if self.report_kind != 'robocasa':
            return None
        value = _cfg_get(self.eval_config, 'task_list')
        if (not isinstance(value, Sequence) or isinstance(value,
                                                          (str, bytes))):
            raise EvaluationEventError(
                'eval_config.task_list must be a sequence for RoboCasa')
        task_list = tuple(
            _require_string(item, f'eval_config.task_list[{index}]')
            for index, item in enumerate(value))
        if not task_list:
            raise EvaluationEventError(
                'eval_config.task_list cannot be empty for RoboCasa')
        if len(task_list) != len(set(task_list)):
            raise EvaluationEventError(
                'eval_config.task_list cannot contain duplicate env names')
        return task_list

    def _validate_robocasa_manifest_entry(self, task_id: str, task_index: int,
                                          metadata: dict) -> None:
        if self.task_list is None or task_index >= len(self.task_list):
            raise EvaluationEventError(
                f'RoboCasa task index {task_index} is outside the configured '
                'task_list')
        if task_id != str(task_index):
            raise EvaluationEventError(
                f'RoboCasa task id {task_id!r} must equal its configured '
                f'index {task_index!r}')
        expected_env = self.task_list[task_index]
        actual_env = metadata.get('env_name', metadata.get('task_name'))
        if actual_env != expected_env:
            raise EvaluationEventError(
                f'RoboCasa task {task_index} env_name {actual_env!r} does '
                f'not match configured task_list entry {expected_env!r}')
        expected_group = _robocasa_group_name(expected_env)
        actual_group = metadata.get('group')
        if actual_group is not None and actual_group != expected_group:
            raise EvaluationEventError(
                f'RoboCasa task {task_index} group {actual_group!r} does not '
                f'match {expected_group!r}')

    def _append_event(self, state, version, event_type, request_id, sequence,
                      payload) -> None:
        record = {
            'event_version': version,
            'event_type': event_type,
            'request_id': request_id,
            'run_session_id': state.session_id,
            'sequence': sequence,
            'recorded_at': datetime.now().astimezone().isoformat(),
            'payload': payload,
        }
        journal_path = state.run_dir / 'events.jsonl'
        journal_existed = journal_path.exists()
        original_size = journal_path.stat().st_size if journal_existed else 0
        try:
            with journal_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                if journal_existed:
                    os.truncate(journal_path, original_size)
                else:
                    journal_path.unlink(missing_ok=True)
            except OSError as rollback_error:
                self._log('[ros-eval] failed to roll back journal append: '
                          f'{rollback_error}')
            raise

    @staticmethod
    def _append_log(state: _RunState, message: str) -> None:
        with (state.run_dir / 'rank0.txt').open(
                'a', encoding='utf-8') as stream:
            stream.write(message)
            stream.flush()

    def _response(self, state: _RunState, status: str) -> dict:
        return {
            'accepted': True,
            'duplicate': False,
            'error': '',
            'run_dir': str(state.run_dir),
            'next_sequence': state.next_sequence,
            'status': status,
        }

    def _log(self, message: str) -> None:
        try:
            self._logger(message)
        # Logging must never fail evaluation reporting.
        except Exception as exc:
            overwatch.warning(f'[ros-eval] logger failed: {exc}; {message}')


__all__ = ['EvaluationEventError', 'FluxVLAROSEvaluationReporter']
