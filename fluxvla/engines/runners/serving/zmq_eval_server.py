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
"""ZMQ transport for FluxThemis evaluation (inference + reporting).

This module replaces the ROS service transport (``ros_server.py``) with a
ZMQ REP server while reusing the transport-independent pieces: the
``FluxVLAROSPolicy`` inference core, the evaluation reporter, and the
episode-affinity policy pool. The wire format is the same msgpack framing as
``zmq_server.PolicyServer`` (``{"endpoint": ..., "data": {...}}``) with
msgpack-numpy array support.

Endpoints:

- ``predict_action`` -- request/reply inference, mirroring ROS PredictAction;
- ``report_event`` -- acknowledged evaluation lifecycle events, mirroring
  ROS ReportEvaluation;
- ``ping`` / ``kill`` -- inherited from :class:`PolicyServer`.
"""

from __future__ import annotations
import copy
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .zmq_server import PolicyServer


class FluxVLAZMQEvalServer(PolicyServer):
    """Serve FluxVLA inference and evaluation reporting over ZMQ REP."""

    def __init__(self,
                 policy: Any,
                 evaluation_reporter: Any = None,
                 host: str = '*',
                 port: int = 5555) -> None:
        super().__init__(host=host, port=port)
        self.policy = policy
        self.evaluation_reporter = evaluation_reporter
        self._request_lock = threading.RLock()
        self._report_lock = threading.RLock()
        self._last_episode_id: str | None = None
        self._last_seed: int | None = None

        self.register_endpoint('predict_action', self._handle_predict)
        self.register_endpoint('report_event', self._handle_report)

    # predict_action
    def _handle_predict(self,
                        observation: Mapping[str, Any],
                        episode_id: str = '',
                        seed: int = 0,
                        unnorm_key: str = '',
                        reset: bool = False,
                        request_id: str = '') -> dict[str, Any]:
        """Preprocess one inference request and return environment units."""
        if not isinstance(observation, Mapping):
            raise TypeError('predict_action observation must be a mapping')
        try:
            if getattr(self.policy, 'supports_episode_affinity', False):
                actions, inference_time_s = self.policy.predict(
                    observation,
                    unnorm_key=unnorm_key,
                    seed=int(seed),
                    episode_id=str(episode_id),
                    reset=bool(reset),
                )
            else:
                with self._request_lock:
                    is_new_episode = (
                        bool(reset) or str(episode_id) != self._last_episode_id
                        or int(seed) != self._last_seed)
                    observation = dict(observation)
                    observation['is_new_episode'] = is_new_episode
                    actions, inference_time_s = self.policy.predict(
                        observation, unnorm_key=unnorm_key, seed=int(seed))
                    self._last_episode_id = str(episode_id)
                    self._last_seed = int(seed)
            actions = self._canonicalize_actions(actions)
            return {
                'ok': True,
                'error': '',
                'request_id': str(request_id),
                'actions': actions,
                'action_horizon': int(actions.shape[0]),
                'action_dim': int(actions.shape[1]),
                'denormalized': True,
                'inference_time_s': float(inference_time_s),
            }
        except Exception as exc:
            return {
                'ok': False,
                'error': f'{type(exc).__name__}: {exc}',
                'request_id': str(request_id),
                'actions': np.zeros((0, 0), dtype=np.float32),
                'action_horizon': 0,
                'action_dim': 0,
                'denormalized': False,
                'inference_time_s': 0.0,
            }

    # report_event
    def _handle_report(self,
                       request_id: str = '',
                       run_id: str = '',
                       event_type: str = '',
                       sequence: int = 0,
                       payload: Mapping[str, Any] | None = None,
                       **kwargs: Any) -> dict[str, Any]:
        """Validate and persist one environment-side evaluation event."""
        if self.evaluation_reporter is None:
            raise RuntimeError('FluxVLAZMQEvalServer reporting was not '
                               'started')
        if not isinstance(payload, Mapping):
            payload = {}
        try:
            with self._report_lock:
                self._validate_parallel_capacity(event_type, payload)
                affinity_episode_id = self._affinity_release_target(
                    event_type, payload)
                result = self.evaluation_reporter.process_event(
                    event_type=event_type,
                    request_id=str(request_id),
                    run_session_id=str(run_id),
                    sequence=int(sequence),
                    payload=dict(payload),
                )
                if not isinstance(result, Mapping):
                    raise TypeError(
                        'Evaluation reporter process_event() must return a '
                        'mapping')
                if bool(result.get('accepted', False)):
                    if affinity_episode_id is not None:
                        self.policy.release_episode(affinity_episode_id)
                    elif (event_type == 'run_end' and getattr(
                            self.policy, 'supports_episode_affinity', False)):
                        self.policy.release_all()
            accepted = bool(result.get('accepted', False))
            return {
                'accepted': accepted,
                'error': str(result.get('error') or ''),
                'run_dir': str(result.get('run_dir') or ''),
                'duplicate': bool(result.get('duplicate', False)),
                'next_sequence': int(result.get('next_sequence', 1)),
                'status': str(result.get('status', 'idle')),
            }
        except Exception as exc:
            return {
                'accepted': False,
                'error': f'{type(exc).__name__}: {exc}',
                'run_dir': '',
                'duplicate': False,
                'next_sequence': 1,
                'status': 'error',
            }

    def _affinity_release_target(self, event_type: str,
                                 payload: Mapping[str, Any]) -> str | None:
        if not getattr(self.policy, 'supports_episode_affinity', False):
            return None
        if event_type != 'episode_end':
            return None
        episode_id = payload.get('episode_id')
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('episode_end payload.episode_id must be a string')
        return episode_id

    def _validate_parallel_capacity(self, event_type: str,
                                    payload: Mapping[str, Any]) -> None:
        if event_type != 'run_start' or not getattr(
                self.policy, 'supports_episode_affinity', False):
            return
        requested = payload.get('parallel_workers', 1)
        if isinstance(requested,
                      bool) or not isinstance(requested, (int, np.integer)):
            raise TypeError('run_start payload.parallel_workers must be an '
                            'integer')
        requested = int(requested)
        available = int(getattr(self.policy, 'worker_count', 1))
        if requested > available:
            raise ValueError(
                f'FluxThemis requested {requested} parallel simulator '
                f'workers, but FluxVLA has only {available} episode-affine '
                'inference replicas. Start FluxVLA with at least '
                f'--num-workers {requested}.')

    @staticmethod
    def _canonicalize_actions(actions: Any) -> np.ndarray:
        value = np.asarray(actions)
        if value.ndim == 3:
            if value.shape[0] != 1:
                raise ValueError(
                    'FluxVLA ZMQ inference only supports batch size 1, got '
                    f'{value.shape}')
            value = value[0]
        elif value.ndim == 1:
            value = value[None, :]
        elif value.ndim != 2:
            raise ValueError(
                f'FluxVLA actions must have shape [A], [T, A], or [1, T, A], '
                f'got {value.shape}')
        try:
            value = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError('FluxVLA actions must be numeric') from exc
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError('FluxVLA returned an empty action chunk')
        if not np.isfinite(value).all():
            raise ValueError('FluxVLA returned NaN or infinite actions')
        return value


def build_zmq_eval_server_from_config(
        cfg: Any,
        ckpt_path: str | None = None,
        device: str | None = None,
        host: str = '*',
        port: int | None = None,
        worker_devices: Sequence[str] | None = None,
        num_workers: int | None = None,
        config_path: str | None = None) -> FluxVLAZMQEvalServer:
    """Build a ZMQ evaluation server from one FluxVLA configuration.

    Mirrors ``build_ros_server_from_config`` but binds the transport-neutral
    policy/reporter/pool assembly to a ZMQ REP server. The reporting channel
    is enabled when ``themis.transport.report_service_name`` is set (the value
    itself is ignored by the ZMQ transport; it only acts as the switch).
    """
    from . import ros_server

    themis_cfg = ros_server._require_mapping(
        ros_server._config_get(cfg, 'themis', ros_server._MISSING),
        'config.themis')
    transport = dict(
        ros_server._require_mapping(
            themis_cfg.get('transport', ros_server._MISSING),
            'themis.transport'))
    server_cfg = dict(
        ros_server._require_mapping(
            themis_cfg.get('ros_server', ros_server._MISSING),
            'themis.ros_server'))

    section_name = server_cfg.get('dataset_section')
    if section_name not in {'eval', 'inference'}:
        raise ValueError('themis.ros_server.dataset_section must be `eval` '
                         'or `inference`')
    section_cfg = ros_server._require_mapping(
        ros_server._config_get(cfg, section_name, ros_server._MISSING),
        f'config.{section_name}')
    resolved_ckpt = ros_server._resolve_checkpoint_path(
        ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    stats_path = ros_server._resolve_statistics_path(
        server_cfg.get('norm_stats_path'), resolved_ckpt)
    model_outputs_environment_actions = server_cfg.get(
        'model_outputs_environment_actions', False)
    if not model_outputs_environment_actions and stats_path is None:
        raise FileNotFoundError(
            'dataset_statistics.json is required to prove the action unit '
            'contract')

    resolved_devices = ros_server._resolve_inference_devices(
        server_cfg=server_cfg,
        device=device,
        worker_devices=worker_devices,
        num_workers=num_workers,
    )
    resolved_device = resolved_devices[0]
    if len(resolved_devices) > 1 and transport.get(
            'report_service_name') is None:
        raise ValueError(
            'Multi-worker ZMQ inference requires '
            'themis.transport.report_service_name so episode affinity can '
            'be released after acknowledged episode_end events')

    evaluation_reporter = None
    if transport.get('report_service_name') is not None:
        reporting_cfg = dict(
            ros_server._require_mapping(
                server_cfg.get('evaluation_reporting', {}),
                'themis.ros_server.evaluation_reporting'))
        resolved_config_path = ros_server._resolve_report_config_path(
            cfg, config_path)
        result_root = ros_server._resolve_report_result_root(
            reporting_cfg.get('result_output_dir'), resolved_config_path)
        reporter_eval_source = ros_server._config_get(cfg, 'eval', section_cfg)
        reporter_eval_config = copy.deepcopy(
            dict(
                ros_server._require_mapping(reporter_eval_source,
                                            'config.eval metadata')))
        runner_cfg = themis_cfg.get('runner')
        if isinstance(runner_cfg, Mapping):
            if 'task_ids' in runner_cfg:
                reporter_eval_config.setdefault('task_ids',
                                                runner_cfg['task_ids'])
            if 'episodes_per_task' in runner_cfg:
                reporter_eval_config.setdefault(
                    'num_trials_per_task', runner_cfg['episodes_per_task'])
            if 'episodes_per_task_overrides' in runner_cfg:
                reporter_eval_config.setdefault(
                    'num_trials_per_task_overrides',
                    runner_cfg['episodes_per_task_overrides'])
            if 'run_name' in runner_cfg:
                reporter_eval_config.setdefault('run_name',
                                                runner_cfg['run_name'])
        reporter_eval_config.setdefault('dataset_section', section_name)
        if 'result_gpu_id' not in reporter_eval_config:
            configured_gpu_id = reporting_cfg.get('result_gpu_id')
            if configured_gpu_id is None:
                report_device = torch.device(resolved_device)
                configured_gpu_id = (
                    report_device.index if report_device.type == 'cuda'
                    and report_device.index is not None else 0)
            reporter_eval_config['result_gpu_id'] = configured_gpu_id

        from .evaluation_reporter import FluxVLAROSEvaluationReporter
        evaluation_reporter = FluxVLAROSEvaluationReporter(
            result_root=result_root,
            config_path=resolved_config_path,
            ckpt_path=resolved_ckpt,
            eval_config=reporter_eval_config,
            logger=None,
            feishu=reporting_cfg.get('feishu'),
            report_kind=reporting_cfg.get('report_kind'))

    workers_cfg = dict(
        ros_server._require_mapping(
            server_cfg.get('workers', {}), 'themis.ros_server.workers'))

    if len(resolved_devices) == 1:
        direct_policy = ros_server.build_ros_policy_from_config(
            cfg,
            ckpt_path=str(resolved_ckpt),
            device=resolved_device,
            service_name=transport.get('service_name'))
        if transport.get('report_service_name') is None:
            policy = direct_policy
        else:
            from .ros_worker_pool import EpisodeAffinityPolicyPool
            policy = EpisodeAffinityPolicyPool(
                backends=[direct_policy],
                lease_timeout_s=workers_cfg.get('lease_timeout_s', 900.0))
    else:
        from .ros_worker_pool import spawn_ros_policy_pool
        policy = spawn_ros_policy_pool(
            cfg=cfg,
            ckpt_path=str(resolved_ckpt),
            devices=resolved_devices,
            service_name=transport.get('service_name'),
            startup_timeout_s=workers_cfg.get('startup_timeout_s', 900.0),
            startup_parallelism=workers_cfg.get('startup_parallelism', 1),
            request_timeout_s=workers_cfg.get('request_timeout_s', 120.0),
            lease_timeout_s=workers_cfg.get('lease_timeout_s', 900.0))

    try:
        return FluxVLAZMQEvalServer(
            policy=policy,
            evaluation_reporter=evaluation_reporter,
            host=host,
            port=int(port) if port is not None else 5555)
    except BaseException:
        close = getattr(policy, 'close', None)
        if callable(close):
            close()
        raise
