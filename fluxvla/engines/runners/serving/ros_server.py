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
"""ROS inference and evaluation-report services used by FluxThemis.

ROS imports are intentionally delayed until :meth:`FluxVLAROSServer.run` so
normal FluxVLA training and evaluation do not require a sourced ROS workspace.
"""
from __future__ import annotations
import copy
import importlib
import json
import os
import random
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

_MISSING = object()


class _RawImageBridge:
    """Decode the ROS ``sensor_msgs/Image`` subset used by FluxThemis.

    ROS 1 Noetic's binary ``cv_bridge`` is tied to the Python ABI it was
    compiled against.  FluxThemis only sends packed 8-bit RGB/BGR images, so
    decoding that small, well-defined subset directly keeps a Python 3.10
    source-built ROS 1 workspace independent of ``cv_bridge``.
    """

    _SUPPORTED_ENCODINGS = frozenset({'rgb8', 'bgr8'})

    def __init__(self, image_type: type | None = None) -> None:
        self._image_type = image_type

    def imgmsg_to_cv2(self, message: Any, desired_encoding: str) -> np.ndarray:
        if self._image_type is not None and not isinstance(
                message, self._image_type):
            raise TypeError(
                'ROS image must be a sensor_msgs/Image instance, got '
                f'{type(message).__name__}')
        if desired_encoding not in self._SUPPORTED_ENCODINGS:
            raise ValueError(
                f'Unsupported desired image encoding {desired_encoding!r}: '
                'expected rgb8 or bgr8')

        encoding = getattr(message, 'encoding', None)
        if encoding not in self._SUPPORTED_ENCODINGS:
            raise ValueError(
                f'Unsupported ROS image encoding {encoding!r}: expected '
                'rgb8 or bgr8')
        if encoding != desired_encoding:
            raise ValueError(
                f'ROS image encoding {encoding!r} does not match requested '
                f'encoding {desired_encoding!r}')

        height = self._positive_integer_field(message, 'height')
        width = self._positive_integer_field(message, 'width')
        step = self._positive_integer_field(message, 'step')
        packed_step = width * 3
        if step < packed_step:
            raise ValueError(
                f'ROS image step {step} is smaller than the packed row size '
                f'{packed_step}')

        try:
            payload = bytes(getattr(message, 'data'))
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError('ROS image data must be a byte sequence') from exc
        expected_size = height * step
        if len(payload) != expected_size:
            raise ValueError(
                f'ROS image data has {len(payload)} bytes, expected exactly '
                f'{expected_size} for height={height} and step={step}')

        rows = np.frombuffer(payload, dtype=np.uint8).reshape(height, step)
        return rows[:, :packed_step].reshape(height, width, 3).copy()

    @staticmethod
    def _positive_integer_field(message: Any, name: str) -> int:
        value = getattr(message, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f'ROS image {name} must be an integer')
        value = int(value)
        if value <= 0:
            raise ValueError(f'ROS image {name} must be positive')
        return value


class FluxVLAROSPolicy:
    """Model, preprocessing and action-unit boundary for the ROS server."""

    def __init__(self,
                 vla: Any,
                 dataset: Any,
                 denormalize_action: Any = None,
                 device: str = 'cuda:0',
                 mixed_precision_dtype: torch.dtype = torch.bfloat16,
                 enable_mixed_precision: bool = True,
                 model_outputs_environment_actions: bool = False,
                 forward_seed: bool = False,
                 denormalize_context: Mapping[str, Any] | None = None,
                 denormalize_per_action: bool = False,
                 expected_unnorm_key: str = '') -> None:
        if not callable(dataset):
            raise TypeError('FluxVLA ROS server dataset must be callable')
        if (not model_outputs_environment_actions
                and not callable(denormalize_action)):
            raise RuntimeError(
                'A denormalize_action transform is required unless '
                'model_outputs_environment_actions=True is explicitly set.')
        if not isinstance(mixed_precision_dtype, torch.dtype):
            raise TypeError('mixed_precision_dtype must be a torch dtype')
        if mixed_precision_dtype not in {
                torch.float32, torch.float16, torch.bfloat16
        }:
            raise ValueError(
                'mixed_precision_dtype must be fp32, fp16, or bf16')
        if not isinstance(enable_mixed_precision, bool):
            raise TypeError('enable_mixed_precision must be a bool')
        if not isinstance(model_outputs_environment_actions, bool):
            raise TypeError('model_outputs_environment_actions must be a bool')
        if not isinstance(forward_seed, bool):
            raise TypeError('forward_seed must be a bool')
        if not isinstance(denormalize_per_action, bool):
            raise TypeError('denormalize_per_action must be a bool')
        if denormalize_context is not None and not isinstance(
                denormalize_context, Mapping):
            raise TypeError('denormalize_context must be a mapping')
        if not isinstance(expected_unnorm_key, str):
            raise TypeError('expected_unnorm_key must be a string')

        self.vla = vla
        self.dataset = dataset
        self.denormalize_action = denormalize_action
        self.device = torch.device(device)
        self.mixed_precision_dtype = mixed_precision_dtype
        self.enable_mixed_precision = enable_mixed_precision
        self.model_outputs_environment_actions = \
            model_outputs_environment_actions
        self.forward_seed = forward_seed
        self.denormalize_context = dict(denormalize_context or {})
        self.denormalize_per_action = denormalize_per_action
        self.expected_unnorm_key = expected_unnorm_key
        self._lock = threading.RLock()

        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.vla.eval()
        self.vla.to(self.device)

    def predict(self, observation: Mapping[str, Any], unnorm_key: str,
                seed: int) -> tuple[np.ndarray, float]:
        """Preprocess one request and return environment-unit ``[T, A]``."""
        if not isinstance(observation, Mapping):
            raise TypeError('observation must be a mapping')
        if (unnorm_key and self.expected_unnorm_key
                and unnorm_key != self.expected_unnorm_key):
            raise ValueError(
                f'Request unnorm_key {unnorm_key!r} does not match the '
                f'configured key {self.expected_unnorm_key!r}')
        unnorm_key = unnorm_key or self.expected_unnorm_key
        with self._lock, self._seed_context(seed):
            result = self.dataset(dict(observation))
            batch = result[0] if isinstance(result, tuple) else result
            if not isinstance(batch, Mapping):
                raise TypeError(
                    'FluxVLA dataset pipeline must return a mapping or a '
                    'tuple whose first item is a mapping')
            batch = dict(batch)
            batch.setdefault('reset_history',
                             bool(observation.get('is_new_episode')))
            if unnorm_key:
                batch['unnorm_key'] = unnorm_key
            if self.forward_seed:
                batch['seed'] = int(seed)
            batch = self._move_to_device(batch)

            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.no_grad(), self._autocast_context():
                raw_actions = self.vla.predict_action(**batch)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            inference_time_s = time.perf_counter() - started
            request_context = {}
            raw_state = getattr(self.dataset, 'last_raw_state', None)
            if raw_state is not None:
                request_context['state'] = np.asarray(
                    raw_state, dtype=np.float32).copy()
            actions = self._to_environment_actions(
                raw_actions,
                unnorm_key=unnorm_key,
                request_context=request_context)
        return actions, inference_time_s

    @contextmanager
    def _seed_context(self, seed: int):
        if not self.forward_seed:
            yield
            return

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cuda_devices = []
        if self.device.type == 'cuda':
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_devices.append(device_index)
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(int(seed))
                np.random.seed(int(seed) % (2**32))
                torch.random.default_generator.manual_seed(int(seed))
                if self.device.type == 'cuda':
                    torch.cuda.manual_seed(int(seed))
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _autocast_context(self):
        enabled = (
            self.enable_mixed_precision and self.device.type == 'cuda'
            and self.mixed_precision_dtype in {torch.bfloat16, torch.float16})
        if not enabled:
            return nullcontext()
        return torch.autocast(
            device_type='cuda', dtype=self.mixed_precision_dtype)

    def _to_environment_actions(
            self,
            raw_actions: Any,
            unnorm_key: str = '',
            request_context: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        raw_array = self._as_numpy(raw_actions)
        if self.model_outputs_environment_actions:
            return self._canonicalize_actions(raw_array)

        context = dict(self.denormalize_context)
        context.update(request_context or {})
        if unnorm_key:
            context.setdefault('unnorm_key', unnorm_key)
            context.setdefault('norm_stats_key', unnorm_key)
            context.setdefault('task_suite_name', unnorm_key)

        if self.denormalize_per_action:
            normalized = self._canonicalize_actions(raw_array)
            denormalized = []
            for action in normalized:
                value = self._call_denormalizer(action, context)
                value = np.asarray(value)
                if value.ndim == 2 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim != 1:
                    raise ValueError(
                        'Per-action denormalizer must return shape [A], got '
                        f'{value.shape}')
                denormalized.append(value)
            return self._canonicalize_actions(np.stack(denormalized))

        value = self._call_denormalizer(raw_array, context)
        return self._canonicalize_actions(value)

    def _call_denormalizer(self, actions: np.ndarray,
                           context: Mapping[str, Any]) -> np.ndarray:
        payload = dict(context)
        payload['action'] = actions
        value = self.denormalize_action(payload)
        if isinstance(value, Mapping):
            if 'action' not in value:
                raise KeyError(
                    'denormalize_action returned a mapping without `action`')
            value = value['action']
        return self._as_numpy(value)

    def _move_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, Mapping):
            return {
                key: self._move_to_device(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _canonicalize_actions(value: Any) -> np.ndarray:
        actions = np.asarray(value)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(
                    'FluxVLA ROS inference only supports batch size 1, got '
                    f'{actions.shape}')
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]
        elif actions.ndim != 2:
            raise ValueError('FluxVLA actions must have shape [A], [T, A], or '
                             f'[1, T, A], got {actions.shape}')
        try:
            actions = np.asarray(actions, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError('FluxVLA actions must be numeric') from exc
        if actions.shape[0] == 0 or actions.shape[1] == 0:
            raise ValueError('FluxVLA returned an empty action chunk')
        if not np.isfinite(actions).all():
            raise ValueError('FluxVLA returned NaN or infinite actions')
        return actions


class FluxVLAROSServer:
    """Expose policy inference and optional evaluation reporting over ROS 1."""

    def __init__(self,
                 policy: Any,
                 transport: Mapping[str, Any],
                 evaluation_reporter: Any = None,
                 node_name: str = 'fluxvla_inference_server',
                 executor_threads: int | None = None) -> None:
        if not isinstance(transport, Mapping):
            raise TypeError('themis.transport must be a mapping')
        self.policy = policy
        self.service_name = self._nonempty_string(
            transport.get('service_name', '/fluxvla/predict_action'),
            'themis.transport.service_name')
        self.image_keys = self._keys(
            transport.get('image_keys'), 'themis.transport.image_keys')
        self.state_keys = self._keys(
            transport.get('state_keys'), 'themis.transport.state_keys')
        self.image_encoding = self._nonempty_string(
            transport.get('image_encoding', 'rgb8'),
            'themis.transport.image_encoding')
        configured_report_service = transport.get('report_service_name')
        self.report_service_name = None
        if configured_report_service is not None:
            self.report_service_name = self._nonempty_string(
                configured_report_service,
                'themis.transport.report_service_name')
        if (self.report_service_name is None) != (evaluation_reporter is None):
            raise ValueError(
                'themis.transport.report_service_name and an evaluation '
                'reporter must be configured together')
        self.node_name = self._nonempty_string(node_name, 'node_name')
        if (executor_threads is not None
                and (isinstance(executor_threads, bool)
                     or not isinstance(executor_threads, int))):
            raise TypeError('executor_threads must be an integer or None')
        if executor_threads is not None and executor_threads < 2:
            raise ValueError('executor_threads must be at least 2')
        self.executor_threads = executor_threads
        self.evaluation_reporter = evaluation_reporter

        self._rospy = None
        self._bridge = None
        self._response_type = None
        self._report_response_type = None
        self._service = None
        self._report_service = None
        self._request_lock = threading.RLock()
        self._report_lock = threading.RLock()
        self._last_episode_id: str | None = None
        self._last_seed: int | None = None

    def run(self) -> None:
        """Import ROS, advertise the service and block in ``rospy.spin``."""
        try:
            rospy = importlib.import_module('rospy')
            sensor_msgs = importlib.import_module('sensor_msgs.msg')
            service_module = importlib.import_module('fluxthemis_msgs.srv')
            bridge = _RawImageBridge(getattr(sensor_msgs, 'Image'))
            service_type = getattr(service_module, 'PredictAction')
            response_type = getattr(service_module, 'PredictActionResponse')
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                'FluxVLA ROS serving requires ROS1 rospy, sensor_msgs/Image '
                'and the generated fluxthemis_msgs/PredictAction service. '
                'Source the ROS 1 workspace before launching the server.') \
                from exc

        report_service_type = None
        report_response_type = None
        if self.report_service_name is not None:
            try:
                report_service_type = getattr(service_module,
                                              'ReportEvaluation')
                report_response_type = getattr(service_module,
                                               'ReportEvaluationResponse')
            except AttributeError as exc:
                raise ImportError(
                    'FluxVLA evaluation reporting is configured, but the '
                    'generated fluxthemis_msgs/ReportEvaluation ROS 1 service '
                    'is unavailable. Rebuild and source the FluxThemis ROS 1 '
                    'interface workspace.') from exc

        rospy.init_node(self.node_name, anonymous=False)
        self._bind_ros(
            rospy,
            bridge,
            response_type,
            report_response_type=report_response_type,
        )
        self._service = rospy.Service(self.service_name, service_type,
                                      self.handle_request)
        rospy.loginfo(f'FluxVLA ROS inference ready on {self.service_name}')
        worker_count = getattr(self.policy, 'worker_count', 1)
        if worker_count > 1:
            devices = ', '.join(getattr(self.policy, 'worker_devices', ()))
            rospy.loginfo(
                f'FluxVLA inference pool: {worker_count} episode-affine '
                f'replicas on {devices}')
        if self.report_service_name is not None:
            self._report_service = rospy.Service(
                self.report_service_name,
                report_service_type,
                self.handle_report_request,
            )
            rospy.loginfo('FluxVLA evaluation reporting ready on '
                          f'{self.report_service_name}')
        rospy.spin()

    def _bind_ros(self,
                  rospy: Any,
                  bridge: Any,
                  response_type: Any,
                  report_response_type: Any = None) -> None:
        """Bind generated ROS types for dependency-free tests."""
        self._rospy = rospy
        self._bridge = bridge
        self._response_type = response_type
        self._report_response_type = report_response_type
        if self.evaluation_reporter is not None:
            set_logger = getattr(self.evaluation_reporter, 'set_logger', None)
            loginfo = getattr(rospy, 'loginfo', None)
            if callable(set_logger) and callable(loginfo):
                set_logger(loginfo)

    def handle_request(self, request: Any, response: Any = None) -> Any:
        if (self._rospy is None or self._bridge is None
                or self._response_type is None):
            raise RuntimeError('FluxVLAROSServer.run() was not called')
        if response is None:
            response = self._response_type()
        request_id = str(getattr(request, 'request_id', ''))
        response.request_id = request_id
        response.header.stamp = self._rospy.Time.now()
        try:
            observation, episode_id, seed, unnorm_key = \
                self._decode_request(request)
            if getattr(self.policy, 'supports_episode_affinity', False):
                actions, inference_time_s = self.policy.predict(
                    observation,
                    unnorm_key=unnorm_key,
                    seed=seed,
                    episode_id=episode_id,
                    reset=bool(request.reset),
                )
            else:
                with self._request_lock:
                    is_new_episode = (
                        bool(request.reset)
                        or episode_id != self._last_episode_id
                        or seed != self._last_seed)
                    observation['is_new_episode'] = is_new_episode
                    actions, inference_time_s = self.policy.predict(
                        observation, unnorm_key=unnorm_key, seed=seed)
                    self._last_episode_id = episode_id
                    self._last_seed = seed

            response.ok = True
            response.error = ''
            response.actions = actions.reshape(-1).tolist()
            response.action_horizon = int(actions.shape[0])
            response.action_dim = int(actions.shape[1])
            response.denormalized = True
            response.inference_time_s = float(inference_time_s)
        except Exception as exc:  # ROS callbacks must return a response.
            response.ok = False
            response.error = f'{type(exc).__name__}: {exc}'
            response.actions = []
            response.action_horizon = 0
            response.action_dim = 0
            response.denormalized = False
            response.inference_time_s = 0.0
            logerr = getattr(self._rospy, 'logerr', None)
            if callable(logerr):
                logerr('FluxVLA ROS request {} failed: {}'.format(
                    request_id or '<missing>', response.error))
        return response

    def handle_report_request(self, request: Any, response: Any = None) -> Any:
        """Validate and persist one environment-side evaluation event."""
        if (self._rospy is None or self._report_response_type is None
                or self.evaluation_reporter is None):
            raise RuntimeError(
                'FluxVLAROSServer evaluation reporting was not started')
        if response is None:
            response = self._report_response_type()
        request_id = str(getattr(request, 'request_id', ''))
        response.request_id = request_id
        response.header.stamp = self._rospy.Time.now()
        response.accepted = False
        response.error = ''
        response.run_dir = ''
        try:
            with self._report_lock:
                event = self._decode_report_request(request)
                self._validate_parallel_capacity(event)
                affinity_episode_id = self._affinity_release_target(event)
                result = self.evaluation_reporter.process_event(
                    event_type=event['event_type'],
                    request_id=event['request_id'],
                    run_session_id=event['run_id'],
                    sequence=event['sequence'],
                    payload=event['payload'],
                )
                if not isinstance(result, Mapping):
                    raise TypeError(
                        'Evaluation reporter process_event() must return a '
                        'mapping')
                if bool(result.get('accepted', False)):
                    if affinity_episode_id is not None:
                        self.policy.release_episode(affinity_episode_id)
                    elif (event['event_type'] == 'run_end' and getattr(
                            self.policy, 'supports_episode_affinity', False)):
                        self.policy.release_all()
            response.accepted = bool(result.get('accepted', False))
            response.error = str(result.get('error') or '')
            response.run_dir = str(result.get('run_dir') or '')
            if not response.accepted and not response.error:
                response.error = 'Evaluation reporter rejected the event'
        # ROS callbacks must always return a response.
        except Exception as exc:
            response.accepted = False
            response.error = f'{type(exc).__name__}: {exc}'
            response.run_dir = ''

        if not response.accepted:
            logerr = getattr(self._rospy, 'logerr', None)
            if callable(logerr):
                logerr('FluxVLA evaluation report {} rejected: {}'.format(
                    request_id or '<missing>', response.error))
        return response

    def close(self) -> None:
        """Release inference workers owned by this server."""
        close = getattr(self.policy, 'close', None)
        if callable(close):
            close()

    def _affinity_release_target(self, event: Mapping[str, Any]) -> str | None:
        if not getattr(self.policy, 'supports_episode_affinity', False):
            return None
        if event.get('event_type') != 'episode_end':
            return None
        payload = event.get('payload')
        if not isinstance(payload, Mapping):
            raise TypeError('episode_end payload must be a mapping')
        episode_id = payload.get('episode_id')
        return self._nonempty_string(
            episode_id,
            'episode_end payload.episode_id',
        )

    def _validate_parallel_capacity(self, event: Mapping[str, Any]) -> None:
        if (event.get('event_type') != 'run_start' or
                not getattr(self.policy, 'supports_episode_affinity', False)):
            return
        payload = event.get('payload')
        if not isinstance(payload, Mapping):
            raise TypeError('run_start payload must be a mapping')
        requested = payload.get('parallel_workers', 1)
        if isinstance(requested,
                      bool) or not isinstance(requested, (int, np.integer)):
            raise TypeError(
                'run_start payload.parallel_workers must be an integer')
        requested = int(requested)
        if requested < 1:
            raise ValueError(
                'run_start payload.parallel_workers must be positive')
        available = int(getattr(self.policy, 'worker_count', 1))
        if requested > available:
            raise ValueError(
                f'FluxThemis requested {requested} parallel simulator '
                f'workers, but FluxVLA has only {available} episode-affine '
                'inference replicas. Start FluxVLA with at least '
                f'--num-workers {requested}, or reduce FluxThemis --workers.')

    def _decode_report_request(self, request: Any) -> dict[str, Any]:
        request_id = self._nonempty_string(
            getattr(request, 'request_id', ''), 'request.request_id')
        run_id = self._nonempty_string(
            getattr(request, 'run_id', ''), 'request.run_id')
        event_type = self._nonempty_string(
            getattr(request, 'event_type', ''), 'request.event_type')
        sequence = getattr(request, 'sequence', None)
        if (isinstance(sequence, bool)
                or not isinstance(sequence, (int, np.integer))):
            raise TypeError('request.sequence must be an integer')
        sequence = int(sequence)
        if sequence < 0:
            raise ValueError('request.sequence must be non-negative')
        payload_json = self._nonempty_string(
            getattr(request, 'payload_json', ''), 'request.payload_json')
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError('request.payload_json must contain valid JSON') \
                from exc
        if not isinstance(payload, Mapping):
            raise TypeError('request.payload_json must encode a JSON object')
        return {
            'request_id': request_id,
            'run_id': run_id,
            'sequence': sequence,
            'event_type': event_type,
            'payload': dict(payload),
        }

    def _decode_request(self,
                        request: Any) -> tuple[dict[str, Any], str, int, str]:
        request_id = self._nonempty_string(
            getattr(request, 'request_id', ''), 'request.request_id')
        del request_id
        episode_id = self._nonempty_string(
            getattr(request, 'episode_id', ''), 'request.episode_id')
        prompt = self._nonempty_string(
            getattr(request, 'prompt', ''), 'request.prompt')

        image_names = list(getattr(request, 'image_names', []))
        image_messages = list(getattr(request, 'images', []))
        if tuple(image_names) != self.image_keys:
            raise ValueError(f'Expected image_names {self.image_keys}, got '
                             f'{tuple(image_names)}')
        if len(image_messages) != len(image_names):
            raise ValueError(
                'request.images and request.image_names must have the same '
                'length')

        observation: dict[str, Any] = {}
        for name, message in zip(image_names, image_messages):
            image = self._bridge.imgmsg_to_cv2(
                message, desired_encoding=self.image_encoding)
            observation[name] = self._validate_image(image, name)

        state_names = list(getattr(request, 'state_names', []))
        state_sizes = list(getattr(request, 'state_sizes', []))
        state_values = np.asarray(
            getattr(request, 'state_values', []), dtype=np.float32)
        if tuple(state_names) != self.state_keys:
            raise ValueError(f'Expected state_names {self.state_keys}, got '
                             f'{tuple(state_names)}')
        if len(state_sizes) != len(state_names):
            raise ValueError(
                'request.state_sizes and request.state_names must have the '
                'same length')
        if state_values.ndim != 1:
            raise ValueError('request.state_values must be flat')
        if not np.isfinite(state_values).all():
            raise ValueError('request.state_values contains NaN or infinity')
        offset = 0
        for name, size_value in zip(state_names, state_sizes):
            size = int(size_value)
            if size <= 0:
                raise ValueError('Every request.state_sizes entry must be '
                                 'positive')
            end = offset + size
            if end > state_values.size:
                raise ValueError(
                    'request.state_values is shorter than state_sizes')
            observation[name] = state_values[offset:end].copy()
            offset = end
        if offset != state_values.size:
            raise ValueError(
                'request.state_values length does not equal sum(state_sizes)')
        observation['task_description'] = prompt

        seed = int(getattr(request, 'seed'))
        unnorm_key = getattr(request, 'unnorm_key', '')
        if not isinstance(unnorm_key, str):
            raise TypeError('request.unnorm_key must be a string')
        return observation, episode_id, seed, unnorm_key

    @staticmethod
    def _validate_image(value: Any, name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.dtype != np.uint8:
            raise TypeError(
                f'Image {name!r} must have dtype uint8, got {image.dtype}')
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'Image {name!r} must have shape [H, W, 3], got '
                             f'{image.shape}')
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f'Image {name!r} cannot be empty')
        return np.ascontiguousarray(image)

    @classmethod
    def _keys(cls, value: Any, name: str) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f'{name} must be a sequence')
        keys = tuple(value)
        if not keys:
            raise ValueError(f'{name} cannot be empty')
        for key in keys:
            cls._nonempty_string(key, f'{name} entry')
        if len(keys) != len(set(keys)):
            raise ValueError(f'{name} cannot contain duplicates')
        return keys

    @staticmethod
    def _nonempty_string(value: Any, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f'{name} must be a string')
        if not value:
            raise ValueError(f'{name} cannot be empty')
        return value


def build_ros_policy_from_config(
        cfg: Any,
        ckpt_path: str | None = None,
        device: str | None = None,
        service_name: str | None = None) -> FluxVLAROSPolicy:
    """Build one complete policy replica from an authoritative config."""
    themis_cfg = _require_mapping(
        _config_get(cfg, 'themis', _MISSING), 'config.themis')
    transport = dict(
        _require_mapping(
            themis_cfg.get('transport', _MISSING), 'themis.transport'))
    server_cfg = dict(
        _require_mapping(
            themis_cfg.get('ros_server', _MISSING), 'themis.ros_server'))
    if service_name is not None:
        transport['service_name'] = service_name

    section_name = server_cfg.get('dataset_section')
    if section_name not in {'eval', 'inference'}:
        raise ValueError('themis.ros_server.dataset_section must be `eval` or '
                         '`inference`')
    section_cfg = _require_mapping(
        _config_get(cfg, section_name, _MISSING), f'config.{section_name}')
    dataset_cfg = copy.deepcopy(
        _require_mapping(
            section_cfg.get('dataset', _MISSING),
            f'config.{section_name}.dataset'))
    resolved_ckpt = _resolve_checkpoint_path(
        ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    stats_path = _resolve_statistics_path(
        server_cfg.get('norm_stats_path'), resolved_ckpt)
    model_outputs_environment_actions = server_cfg.get(
        'model_outputs_environment_actions', False)
    if not model_outputs_environment_actions and stats_path is None:
        raise FileNotFoundError(
            'dataset_statistics.json is required to prove the ROS action '
            'unit contract')

    from fluxvla.engines import (build_dataset_from_cfg,
                                 build_transform_from_cfg, build_vla_from_cfg)
    from fluxvla.engines.utils import str_to_dtype

    model_cfg = _config_get(cfg, 'inference_model', None)
    if model_cfg is None:
        model_cfg = _config_get(cfg, 'model', _MISSING)
    if model_cfg is _MISSING:
        raise KeyError('FluxVLA config must define `model` or '
                       '`inference_model`')
    vla = build_vla_from_cfg(model_cfg)
    checkpoint_model_cfg = _config_get(cfg, 'model', model_cfg)
    _load_checkpoint(
        vla,
        resolved_ckpt,
        name_mapping=_config_get(checkpoint_model_cfg, 'name_mapping'),
    )
    if stats_path is not None:
        with stats_path.open('r', encoding='utf-8') as stream:
            vla.norm_stats = json.load(stream)

    _prepare_dataset_config(
        dataset_cfg=dataset_cfg,
        section_cfg=section_cfg,
        transport=transport,
        stats_path=stats_path,
        model_root=resolved_ckpt.parent.parent,
    )
    dataset = build_dataset_from_cfg(dataset_cfg)

    denormalize_action = None
    denormalize_context = dict(
        _require_mapping(
            server_cfg.get('denormalize_context', {}),
            'themis.ros_server.denormalize_context'))
    if not model_outputs_environment_actions:
        denorm_cfg = copy.deepcopy(
            _require_mapping(
                section_cfg.get('denormalize_action', _MISSING),
                f'config.{section_name}.denormalize_action'))
        denorm_cfg['norm_stats'] = str(stats_path)
        denormalize_action = build_transform_from_cfg(denorm_cfg)
        _prepare_denormalize_context(denormalize_context, denorm_cfg,
                                     section_cfg, transport)
        _validate_denormalization_stats(denormalize_action,
                                        denormalize_context)

    resolved_device = device or server_cfg.get('device', 'cuda:0')
    dtype_name = server_cfg.get('mixed_precision_dtype', 'bf16')
    return FluxVLAROSPolicy(
        vla=vla,
        dataset=dataset,
        denormalize_action=denormalize_action,
        device=resolved_device,
        mixed_precision_dtype=str_to_dtype(str(dtype_name)),
        enable_mixed_precision=server_cfg.get('enable_mixed_precision', True),
        model_outputs_environment_actions=model_outputs_environment_actions,
        forward_seed=server_cfg.get('forward_seed', False),
        denormalize_context=denormalize_context,
        denormalize_per_action=server_cfg.get('denormalize_per_action', False),
        expected_unnorm_key=str(transport.get('unnorm_key', '')),
    )


def build_ros_server_from_config(
        cfg: Any,
        ckpt_path: str | None = None,
        device: str | None = None,
        worker_devices: Sequence[str] | None = None,
        num_workers: int | None = None,
        service_name: str | None = None,
        node_name: str | None = None,
        ros_version: int | str | None = None,
        config_path: str | os.PathLike | None = None) -> FluxVLAROSServer:
    """Build a ROS 1 or ROS 2 server from one FluxVLA configuration.

    ``ros_version`` overrides ``themis.ros_server.ros_version``.  ROS 1 stays
    the default so existing launch commands and configurations are unchanged.
    """
    themis_cfg = _require_mapping(
        _config_get(cfg, 'themis', _MISSING), 'config.themis')
    transport = dict(
        _require_mapping(
            themis_cfg.get('transport', _MISSING), 'themis.transport'))
    server_cfg = dict(
        _require_mapping(
            themis_cfg.get('ros_server', _MISSING), 'themis.ros_server'))
    configured_ros_version = server_cfg.get('ros_version', 1)
    requested_ros_version = (
        configured_ros_version if ros_version is None else ros_version)
    resolved_ros_version = _resolve_ros_version(requested_ros_version)
    if service_name is not None:
        transport['service_name'] = service_name

    section_name = server_cfg.get('dataset_section')
    if section_name not in {'eval', 'inference'}:
        raise ValueError('themis.ros_server.dataset_section must be `eval` or '
                         '`inference`')
    section_cfg = _require_mapping(
        _config_get(cfg, section_name, _MISSING), f'config.{section_name}')
    resolved_ckpt = _resolve_checkpoint_path(
        ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    stats_path = _resolve_statistics_path(
        server_cfg.get('norm_stats_path'), resolved_ckpt)
    model_outputs_environment_actions = server_cfg.get(
        'model_outputs_environment_actions', False)
    if not model_outputs_environment_actions and stats_path is None:
        raise FileNotFoundError(
            'dataset_statistics.json is required to prove the ROS action '
            'unit contract')
    resolved_devices = _resolve_inference_devices(
        server_cfg=server_cfg,
        device=device,
        worker_devices=worker_devices,
        num_workers=num_workers,
    )
    resolved_device = resolved_devices[0]
    if (len(resolved_devices) > 1
            and transport.get('report_service_name') is None):
        raise ValueError(
            'Multi-worker ROS inference requires '
            'themis.transport.report_service_name so episode affinity can be '
            'released after acknowledged episode_end events')
    evaluation_reporter = None
    if transport.get('report_service_name') is not None:
        reporting_cfg = _require_mapping(
            server_cfg.get('evaluation_reporting', {}),
            'themis.ros_server.evaluation_reporting',
        )
        resolved_config_path = _resolve_report_config_path(cfg, config_path)
        result_root = _resolve_report_result_root(
            reporting_cfg.get('result_output_dir'),
            resolved_config_path,
        )
        reporter_eval_source = _config_get(cfg, 'eval', section_cfg)
        reporter_eval_config = copy.deepcopy(
            dict(
                _require_mapping(reporter_eval_source,
                                 'config.eval metadata')))
        runner_cfg = themis_cfg.get('runner')
        if isinstance(runner_cfg, Mapping):
            if 'task_ids' in runner_cfg:
                reporter_eval_config.setdefault('task_ids',
                                                runner_cfg['task_ids'])
            if 'episodes_per_task' in runner_cfg:
                reporter_eval_config.setdefault(
                    'num_trials_per_task', runner_cfg['episodes_per_task'])
            if 'run_name' in runner_cfg:
                reporter_eval_config.setdefault('run_name',
                                                runner_cfg['run_name'])
        reporter_eval_config.setdefault('dataset_section', section_name)
        reporter_eval_config.setdefault('service_name',
                                        transport.get('service_name'))
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
            report_kind=reporting_cfg.get('report_kind'),
        )

    workers_cfg = dict(
        _require_mapping(
            server_cfg.get('workers', {}),
            'themis.ros_server.workers',
        ))
    if len(resolved_devices) == 1:
        direct_policy = build_ros_policy_from_config(
            cfg,
            ckpt_path=str(resolved_ckpt),
            device=resolved_device,
            service_name=transport.get('service_name'),
        )
        if transport.get('report_service_name') is None:
            policy = direct_policy
        else:
            # Reporting supplies the durable episode_end signal.  Use the
            # same affinity contract for one replica so concurrent simulator
            # clients cannot interleave stateful policy history.
            from .ros_worker_pool import EpisodeAffinityPolicyPool
            policy = EpisodeAffinityPolicyPool(
                backends=[direct_policy],
                lease_timeout_s=workers_cfg.get('lease_timeout_s', 900.0),
            )
    else:
        from .ros_worker_pool import spawn_ros_policy_pool
        policy = spawn_ros_policy_pool(
            cfg=cfg,
            ckpt_path=str(resolved_ckpt),
            devices=resolved_devices,
            service_name=transport.get('service_name'),
            startup_timeout_s=workers_cfg.get('startup_timeout_s', 900.0),
            request_timeout_s=workers_cfg.get('request_timeout_s', 120.0),
            lease_timeout_s=workers_cfg.get('lease_timeout_s', 900.0),
        )
    try:
        server_type = FluxVLAROSServer
        if resolved_ros_version == 2:
            from .ros2_server import FluxVLAROS2Server
            server_type = FluxVLAROS2Server
        return server_type(
            policy=policy,
            transport=transport,
            evaluation_reporter=evaluation_reporter,
            node_name=node_name
            or server_cfg.get('node_name', 'fluxvla_inference_server'),
            executor_threads=server_cfg.get('executor_threads'),
        )
    except BaseException:
        close = getattr(policy, 'close', None)
        if callable(close):
            close()
        raise


def _resolve_inference_devices(server_cfg: Mapping[str,
                                                   Any], device: str | None,
                               worker_devices: Sequence[str] | None,
                               num_workers: int | None) -> tuple[str, ...]:
    """Resolve CLI/config worker devices while preserving legacy defaults."""
    workers_cfg = _require_mapping(
        server_cfg.get('workers', {}),
        'themis.ros_server.workers',
    )
    if worker_devices is not None and num_workers is not None:
        raise ValueError(
            'worker_devices and num_workers are mutually exclusive')

    values: Any
    if worker_devices is not None:
        values = worker_devices
    elif num_workers is not None:
        count = _positive_worker_count(num_workers, 'num_workers')
        if count == 1:
            values = [device or server_cfg.get('device', 'cuda:0')]
        else:
            if device is not None:
                raise ValueError(
                    '--device cannot be combined with --num-workers greater '
                    'than one; use --devices for explicit placement')
            values = [f'cuda:{index}' for index in range(count)]
    elif device is not None:
        values = [device]
    elif workers_cfg.get('devices') is not None:
        values = workers_cfg['devices']
    elif workers_cfg.get('num_workers') is not None:
        count = _positive_worker_count(
            workers_cfg['num_workers'],
            'themis.ros_server.workers.num_workers',
        )
        values = [f'cuda:{index}' for index in range(count)]
    else:
        values = [server_cfg.get('device', 'cuda:0')]

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError('inference worker devices must be a sequence')
    if not values:
        raise ValueError('inference worker devices cannot be empty')
    normalized = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(
                f'inference worker device {index} must be a non-empty string')
        value = value.strip()
        try:
            torch.device(value)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f'Invalid inference worker device {value!r}') from exc
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError('inference worker devices cannot contain duplicates')
    return tuple(normalized)


def _positive_worker_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an integer')
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _resolve_ros_version(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError('ROS version must be 1 or 2, not a bool')
    if isinstance(value, str):
        value = value.strip()
    elif not isinstance(value, int):
        raise TypeError('ROS version must be an integer or string')
    if value == 1 or value == '1':
        return 1
    if value == 2 or value == '2':
        return 2
    raise ValueError('ROS version must be 1 or 2')


def _resolve_report_config_path(
        cfg: Any, explicit_path: str | os.PathLike | None) -> Path:
    value = explicit_path
    if value is None:
        value = getattr(cfg, 'filename', None)
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(
            'Evaluation reporting requires the authoritative FluxVLA config '
            'path')
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'FluxVLA config not found: {path}')
    return path


def _resolve_report_result_root(value: Any, config_path: Path) -> Path:
    fluxvla_root = _find_fluxvla_root(config_path)
    work_dirs_root = (fluxvla_root / 'work_dirs').resolve()
    if value is None:
        candidate = work_dirs_root / 'fluxthemis'
    else:
        if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
            raise TypeError(
                'themis.ros_server.evaluation_reporting.result_output_dir '
                'must be a non-empty path')
        requested = Path(value).expanduser()
        if requested.is_absolute():
            candidate = requested
        elif requested.parts and requested.parts[0] == 'work_dirs':
            candidate = fluxvla_root / requested
        else:
            candidate = work_dirs_root / requested
    resolved = candidate.resolve()
    try:
        resolved.relative_to(work_dirs_root)
    except ValueError as exc:
        raise ValueError(
            'Evaluation reporting artifacts must stay inside FluxVLA '
            f'work_dirs: {work_dirs_root}') from exc
    return resolved


def _find_fluxvla_root(config_path: Path) -> Path:
    for candidate in config_path.parent.parents:
        if ((candidate / 'fluxvla' / '__init__.py').is_file()
                and (candidate / 'configs').is_dir()):
            try:
                config_path.relative_to((candidate / 'configs').resolve())
            except ValueError:
                continue
            return candidate.resolve()
    raise ValueError(
        'Evaluation reporting config must be located under a FluxVLA '
        f'configs directory: {config_path}')


def _resolve_checkpoint_path(value: Any) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(
            'Checkpoint path is required via --ckpt-path, '
            'themis.ros_server.ckpt_path, or the selected section.ckpt_path')
    path = Path(value).expanduser().resolve()
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def _resolve_statistics_path(value: Any, checkpoint_path: Path) -> Path | None:
    explicit = value is not None
    path = (
        Path(value).expanduser().resolve() if explicit else
        checkpoint_path.parent.parent / 'dataset_statistics.json')
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(
                f'Configured norm_stats_path not found: {path}')
        return None
    return path


def _load_checkpoint(vla: Any,
                     checkpoint_path: Path,
                     name_mapping: Mapping[str, str] | None = None) -> None:
    if checkpoint_path.is_dir():
        from safetensors.torch import load_file
        shards = sorted(checkpoint_path.glob('model-*.safetensors'))
        if not shards:
            raise FileNotFoundError(
                f'No model-*.safetensors files found in {checkpoint_path}')
        state_dict = {}
        for shard in shards:
            state_dict.update(load_file(str(shard), device='cpu'))
    elif checkpoint_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path), device='cpu')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = (
            checkpoint['model'] if isinstance(checkpoint, Mapping)
            and 'model' in checkpoint else checkpoint)

    if name_mapping:
        if not isinstance(name_mapping, Mapping):
            raise TypeError('model.name_mapping must be a mapping')

        def has_prefix(key: str, prefix: str) -> bool:
            return key == prefix or key.startswith(f'{prefix}.')

        native_prefixes = tuple(name_mapping)
        source_prefixes = tuple(name_mapping.values())
        has_native_keys = any(
            any(has_prefix(key, prefix) for key in state_dict)
            for prefix in native_prefixes)
        needs_mapping = (
            any(
                any(has_prefix(key, prefix) for key in state_dict)
                for prefix in source_prefixes) and not has_native_keys)
        if needs_mapping:
            mapped_state_dict = {}
            for native_prefix, source_prefix in name_mapping.items():
                if not isinstance(native_prefix, str) or not isinstance(
                        source_prefix, str):
                    raise TypeError(
                        'model.name_mapping keys and values must be strings')
                for key, value in state_dict.items():
                    if has_prefix(key, source_prefix):
                        mapped_key = native_prefix + key[len(source_prefix):]
                        mapped_state_dict[mapped_key] = value
            state_dict = mapped_state_dict

    from fluxvla.engines.utils.checkpoint_utils import handle_shared_tensors
    state_dict = handle_shared_tensors(state_dict, vla.state_dict())
    vla.load_state_dict(state_dict, strict=True)


def _prepare_dataset_config(dataset_cfg: dict, section_cfg: Mapping[str, Any],
                            transport: Mapping[str, Any],
                            stats_path: Path | None, model_root: Path) -> None:
    if stats_path is not None:
        dataset_cfg['norm_stats'] = str(stats_path)
    dataset_type = dataset_cfg.get('type', '')
    dataset_type_name = (
        dataset_type if isinstance(dataset_type, str) else getattr(
            dataset_type, '__name__', str(dataset_type)))
    task_suite_name = section_cfg.get('task_suite_name')

    if 'Libero' in dataset_type_name:
        if not task_suite_name:
            raise KeyError(
                'Libero ROS serving requires section.task_suite_name')
        dataset_cfg.setdefault('task_suite_name', task_suite_name)
        dataset_cfg.setdefault(
            'norm_stats_key',
            section_cfg.get('norm_stats_key') or f'{task_suite_name}_no_noops')
    if 'PrivateInferenceDataset' in dataset_type_name:
        dataset_cfg.setdefault('model_path', str(model_root))
    if 'Robocasa' in dataset_type_name:
        unnorm_key = transport.get('unnorm_key')
        if unnorm_key:
            dataset_cfg.setdefault('unnorm_key', unnorm_key)


def _prepare_denormalize_context(context: dict[str,
                                               Any], denorm_cfg: Mapping[str,
                                                                         Any],
                                 section_cfg: Mapping[str, Any],
                                 transport: Mapping[str, Any]) -> None:
    transform_type = denorm_cfg.get('type', '')
    transform_name = (
        transform_type if isinstance(transform_type, str) else getattr(
            transform_type, '__name__', str(transform_type)))
    if 'Robocasa' in transform_name:
        section_key = section_cfg.get('unnorm_key')
        transport_key = transport.get('unnorm_key')
        if section_key and transport_key and section_key != transport_key:
            raise ValueError('RoboCasa section.unnorm_key and '
                             'themis.transport.unnorm_key must match')
        stats_key = transport_key or section_key
        if not stats_key:
            raise KeyError(
                'RoboCasa ROS serving requires an unnorm_key in the eval '
                'section or themis.transport')
        configured_key = context.get('task_suite_name')
        if configured_key and configured_key != stats_key:
            raise ValueError(
                'RoboCasa denormalize_context.task_suite_name must match '
                'the configured unnorm_key')
        context['task_suite_name'] = stats_key
        return
    task_suite_name = section_cfg.get('task_suite_name')
    if task_suite_name:
        context.setdefault('task_suite_name', task_suite_name)
    if 'Libero' in transform_name:
        context.setdefault(
            'norm_stats_key',
            section_cfg.get('norm_stats_key')
            or (f'{task_suite_name}_no_noops'
                if task_suite_name else transport.get('unnorm_key', '')))


def _validate_denormalization_stats(transform: Any,
                                    context: Mapping[str, Any]) -> None:
    norm_stats = getattr(transform, 'norm_stats', None)
    transform_name = type(transform).__name__
    if 'Libero' in transform_name:
        stats_key = context.get('norm_stats_key')
    else:
        stats_key = (
            context.get('task_suite_name') or context.get('norm_stats_key'))
    if not isinstance(norm_stats, Mapping) or not stats_key:
        return
    if stats_key not in norm_stats:
        raise KeyError(
            f'Normalization statistics key {stats_key!r} is missing. '
            f'available keys: {list(norm_stats)}')
    stats = norm_stats[stats_key]
    if not isinstance(stats, Mapping) or 'action' not in stats:
        raise KeyError(
            f'Normalization statistics {stats_key!r} must contain action')


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is _MISSING:
        raise KeyError(f'{name} is required')
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    return value
