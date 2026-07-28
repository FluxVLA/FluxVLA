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

import time
import traceback
from dataclasses import dataclass
from threading import Event, Lock, Thread
from types import SimpleNamespace

import numpy as np
import torch

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .oli_inference_runner import OliInferenceRunner

overwatch = initialize_overwatch(__name__)


@dataclass
class OliRTCInferenceRequest:
    """Snapshot of the active Oli action chunk at inference start."""

    prefix_actions: np.ndarray | None
    prefix_len: int
    remaining_actions: int
    active_chunk_id: int | None
    active_index_at_request: int


class OliRTCChunkScheduler:
    """Oli-specific active/pending scheduler for asynchronous RTC.

    Oli publishes at the model action rate and does not interpolate actions,
    so normalized model actions and denormalized robot actions share the same
    time index. Keeping this scheduler local avoids coupling Oli execution to
    the Teleop02 runner or its WBT interpolation/state-machine behavior.
    """

    def __init__(self, remaining_actions_threshold: int,
                 rtc_prefix_length: int):
        self.remaining_actions_threshold = max(
            0, int(remaining_actions_threshold))
        self.rtc_prefix_length = max(0, int(rtc_prefix_length))
        self.lock = Lock()

        self.active_original: np.ndarray | None = None
        self.active_processed: np.ndarray | None = None
        self.active_index = 0
        self.active_chunk_id: int | None = None

        self.pending_original: np.ndarray | None = None
        self.pending_processed: np.ndarray | None = None
        self.pending_start_index = 0
        self.pending_activation_index: int | None = None
        self.pending_chunk_id: int | None = None

        self._inference_in_flight = False
        self._last_commit_debug = {}

    def _remaining_active_locked(self) -> int:
        if self.active_processed is None:
            return 0
        return max(0, len(self.active_processed) - self.active_index)

    def _pending_executable_locked(self) -> int:
        if self.pending_processed is None:
            return 0
        return max(0, len(self.pending_processed) - self.pending_start_index)

    def _remaining_before_switch_locked(self) -> int:
        if self.active_processed is None:
            return 0
        if (self.pending_processed is None
                or self.pending_activation_index is None):
            return self._remaining_active_locked()
        return max(0, self.pending_activation_index - self.active_index)

    def _activate_pending_locked(self) -> None:
        if self.pending_processed is None or self.pending_original is None:
            self.active_original = None
            self.active_processed = None
            self.active_index = 0
            self.active_chunk_id = None
            return

        self.active_original = self.pending_original
        self.active_processed = self.pending_processed
        self.active_index = min(self.pending_start_index,
                                len(self.active_processed))
        self.active_chunk_id = self.pending_chunk_id

        self.pending_original = None
        self.pending_processed = None
        self.pending_start_index = 0
        self.pending_activation_index = None
        self.pending_chunk_id = None

    def prepare_inference_request(
            self, use_rtc: bool) -> OliRTCInferenceRequest | None:
        with self.lock:
            if self._inference_in_flight:
                return None

            remaining_actions = self._remaining_active_locked()
            should_request = False
            if self.active_processed is None:
                should_request = True
            elif remaining_actions <= 0:
                should_request = self.pending_processed is None
            elif (self.pending_processed is None
                  and remaining_actions <= self.remaining_actions_threshold):
                if use_rtc and remaining_actions < self.rtc_prefix_length:
                    return None
                should_request = True

            if not should_request:
                return None

            prefix_actions = None
            prefix_len = 0
            if (use_rtc and self.active_original is not None
                    and remaining_actions >= self.rtc_prefix_length
                    and self.rtc_prefix_length > 0):
                start = self.active_index
                stop = start + self.rtc_prefix_length
                prefix_actions = self.active_original[start:stop].copy()
                prefix_len = len(prefix_actions)
                if prefix_len == 0:
                    prefix_actions = None

            self._inference_in_flight = True
            return OliRTCInferenceRequest(
                prefix_actions=prefix_actions,
                prefix_len=prefix_len,
                remaining_actions=remaining_actions,
                active_chunk_id=self.active_chunk_id,
                active_index_at_request=self.active_index,
            )

    def commit_inference_result(self,
                                request: OliRTCInferenceRequest,
                                original_actions: np.ndarray,
                                processed_actions: np.ndarray,
                                chunk_id: int | None = None) -> bool:
        with self.lock:
            self._inference_in_flight = False
            if request.active_chunk_id != self.active_chunk_id:
                self._last_commit_debug = {
                    'reason': 'active_chunk_changed',
                    'request_active_chunk_id': request.active_chunk_id,
                    'current_active_chunk_id': self.active_chunk_id,
                    'request_prefix_len': request.prefix_len,
                    'request_active_index': request.active_index_at_request,
                    'current_active_index': self.active_index,
                }
                return False

            clipped_prefix_len = min(
                max(0, request.prefix_len), len(original_actions))
            executed_since_request = max(
                0, self.active_index - request.active_index_at_request)
            if executed_since_request > clipped_prefix_len:
                self._last_commit_debug = {
                    'reason': 'prefix_window_passed',
                    'request_prefix_len': request.prefix_len,
                    'clipped_prefix_len': clipped_prefix_len,
                    'executed_since_request': executed_since_request,
                    'request_active_index': request.active_index_at_request,
                    'current_active_index': self.active_index,
                }
                return False

            if executed_since_request == clipped_prefix_len:
                self.active_original = original_actions.copy()
                self.active_processed = processed_actions.copy()
                self.active_index = clipped_prefix_len
                self.active_chunk_id = chunk_id
                self.pending_original = None
                self.pending_processed = None
                self.pending_start_index = 0
                self.pending_activation_index = None
                self.pending_chunk_id = None
                self._last_commit_debug = {
                    'reason': 'accepted_replace_active',
                    'request_prefix_len': request.prefix_len,
                    'executed_since_request': executed_since_request,
                    'active_index': self.active_index,
                }
                return True

            self.pending_original = original_actions.copy()
            self.pending_processed = processed_actions.copy()
            self.pending_start_index = clipped_prefix_len
            self.pending_activation_index = (
                request.active_index_at_request + clipped_prefix_len)
            self.pending_chunk_id = chunk_id
            self._last_commit_debug = {
                'reason': 'accepted_pending',
                'request_prefix_len': request.prefix_len,
                'executed_since_request': executed_since_request,
                'pending_start_index': self.pending_start_index,
                'pending_activation_index': self.pending_activation_index,
            }
            return True

    def cancel_inference_request(self) -> None:
        with self.lock:
            self._inference_in_flight = False

    def last_commit_debug(self) -> dict:
        with self.lock:
            return dict(self._last_commit_debug)

    def pop_next(self) -> tuple[np.ndarray | None, int | None]:
        with self.lock:
            if (self.pending_processed is not None
                    and self.pending_activation_index is not None
                    and self.active_index >= self.pending_activation_index):
                self._activate_pending_locked()

            if self._remaining_before_switch_locked() <= 0:
                self._activate_pending_locked()
            if self._remaining_before_switch_locked() <= 0:
                return None, self.active_chunk_id

            action = self.active_processed[self.active_index].copy()
            chunk_id = self.active_chunk_id
            self.active_index += 1
            return action, chunk_id

    def qsize(self) -> int:
        with self.lock:
            return (self._remaining_before_switch_locked() +
                    self._pending_executable_locked())


@RUNNERS.register_module()
class OliRTCInferenceRunner(OliInferenceRunner):
    """Oli runner with bounded asynchronous RTC prefix conditioning.

    Prompt selection and execution-count handling come from
    :class:`OliInferenceRunner`.  For one prompt selection, the requested
    execution count is interpreted as the number of model action chunks to
    generate.  A producer predicts the next chunk while an actor is still
    executing the current chunk.  The unexecuted tail is supplied to the next
    model call through ``prev_actions`` / ``prefix_len``.
    """

    def __init__(self,
                 rtc_config: dict = None,
                 async_remaining_actions_threshold: int = 8,
                 *args,
                 **kwargs):
        self.rtc_config = rtc_config or {
            'enabled': True,
            'method': 'prefix',
            'prefix_len': 7,
        }
        self.async_remaining_actions_threshold = max(
            0, int(async_remaining_actions_threshold))
        self._observation_lock = Lock()
        self._action_lock = Lock()
        self._rtc_stop_event = None
        self._last_rtc_chunk_count = 0
        self._last_rtc_action_count = 0
        super().__init__(*args, **kwargs)

        method = self.rtc_config.get('method', 'prefix')
        if method not in {'prefix', 'guidance'}:
            raise ValueError(f'Unsupported RTC method: {method}')
        prefix_len = self._rtc_prefix_len()
        if self._rtc_enabled() and prefix_len <= 0:
            raise ValueError('RTC prefix_len must be positive when enabled')
        if (self.execute_horizon is not None
                and prefix_len > self.execute_horizon):
            raise ValueError('RTC prefix_len must not exceed execute_horizon')
        if (self._rtc_enabled()
                and self.async_remaining_actions_threshold < prefix_len):
            raise ValueError(
                'async_remaining_actions_threshold must be >= RTC prefix_len')
        if self._rtc_enabled() and self._use_remote:
            raise ValueError('Oli RTC currently requires local inference')

    def _rtc_enabled(self) -> bool:
        return bool(self.rtc_config.get('enabled', False))

    def _rtc_prefix_len(self) -> int:
        if not self._rtc_enabled():
            return 0
        prefix_len = self.rtc_config.get('prefix_len', 0)
        if prefix_len is None:
            return 0
        return max(0, int(prefix_len))

    def _use_vjp_guidance(self) -> bool:
        return (self._rtc_enabled()
                and self.rtc_config.get('method', 'prefix') == 'guidance'
                and self.rtc_config.get('use_vjp', False))

    def _torch_mode_context(self):
        return (torch.no_grad()
                if self._use_vjp_guidance() else torch.inference_mode())

    def _make_scheduler(self) -> OliRTCChunkScheduler:
        return OliRTCChunkScheduler(
            remaining_actions_threshold=(
                self.async_remaining_actions_threshold),
            rtc_prefix_length=self._rtc_prefix_len(),
        )

    def run(self, initial_instruction='pour water into the cup'):
        overwatch.info('Starting Oli RTC inference runner')
        while self._running:
            self._run_episode(initial_instruction)

    def _run_episode(self, default_instruction):
        instructions = self._get_user_task_instruction(default_instruction)
        if not instructions or not self._running:
            return
        self._prev_ctx = None
        self._run_rtc_instruction(
            instruction=instructions[0], chunk_count=len(instructions))

    def _run_rtc_instruction(self, instruction: str, chunk_count: int):
        if chunk_count <= 0:
            return

        scheduler = self._make_scheduler()
        stop_event = Event()
        producer_done = Event()
        self._rtc_stop_event = stop_event
        self._last_rtc_chunk_count = 0
        self._last_rtc_action_count = 0

        producer = Thread(
            target=self._producer_loop,
            args=(instruction, chunk_count, scheduler, stop_event,
                  producer_done),
            daemon=True,
            name='OliRTCProducer')
        actor = Thread(
            target=self._actor_loop,
            args=(scheduler, stop_event, producer_done),
            daemon=True,
            name='OliRTCActor')
        producer.start()
        actor.start()
        overwatch.info(
            f'Started Oli RTC producer/actor for {chunk_count} chunk(s)')

        horizon = self.execute_horizon or self.action_chunk
        timeout = chunk_count * horizon * self._dt + 30.0
        deadline = time.monotonic() + timeout
        while self._running and (producer.is_alive() or actor.is_alive()):
            if time.monotonic() >= deadline:
                overwatch.error(
                    f'Oli RTC execution timed out after {timeout:.1f}s')
                stop_event.set()
                break
            stop_event.wait(0.05)

        if not self._running:
            stop_event.set()
        producer.join(timeout=2.0)
        actor.join(timeout=2.0)
        self._rtc_stop_event = None

    def _producer_loop(self, instruction: str, chunk_count: int,
                       scheduler: OliRTCChunkScheduler, stop_event: Event,
                       producer_done: Event):
        committed_chunks = 0
        next_chunk_id = 0
        try:
            while (self._running and not stop_event.is_set()
                   and committed_chunks < chunk_count):
                request = scheduler.prepare_inference_request(
                    use_rtc=self._rtc_enabled())
                if request is None:
                    time.sleep(0.005)
                    continue

                next_chunk_id += 1
                try:
                    self._action_ctx = SimpleNamespace(instruction=instruction)
                    raw_action, actions = self._infer_action_chunk(
                        instruction, request)
                    original_actions = self._raw_action_to_numpy(raw_action)
                    original_actions = original_actions[:len(actions)]
                    horizon = self.execute_horizon
                    if horizon is not None:
                        actions = actions[:horizon]
                        original_actions = original_actions[:horizon]
                    accepted = scheduler.commit_inference_result(
                        request,
                        original_actions,
                        np.asarray(actions),
                        chunk_id=next_chunk_id)
                    if accepted:
                        committed_chunks += 1
                        self._prev_ctx = self._action_ctx
                        self._last_rtc_chunk_count = committed_chunks
                        overwatch.info(f'[OliRTC] committed chunk '
                                       f'{committed_chunks}/{chunk_count} '
                                       f'prefix_len={request.prefix_len}')
                    else:
                        debug = scheduler.last_commit_debug()
                        overwatch.warning(
                            f'[OliRTC] dropped chunk {next_chunk_id}: '
                            f'{debug}')
                except Exception:
                    scheduler.cancel_inference_request()
                    raise
        except Exception as exc:
            overwatch.error(f'[OliRTC producer] fatal error: {exc}')
            overwatch.error(traceback.format_exc())
            self._running = False
            stop_event.set()
        finally:
            producer_done.set()

    def _infer_action_chunk(self, instruction: str,
                            request: OliRTCInferenceRequest):
        with self._observation_lock:
            inputs = self._preprocess(instruction)
        self._apply_request_prefix(inputs, request)

        inference_start = time.perf_counter()
        with self._torch_mode_context():
            with torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=(self.enable_mixed_precision
                             and not self._use_remote)):
                raw_action = self._predict_action(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        actions = self._postprocess_actions(raw_action)
        overwatch.info(
            f'[OliRTC] inference '
            f'{(time.perf_counter() - inference_start) * 1000.0:.1f}ms '
            f'prefix_len={request.prefix_len} '
            f'remaining={request.remaining_actions}')
        return raw_action, actions

    def _apply_request_prefix(self, inputs: dict,
                              request: OliRTCInferenceRequest):
        if (not self._rtc_enabled() or request.prefix_actions is None
                or request.prefix_len <= 0):
            return
        if self._use_remote:
            raise RuntimeError('Oli RTC prefix mode requires local inference')

        prefix = torch.from_numpy(request.prefix_actions[None]).to(
            device=inputs['states'].device, dtype=inputs['states'].dtype)
        inputs['prev_actions'] = prefix
        inputs['prefix_len'] = request.prefix_len
        inputs['rtc_config'] = self.rtc_config

    @staticmethod
    def _raw_action_to_numpy(raw_action) -> np.ndarray:
        raw_np = raw_action.detach().cpu().numpy()
        if raw_np.ndim >= 3 and raw_np.shape[0] == 1:
            raw_np = raw_np[0]
        return raw_np

    def _actor_loop(self, scheduler: OliRTCChunkScheduler, stop_event: Event,
                    producer_done: Event):
        actor_dt = self._dt
        action_count = 0
        last_chunk_id = None
        try:
            while self._running and not stop_event.is_set():
                loop_start = time.perf_counter()
                action, chunk_id = scheduler.pop_next()
                if action is not None:
                    if not self.disable_puppet_arm:
                        with self._action_lock:
                            self.ros_operator.send_action(action)
                    action_count += 1
                    self._last_rtc_action_count = action_count
                    if chunk_id != last_chunk_id:
                        overwatch.info(f'[OliRTC actor] chunk={chunk_id} '
                                       f'action_count={action_count}')
                        last_chunk_id = chunk_id
                    if (self.max_publish_step
                            and action_count >= self.max_publish_step):
                        stop_event.set()
                elif producer_done.is_set() and scheduler.qsize() == 0:
                    break

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, actor_dt - elapsed))
        except Exception as exc:
            overwatch.error(f'[OliRTC actor] fatal error: {exc}')
            overwatch.error(traceback.format_exc())
            self._running = False
            stop_event.set()

    def cleanup(self):
        if self._rtc_stop_event is not None:
            self._rtc_stop_event.set()
        super().cleanup()
