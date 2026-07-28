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

from ..operators.teleop02_wbt_operator import interpolate_wbt_actions
from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .teleop02_wbt_inference_runner import (Teleop02WbtInferenceRunner,
                                            decide_advance)

overwatch = initialize_overwatch(__name__)


@dataclass
class InferenceRequest:
    prefix_actions: np.ndarray | None
    prefix_len: int
    remaining_actions: int
    active_chunk_id: int | None
    active_index_at_request: int
    active_source_index_at_request: int = 0


class ChunkScheduler:
    """PI0.5-style active/pending scheduler for chunked async inference."""

    def __init__(self,
                 async_enabled: bool = False,
                 remaining_actions_threshold: int = 0,
                 rtc_prefix_length: int = 0,
                 source_hz: float = 30,
                 processed_hz: float | None = None):
        self.async_enabled = async_enabled
        self.remaining_actions_threshold = max(
            0, int(remaining_actions_threshold))
        self.rtc_prefix_length = max(0, int(rtc_prefix_length))
        self.source_hz = max(float(source_hz), 1e-6)
        self.processed_hz = max(
            float(processed_hz if processed_hz is not None else source_hz),
            1e-6)
        self.lock = Lock()

        self.active_original: np.ndarray | None = None
        self.active_processed: np.ndarray | None = None
        self.active_processed_base_pos: np.ndarray | None = None
        self.active_processed_base_quat: np.ndarray | None = None
        self.active_index = 0
        self.active_chunk_id: int | None = None

        self.pending_original: np.ndarray | None = None
        self.pending_processed: np.ndarray | None = None
        self.pending_processed_base_pos: np.ndarray | None = None
        self.pending_processed_base_quat: np.ndarray | None = None
        self.pending_start_index = 0
        self.pending_activation_index: int | None = None
        self.pending_chunk_id: int | None = None

        self._inference_in_flight = False
        self._last_commit_debug = {}

    def _remaining_active_locked(self) -> int:
        if self.active_processed is None:
            return 0
        return max(0, len(self.active_processed) - self.active_index)

    def _processed_to_source_index(self, processed_index: int,
                                   original_actions: np.ndarray | None) -> int:
        if original_actions is None:
            return 0
        original_len = len(original_actions)
        if original_len <= 0 or processed_index <= 0:
            return 0
        elapsed = processed_index / self.processed_hz
        source_index = int(np.ceil(elapsed * self.source_hz - 1e-9))
        return min(max(source_index, 0), original_len)

    def _source_to_processed_index(
            self, source_index: int,
            processed_actions: np.ndarray | None) -> int:
        if processed_actions is None:
            return 0
        processed_len = len(processed_actions)
        if processed_len <= 0 or source_index <= 0:
            return 0
        elapsed = source_index / self.source_hz
        processed_index = int(np.ceil(elapsed * self.processed_hz - 1e-9))
        return min(max(processed_index, 0), processed_len)

    def _active_source_index_locked(self) -> int:
        return self._processed_to_source_index(self.active_index,
                                               self.active_original)

    def _remaining_active_source_locked(self) -> int:
        if self.active_original is None:
            return 0
        return max(
            0,
            len(self.active_original) - self._active_source_index_locked())

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

    def _has_active_remaining_locked(self) -> bool:
        return self._remaining_before_switch_locked() > 0

    def _activate_pending_locked(self) -> None:
        if self.pending_processed is None or self.pending_original is None:
            self.active_original = None
            self.active_processed = None
            self.active_processed_base_pos = None
            self.active_processed_base_quat = None
            self.active_index = 0
            self.active_chunk_id = None
            return

        self.active_original = self.pending_original
        self.active_processed = self.pending_processed
        self.active_processed_base_pos = self.pending_processed_base_pos
        self.active_processed_base_quat = self.pending_processed_base_quat
        self.active_index = min(self.pending_start_index,
                                len(self.active_processed))
        self.active_chunk_id = self.pending_chunk_id

        self.pending_original = None
        self.pending_processed = None
        self.pending_processed_base_pos = None
        self.pending_processed_base_quat = None
        self.pending_start_index = 0
        self.pending_activation_index = None
        self.pending_chunk_id = None

    def prepare_inference_request(self,
                                  use_rtc: bool) -> InferenceRequest | None:
        with self.lock:
            if self._inference_in_flight:
                return None

            remaining_actions = self._remaining_active_source_locked()
            should_request = False
            if self.active_processed is None:
                should_request = True
            elif remaining_actions <= 0:
                should_request = self.pending_processed is None
            elif (self.async_enabled and self.pending_processed is None
                  and remaining_actions <= self.remaining_actions_threshold):
                if use_rtc and remaining_actions < self.rtc_prefix_length:
                    return None
                should_request = True

            if not should_request:
                return None

            prefix_actions = None
            prefix_len = 0
            source_index = self._active_source_index_locked()
            if (use_rtc and self.active_original is not None
                    and remaining_actions >= self.rtc_prefix_length
                    and self.rtc_prefix_length > 0):
                prefix_actions = self.active_original[
                    source_index:source_index + self.rtc_prefix_length].copy()
                prefix_len = prefix_actions.shape[0]
                if prefix_len == 0:
                    prefix_actions = None

            self._inference_in_flight = True
            return InferenceRequest(
                prefix_actions=prefix_actions,
                prefix_len=prefix_len,
                remaining_actions=remaining_actions,
                active_chunk_id=self.active_chunk_id,
                active_index_at_request=self.active_index,
                active_source_index_at_request=source_index,
            )

    def commit_inference_result(
            self,
            request: InferenceRequest,
            original_actions: np.ndarray,
            processed_actions: np.ndarray,
            chunk_id: int | None = None,
            processed_base_pos: np.ndarray | None = None,
            processed_base_quat: np.ndarray | None = None) -> bool:
        with self.lock:
            self._inference_in_flight = False
            if request.active_chunk_id != self.active_chunk_id:
                self._last_commit_debug = {
                    'reason': 'active_chunk_changed',
                    'request_active_chunk_id': request.active_chunk_id,
                    'current_active_chunk_id': self.active_chunk_id,
                    'request_prefix_len': request.prefix_len,
                    'request_remaining_actions': request.remaining_actions,
                    'request_active_index': request.active_index_at_request,
                    'current_active_index': self.active_index,
                }
                return False

            clipped_prefix_len = min(
                max(0, request.prefix_len), len(original_actions))
            source_index = self._active_source_index_locked()
            request_source_index = getattr(request,
                                           'active_source_index_at_request',
                                           request.active_index_at_request)
            executed_since_request = max(0,
                                         source_index - request_source_index)

            if executed_since_request > clipped_prefix_len:
                self._last_commit_debug = {
                    'reason': 'prefix_window_passed',
                    'request_prefix_len': request.prefix_len,
                    'clipped_prefix_len': clipped_prefix_len,
                    'request_remaining_actions': request.remaining_actions,
                    'request_source_index': request_source_index,
                    'current_source_index': source_index,
                    'executed_since_request': executed_since_request,
                    'request_active_index': request.active_index_at_request,
                    'current_active_index': self.active_index,
                    'source_hz': self.source_hz,
                    'processed_hz': self.processed_hz,
                    'original_actions_len': len(original_actions),
                    'processed_actions_len': len(processed_actions),
                }
                return False

            if executed_since_request == clipped_prefix_len:
                self.active_original = original_actions.copy()
                self.active_processed = processed_actions.copy()
                self.active_processed_base_pos = (
                    None if processed_base_pos is None else
                    processed_base_pos.copy())
                self.active_processed_base_quat = (
                    None if processed_base_quat is None else
                    processed_base_quat.copy())
                self.active_index = self._source_to_processed_index(
                    clipped_prefix_len, self.active_processed)
                self.active_chunk_id = chunk_id
                self.pending_original = None
                self.pending_processed = None
                self.pending_processed_base_pos = None
                self.pending_processed_base_quat = None
                self.pending_start_index = 0
                self.pending_activation_index = None
                self.pending_chunk_id = None
                self._last_commit_debug = {
                    'reason': 'accepted_replace_active',
                    'request_prefix_len': request.prefix_len,
                    'clipped_prefix_len': clipped_prefix_len,
                    'executed_since_request': executed_since_request,
                    'current_source_index': source_index,
                    'active_index': self.active_index,
                }
                return True

            self.pending_original = original_actions.copy()
            self.pending_processed = processed_actions.copy()
            self.pending_processed_base_pos = (None
                                               if processed_base_pos is None
                                               else processed_base_pos.copy())
            self.pending_processed_base_quat = (
                None
                if processed_base_quat is None else processed_base_quat.copy())
            self.pending_start_index = self._source_to_processed_index(
                clipped_prefix_len, self.pending_processed)
            self.pending_activation_index = self._source_to_processed_index(
                request_source_index + clipped_prefix_len,
                self.active_processed)
            self.pending_chunk_id = chunk_id
            self._last_commit_debug = {
                'reason': 'accepted_pending',
                'request_prefix_len': request.prefix_len,
                'clipped_prefix_len': clipped_prefix_len,
                'executed_since_request': executed_since_request,
                'request_source_index': request_source_index,
                'current_source_index': source_index,
                'pending_start_index': self.pending_start_index,
                'pending_activation_index': self.pending_activation_index,
            }
            return True

    def last_commit_debug(self) -> dict:
        with self.lock:
            return dict(self._last_commit_debug)

    def cancel_inference_request(self) -> None:
        with self.lock:
            self._inference_in_flight = False

    def pop_next_command(
        self
    ) -> tuple[np.ndarray | None, int | None, np.ndarray | None, np.ndarray
               | None]:
        with self.lock:
            if (self.pending_processed is not None
                    and self.pending_activation_index is not None
                    and self.active_index >= self.pending_activation_index):
                self._activate_pending_locked()

            if not self._has_active_remaining_locked():
                self._activate_pending_locked()

            if not self._has_active_remaining_locked():
                return None, self.active_chunk_id, None, None

            action = self.active_processed[self.active_index].copy()
            base_pos = (
                None if self.active_processed_base_pos is None else
                self.active_processed_base_pos[self.active_index].copy())
            base_quat = (
                None if self.active_processed_base_quat is None else
                self.active_processed_base_quat[self.active_index].copy())
            chunk_id = self.active_chunk_id
            self.active_index += 1
            return action, chunk_id, base_pos, base_quat

    def pop_next(self) -> tuple[np.ndarray | None, int | None]:
        action, chunk_id, _, _ = self.pop_next_command()
        return action, chunk_id

    def qsize(self) -> int:
        with self.lock:
            return (self._remaining_before_switch_locked() +
                    self._pending_executable_locked())


@RUNNERS.register_module()
class Teleop02WbtRTCInferenceRunner(Teleop02WbtInferenceRunner):
    """Teleop02 WBT inference runner with RTC prefix conditioning.

    Extends Teleop02WbtInferenceRunner by adding RTC support to
    _predict_action, which conditions the model on previously predicted
    actions for smoother trajectory stitching across inference chunks.

    Args:
        rtc_config (dict, optional): RTC configuration dict. Expected keys:
            - enabled (bool): Whether RTC is active.
            - method (str): 'prefix' or 'guidance'.
            - prefix_len (int, optional): Number of prefix steps. If None,
              estimated from last inference time.
    """

    def __init__(self,
                 rtc_config: dict = None,
                 async_remaining_actions_threshold: int = 6,
                 *args,
                 **kwargs):
        self.rtc_config = rtc_config
        self.async_remaining_actions_threshold = (
            async_remaining_actions_threshold)
        self._observation_lock = Lock()
        self._action_lock = Lock()
        super().__init__(*args, **kwargs)

    def run(self, initial_instruction: str = 'pour water into the cup'):
        """Run inference loop with mode selected by RTC config.

        If RTC guidance uses VJP, run under no_grad and let guidance internals
        enable gradients only where needed; otherwise use inference_mode.
        """
        overwatch.info('Starting Teleop02 WBT RTC inference runner')

        with self._torch_mode_context():
            self._warmup_model(initial_instruction)
        while self._running:
            self._run_episode(initial_instruction)

    def _use_vjp_guidance(self) -> bool:
        return (self.rtc_config and self.rtc_config.get('enabled', False)
                and self.rtc_config.get('method', 'prefix') == 'guidance'
                and self.rtc_config.get('use_vjp', False))

    def _torch_mode_context(self):
        return (torch.no_grad()
                if self._use_vjp_guidance() else torch.inference_mode())

    def _rtc_enabled(self) -> bool:
        return bool(self.rtc_config and self.rtc_config.get('enabled', False))

    def _rtc_prefix_len(self) -> int:
        if not self._rtc_enabled():
            return 0
        prefix_len = self.rtc_config.get('prefix_len', 0)
        if prefix_len is None:
            return 0
        return max(0, int(prefix_len))

    def _actor_publish_rate(self) -> float:
        if self._should_interpolate_actor_actions():
            return float(self.target_hz)
        return float(getattr(self, 'publish_rate', 30.0))

    def _should_interpolate_actor_actions(self) -> bool:
        target_hz = getattr(self, 'target_hz', None)
        publish_rate = getattr(self, 'publish_rate', None)
        return (target_hz is not None and publish_rate is not None
                and target_hz != publish_rate)

    def _actor_interpolation_origin(self) -> tuple[np.ndarray | None, float]:
        operator = getattr(self, 'ros_operator', None)
        if operator is None:
            return None, 0.0

        init_base_pos = getattr(operator, '_accum_base_pos', None)
        if init_base_pos is not None:
            init_base_pos = np.asarray(init_base_pos, dtype=np.float64).copy()

        init_base_yaw = getattr(operator, '_accum_base_yaw', None)
        if init_base_yaw is None:
            accum_base_rot = getattr(operator, '_accum_base_rot', None)
            if accum_base_rot is not None:
                init_base_yaw = float(accum_base_rot.as_euler('ZYX')[0])
            else:
                init_base_yaw = 0.0
        return init_base_pos, float(init_base_yaw)

    def _prepare_actor_actions(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if not self._should_interpolate_actor_actions():
            return actions, None, None

        init_base_pos, init_base_yaw = self._actor_interpolation_origin()
        actions_interp, base_pos_interp, base_quat_interp = (
            interpolate_wbt_actions(
                actions,
                source_hz=self.publish_rate,
                target_hz=self.target_hz,
                method=self.interpolation_method,
                init_base_pos=init_base_pos,
                init_base_yaw=init_base_yaw))
        return (actions_interp.astype(actions.dtype, copy=False),
                base_pos_interp, base_quat_interp)

    def _make_scheduler(self) -> ChunkScheduler:
        return ChunkScheduler(
            async_enabled=self.async_execution,
            remaining_actions_threshold=(
                self.async_remaining_actions_threshold),
            rtc_prefix_length=self._rtc_prefix_len(),
            source_hz=self.publish_rate,
            processed_hz=self._actor_publish_rate(),
        )

    def _run_async_episode_legacy(self, default_instruction):
        instructions = self._get_user_task_instruction(default_instruction)
        for instruction in instructions:
            if not self._running:
                break
            self._current_idx = getattr(self, '_current_idx', 0)
            self._skip_done_check = getattr(self, '_skip_done_check', 0)
            self._run_async_instruction(instruction, Event(), Event())

    def _run_episode(self, default_instruction):
        if getattr(self, 'interactive', False):
            return self._run_async_episode_legacy(default_instruction)
        if not getattr(self, 'use_done_state_machine', True):
            self._current_idx = int(getattr(self, '_current_idx', 0))
            if self._current_idx >= len(self.done_subtask_order):
                self._current_idx = len(self.done_subtask_order) - 1
            task_id = self.done_subtask_order[self._current_idx]
            instruction = self._get_task_description(task_id)
            overwatch.info(f'[NO_DONE] starting task_idx={self._current_idx} '
                           f'task_id={task_id}')
            self._run_async_instruction(instruction, Event(), Event())
            return

        last_idx = int(getattr(self, '_current_idx', 0))
        max_reached_idx = int(getattr(self, '_max_reached_task_idx', last_idx))
        self._current_idx = min(
            max(0, max(last_idx, max_reached_idx)),
            len(self.done_subtask_order) - 1)
        self._max_reached_task_idx = self._current_idx
        self._skip_done_check = getattr(self, '_skip_done_check', 0)
        while self._running and self._current_idx < len(
                self.done_subtask_order):
            task_id = self.done_subtask_order[self._current_idx]
            instruction = self._get_task_description(task_id)
            advance_event = Event()
            final_done_event = Event()
            overwatch.info(
                f'[STATE_MACHINE] starting task_idx={self._current_idx} '
                f'task_id={task_id}')
            self._run_async_instruction(instruction, advance_event,
                                        final_done_event)
            is_last = (self._current_idx == len(self.done_subtask_order) - 1)
            if advance_event.is_set() and not is_last:
                self._current_idx += 1
                self._max_reached_task_idx = max(self._max_reached_task_idx,
                                                 self._current_idx)
            elif final_done_event.is_set():
                if self.stop_on_final_done:
                    self._running = False
                break
            else:
                break

    def _run_async_instruction(self, instruction: str, advance_event: Event,
                               final_done_event: Event):
        scheduler = self._make_scheduler()
        stop_event = Event()

        producer_thread = Thread(
            target=self._get_actions_loop,
            args=(instruction, scheduler, stop_event, advance_event,
                  final_done_event, self._current_idx),
            daemon=True,
            name='Teleop02WbtGetActions')
        actor_thread = Thread(
            target=self._actor_control_loop,
            args=(scheduler, stop_event),
            daemon=True,
            name='Teleop02WbtActor')

        producer_thread.start()
        actor_thread.start()
        overwatch.info('Started PI0.5-style GR00T RTC producer/actor threads')

        start_time = time.time()
        last_log_time = start_time
        while (self._running and not stop_event.is_set()
               and not advance_event.is_set()
               and not final_done_event.is_set()):
            stop_event.wait(0.5)
            now = time.time()
            if now - last_log_time >= 10.0:
                overwatch.info(
                    f'[MAIN] Action queue size: {scheduler.qsize()}')
                last_log_time = now
            if self.max_publish_step and (
                    now - start_time >
                    self.max_publish_step * self._dt + 30.0):
                stop_event.set()

        if advance_event.is_set() or final_done_event.is_set():
            stop_event.set()
        stop_event.set()
        producer_thread.join(timeout=2.0)
        actor_thread.join(timeout=2.0)

    def _get_actions_loop(self, instruction: str, scheduler: ChunkScheduler,
                          stop_event: Event, advance_event: Event,
                          final_done_event: Event, segment_idx: int):
        chunk_id = 0
        try:
            while self._running and not stop_event.is_set():
                request = scheduler.prepare_inference_request(
                    use_rtc=self._rtc_enabled())
                if request is None:
                    time.sleep(0.005)
                    continue

                chunk_id += 1
                try:
                    self._action_ctx = SimpleNamespace(instruction=instruction)
                    raw_action, actions = self._infer_action_chunk(
                        instruction, request)
                    original_actions = self._raw_action_to_numpy(raw_action)
                    original_actions = original_actions[:len(actions)]
                    # Shrink the effective chunk to execute_horizon so that
                    # downstream RTC scheduling (remaining_actions / prefix
                    # extraction / chunk switching) treats the chunk as if
                    # its true length is execute_horizon. The tail beyond
                    # execute_horizon is discarded.
                    if (self.execute_horizon is not None
                            and self.execute_horizon > 0):
                        actions = actions[:self.execute_horizon]
                        original_actions = original_actions[:self.
                                                            execute_horizon]
                    actor_actions, actor_base_pos, actor_base_quat = (
                        self._prepare_actor_actions(actions))
                    accepted = scheduler.commit_inference_result(
                        request,
                        original_actions,
                        actor_actions,
                        chunk_id=chunk_id,
                        processed_base_pos=actor_base_pos,
                        processed_base_quat=actor_base_quat)
                    if not accepted:
                        commit_debug = scheduler.last_commit_debug()
                        debug_text = ', '.join(
                           f'{key}={value}'
                           for key, value in commit_debug.items())
                        overwatch.warning(
                           f'[GET_ACTIONS] Dropping inferred chunk '
                           f'{chunk_id} because the prefix window has '
                           f'already passed ({debug_text})')
                    else:
                        if not getattr(self, 'use_done_state_machine', True):
                            continue
                        done_chunk = getattr(self._action_ctx, 'done_chunk',
                                             None)
                        if done_chunk is not None:
                            transition = decide_advance(
                                done_chunk=done_chunk,
                                current_idx=segment_idx,
                                num_subtasks=len(self.done_subtask_order),
                                skip_done_check=self._skip_done_check,
                                done_window=self.done_window,
                                done_threshold=self.done_threshold,
                                done_advance_cooldown=(
                                    self.done_advance_cooldown),
                            )
                            self._skip_done_check = (
                                transition.next_skip_done_check)
                            window = max(1, int(self.done_window))
                            done_score = float(
                                np.asarray(done_chunk[-window:]).mean())
                            overwatch.info(f'[STATE_MACHINE] '
                                           f'task_idx={segment_idx} '
                                           f'done_score={done_score:.3f} '
                                           f'threshold={self.done_threshold} '
                                           f'transition={transition.kind}')
                            if transition.kind == 'advance':
                                advance_event.set()
                                stop_event.set()
                                return
                            if transition.kind == 'final_done':
                                final_done_event.set()
                                stop_event.set()
                                return
                except Exception:
                    scheduler.cancel_inference_request()
                    raise
        except Exception as exc:
            overwatch.error(
                f'[GET_ACTIONS] Fatal exception in get_actions thread: '
                f'{exc}')
            overwatch.error(traceback.format_exc())
            self._running = False
            stop_event.set()

    def _infer_action_chunk(self, instruction: str, request: InferenceRequest):
        chunk_start = time.perf_counter()
        with self._observation_lock:
            inputs = self._preprocess(instruction)
        self._apply_request_prefix(inputs, request)

        predict_start = time.perf_counter()
        with self._torch_mode_context():
            with torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision):
                raw_action = self._predict_action(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predict_ms = (time.perf_counter() - predict_start) * 1000.0

        postprocess_start = time.perf_counter()
        actions = self._postprocess_actions(raw_action)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        total_ms = (time.perf_counter() - chunk_start) * 1000.0
        overwatch.info(
            f'[GET_ACTIONS] Chunk inference took {predict_ms:.1f} ms '
            f'(total {total_ms:.1f} ms, postprocess '
            f'{postprocess_ms:.1f} ms, prefix_len={request.prefix_len}, '
            f'remaining_actions={request.remaining_actions})')
        return raw_action, actions

    def _apply_request_prefix(self, inputs: dict,
                              request: InferenceRequest) -> None:
        if (not self._rtc_enabled() or request.prefix_actions is None
                or request.prefix_len <= 0):
            return

        prefix_actions = request.prefix_actions[None]
        inputs['prev_actions'] = torch.from_numpy(prefix_actions).to(
            device=inputs['states'].device, dtype=inputs['states'].dtype)
        inputs['prefix_len'] = request.prefix_len
        inputs['rtc_config'] = self.rtc_config

    def _predict_action(self, inputs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_start = time.perf_counter()
        raw_action = self.vla.predict_action(**inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - inference_start
        # print(f'Inference time: {elapsed:.3f}s', flush=True)
        # print(
        #     f'Using prefix_len={inputs.get("prefix_len", "N/A")}',
        #     flush=True)
        return raw_action

    def _raw_action_to_numpy(self, raw_action) -> np.ndarray:
        raw_np = raw_action.detach().cpu().numpy()
        if raw_np.ndim >= 3 and raw_np.shape[0] == 1:
            raw_np = raw_np[0]
        return raw_np

    def _actor_control_loop(self, scheduler: ChunkScheduler,
                            stop_event: Event):
        action_count = 0
        last_chunk_id = None
        actor_dt = 1.0 / self._actor_publish_rate()
        try:
            while self._running and not stop_event.is_set():
                start_time = time.perf_counter()
                if hasattr(scheduler, 'pop_next_command'):
                    action, chunk_id, base_pos, base_quat = (
                        scheduler.pop_next_command())
                else:
                    action, chunk_id = scheduler.pop_next()
                    base_pos = None
                    base_quat = None

                if action is not None:
                    with self._action_lock:
                        # print('actions:', action, flush=True)
                        if base_pos is not None and base_quat is not None:
                            self.ros_operator.send_action_absolute(
                                action, base_pos, base_quat)
                        else:
                            self.ros_operator.send_action(action)
                    if chunk_id != last_chunk_id:
                        overwatch.info(f'[ACTOR] Sent chunk {chunk_id} '
                                       f'at action_count={action_count}')
                        last_chunk_id = chunk_id
                    action_count += 1
                    if (self.max_publish_step
                            and action_count >= self.max_publish_step):
                        stop_event.set()

                elapsed = time.perf_counter() - start_time
                if elapsed > actor_dt:
                    overwatch.warning(
                        f'[ACTOR] publish loop overrun '
                        f'elapsed={elapsed * 1000.0:.2f} ms '
                        f'budget={actor_dt * 1000.0:.2f} ms '
                        f'action_count={action_count} '
                        f'chunk_id={chunk_id}')
                time.sleep(max(0.0, actor_dt - elapsed))
        except Exception as exc:
            overwatch.error(
                f'[ACTOR] Fatal exception in actor_control thread: {exc}')
            overwatch.error(traceback.format_exc())
            self._running = False
            stop_event.set()
