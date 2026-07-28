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

import signal
import time
import unicodedata
from collections import deque
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch

from ..utils.root import RUNNERS
from .base_inference_runner import BaseInferenceRunner


class _ShutdownRequested(Exception):
    """Raised internally to unwind the inference loop on shutdown."""


@RUNNERS.register_module()
class OliInferenceRunner(BaseInferenceRunner):
    """Runner for Oli whole-body (loco-manipulation) inference.

    Supports one or two cameras, a 33-dim state (31 joints + 2 hand-closed),
    and a 42-dim action (31 joint q + 9 base pose + 2 hand-closed). Each
    predicted action step is sent to ``OliOperator`` with time-based control.

    No RTC, interpolation, async execution, or done-driven prompt switching.
    Interactive execution selects a prompt ID and a positive execution count;
    one execution corresponds to one predicted action chunk (optionally
    truncated by ``execute_horizon``).
    """

    def __init__(self,
                 execute_horizon: int = None,
                 interactive: bool = True,
                 default_prompt_id: str = None,
                 default_execution_count: int = 1,
                 apply_jpeg_compression: bool = False,
                 zero_prompt_resets: bool = False,
                 initial_state=None,
                 reset_duration_sec: float = 5.0,
                 *args,
                 **kwargs):
        self.execute_horizon = execute_horizon
        self.interactive = bool(interactive)
        self.default_prompt_id = default_prompt_id
        self.default_execution_count = int(default_execution_count)
        self.apply_jpeg_compression = bool(apply_jpeg_compression)
        self.zero_prompt_resets = bool(zero_prompt_resets)
        self.initial_state = (None if initial_state is None else np.asarray(
            initial_state, dtype=np.float64))
        self.reset_duration_sec = float(reset_duration_sec)
        if self.execute_horizon is not None and self.execute_horizon <= 0:
            raise ValueError('execute_horizon must be positive or None')
        if self.default_execution_count <= 0:
            raise ValueError('default_execution_count must be positive')
        if self.initial_state is not None:
            if self.initial_state.shape != (33, ):
                raise ValueError(
                    'Oli initial_state must contain 31 joints and 2 hand '
                    f'flags, got shape {self.initial_state.shape}')
            if not np.all(np.isfinite(self.initial_state)):
                raise ValueError('Oli initial_state must be finite')
        if self.reset_duration_sec <= 0:
            raise ValueError('reset_duration_sec must be positive')

        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = ['head']

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'OliOperator',
                'head_rgb_topic': '/head/color/image_raw/compressed',
                'joint_state_topic': '/joint/state',
                'robot_ip': '10.192.1.2',
                'ws_port': 5000,
            }

        if 'task_descriptions' not in kwargs or \
                kwargs['task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1': 'pour water into the cup',
            }

        super().__init__(*args, **kwargs)

        if not self.task_descriptions:
            raise ValueError('task_descriptions must not be empty')
        if self.default_prompt_id is None:
            self.default_prompt_id = next(iter(self.task_descriptions))
        self.default_prompt_id = str(self.default_prompt_id)
        if self.default_prompt_id not in self.task_descriptions:
            raise ValueError(
                f'default_prompt_id {self.default_prompt_id!r} is not in '
                f'task_descriptions={list(self.task_descriptions)}')
        if self.zero_prompt_resets:
            if self.initial_state is None:
                raise ValueError(
                    'initial_state is required when zero_prompt_resets=True')
            if self.default_prompt_id == '0':
                raise ValueError(
                    'default_prompt_id cannot be 0 when 0 is the reset '
                    'command')

        self._running = True
        self._dt = 1.0 / self.publish_rate
        self._selected_prompt_id = self.default_prompt_id
        self._selected_execution_count = self.default_execution_count

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle SIGINT for graceful shutdown."""
        print('\nShutdown requested...')
        self._running = False

    def _get_task_description(self, task_id: str) -> str:
        """Fall back to the first configured Oli task rather than the base
        class's unrelated default description."""
        if task_id in self.task_descriptions:
            return self.task_descriptions[task_id]
        return next(iter(self.task_descriptions.values()))

    @staticmethod
    def _normalize_input(value: str) -> str:
        return unicodedata.normalize('NFKC', value).strip()

    def _get_user_task_instruction(self, default_instruction: str):
        """Select a prompt ID and the number of action chunks to execute."""
        del default_instruction
        if not self.interactive:
            prompt_id = self.default_prompt_id
            self._selected_prompt_id = prompt_id
            self._selected_execution_count = self.default_execution_count
            description = self._get_task_description(prompt_id)
            return [description] * self.default_execution_count

        prompt_ids = ', '.join(self.task_descriptions)
        reset_hint = '; 0 resets robot' if self.zero_prompt_resets else ''
        while self._running:
            try:
                value = input(f'Prompt ID [{self.default_prompt_id}] '
                              f'(available: {prompt_ids}{reset_hint}; '
                              'q to quit): ')
            except (EOFError, KeyboardInterrupt):
                self._running = False
                return []
            prompt_id = self._normalize_input(value)
            if prompt_id.lower() in {'q', 'quit', 'exit'}:
                self._running = False
                return []
            if prompt_id == '':
                prompt_id = self.default_prompt_id
            if prompt_id == '0' and self.zero_prompt_resets:
                self._move_to_prepare_pose()
                print('[reset] Oli initial state restored.', flush=True)
                continue
            if prompt_id in self.task_descriptions:
                break
            print(
                f'Unknown prompt ID {prompt_id!r}; available IDs: '
                f'{prompt_ids}',
                flush=True)

        while self._running:
            try:
                value = input(
                    f'Execution count [{self.default_execution_count}] '
                    '(q to quit): ')
            except (EOFError, KeyboardInterrupt):
                self._running = False
                return []
            value = self._normalize_input(value)
            if value.lower() in {'q', 'quit', 'exit'}:
                self._running = False
                return []
            if value == '':
                execution_count = self.default_execution_count
                break
            try:
                execution_count = int(value)
            except ValueError:
                print(
                    'Execution count must be a positive integer.', flush=True)
                continue
            if execution_count <= 0:
                print(
                    'Execution count must be a positive integer.', flush=True)
                continue
            break

        self._selected_prompt_id = prompt_id
        self._selected_execution_count = execution_count
        description = self._get_task_description(prompt_id)
        print(
            f'[prompt] id={prompt_id} executions={execution_count} '
            f'description={description!r}',
            flush=True)
        return [description] * execution_count

    def run(self, initial_instruction='pour water into the cup'):
        """Main inference loop using time-based rate control.

        Args:
            initial_instruction (str): Default task instruction.
        """
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Starting Oli whole-body inference runner')

        with torch.inference_mode():
            try:
                while self._running:
                    self._run_episode(initial_instruction)
            except _ShutdownRequested:
                pass

    def _run_episode(self, default_instruction):
        """Execute the selected prompt for the requested chunk count."""
        instructions = self._get_user_task_instruction(default_instruction)
        self._prev_ctx = None
        published_steps = 0

        for execution_index, instruction in enumerate(instructions, start=1):
            if not self._running or published_steps >= self.max_publish_step:
                break
            self._action_ctx = SimpleNamespace()
            self._action_ctx.instruction = instruction
            inputs = self._preprocess(instruction)

            with torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=(self.enable_mixed_precision
                             and not self._use_remote)):
                raw_action = self._predict_action(inputs)

            actions = self._postprocess_actions(raw_action)
            sent_steps = self._execute_actions(actions, None)
            self._prev_ctx = self._action_ctx
            published_steps += sent_steps
            print(
                f'[execution] prompt_id={self._selected_prompt_id} '
                f'{execution_index}/{self._selected_execution_count} '
                f'published_steps={sent_steps}',
                flush=True)

    def get_ros_observation(self):
        """Poll the operator until a synchronized observation is available.

        Returns:
            tuple: Camera images followed by ``state_33d``, or ``None``.
        """
        last_wait_print = 0.0
        while self._running:
            result = self.ros_operator.get_frame()
            if result is not False:
                return result
            now = time.monotonic()
            if now - last_wait_print >= 2.0:
                print(
                    '[waiting] No complete Oli observation received yet.',
                    flush=True)
                last_wait_print = now
            time.sleep(0.01)
        return None

    def update_observation_window(self) -> Dict:
        """Update the observation window with the latest sensor data.

        Returns:
            Dict: Latest observation with ``qpos`` and configured images.
        """
        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)
            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        result = self.get_ros_observation()
        if result is None:
            # Shutdown requested while waiting for the first observation.
            raise _ShutdownRequested()

        images = list(result[:-1])
        state = result[-1]
        if len(images) != len(self.camera_names):
            raise ValueError(
                f'OliOperator returned {len(images)} image(s), but '
                f'camera_names={self.camera_names}')

        observation = {'qpos': state}
        for camera_name, image in zip(self.camera_names, images):
            if self.apply_jpeg_compression:
                bgr = image[:, :, ::-1]
                image = self._apply_jpeg_compression(bgr)[:, :, ::-1].copy()
            observation[camera_name] = image
        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _execute_actions(self, actions: np.ndarray, rate):
        """Send each 42-dim action to the operator with rate control."""
        del rate
        if self.disable_puppet_arm:
            return 0
        if self.execute_horizon is not None:
            actions = actions[:self.execute_horizon]
        sent_steps = 0
        for action in actions:
            if not self._running:
                break
            self.ros_operator.send_action(action)
            sent_steps += 1
            time.sleep(self._dt)
        return sent_steps

    def _move_to_prepare_pose(self):
        """Smoothly restore the configured 33-dim Oli initial state."""
        if self.initial_state is None:
            raise RuntimeError('No Oli initial_state is configured')
        if self.disable_puppet_arm:
            print(
                '[reset] disable_puppet_arm=True; reset not sent.', flush=True)
            return self.initial_state.copy()
        target = self.ros_operator.gohome(
            self.initial_state,
            duration_sec=self.reset_duration_sec,
            publish_rate=self.publish_rate,
            running_flag_fn=lambda: self._running)
        self.observation_window = None
        return target

    def cleanup(self):
        """Clean up resources."""
        print('Cleaning up OliInferenceRunner')
        self._running = False
        if hasattr(self.ros_operator, 'stop_trajectory'):
            self.ros_operator.stop_trajectory()
        if hasattr(self.ros_operator, 'close'):
            self.ros_operator.close()
        super().cleanup()
