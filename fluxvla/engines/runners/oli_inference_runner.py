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

    Single head camera, 33-dim state (31 joints + 2 hand-closed), 42-dim
    action (31 joint q + 9 base pose + 2 hand-closed). Each predicted action
    step is sent to ``OliOperator`` with simple time-based rate control.

    No RTC, interpolation, async execution, or done-driven prompt switching.
    """

    def __init__(self, *args, **kwargs):
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

        self._running = True
        self._dt = 1.0 / self.publish_rate

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
        """Run a single episode: observe, predict, execute, repeat."""
        t = 0

        while t < self.max_publish_step and self._running:
            instructions = self._get_user_task_instruction(default_instruction)
            self._prev_ctx = None
            for instruction in instructions:
                if not self._running:
                    break
                self._action_ctx = SimpleNamespace()
                self._action_ctx.instruction = instruction
                inputs = self._preprocess(instruction)

                with torch.autocast(
                        'cuda',
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision):
                    raw_action = self._predict_action(inputs)

                actions = self._postprocess_actions(raw_action)
                self._execute_actions(actions, None)

                self._prev_ctx = self._action_ctx
                t += self.action_chunk
                print(f'Published Step {t}')

    def get_ros_observation(self):
        """Poll the operator until a synchronized observation is available.

        Returns:
            tuple: ``(head_img_rgb, state_33d)`` or ``None`` on shutdown.
        """
        while self._running:
            result = self.ros_operator.get_frame()
            if result is not False:
                return result
            time.sleep(0.01)
        return None

    def update_observation_window(self) -> Dict:
        """Update the observation window with the latest sensor data.

        Returns:
            Dict: Latest observation with ``qpos`` (33d) and ``head`` image.
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

        head_img, state = result
        observation = {
            'qpos': state,
            self.camera_names[0]: head_img,  # 'head'
        }
        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _execute_actions(self, actions: np.ndarray, rate):
        """Send each 42-dim action to the operator with rate control."""
        if self.disable_puppet_arm:
            return
        for action in actions:
            if not self._running:
                break
            self.ros_operator.send_action(action)
            time.sleep(self._dt)

    def _move_to_prepare_pose(self):
        """No-op for Oli (teleop-controlled robot)."""
        pass

    def cleanup(self):
        """Clean up resources."""
        print('Cleaning up OliInferenceRunner')
        self._running = False
        if hasattr(self.ros_operator, 'close'):
            self.ros_operator.close()
        super().cleanup()
