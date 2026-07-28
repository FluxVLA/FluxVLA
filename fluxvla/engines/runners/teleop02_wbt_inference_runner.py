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

import os
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import cv2
import numpy as np
import torch

from ..utils.root import RUNNERS
from .aloha_inference_runner import resample_remaining
from .base_inference_runner import BaseInferenceRunner


@dataclass(frozen=True)
class Transition:
    kind: str
    next_skip_done_check: int


def decide_advance(
    done_chunk,
    current_idx: int,
    num_subtasks: int,
    skip_done_check: int,
    done_window: int,
    done_threshold: float,
    done_advance_cooldown: int,
):
    """Decide whether the done signal advances to the next subtask."""
    if skip_done_check > 0:
        return Transition('none', skip_done_check - 1)

    window = max(1, int(done_window))
    score = float(np.asarray(done_chunk[-window:]).mean())

    is_last = current_idx >= num_subtasks - 1
    if score >= done_threshold:
        if is_last:
            return Transition('final_done', 0)
        return Transition('advance', int(done_advance_cooldown))
    return Transition('none', 0)


@RUNNERS.register_module()
class Teleop02WbtInferenceRunner(BaseInferenceRunner):
    """Runner for Teleop02 WBT (whole-body tracking) loco-mani inference.

    Uses mros middleware instead of rospy. Sends joint-level commands
    plus base_link pose via /teleop_cmd_WBT, matching the WBT action
    space (42-dim).

    The robot has:
        - 1 head camera + 1 left wrist camera
        - 33-dim state (31 joints + 2 hand_closed)
        - 42-dim action (31 joint q + 9 base_pose + 2 hand_closed)
    """

    def __init__(self,
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 target_hz: int = None,
                 interpolation_method: str = 'cubic',
                 debug_jpeg_dump_dir: str = None,
                 debug_jpeg_dump_max_frames: int = 10,
                 done_dim_index: int = 42,
                 done_threshold: float = 0.5,
                 done_window: int = 1,
                 done_advance_cooldown: int = 1,
                 done_subtask_order=None,
                 stop_on_final_done: bool = False,
                 use_done_state_machine: bool = True,
                 interactive: bool = False,
                 *args,
                 **kwargs):
        self.async_execution = async_execution
        self.execute_horizon = execute_horizon
        self.target_hz = target_hz
        self.interpolation_method = interpolation_method
        self.debug_jpeg_dump_dir = debug_jpeg_dump_dir
        self.debug_jpeg_dump_max_frames = debug_jpeg_dump_max_frames
        self._debug_jpeg_dump_count = 0
        self.done_dim_index = done_dim_index
        self.done_threshold = done_threshold
        self.done_window = done_window
        self.done_advance_cooldown = done_advance_cooldown
        self._done_subtask_order_arg = done_subtask_order
        self.stop_on_final_done = stop_on_final_done
        self.use_done_state_machine = use_done_state_machine
        self.interactive = interactive

        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = ['head', 'left_wrist']

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'Teleop02WbtOperator',
                'head_rgb_topic': '/head/color/image_raw/compressed',
                'left_wrist_rgb_topic':
                '/left_wrist_camera/color/image_raw/compressed',
                'joint_state_topic': '/joint/state',
                'finger_state_topic': '/brainco1/hand/state',
                'finger_cmd_topic': '/brainco1/hand/cmd',
                'teleop_wbt_topic': '/teleop_cmd_WBT',
                'cmd_vel_topic': '/sdk_cmd_vel_vla',
            }

        if 'task_descriptions' not in kwargs or \
                kwargs['task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1': 'pour water into the cup',
            }

        super().__init__(*args, **kwargs)

        self.done_subtask_order = (
            self._done_subtask_order_arg if self._done_subtask_order_arg
            is not None else list(self.task_descriptions.keys()))
        assert len(self.done_subtask_order) > 0, (
            'done_subtask_order must be non-empty')
        unknown = [
            tid for tid in self.done_subtask_order
            if tid not in self.task_descriptions
        ]
        assert not unknown, (
            f'done_subtask_order has unknown task_ids: {unknown}; '
            f'valid keys: {list(self.task_descriptions.keys())}')

        if self.use_done_state_machine and hasattr(
                self, 'denormalize_action') and getattr(
                    self.denormalize_action, 'norm_stats', None) is not None:
            stat_name = getattr(self.denormalize_action, 'statistic_name',
                                'private')
            stats_action = self.denormalize_action.norm_stats[stat_name].get(
                'action', {})
            if 'min' in stats_action and 'max' in stats_action:
                assert len(stats_action['min']) > self.done_dim_index, (
                    f'action stats min has {len(stats_action["min"])} '
                    f'entries, need > {self.done_dim_index}')
                assert abs(stats_action['min'][self.done_dim_index] -
                           0.0) < 1e-6, (
                               f'done dim min must be 0.0, got '
                               f'{stats_action["min"][self.done_dim_index]}')
                assert abs(stats_action['max'][self.done_dim_index] -
                           1.0) < 1e-6, (
                               f'done dim max must be 1.0, got '
                               f'{stats_action["max"][self.done_dim_index]}')

        if self.use_done_state_machine and self.execute_horizon is not None:
            assert self.done_window <= self.execute_horizon, (
                f'done_window {self.done_window} > execute_horizon '
                f'{self.execute_horizon}')
        if self.use_done_state_machine:
            assert self.done_window <= self.action_chunk, (
                f'done_window {self.done_window} > action_chunk '
                f'{self.action_chunk}')

        self._running = True
        self._dt = 1.0 / getattr(self, 'publish_rate',
                                 kwargs.get('publish_rate'))
        self._model_warmed_up = False

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle SIGINT for graceful shutdown."""
        print('\nShutdown requested...')
        self._running = False

    def run(self, initial_instruction='pour water into the cup'):
        """Main inference loop using time-based rate control.

        Replaces rospy-based loop with pure Python time.sleep.

        Args:
            initial_instruction (str): Default task instruction.
        """
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Starting Teleop02 WBT inference runner')

        with torch.inference_mode():
            self._warmup_model(initial_instruction)
            while self._running:
                self._run_episode(initial_instruction)

    def _warmup_model(self, instruction: str):
        """Run one dummy inference before waiting for real observations."""
        if self._model_warmed_up:
            return

        # Wait for the first real observation before warm-up.
        print('[warm-up] Waiting for first image input...', flush=True)
        result = None
        while self._running:
            result = self.ros_operator.get_frame()
            if result is not False:
                break
            print(
                '[warm-up] No image received yet. '
                'Waiting for image input from MROS topics...',
                flush=True)
            time.sleep(1.0)

        if not self._running or result is None:
            return

        head_img, left_wrist_img, state = result

        warmup_start = time.perf_counter()
        print(
            '[warm-up] First image received. Starting model warm-up...',
            flush=True)

        warmup_obs = {'qpos': state}
        imgs = [head_img, left_wrist_img]
        for i, camera_name in enumerate(self.camera_names):
            if i < len(imgs):
                warmup_obs[camera_name] = imgs[i]
        warmup_obs['task_description'] = instruction

        dataset_start = time.perf_counter()
        inputs = self.dataset(warmup_obs)
        print(
            f'[warm-up] dataset_transform='
            f'{(time.perf_counter() - dataset_start) * 1000.0:.3f} ms',
            flush=True)

        predict_start = time.perf_counter()
        with torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision):
            _ = self.vla.predict_action(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(
            f'[warm-up] predict_action='
            f'{(time.perf_counter() - predict_start) * 1000.0:.3f} ms',
            flush=True)

        self._model_warmed_up = True
        print(
            f'[warm-up] Dummy model warm-up completed in '
            f'{(time.perf_counter() - warmup_start) * 1000.0:.3f} ms; '
            f'action discarded. Now waiting for real observation data.',
            flush=True)

    def _run_episode_legacy(self, default_instruction):
        """Run a single episode without rospy dependency.

        Args:
            default_instruction (str): Default task instruction.
        """
        t = 0

        print('run episode()')

        while t < self.max_publish_step and self._running:
            instructions = self._get_user_task_instruction(default_instruction)
            self._prev_ctx = None
            for instruction in instructions:
                if not self._running:
                    break
                self._action_ctx = SimpleNamespace()
                self._action_ctx.instruction = instruction
                chunk_start = time.perf_counter()
                print(
                    f'[timing] chunk_start step={t} '
                    f'instruction={instruction!r}',
                    flush=True)

                stage_start = time.perf_counter()
                inputs = self._preprocess(instruction)
                print(
                    f'[timing] preprocess_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                stage_start = time.perf_counter()
                with torch.autocast(
                        'cuda',
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision):
                    raw_action = self._predict_action(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(
                    f'[timing] predict_action_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                stage_start = time.perf_counter()
                actions = self._postprocess_actions(raw_action)
                print(
                    f'[timing] postprocess_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                stage_start = time.perf_counter()
                self._execute_actions(actions, None)
                print(
                    f'[timing] execute_actions_total='
                    f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
                    flush=True)

                self._prev_ctx = self._action_ctx
                t += self.action_chunk
                print(
                    f'[timing] chunk_total='
                    f'{(time.perf_counter() - chunk_start) * 1000.0:.3f} ms',
                    flush=True)
                print(f'Published Step {t}')

    def _run_episode(self, default_instruction):
        """Run one episode, using done-driven prompt switching by default."""
        if getattr(self, 'interactive', False):
            return self._run_episode_legacy(default_instruction)
        if not getattr(self, 'use_done_state_machine', True):
            return self._run_episode_no_done(default_instruction)

        t = 0
        self._prev_ctx = None
        current_idx = 0
        skip_done_check = 0

        while t < self.max_publish_step and self._running:
            if current_idx >= len(self.done_subtask_order):
                current_idx = len(self.done_subtask_order) - 1
            task_id = self.done_subtask_order[current_idx]
            instruction = self._get_task_description(task_id)

            self._action_ctx = SimpleNamespace()
            self._action_ctx.instruction = instruction

            inputs = self._preprocess(instruction)
            with torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision):
                raw_action = self._predict_action(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            actions = self._postprocess_actions(raw_action)
            self._execute_actions(actions, None)
            self._prev_ctx = self._action_ctx
            t += self.action_chunk

            done_chunk = self._action_ctx.done_chunk
            transition = decide_advance(
                done_chunk=done_chunk,
                current_idx=current_idx,
                num_subtasks=len(self.done_subtask_order),
                skip_done_check=skip_done_check,
                done_window=self.done_window,
                done_threshold=self.done_threshold,
                done_advance_cooldown=self.done_advance_cooldown,
            )
            done_score = float(
                np.asarray(done_chunk[-max(1, int(self.done_window)):]).mean())
            print(
                f'[state_machine] task_id={task_id} '
                f'done_score={done_score:.3f} '
                f'threshold={self.done_threshold} '
                f'transition={transition.kind}',
                flush=True)

            if transition.kind == 'advance':
                current_idx += 1
                skip_done_check = transition.next_skip_done_check
                self._prev_ctx = None
            elif transition.kind == 'final_done':
                skip_done_check = transition.next_skip_done_check
                if self.stop_on_final_done:
                    self._running = False
            else:
                skip_done_check = transition.next_skip_done_check

    def _run_episode_no_done(self, default_instruction):
        """Run one episode with a fixed prompt and no done/progress signal."""
        t = 0
        self._prev_ctx = None
        task_id = self.done_subtask_order[
            0] if self.done_subtask_order else None
        instruction = (
            self._get_task_description(task_id)
            if task_id is not None else default_instruction)

        while t < self.max_publish_step and self._running:
            self._action_ctx = SimpleNamespace()
            self._action_ctx.instruction = instruction

            inputs = self._preprocess(instruction)
            with torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision):
                raw_action = self._predict_action(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            actions = self._postprocess_actions(raw_action)
            self._execute_actions(actions, None)
            self._prev_ctx = self._action_ctx
            t += self.action_chunk

    def get_ros_observation(self):
        """Get observation from Teleop02WbtOperator via mros.

        Polls operator.get_frame() until data is available.

        Returns:
            tuple: (head_img_rgb, left_wrist_img_rgb, state_33d)
        """
        last_wait_print = 0
        while self._running:
            get_frame_start = time.perf_counter()
            result = self.ros_operator.get_frame()
            get_frame_elapsed_ms = (time.perf_counter() -
                                    get_frame_start) * 1000.0
            if result is not False:
                # print(
                #     f'get_frame() elapsed: {get_frame_elapsed_ms:.3f} ms, '
                #     f'valid: True',
                #     flush=True)
                return result
            now = time.monotonic()
            if now - last_wait_print > 2.0:
                print(
                    '[waiting] No image received. '
                    'Waiting for image input from MROS topics...',
                    flush=True)
                last_wait_print = now
            time.sleep(0.01)
        return None

    def _write_debug_jpeg_image(self, path, img):
        """Write one post-JPEG debug image to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, img[:, :, ::-1])

    def _apply_jpeg_compression_rgb(self, img):
        """Apply base BGR JPEG compression while preserving RGB API."""
        bgr_img = img[:, :, ::-1]
        compressed_bgr = self._apply_jpeg_compression(bgr_img)
        return compressed_bgr[:, :, ::-1].copy()

    def _dump_debug_jpeg_images(self, images):
        """Save post-JPEG images for visual dataset comparison."""
        dump_dir = getattr(self, 'debug_jpeg_dump_dir', None)
        if not dump_dir:
            return

        dump_count = getattr(self, '_debug_jpeg_dump_count', 0)
        max_frames = getattr(self, 'debug_jpeg_dump_max_frames', 10)
        if max_frames is not None and dump_count >= max_frames:
            return

        for camera_name, image in images.items():
            if image is None:
                continue
            filename = f'frame_{dump_count:06d}_{camera_name}.png'
            path = os.path.join(dump_dir, filename)
            self._write_debug_jpeg_image(path, image)
            print(f'[debug] dumped post-JPEG image: {path}', flush=True)

        self._debug_jpeg_dump_count = dump_count + 1

    def update_observation_window(self) -> Dict:
        """Update observation window with latest sensor data.

        Returns:
            Dict: Latest observation with 'qpos' (33d), 'head' image, and
            'left_wrist' image.
        """
        if self.observation_window is None:
            window_init_start = time.perf_counter()
            self.observation_window = deque(maxlen=2)
            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)
            # print(
            #     f'[timing] observation_window_init='
            #     f'{(time.perf_counter() - window_init_start) * 1000.0:.3f} ms',
            #     flush=True)

        stage_start = time.perf_counter()
        result = self.get_ros_observation()
        # print(
        #     f'[timing] get_ros_observation_total='
        #     f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
        #     flush=True)
        if result is None:
            return self.observation_window[-1]

        head_img, left_wrist_img, state = result

        # Apply JPEG compression to match training conditions
        stage_start = time.perf_counter()
        head_img = self._apply_jpeg_compression_rgb(head_img)
        left_wrist_img = self._apply_jpeg_compression_rgb(left_wrist_img)

        debug_images = {'head': head_img, 'left_wrist': left_wrist_img}

        self._dump_debug_jpeg_images(debug_images)
        # print(
        #     f'[timing] jpeg_compression='
        #     f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
        #     flush=True)

        observation = {
            'qpos': state,
            self.camera_names[0]: head_img,  # 'head'
            self.camera_names[1]: left_wrist_img,  # 'left_wrist'
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _preprocess(self, instruction: str) -> dict:
        """Observe environment and build model inputs with timing logs."""
        stage_start = time.perf_counter()
        obs = self.update_observation_window()
        # print(
        #     f'[timing] update_observation_window='
        #     f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
        #     flush=True)

        obs['task_description'] = instruction

        stage_start = time.perf_counter()
        inputs = self.dataset(obs)
        # print(
        #     f'[timing] dataset_transform='
        #     f'{(time.perf_counter() - stage_start) * 1000.0:.3f} ms',
        #     flush=True)
        return inputs

    def _predict_action(self, inputs):
        """Run model inference with timing instrumentation."""
        self._action_ctx.inference_start = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        predict_start = time.perf_counter()
        raw_action = self.vla.predict_action(**inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        # print(
        #     f'[timing] vla_predict_call_returned='
        #     f'{(time.perf_counter() - predict_start) * 1000.0:.3f} ms',
        #     flush=True)
        return raw_action

    def _postprocess_actions(self, raw_action):
        """Split done from denormalized WBT actions before execution."""
        denormalized = self.denormalize_action(
            dict(action=raw_action.cpu().numpy()))
        denormalized = denormalized[:self.action_chunk]
        if not getattr(self, 'use_done_state_machine', True):
            return denormalized
        joint_actions = denormalized[:, :self.done_dim_index]
        self._action_ctx.done_chunk = denormalized[:, self.done_dim_index]
        return joint_actions

    def _execute_actions(self, actions: np.ndarray, rate):
        """Execute actions (sync or async), with optional interpolation.

        When target_hz is set and differs from publish_rate, uses
        frequency interpolation to upsample the trajectory.

        Args:
            actions (np.ndarray): Array of denormalized 42-dim actions.
            rate: Unused (kept for interface compatibility).
        """
        if not self._running:
            return

        ctx = self._action_ctx

        if self.async_execution and self._prev_ctx is not None:
            ctx.action_timestamp = ctx.inference_start
            offset = (time.time() - ctx.action_timestamp) / self._dt
            actions = resample_remaining(actions, offset)
        else:
            ctx.action_timestamp = time.time()
            if self.execute_horizon is not None:
                actions = actions[:self.execute_horizon]

        # print('actions:', actions, flush=True)
        # Use interpolated execution if target_hz differs from model rate
        if (self.target_hz is not None
                and self.target_hz != self.publish_rate):
            self.ros_operator.execute_trajectory_interpolated(
                actions,
                source_hz=self.publish_rate,
                target_hz=self.target_hz,
                method=self.interpolation_method,
                async_exec=self.async_execution,
                running_flag_fn=lambda: self._running)
        else:
            self.ros_operator.execute_trajectory(
                actions,
                dt=self._dt,
                async_exec=self.async_execution,
                running_flag_fn=lambda: self._running)

        if self.async_execution and self.execute_horizon is not None:
            time.sleep(self.execute_horizon * self._dt)

    def _move_to_prepare_pose(self):
        """No-op for Teleop02 WBT (teleop-controlled robot)."""
        pass

    def cleanup(self):
        """Clean up resources."""
        print('Cleaning up Teleop02WbtInferenceRunner')
        self._running = False

        if hasattr(self.ros_operator, 'stop_trajectory'):
            self.ros_operator.stop_trajectory()

        super().cleanup()
