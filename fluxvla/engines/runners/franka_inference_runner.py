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
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.root import RUNNERS
from .base_inference_runner import BaseInferenceRunner


def resample_remaining(traj, offset):
    """Linearly interpolate remaining trajectory from a fractional offset."""
    N = traj.shape[0]
    M = N - int(offset)
    if M <= 0:
        return traj[:0]
    idx = np.clip(offset + np.arange(M), 0.0, N - 1.0)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, N - 1)
    alpha = (idx - lo)[:, np.newaxis]
    return traj[lo] + alpha * (traj[hi] - traj[lo])


@RUNNERS.register_module()
class FrankaInferenceRunner(BaseInferenceRunner):
    """Runner for dual Franka robot inference tasks.

    This runner handles real-time inference tasks for dual-arm Franka robotic
    manipulation using Vision-Language-Action (VLA) models. It manages ROS
    communication, observation collection, action prediction, and robot control
    for both Franka arms in a synchronized manner.

    Args:
        gripper_threshold (float, optional): Threshold for gripper action.
            Defaults to 0.05.
        prepare_pose (List[float], optional): Prepare pose for the robot.
            Defaults to None.
        async_execution (bool, optional): Whether to execute actions
            asynchronously.
            Defaults to False.
        execute_horizon (int, optional): Number of steps to execute from the
            action chunk.
            Defaults to None (execute all).
    """

    def __init__(self,
                 gripper_threshold: float = 0.05,
                 prepare_pose: List[float] = None,
                 action_mode: str = 'cartesian',
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 observation_timeout: float = 15.0,
                 *args,
                 **kwargs):
        self.gripper_threshold = gripper_threshold
        if action_mode not in {'cartesian', 'joint'}:
            raise ValueError(f'Unsupported Franka action_mode: {action_mode}')
        self.action_mode = action_mode
        self.async_execution = async_execution
        self.execute_horizon = execute_horizon
        self.observation_timeout = observation_timeout

        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = [
                'cam_front', 'cam_wrist_left', 'cam_wrist_right'
            ]

        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = {
                'type': 'FrankaDualOperator',
                'img_front_topic': '/camera_front/color/image_raw',
                'img_left_topic': '/camera_left/color/image_raw',
                'img_right_topic': '/camera_right/color/image_raw',
                'puppet_arm_left_topic': '/left_arm/joint_states',
                'puppet_arm_right_topic': '/right_arm/joint_states',
                'puppet_gripper_left_topic':
                '/left_arm/franka_gripper/joint_states',
                'puppet_gripper_right_topic':
                '/right_arm/franka_gripper/joint_states',
                'puppet_franka_state_left_topic':
                '/left_arm/franka_state_controller/franka_states',
                'puppet_franka_state_right_topic':
                '/right_arm/franka_state_controller/franka_states',
                'cartesian_cmd_left_topic':
                '/left_arm/cartesian_impedance_controller/equilibrium_pose',
                'cartesian_cmd_right_topic':
                '/right_arm/cartesian_impedance_controller/equilibrium_pose',
                'gripper_action_left_name': '/left_arm/franka_gripper/move',
                'gripper_action_right_name': '/right_arm/franka_gripper/move',
            }

        if 'task_descriptions' not in kwargs or kwargs[
                'task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1':
                'The right arm picks up the shuttlecock bucket, hands it to '
                'the left arm, and places it on the plate.'
            }

        super().__init__(*args, **kwargs)

        self.dt = 1.0 / self.publish_rate

        if prepare_pose is None:
            self.prepare_pose = None
        else:
            self.prepare_pose = prepare_pose
        self._remaining_instruction_chunks = None

    @staticmethod
    def _joint_state_to_arm_qpos(joint_state, gripper_width):
        positions = np.asarray(joint_state.position, dtype=np.float32)
        if positions.shape[0] < 7:
            raise ValueError(
                'Franka joint state must contain at least 7 arm joints, '
                f'got {positions.shape[0]}')
        return np.concatenate(
            (positions[:7], np.array([gripper_width], dtype=np.float32)),
            axis=0)

    def get_ros_observation(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any, float, float]:
        """Get synchronized observation data from ROS topics.

        Returns:
            Tuple containing:
                - img_front (np.ndarray): Front camera RGB image
                - img_left (np.ndarray): Left wrist camera RGB image
                - img_right (np.ndarray): Right wrist camera RGB image
                - puppet_arm_left (JointState): Left arm joint state
                - puppet_arm_right (JointState): Right arm joint state
                - puppet_ee_pose_left (PoseStamped | None): Left arm
                  end-effector pose when available
                - puppet_ee_pose_right (PoseStamped | None): Right arm
                  end-effector pose when available
                - gripper_left_width (float): Left gripper width
                - gripper_right_width (float): Right gripper width
        """
        import rospy

        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)

        rate = rospy.Rate(self.publish_rate)
        print_flag = True
        started_at = time.monotonic()
        last_status_at = 0.0
        rate.sleep()

        while not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    overwatch.info(
                        'Synchronization failed in get_ros_observation')
                    print_flag = False
                now = time.monotonic()
                if now - last_status_at > 2.0:
                    if hasattr(self.ros_operator, 'get_queue_status'):
                        overwatch.info(
                            f'Waiting for synchronized Franka observation: '
                            f'{self.ros_operator.get_queue_status()}')
                    last_status_at = now
                if (self.observation_timeout is not None
                        and now - started_at > self.observation_timeout):
                    queue_status = {}
                    if hasattr(self.ros_operator, 'get_queue_status'):
                        queue_status = self.ros_operator.get_queue_status()
                    raise TimeoutError(
                        'Timed out waiting for synchronized Franka '
                        'observation. '
                        f'queue_status={queue_status}')
                rate.sleep()
                continue

            print_flag = True
            (img_front, img_left, img_right, img_front_depth, img_left_depth,
             img_right_depth, puppet_arm_left, puppet_arm_right,
             puppet_ee_pose_left, puppet_ee_pose_right, puppet_gripper_left,
             puppet_gripper_right) = result

            return (img_front, img_left, img_right, puppet_arm_left,
                    puppet_arm_right, puppet_ee_pose_left,
                    puppet_ee_pose_right, puppet_gripper_left.data,
                    puppet_gripper_right.data)

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        Returns:
            Dict: Latest observation containing:
                - 'qpos': Robot state for both arms. In joint mode this is
                  16 dimensions: 2 arms x (7 joints + gripper).
                - Camera images keyed by camera names
        """
        from collections import deque

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        (img_front, img_left, img_right, puppet_arm_left, puppet_arm_right,
         puppet_ee_pose_left, puppet_ee_pose_right, gripper_left_width,
         gripper_right_width) = self.get_ros_observation()

        img_front = self._apply_jpeg_compression(img_front)
        img_left = self._apply_jpeg_compression(img_left)
        img_right = self._apply_jpeg_compression(img_right)

        if self.action_mode == 'joint':
            qpos_left = self._joint_state_to_arm_qpos(puppet_arm_left,
                                                      gripper_left_width)
            qpos_right = self._joint_state_to_arm_qpos(puppet_arm_right,
                                                       gripper_right_width)
        else:
            if puppet_ee_pose_left is None or puppet_ee_pose_right is None:
                raise ValueError(
                    'End-effector poses are required in cartesian mode')
            left_pose = puppet_ee_pose_left.pose
            right_pose = puppet_ee_pose_right.pose

            qpos_left = np.array([
                left_pose.position.x, left_pose.position.y,
                left_pose.position.z, left_pose.orientation.x,
                left_pose.orientation.y, left_pose.orientation.z,
                left_pose.orientation.w, gripper_left_width
            ])

            qpos_right = np.array([
                right_pose.position.x, right_pose.position.y,
                right_pose.position.z, right_pose.orientation.x,
                right_pose.orientation.y, right_pose.orientation.z,
                right_pose.orientation.w, gripper_right_width
            ])

        qpos = np.concatenate((qpos_left, qpos_right), axis=0)

        observation = {
            'qpos': qpos,
            self.camera_names[0]: img_front,
            self.camera_names[1]: img_left,
            self.camera_names[2]: img_right,
        }

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _move_to_prepare_pose(self):
        """Move robot to predefined preparation pose.

        The prepare_pose should be a tuple of two 8-element arrays.

        In joint mode each arm is [joint1..joint7, gripper_width].
        In cartesian mode each arm is [x, y, z, qx, qy, qz, qw,
        gripper_width].
        """
        from ..utils import initialize_overwatch
        overwatch = initialize_overwatch(__name__)

        if hasattr(self.ros_operator, 'stop_trajectory'):
            try:
                self.ros_operator.stop_trajectory(join_timeout=1.0)
            except TypeError:
                self.ros_operator.stop_trajectory()

        if self.prepare_pose is None and hasattr(self.ros_operator,
                                                 'home_both_arms'):
            overwatch.info('Returning Franka arms to home pose...')
            self.ros_operator.home_both_arms()
            self.observation_window = None
            overwatch.info('Franka home pose reached')
            return

        if self.prepare_pose is not None:
            overwatch.info('Moving to prepare pose...')
            left_pose, right_pose = self.prepare_pose

            # Validate pose dimensions
            if len(left_pose) != 8 or len(right_pose) != 8:
                raise ValueError(
                    f'Each prepare pose must have 8 elements '
                    f'for the configured action mode, '
                    f'got left={len(left_pose)}, right={len(right_pose)}')

            self.ros_operator.move_to_joints(left_pose, right_pose)
            self.observation_window = None
            overwatch.info('Prepare pose reached')
            return

        overwatch.warning(
            'No prepare_pose is configured and the operator has no '
            'home_both_arms method')

    def _get_user_task_instruction(self,
                                   default_instruction: str) -> List[str]:
        """Read Franka task input without changing the shared base runner."""
        while True:
            task_id = self._prompt_task_id()
            while self._is_reset_command(task_id):
                self._move_to_prepare_pose()
                task_id = self._prompt_task_id('Enter task ID after reset: ')

            if task_id in self.task_pose_sequences:
                self.execute_task_pose(task_id)
                task_id = self._prompt_task_id()

            num_times = self._prompt_repeat_count()
            if num_times is None:
                self._move_to_prepare_pose()
                continue

            task_description = self._get_task_description(task_id)
            self._remaining_instruction_chunks = num_times
            return [task_description] * num_times

    def _prompt_task_id(
        self,
        prompt:
        str = 'Enter task ID (or press Enter for default, 0/home to reset): '
    ) -> str:
        task_id = input(prompt).strip()
        return unicodedata.normalize('NFKC', task_id).strip()

    def _is_reset_command(self, value: str) -> bool:
        return value.lower() in {'0', 'home', 'h', 'reset'}

    def _prompt_repeat_count(self) -> Optional[int]:
        from ..utils import initialize_overwatch
        overwatch = initialize_overwatch(__name__)

        while True:
            value = input(
                'Number of times to repeat the task [1] (0/home to reset): '
            ).strip()
            value = unicodedata.normalize('NFKC', value).strip()
            if value == '':
                return 1
            if self._is_reset_command(value):
                return None
            try:
                num_times = int(value)
            except ValueError:
                overwatch.warning(
                    f'Invalid repeat count "{value}", please enter a '
                    f'positive integer.')
                continue
            if num_times <= 0:
                overwatch.warning('Repeat count must be a positive integer.')
                continue
            return num_times

    def _predict_action(self, inputs: dict):
        self._action_ctx.inference_start = time.time()
        raw_action = self.vla.predict_action(**inputs)
        return raw_action

    LEFT_GRIPPER_COL = 7
    RIGHT_GRIPPER_COL = 15
    GRIPPER_CLOSED = 0.0

    def _postprocess_actions(self, raw_action):
        """Denormalize and snap near-closed grippers to fully closed."""
        actions = super()._postprocess_actions(raw_action)
        for col in (self.LEFT_GRIPPER_COL, self.RIGHT_GRIPPER_COL):
            actions[:,
                    col] = np.where(actions[:, col] < self.gripper_threshold,
                                    self.GRIPPER_CLOSED, actions[:, col])
        return actions

    def _execute_actions(self, actions, rate):
        """Execute dual-arm actions (sync or async)."""
        if self.disable_puppet_arm:
            return

        ctx = self._action_ctx
        final_chunk = False
        if self._remaining_instruction_chunks is not None:
            final_chunk = self._remaining_instruction_chunks <= 1

        if self.async_execution and self._prev_ctx is not None:
            ctx.action_timestamp = ctx.inference_start
            offset = (time.time() - ctx.action_timestamp) / self.dt
            actions = resample_remaining(actions, offset)
        else:
            ctx.action_timestamp = time.time()
            if self.execute_horizon is not None:
                actions = actions[:self.execute_horizon]

        self.ros_operator.execute_trajectory(
            actions[:, :8],
            actions[:, 8:16],
            dt=self.dt,
            async_exec=self.async_execution)

        if self.async_execution and self.execute_horizon is not None:
            time.sleep(self.execute_horizon * self.dt)
            if final_chunk and hasattr(self.ros_operator, 'stop_trajectory'):
                try:
                    self.ros_operator.stop_trajectory(
                        join_timeout=1.0, hold_current=True)
                except TypeError:
                    self.ros_operator.stop_trajectory(join_timeout=1.0)

        if self._remaining_instruction_chunks is not None:
            self._remaining_instruction_chunks -= 1
            if self._remaining_instruction_chunks <= 0:
                self._remaining_instruction_chunks = None

    def cleanup(self):
        """Clean up resources."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Cleaning up FrankaInferenceRunner')

        if hasattr(self.ros_operator, 'stop_trajectory'):
            self.ros_operator.stop_trajectory()

        super().cleanup()

        overwatch.info('FrankaInferenceRunner cleanup completed')
