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
from ..utils.trajectory_utils import resample_remaining
from .base_inference_runner import BaseInferenceRunner


@RUNNERS.register_module()
class FrankaInferenceRunner(BaseInferenceRunner):
    """Runner for single-arm or dual-arm Franka inference tasks.

    This runner handles real-time inference tasks for Franka robotic
    manipulation using Vision-Language-Action (VLA) models. It manages ROS
    communication, observation collection, action prediction, and robot control
    for the configured active arm(s) in a synchronized manner.

    Args:
        gripper_threshold (float, optional): Threshold for gripper action.
            Defaults to 0.05.
        prepare_pose (List[float], optional): Prepare pose for the robot.
            Defaults to None.
        active_arms (Tuple[str, ...], optional): Ordered active Franka arms.
            Defaults to ('left', 'right').
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
                 active_arms: Tuple[str, ...] = ('left', 'right'),
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 observation_timeout: float = 15.0,
                 *args,
                 **kwargs):
        self.gripper_threshold = gripper_threshold
        if action_mode not in {'cartesian', 'joint'}:
            raise ValueError(f'Unsupported Franka action_mode: {action_mode}')
        self.action_mode = action_mode
        self.active_arms = self._validate_active_arms(active_arms)
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
                'command_mode': self.action_mode,
                'img_front_topic': '/camera_front/color/image_raw',
                'img_left_topic': '/camera_left/color/image_raw',
                'img_right_topic': '/camera_right/color/image_raw',
                'puppet_arm_left_topic': '/left_arm/joint_states',
                'puppet_arm_right_topic': '/right_arm/joint_states',
                'puppet_franka_state_left_topic':
                '/left_arm/franka_state_controller/franka_states',
                'puppet_franka_state_right_topic':
                '/right_arm/franka_state_controller/franka_states',
                'cartesian_cmd_left_topic':
                '/left_arm/cartesian_impedance_controller/equilibrium_pose',
                'cartesian_cmd_right_topic':
                '/right_arm/cartesian_impedance_controller/equilibrium_pose',
                'joint_cmd_left_topic':
                '/left_arm/ruckig_joint_impedance_controller/target_joint_state',  # noqa: E501
                'joint_cmd_right_topic':
                '/right_arm/ruckig_joint_impedance_controller/target_joint_state',  # noqa: E501
                'gripper_left_topic':
                '/left_arm/franka_gripper/move/goal',
                'gripper_right_topic':
                '/right_arm/franka_gripper/move/goal',
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
    def _validate_active_arms(active_arms):
        if isinstance(active_arms, str):
            active_arms = (active_arms, )
        active_arms = tuple(active_arms)
        if not active_arms:
            raise ValueError('active_arms must contain at least one arm')
        unsupported = set(active_arms) - {'left', 'right'}
        if unsupported:
            raise ValueError(f'Unsupported Franka active_arms: {unsupported}')
        if len(set(active_arms)) != len(active_arms):
            raise ValueError(f'Duplicate Franka active_arms: {active_arms}')
        return active_arms

    @staticmethod
    def _arm_action_slice(arm_index):
        start = arm_index * 8
        return slice(start, start + 8)

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

    @staticmethod
    def _gripper_width_from_joint_state(joint_state):
        positions = dict(zip(joint_state.name, joint_state.position))
        finger1 = positions.get('panda_finger_joint1')
        finger2 = positions.get('panda_finger_joint2')
        if finger1 is None or finger2 is None:
            raise ValueError(
                'Franka JointState must contain panda_finger_joint1 and '
                'panda_finger_joint2 for gripper width')
        return float(finger1 + finger2)

    def _pose_from_frame(self, frame, side):
        pose = frame.get(f'{side}_pose')
        if pose is not None:
            return pose

        franka_state = frame.get(f'{side}_franka_state')
        if franka_state is not None:
            return self._franka_state_to_pose_stamped(franka_state)
        return None

    @staticmethod
    def _franka_state_to_pose_stamped(msg):
        from geometry_msgs.msg import Point, PoseStamped, Quaternion
        from tf.transformations import quaternion_from_matrix

        transform = np.asarray(
            msg.O_T_EE, dtype=np.float64).reshape((4, 4), order='F')
        quat = quaternion_from_matrix(transform)

        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose.position = Point(
            x=float(transform[0, 3]),
            y=float(transform[1, 3]),
            z=float(transform[2, 3]))
        pose_msg.pose.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]))
        return pose_msg

    def get_ros_observation(self) -> Dict[str, Dict[str, Any]]:
        """Get synchronized observation data from ROS topics."""
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
            images = self._images_from_frame(result)
            arms = {
                arm: self._arm_observation_from_frame(result, arm)
                for arm in self.active_arms
            }

            return {'images': images, 'arms': arms}

    def _images_from_frame(self, frame):
        image_key_pairs = (
            ('img_front', self.camera_names[0]),
            ('img_left', self.camera_names[1]
             if len(self.camera_names) > 1 else None),
            ('img_right', self.camera_names[2]
             if len(self.camera_names) > 2 else None),
        )
        return {
            camera_name: self._apply_jpeg_compression(frame[frame_key])
            for frame_key, camera_name in image_key_pairs
            if camera_name is not None and frame_key in frame
        }

    def _arm_observation_from_frame(self, frame, arm):
        joint_state = frame[f'{arm}_arm']
        return {
            'joint_state': joint_state,
            'pose': self._pose_from_frame(frame, arm),
            'gripper_width': self._gripper_width_from_joint_state(joint_state),
        }

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        Returns:
            Dict: Latest observation containing:
                - 'qpos': Robot state for active arms.
                - Camera images keyed by camera names
        """
        from collections import deque

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        ros_obs = self.get_ros_observation()
        qpos = np.concatenate([
            self._qpos_from_arm_observation(arm, ros_obs['arms'][arm])
            for arm in self.active_arms
        ],
                              axis=0)

        observation = {'qpos': qpos}
        observation.update(ros_obs['images'])

        self.observation_window.append(observation)
        return self.observation_window[-1]

    def _qpos_from_arm_observation(self, arm, arm_obs):
        if self.action_mode == 'joint':
            return self._joint_state_to_arm_qpos(arm_obs['joint_state'],
                                                 arm_obs['gripper_width'])

        pose_msg = arm_obs['pose']
        if pose_msg is None:
            raise ValueError(
                f'End-effector pose is required for {arm} in cartesian mode')
        pose = pose_msg.pose
        return np.array([
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z,
            pose.orientation.w, arm_obs['gripper_width']
        ])

    def _move_to_prepare_pose(self):
        """Move robot to predefined preparation pose.

        Each active arm pose has 8 elements. In joint mode each arm is
        [joint1..joint7, gripper_width]. In cartesian mode each arm is
        [x, y, z, qx, qy, qz, qw, gripper_width].
        """
        from ..utils import initialize_overwatch
        overwatch = initialize_overwatch(__name__)

        self.ros_operator.stop_trajectory()

        if self.prepare_pose is None:
            overwatch.info('Returning Franka arms to home pose...')
            self.ros_operator.gohome()
            self.observation_window = None
            overwatch.info('Franka home pose reached')
            return

        if self.prepare_pose is not None:
            overwatch.info('Moving to prepare pose...')
            poses = self._prepare_pose_by_arm()
            self._send_prepare_pose(poses)
            self.observation_window = None
            overwatch.info('Prepare pose reached')
            return

        overwatch.warning('No prepare_pose is configured')

    def _prepare_pose_by_arm(self):
        if len(self.active_arms) == 1:
            arm = self.active_arms[0]
            pose = self.prepare_pose
            if (len(pose) == 1 and isinstance(pose[0],
                                              (list, tuple, np.ndarray))):
                pose = pose[0]
            if len(pose) != 8:
                raise ValueError(
                    f'Prepare pose for {arm} must have 8 elements, '
                    f'got {len(pose)}')
            return {arm: pose}

        if len(self.prepare_pose) != len(self.active_arms):
            raise ValueError(
                f'prepare_pose must contain {len(self.active_arms)} arm poses, '
                f'got {len(self.prepare_pose)}')

        poses = {}
        for arm, pose in zip(self.active_arms, self.prepare_pose):
            if len(pose) != 8:
                raise ValueError(
                    f'Prepare pose for {arm} must have 8 elements, '
                    f'got {len(pose)}')
            poses[arm] = pose
        return poses

    def _send_prepare_pose(self, poses):
        arm_targets = {
            arm: pose[:7]
            for arm, pose in poses.items()
        }
        if self.action_mode == 'cartesian':
            self.ros_operator.send_eepose(arm_targets)
        else:
            self.ros_operator.send_joints(arm_targets)

        self.ros_operator.send_gripper(
            {
                arm: pose[7]
                for arm, pose in poses.items()
            },
            wait=True)

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

    GRIPPER_CLOSED = 0.0

    def _postprocess_actions(self, raw_action):
        """Denormalize and snap near-closed grippers to fully closed."""
        actions = super()._postprocess_actions(raw_action)
        for arm_index in range(len(self.active_arms)):
            col = self._arm_action_slice(arm_index).start + 7
            if col >= actions.shape[1]:
                continue
            actions[:,
                    col] = np.where(actions[:, col] < self.gripper_threshold,
                                    self.GRIPPER_CLOSED, actions[:, col])
        return actions

    def _execute_actions(self, actions, rate):
        """Execute active-arm actions (sync or async)."""
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

        self._validate_action_width(actions)
        arm_trajectories = {}
        gripper_trajectories = {}
        for arm_index, arm in enumerate(self.active_arms):
            arm_slice = self._arm_action_slice(arm_index)
            arm_trajectories[arm] = actions[:, arm_slice.start:arm_slice.start
                                            + 7]
            gripper_trajectories[arm] = actions[:, arm_slice.start + 7]

        self.ros_operator.execute_trajectory(
            arm_trajectories=arm_trajectories,
            gripper_trajectories=gripper_trajectories,
            dt=self.dt,
            async_exec=self.async_execution)

        if self.async_execution and self.execute_horizon is not None:
            time.sleep(self.execute_horizon * self.dt)
            if final_chunk:
                self.ros_operator.stop_trajectory()

        if self._remaining_instruction_chunks is not None:
            self._remaining_instruction_chunks -= 1
            if self._remaining_instruction_chunks <= 0:
                self._remaining_instruction_chunks = None

    def _validate_action_width(self, actions):
        required = len(self.active_arms) * 8
        if actions.shape[1] < required:
            raise ValueError(
                f'Franka actions must have at least {required} columns for '
                f'active_arms={self.active_arms}, got {actions.shape[1]}')

    def cleanup(self):
        """Clean up resources."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Cleaning up FrankaInferenceRunner')

        self.ros_operator.stop_trajectory()

        super().cleanup()

        overwatch.info('FrankaInferenceRunner cleanup completed')
