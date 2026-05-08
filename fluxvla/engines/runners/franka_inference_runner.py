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
from typing import Dict, List, Tuple

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
        async_execution (bool, optional): Whether to execute actions asynchronously.
            Defaults to False.
        execute_horizon (int, optional): Number of steps to execute from action chunk.
            Defaults to None (execute all).
    """

    def __init__(self,
                 gripper_threshold: float = 0.05,
                 prepare_pose: List[float] = None,
                 async_execution: bool = False,
                 execute_horizon: int = None,
                 *args,
                 **kwargs):
        self.gripper_threshold = gripper_threshold
        self.async_execution = async_execution
        self.execute_horizon = execute_horizon

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
                'puppet_gripper_left_topic': '/left_arm/franka_gripper/joint_states',
                'puppet_gripper_right_topic': '/right_arm/franka_gripper/joint_states',
                'puppet_franka_state_left_topic': '/left_arm/franka_state_controller/franka_states',
                'puppet_franka_state_right_topic': '/right_arm/franka_state_controller/franka_states',
                'cartesian_cmd_left_topic': '/left_arm/cartesian_impedance_controller/equilibrium_pose',
                'cartesian_cmd_right_topic': '/right_arm/cartesian_impedance_controller/equilibrium_pose',
                'gripper_action_left_name': '/left_arm/franka_gripper/move',
                'gripper_action_right_name': '/right_arm/franka_gripper/move',
            }

        if 'task_descriptions' not in kwargs or kwargs['task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1': 'The right arm picks up the shuttlecock bucket, hands it to the left arm, and places it on the plate.'
            }

        super().__init__(*args, **kwargs)

        self.dt = 1.0 / self.publish_rate

        if prepare_pose is None:
            self.prepare_pose = None
        else:
            self.prepare_pose = prepare_pose

    def get_ros_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 'PoseStamped', 'PoseStamped', float, float]:
        """Get synchronized observation data from ROS topics.

        Returns:
            Tuple containing:
                - img_front (np.ndarray): Front camera RGB image
                - img_left (np.ndarray): Left wrist camera RGB image
                - img_right (np.ndarray): Right wrist camera RGB image
                - puppet_ee_pose_left (PoseStamped): Left arm end-effector pose
                - puppet_ee_pose_right (PoseStamped): Right arm end-effector pose
                - gripper_left_width (float): Left gripper width
                - gripper_right_width (float): Right gripper width
        """
        import rospy
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)

        rate = rospy.Rate(self.publish_rate)
        print_flag = True
        rate.sleep()

        while not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    overwatch.info('Synchronization failed in get_ros_observation')
                    print_flag = False
                rate.sleep()
                continue

            print_flag = True
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, puppet_ee_pose_left, puppet_ee_pose_right,
             puppet_gripper_left, puppet_gripper_right) = result

            return (img_front, img_left, img_right, puppet_ee_pose_left, puppet_ee_pose_right,
                    puppet_gripper_left.data, puppet_gripper_right.data)

    def update_observation_window(self) -> Dict:
        """Update the observation window with latest sensor data.

        Returns:
            Dict: Latest observation containing:
                - 'qpos': End-effector poses from both arms (14 dimensions: 2 arms × 7 DOF)
                - Camera images keyed by camera names
        """
        from collections import deque

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)

            dummy_obs = {'qpos': None}
            for camera_name in self.camera_names:
                dummy_obs[camera_name] = None
            self.observation_window.append(dummy_obs)

        (img_front, img_left, img_right, puppet_ee_pose_left, puppet_ee_pose_right,
         gripper_left_width, gripper_right_width) = self.get_ros_observation()

        img_front = self._apply_jpeg_compression(img_front)
        img_left = self._apply_jpeg_compression(img_left)
        img_right = self._apply_jpeg_compression(img_right)

        left_pose = puppet_ee_pose_left.pose
        right_pose = puppet_ee_pose_right.pose

        qpos_left = np.array([
            left_pose.position.x, left_pose.position.y, left_pose.position.z,
            left_pose.orientation.x, left_pose.orientation.y, left_pose.orientation.z,
            left_pose.orientation.w,
            gripper_left_width
        ])

        qpos_right = np.array([
            right_pose.position.x, right_pose.position.y, right_pose.position.z,
            right_pose.orientation.x, right_pose.orientation.y, right_pose.orientation.z,
            right_pose.orientation.w,
            gripper_right_width
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

        The prepare_pose should be a tuple of two 8-element arrays:
        (left_arm_pose, right_arm_pose)

        Each pose: [x, y, z, qx, qy, qz, qw, gripper_width]
        """
        if self.prepare_pose is not None:
            from ..utils import initialize_overwatch
            overwatch = initialize_overwatch(__name__)

            overwatch.info('Moving to prepare pose...')
            left_pose, right_pose = self.prepare_pose

            # Validate pose dimensions
            if len(left_pose) != 8 or len(right_pose) != 8:
                raise ValueError(
                    f'Each prepare pose must have 8 elements [x,y,z,qx,qy,qz,qw,gripper], '
                    f'got left={len(left_pose)}, right={len(right_pose)}')

            self.ros_operator.move_to_joints(left_pose, right_pose)
            overwatch.info('Prepare pose reached')

    def _predict_action(self, inputs: dict):
        self._action_ctx.inference_start = time.time()
        raw_action = self.vla.predict_action(**inputs)
        return raw_action

    LEFT_GRIPPER_COL = 7
    RIGHT_GRIPPER_COL = 15
    GRIPPER_CLOSED = -0.01

    def _postprocess_actions(self, raw_action):
        """Denormalize and snap near-closed grippers to fully closed."""
        actions = super()._postprocess_actions(raw_action)
        for col in (self.LEFT_GRIPPER_COL, self.RIGHT_GRIPPER_COL):
            actions[:, col] = np.where(actions[:, col] < self.gripper_threshold,
                                       self.GRIPPER_CLOSED, actions[:, col])
        return actions

    def _execute_actions(self, actions, rate):
        """Execute dual-arm actions (sync or async)."""
        if self.disable_puppet_arm:
            return

        ctx = self._action_ctx

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

    def cleanup(self):
        """Clean up resources."""
        from ..utils import initialize_overwatch

        overwatch = initialize_overwatch(__name__)
        overwatch.info('Cleaning up FrankaInferenceRunner')

        if hasattr(self.ros_operator, 'stop_trajectory'):
            self.ros_operator.stop_trajectory()

        super().cleanup()

        overwatch.info('FrankaInferenceRunner cleanup completed')
