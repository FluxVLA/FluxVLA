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
"""ROS operator for a dual-arm reBot."""

import time

import numpy as np

from fluxvla.engines.operators.base_operator import BaseOperator
from fluxvla.engines.utils.root import OPERATORS

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_yaw',
    'wrist_roll',
    'gripper',
]


@OPERATORS.register_module()
class RebotDualOperator(BaseOperator):
    """Synchronize reBot observations and publish 7-D commands per arm."""

    def __init__(
        self,
        img_left_topic='/camera_left_wrist/color/image_raw',
        img_right_topic='/camera_right_wrist/color/image_raw',
        img_front_topic='/camera_front/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_arm_right_topic='/right_arm/joint_states',
        puppet_arm_left_cmd_topic=(
            '/left_arm/rebot_joint_controller/target_joint_state'),
        puppet_arm_right_cmd_topic=(
            '/right_arm/rebot_joint_controller/target_joint_state'),
        sync_slop=0.05,
        sync_warning_enabled=True,
        sync_warning_target_hz=30.0,
        publish_rate=30,
        arm_steps_length=None,
        home_timeout=15.0,
        image_encoding='rgb8',
    ):
        self.img_left_topic = img_left_topic
        self.img_right_topic = img_right_topic
        self.img_front_topic = img_front_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_arm_right_topic = puppet_arm_right_topic
        self.puppet_arm_left_cmd_topic = puppet_arm_left_cmd_topic
        self.puppet_arm_right_cmd_topic = puppet_arm_right_cmd_topic
        self.command_mode = 'joint'
        self.publish_rate = int(publish_rate)
        self.home_timeout = float(home_timeout)
        if self.publish_rate <= 0:
            raise ValueError('publish_rate must be positive')
        if self.home_timeout <= 0:
            raise ValueError('home_timeout must be positive')
        if arm_steps_length is None:
            arm_steps_length = [0.02] * len(JOINT_NAMES)
        self.arm_steps_length = np.asarray(arm_steps_length, dtype=np.float32)
        if self.arm_steps_length.shape != (len(JOINT_NAMES), ):
            raise ValueError('arm_steps_length must contain seven values')
        if np.any(self.arm_steps_length <= 0):
            raise ValueError('arm_steps_length values must be positive')

        super().__init__(
            sync_slop=sync_slop,
            sync_queue_size=30,
            synced_frame_queue_size=10,
            sync_warning_enabled=sync_warning_enabled,
            sync_warning_target_hz=sync_warning_target_hz,
            image_encoding=image_encoding)
        self.left_joint_pub = None
        self.right_joint_pub = None
        self._init_ros()

    def _init_ros(self):
        """Initialize synchronized subscribers and command publishers."""
        import rospy
        from sensor_msgs.msg import JointState

        rospy.init_node('rebot_dual_operator', anonymous=True)
        self.setup_observation_sync(self.build_observation_specs())
        self.left_joint_pub = rospy.Publisher(
            self.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.right_joint_pub = rospy.Publisher(
            self.puppet_arm_right_cmd_topic, JointState, queue_size=10)

    def build_observation_specs(self):
        """Return the five streams that form one observation."""
        from sensor_msgs.msg import Image, JointState

        return [
            ('img_front', self.img_front_topic, Image),
            ('img_left', self.img_left_topic, Image),
            ('img_right', self.img_right_topic, Image),
            ('left_arm', self.puppet_arm_left_topic, JointState),
            ('right_arm', self.puppet_arm_right_topic, JointState),
        ]

    @staticmethod
    def joint_positions(message):
        """Return a JointState in the canonical reBot order."""
        if message.name:
            positions_by_name = dict(zip(message.name, message.position))
            if all(name in positions_by_name for name in JOINT_NAMES):
                values = [positions_by_name[name] for name in JOINT_NAMES]
            elif all(f'joint{index}' in positions_by_name
                     for index in range(len(JOINT_NAMES))):
                values = [
                    positions_by_name[f'joint{index}']
                    for index in range(len(JOINT_NAMES))
                ]
            else:
                raise ValueError(
                    'reBot JointState names must be shoulder_pan..gripper or '
                    'joint0..joint6')
        else:
            values = list(message.position)
        if len(values) != len(JOINT_NAMES):
            raise ValueError('reBot JointState must contain seven positions')
        positions = np.asarray(values, dtype=np.float32)
        if not np.isfinite(positions).all():
            raise ValueError('reBot JointState contains NaN or infinity')
        return positions

    def send_joints(self, arm_targets):
        """Publish left and/or right joint targets in canonical order."""
        unsupported = set(arm_targets) - {'left', 'right'}
        if unsupported:
            raise ValueError(f'Unsupported reBot arm target(s): {unsupported}')

        import rospy

        stamp = rospy.Time.now()
        if 'left' in arm_targets:
            self.left_joint_pub.publish(
                self._build_joint_state(arm_targets['left'], stamp))
        if 'right' in arm_targets:
            self.right_joint_pub.publish(
                self._build_joint_state(arm_targets['right'], stamp))

    @staticmethod
    def _build_joint_state(positions, stamp):
        from sensor_msgs.msg import JointState

        if len(positions) != len(JOINT_NAMES):
            raise ValueError('reBot joint command must contain seven values')
        message = JointState()
        message.header.stamp = stamp
        message.name = list(JOINT_NAMES)
        message.position = [float(value) for value in positions]
        return message

    def gohome(self, prepare_pose):
        """Interpolate both arms to an explicitly configured prepare pose."""
        import rospy

        targets = np.asarray(prepare_pose, dtype=np.float32)
        if targets.shape != (2, len(JOINT_NAMES)):
            raise ValueError('prepare_pose must have shape [2, 7]')

        deadline = time.monotonic() + self.home_timeout
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            frame = self.get_frame()
            if frame:
                current = np.stack([
                    self.joint_positions(frame['left_arm']),
                    self.joint_positions(frame['right_arm']),
                ])
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    'Timed out waiting for a synchronized reBot observation')
            rate.sleep()
        else:
            return

        max_step_count = np.max(
            np.abs(targets - current) / self.arm_steps_length)
        steps = max(1, int(np.ceil(max_step_count)))
        fractions = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[1:]
        trajectories = current[None] + fractions[:, None, None] * (
            targets - current)[None]
        self.clear_observation_queues()
        self.execute_trajectory(
            arm_trajectories={
                'left': trajectories[:, 0],
                'right': trajectories[:, 1],
            },
            dt=1.0 / self.publish_rate)
        self.clear_observation_queues()
