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
"""Inference runner for a dual-arm reBot."""

from typing import Dict

import numpy as np

from fluxvla.engines.utils.root import RUNNERS
from .base_inference_runner import BaseInferenceRunner


@RUNNERS.register_module()
class RebotDualInferenceRunner(BaseInferenceRunner):
    """Run the reBot 14-D observation and joint-command loop."""

    def __init__(self, prepare_pose=None, *args, **kwargs):
        if 'camera_names' not in kwargs or kwargs['camera_names'] is None:
            kwargs['camera_names'] = [
                'cam_front', 'cam_wrist_left', 'cam_wrist_right'
            ]
        if 'operator' not in kwargs or kwargs['operator'] is None:
            kwargs['operator'] = dict(
                type='RebotDualOperator',
                img_front_topic='/camera_front/color/image_raw',
                img_left_topic='/camera_left_wrist/color/image_raw',
                img_right_topic='/camera_right_wrist/color/image_raw',
                puppet_arm_left_topic='/left_arm/joint_states',
                puppet_arm_right_topic='/right_arm/joint_states',
                puppet_arm_left_cmd_topic=(
                    '/left_arm/rebot_joint_controller/target_joint_state'),
                puppet_arm_right_cmd_topic=(
                    '/right_arm/rebot_joint_controller/target_joint_state'))
        if 'task_descriptions' not in kwargs or kwargs[
                'task_descriptions'] is None:
            kwargs['task_descriptions'] = {
                '1': 'Complete the requested bimanual task.'
            }

        super().__init__(*args, **kwargs)
        if len(self.camera_names) != 3:
            raise ValueError('reBot inference requires exactly three cameras')
        self.dt = 1.0 / self.publish_rate
        self.prepare_pose = prepare_pose

    def get_ros_observation(self) -> Dict:
        """Wait for one synchronized three-camera, dual-arm observation."""
        import rospy

        from ..utils import initialize_overwatch

        logger = initialize_overwatch(__name__)
        rate = rospy.Rate(self.publish_rate)
        should_log = True
        rate.sleep()
        while not rospy.is_shutdown():
            frame = self.ros_operator.get_frame()
            if frame:
                return frame
            if should_log:
                logger.info('Synchronization failed in get_ros_observation')
                should_log = False
            rate.sleep()
        return {}

    def update_observation_window(self) -> Dict:
        """Build the model observation using the canonical left-right order."""
        from collections import deque

        if self.observation_window is None:
            self.observation_window = deque(maxlen=2)
            dummy = {'qpos': None}
            dummy.update({name: None for name in self.camera_names})
            self.observation_window.append(dummy)

        frame = self.get_ros_observation()
        qpos = np.concatenate([
            self.ros_operator.joint_positions(frame['left_arm']),
            self.ros_operator.joint_positions(frame['right_arm']),
        ])
        observation = {
            'qpos': qpos,
            self.camera_names[0]:
            self._apply_jpeg_compression(frame['img_front']),
            self.camera_names[1]:
            self._apply_jpeg_compression(frame['img_left']),
            self.camera_names[2]:
            self._apply_jpeg_compression(frame['img_right']),
        }
        self.observation_window.append(observation)
        return observation

    def _move_to_prepare_pose(self):
        """Move only when a deployment supplies a verified prepare pose."""
        if self.prepare_pose is not None:
            self.ros_operator.gohome(self.prepare_pose)
            self.observation_window = None

    def _execute_actions(self, actions, rate):
        """Execute a 14-D action chunk as two seven-joint trajectories."""
        del rate
        if self.disable_puppet_arm:
            return
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] < 14:
            raise ValueError(
                f'reBot actions must have shape [T, >=14], got {actions.shape}'
            )
        self.ros_operator.execute_trajectory(
            arm_trajectories={
                'left': actions[:, :7],
                'right': actions[:, 7:14],
            },
            dt=self.dt)

    def cleanup(self):
        """Stop an active trajectory before base runner cleanup."""
        self.ros_operator.stop_trajectory()
        super().cleanup()
