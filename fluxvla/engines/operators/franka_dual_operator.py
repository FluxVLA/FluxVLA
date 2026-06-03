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

import atexit
import csv
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fluxvla.engines.utils.root import OPERATORS

DEFAULT_HOME_JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]

DEFAULT_HOME_JOINT_POSITIONS = [
    0.0,
    -0.785398163397,
    0.0,
    -2.35619449019,
    0.0,
    1.57079632679,
    0.785398163397,
]

CARTESIAN_IMPEDANCE_CONTROLLER = 'cartesian_impedance_controller'
JOINT_IMPEDANCE_CONTROLLER = 'joint_impedance_controller'
JOINT_RUCKIG_POSITION_CONTROLLER = 'joint_ruckig_position_controller'
HOMING_CONTROLLER = 'joint_position_controller'
ALL_MANAGED_CONTROLLERS = (
    CARTESIAN_IMPEDANCE_CONTROLLER,
    JOINT_IMPEDANCE_CONTROLLER,
    JOINT_RUCKIG_POSITION_CONTROLLER,
    HOMING_CONTROLLER,
)


def replace_last_segment(input_string, new_segment='camera_info'):
    """Replace the last segment of a path-like string."""
    last_slash_index = input_string.rfind('/')
    if last_slash_index != -1:
        return input_string[:last_slash_index + 1] + new_segment
    return new_segment


@OPERATORS.register_module()
class FrankaDualOperator:
    """Dual Franka operator for ROS-based dual-arm control.

    This operator handles dual Franka arm control, multi-camera sensor data
    collection, and synchronization for dual Franka robotic systems in a ROS
    environment. Supports RGB and depth image streams from multiple cameras,
    end-effector poses and joint states for dual arms.
    """

    def __init__(
            self,
            img_left_topic,
            img_right_topic,
            img_front_topic,
            puppet_arm_left_topic,
            puppet_arm_right_topic,
            puppet_gripper_left_topic,
            puppet_gripper_right_topic,
            puppet_ee_pose_left_topic=None,
            puppet_ee_pose_right_topic=None,
            puppet_franka_state_left_topic=None,
            puppet_franka_state_right_topic=None,
            cartesian_cmd_left_topic=(
                '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
            cartesian_cmd_right_topic=(
                '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
            joint_cmd_left_topic=None,
            joint_cmd_right_topic=None,
            command_mode='cartesian',
            joint_names=None,
            command_controller_name=None,
            gripper_action_left_name='/left_arm/franka_gripper/move',
            gripper_action_right_name='/right_arm/franka_gripper/move',
            use_depth_image=False,
            img_left_depth_topic=None,
            img_right_depth_topic=None,
            img_front_depth_topic=None,
            gripper_speed=0.08,
            gripper_open_width=0.08,
            base_frame_id='',
            publish_rate=30,
            left_arm_ns='/left_arm',
            right_arm_ns='/right_arm',
            use_home_service=True,
            left_home_service='/left_arm/vive_controller_franka/home_arm',
            right_home_service='/right_arm/vive_controller_franka/home_arm',
            home_service_wait_timeout=1.0,
            home_joint_names=None,
            home_target_joint_positions=None,
            home_joint_tolerance=0.01,
            home_joint_velocity_tolerance=0.02,
            home_min_duration=11.0,
            home_settle_samples=5,
            home_timeout=25.0,
            left_home_joint_state_topic=None,
            right_home_joint_state_topic=None,
            enable_joint_tracking_plot=True,
            joint_tracking_output_dir='/home/franka/Data/raw_data/vla_tracking',
            qpos_stream_rate=50.0,
            qpos_smoothing_tau=0.0,
            joint_deadband=0.005):
        self.img_left_topic = img_left_topic
        self.img_right_topic = img_right_topic
        self.img_front_topic = img_front_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_arm_right_topic = puppet_arm_right_topic
        self.puppet_gripper_left_topic = puppet_gripper_left_topic
        self.puppet_gripper_right_topic = puppet_gripper_right_topic
        self.puppet_ee_pose_left_topic = puppet_ee_pose_left_topic
        self.puppet_ee_pose_right_topic = puppet_ee_pose_right_topic
        self.puppet_franka_state_left_topic = puppet_franka_state_left_topic
        self.puppet_franka_state_right_topic = puppet_franka_state_right_topic
        if command_mode not in {'cartesian', 'joint'}:
            raise ValueError(
                f'Unsupported Franka command_mode: {command_mode}')
        self.command_mode = command_mode
        self.left_arm_ns = left_arm_ns
        self.right_arm_ns = right_arm_ns
        self.cartesian_cmd_left_topic = cartesian_cmd_left_topic
        self.cartesian_cmd_right_topic = cartesian_cmd_right_topic
        self.joint_cmd_left_topic = (
            joint_cmd_left_topic or
            f'{self.left_arm_ns}/joint_ruckig_position_controller/target_joint_state'  # noqa: E501
        )
        self.joint_cmd_right_topic = (
            joint_cmd_right_topic or
            f'{self.right_arm_ns}/joint_ruckig_position_controller/target_joint_state'  # noqa: E501
        )
        self.joint_names = joint_names or DEFAULT_HOME_JOINT_NAMES
        self.command_controller_name = (
            command_controller_name
            or (JOINT_RUCKIG_POSITION_CONTROLLER if self.command_mode
                == 'joint' else CARTESIAN_IMPEDANCE_CONTROLLER))
        self.gripper_action_left_name = gripper_action_left_name
        self.gripper_action_right_name = gripper_action_right_name
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
        self.img_right_depth_topic = img_right_depth_topic
        self.img_front_depth_topic = img_front_depth_topic
        self.gripper_speed = gripper_speed
        self.gripper_open_width = gripper_open_width
        self.base_frame_id = base_frame_id
        self.publish_rate = publish_rate
        self.use_home_service = use_home_service
        self.left_home_service = left_home_service
        self.right_home_service = right_home_service
        self.home_service_wait_timeout = home_service_wait_timeout
        self.home_joint_names = home_joint_names or DEFAULT_HOME_JOINT_NAMES
        self.home_target_joint_positions = (
            home_target_joint_positions or DEFAULT_HOME_JOINT_POSITIONS)
        self.home_joint_tolerance = home_joint_tolerance
        self.home_joint_velocity_tolerance = home_joint_velocity_tolerance
        self.home_min_duration = home_min_duration
        self.home_settle_samples = home_settle_samples
        self.home_timeout = home_timeout
        self.left_home_joint_state_topic = (
            left_home_joint_state_topic
            or f'{self.left_arm_ns}/franka_state_controller/joint_states')
        self.right_home_joint_state_topic = (
            right_home_joint_state_topic
            or f'{self.right_arm_ns}/franka_state_controller/joint_states')
        self.enable_joint_tracking_plot = enable_joint_tracking_plot
        self.joint_tracking_output_dir = joint_tracking_output_dir
        self.qpos_stream_rate = float(qpos_stream_rate)
        self.qpos_smoothing_tau = float(qpos_smoothing_tau)
        self.joint_deadband = float(joint_deadband)

        if len(self.home_joint_names) != 7 or len(
                self.home_target_joint_positions) != 7:
            raise ValueError(
                'Home joint configuration must contain exactly 7 joints')
        if len(self.joint_names) != 7:
            raise ValueError(
                'Joint command configuration must contain exactly 7 joints')

        if self.use_depth_image:
            if not all([
                    img_left_depth_topic, img_right_depth_topic,
                    img_front_depth_topic
            ]):
                raise ValueError(
                    'When use_depth_image=True, all depth topics must be '
                    'provided')

        if (self.command_mode == 'cartesian'
                and self.puppet_ee_pose_left_topic is None
                and self.puppet_franka_state_left_topic is None):
            raise ValueError('Either puppet_ee_pose_left_topic or '
                             'puppet_franka_state_left_topic must be provided')

        if (self.command_mode == 'cartesian'
                and self.puppet_ee_pose_right_topic is None
                and self.puppet_franka_state_right_topic is None):
            raise ValueError(
                'Either puppet_ee_pose_right_topic or '
                'puppet_franka_state_right_topic must be provided')

        self._init()
        self._init_ros()
        atexit.register(self.save_joint_tracking_outputs)
        atexit.register(self.save_joint_action_outputs)

    def _init(self):
        from cv_bridge import CvBridge

        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.puppet_ee_pose_left_deque = deque()
        self.puppet_ee_pose_right_deque = deque()
        self.puppet_gripper_left_deque = deque()
        self.puppet_gripper_right_deque = deque()

        self.movegrip_left_client = None
        self.movegrip_right_client = None
        self._last_left_gripper_cmd = None
        self._last_right_gripper_cmd = None
        self._last_left_gripper_cmd_time = 0.0
        self._last_right_gripper_cmd_time = 0.0
        self.cam_info_dict = {}
        self._traj_thread = None
        self._traj_stop_event = threading.Event()
        self._joint_tracking_rows = []
        self._joint_tracking_start_time = None
        self._joint_tracking_saved = False
        self._joint_tracking_base = None
        self._last_joint_tracking_plot_time = 0.0
        self._joint_action_rows = []
        self._joint_action_start_time = None
        self._joint_action_saved = False
        self._last_joint_action_plot_time = 0.0
        self._last_stable_left_qpos = None
        self._last_stable_right_qpos = None

    def get_frame(self, slop=0.7):
        """Get synchronized frame data from all sensors."""
        required_queues_empty = (
            len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or len(self.puppet_arm_left_deque) == 0
            or len(self.puppet_arm_right_deque) == 0
            or len(self.puppet_gripper_left_deque) == 0
            or len(self.puppet_gripper_right_deque) == 0)
        if self.command_mode == 'cartesian':
            required_queues_empty = (
                required_queues_empty
                or len(self.puppet_ee_pose_left_deque) == 0
                or len(self.puppet_ee_pose_right_deque) == 0)

        depth_queues_empty = (
            self.use_depth_image and (len(self.img_left_depth_deque) == 0
                                      or len(self.img_right_depth_deque) == 0
                                      or len(self.img_front_depth_deque) == 0))

        if required_queues_empty or depth_queues_empty:
            return False

        frame_time = self._calculate_frame_time()
        if not self._check_sensor_data_availability(frame_time):
            return False

        frame_time_max = self._synchronize_queues(frame_time)
        if abs(frame_time_max - frame_time) > slop:
            self._flush_outdated_data(frame_time)
            return False

        return self._extract_synchronized_data()

    def _calculate_frame_time(self):
        timestamps = [
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec(),
            self.puppet_arm_left_deque[-1].header.stamp.to_sec(),
            self.puppet_arm_right_deque[-1].header.stamp.to_sec(),
            self.puppet_gripper_left_deque[-1].header.stamp.to_sec(),
            self.puppet_gripper_right_deque[-1].header.stamp.to_sec(),
        ]
        if self.command_mode == 'cartesian':
            timestamps.extend([
                self.puppet_ee_pose_left_deque[-1].header.stamp.to_sec(),
                self.puppet_ee_pose_right_deque[-1].header.stamp.to_sec(),
            ])
        if self.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_right_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec(),
            ])
        return min(timestamps)

    def _check_sensor_data_availability(self, frame_time):
        checks = [
            self.img_left_deque, self.img_right_deque, self.img_front_deque,
            self.puppet_arm_left_deque, self.puppet_arm_right_deque,
            self.puppet_gripper_left_deque, self.puppet_gripper_right_deque
        ]
        if self.command_mode == 'cartesian':
            checks.extend([
                self.puppet_ee_pose_left_deque,
                self.puppet_ee_pose_right_deque,
            ])
        for deque_obj in checks:
            if (len(deque_obj) == 0
                    or deque_obj[-1].header.stamp.to_sec() < frame_time):
                return False

        if self.use_depth_image:
            depth_checks = [
                self.img_left_depth_deque, self.img_right_depth_deque,
                self.img_front_depth_deque
            ]
            for deque_obj in depth_checks:
                if (len(deque_obj) == 0
                        or deque_obj[-1].header.stamp.to_sec() < frame_time):
                    return False
        return True

    def _synchronize_queues(self, frame_time):
        frame_time_max = 0
        queues_to_sync = [
            self.img_left_deque, self.img_right_deque, self.img_front_deque,
            self.puppet_arm_left_deque, self.puppet_arm_right_deque,
            self.puppet_gripper_left_deque, self.puppet_gripper_right_deque
        ]
        if self.command_mode == 'cartesian':
            queues_to_sync.extend([
                self.puppet_ee_pose_left_deque,
                self.puppet_ee_pose_right_deque,
            ])
        for queue in queues_to_sync:
            while queue[0].header.stamp.to_sec() < frame_time:
                queue.popleft()
            frame_time_max = max(frame_time_max,
                                 queue[0].header.stamp.to_sec())

        if self.use_depth_image:
            depth_queues = [
                self.img_left_depth_deque, self.img_right_depth_deque,
                self.img_front_depth_deque
            ]
            for queue in depth_queues:
                while queue[0].header.stamp.to_sec() < frame_time:
                    queue.popleft()
                frame_time_max = max(frame_time_max,
                                     queue[0].header.stamp.to_sec())
        return frame_time_max

    def _flush_outdated_data(self, frame_time):
        queues_to_flush = [
            self.img_left_deque, self.img_right_deque, self.img_front_deque,
            self.img_left_depth_deque, self.img_right_depth_deque,
            self.img_front_depth_deque, self.puppet_arm_left_deque,
            self.puppet_arm_right_deque, self.puppet_gripper_left_deque,
            self.puppet_gripper_right_deque
        ]
        if self.command_mode == 'cartesian':
            queues_to_flush.extend([
                self.puppet_ee_pose_left_deque,
                self.puppet_ee_pose_right_deque,
            ])
        for queue in queues_to_flush:
            while (len(queue) > 0
                   and queue[0].header.stamp.to_sec() <= frame_time):
                queue.popleft()

    def _extract_synchronized_data(self):
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(),
                                              'passthrough')
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(),
                                             'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(),
                                              'passthrough')

        puppet_arm_left = self.puppet_arm_left_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()
        if self.command_mode == 'cartesian':
            puppet_ee_pose_left = self.puppet_ee_pose_left_deque.popleft()
            puppet_ee_pose_right = self.puppet_ee_pose_right_deque.popleft()
        else:
            puppet_ee_pose_left = None
            puppet_ee_pose_right = None
        puppet_gripper_left = self.puppet_gripper_left_deque.popleft()
        puppet_gripper_right = self.puppet_gripper_right_deque.popleft()

        img_left_depth = None
        img_right_depth = None
        img_front_depth = None
        if self.use_depth_image:
            img_left_depth = self.bridge.imgmsg_to_cv2(
                self.img_left_depth_deque.popleft(), 'passthrough')
            img_right_depth = self.bridge.imgmsg_to_cv2(
                self.img_right_depth_deque.popleft(), 'passthrough')
            img_front_depth = self.bridge.imgmsg_to_cv2(
                self.img_front_depth_deque.popleft(), 'passthrough')

        return (img_front, img_left, img_right, img_front_depth,
                img_left_depth, img_right_depth, puppet_arm_left,
                puppet_arm_right, puppet_ee_pose_left, puppet_ee_pose_right,
                puppet_gripper_left, puppet_gripper_right)

    def _append_with_limit(self, deque_obj, msg, max_len=2000):
        if len(deque_obj) >= max_len:
            deque_obj.popleft()
        deque_obj.append(msg)

    def clear_observation_queues(self):
        for queue in self._all_observation_queues():
            queue.clear()

    def get_queue_status(self):
        return {
            'img_left': len(self.img_left_deque),
            'img_right': len(self.img_right_deque),
            'img_front': len(self.img_front_deque),
            'img_left_depth': len(self.img_left_depth_deque),
            'img_right_depth': len(self.img_right_depth_deque),
            'img_front_depth': len(self.img_front_depth_deque),
            'left_joint': len(self.puppet_arm_left_deque),
            'right_joint': len(self.puppet_arm_right_deque),
            'left_pose': len(self.puppet_ee_pose_left_deque),
            'right_pose': len(self.puppet_ee_pose_right_deque),
            'left_gripper': len(self.puppet_gripper_left_deque),
            'right_gripper': len(self.puppet_gripper_right_deque),
        }

    def _all_observation_queues(self):
        return (
            self.img_left_deque,
            self.img_right_deque,
            self.img_front_deque,
            self.img_left_depth_deque,
            self.img_right_depth_deque,
            self.img_front_depth_deque,
            self.puppet_arm_left_deque,
            self.puppet_arm_right_deque,
            self.puppet_ee_pose_left_deque,
            self.puppet_ee_pose_right_deque,
            self.puppet_gripper_left_deque,
            self.puppet_gripper_right_deque,
        )

    def img_left_callback(self, msg):
        self._append_with_limit(self.img_left_deque, msg)

    def img_right_callback(self, msg):
        self._append_with_limit(self.img_right_deque, msg)

    def img_front_callback(self, msg):
        self._append_with_limit(self.img_front_deque, msg)

    def img_left_depth_callback(self, msg):
        self._append_with_limit(self.img_left_depth_deque, msg)

    def img_right_depth_callback(self, msg):
        self._append_with_limit(self.img_right_depth_deque, msg)

    def img_front_depth_callback(self, msg):
        self._append_with_limit(self.img_front_depth_deque, msg)

    def puppet_arm_left_callback(self, msg):
        self._append_with_limit(self.puppet_arm_left_deque, msg)

    def puppet_arm_right_callback(self, msg):
        self._append_with_limit(self.puppet_arm_right_deque, msg)

    def puppet_ee_pose_left_callback(self, msg):
        self._append_with_limit(self.puppet_ee_pose_left_deque, msg)

    def puppet_ee_pose_right_callback(self, msg):
        self._append_with_limit(self.puppet_ee_pose_right_deque, msg)

    def puppet_gripper_left_callback(self, msg):
        stamped_width = self._joint_state_to_stamped_width(msg)
        self._append_with_limit(self.puppet_gripper_left_deque, stamped_width)

    def puppet_gripper_right_callback(self, msg):
        stamped_width = self._joint_state_to_stamped_width(msg)
        self._append_with_limit(self.puppet_gripper_right_deque, stamped_width)

    def puppet_franka_state_left_callback(self, msg):
        pose_msg = self._franka_state_to_pose_stamped(msg)
        self._append_with_limit(self.puppet_ee_pose_left_deque, pose_msg)

    def puppet_franka_state_right_callback(self, msg):
        pose_msg = self._franka_state_to_pose_stamped(msg)
        self._append_with_limit(self.puppet_ee_pose_right_deque, pose_msg)

    def _joint_state_to_stamped_width(self, msg):
        from std_msgs.msg import Header

        if not msg.position:
            gripper_width = 0.0
        elif len(msg.position) >= 2:
            gripper_width = float(msg.position[0] + msg.position[1])
        else:
            gripper_width = float(msg.position[0])

        stamped_width = SimpleNamespace()
        stamped_width.header = Header()
        stamped_width.header.stamp = msg.header.stamp
        stamped_width.data = gripper_width
        return stamped_width

    def _franka_state_to_pose_stamped(self, msg):
        from geometry_msgs.msg import Point, PoseStamped, Quaternion
        from tf.transformations import quaternion_from_matrix

        transform = np.array(
            msg.O_T_EE, dtype=np.float64).reshape((4, 4), order='F')
        quat = quaternion_from_matrix(transform)

        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        if self.base_frame_id and not pose_msg.header.frame_id:
            pose_msg.header.frame_id = self.base_frame_id
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

    def _init_ros(self):
        import warnings

        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image, JointState

        rospy.init_node('franka_dual_operator', anonymous=True)
        warnings.filterwarnings(
            'ignore',
            message=r'notifyAll\(\) is deprecated, use notify_all\(\) instead',
            category=DeprecationWarning,
            module=r'actionlib\.simple_action_client')
        camera_info_topics = []

        rospy.Subscriber(
            self.img_left_topic,
            Image,
            self.img_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_left_topic))

        rospy.Subscriber(
            self.img_right_topic,
            Image,
            self.img_right_callback,
            queue_size=1000,
            tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_right_topic))

        rospy.Subscriber(
            self.img_front_topic,
            Image,
            self.img_front_callback,
            queue_size=1000,
            tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_front_topic))

        if self.use_depth_image:
            rospy.Subscriber(
                self.img_left_depth_topic,
                Image,
                self.img_left_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)
            camera_info_topics.append(
                replace_last_segment(self.img_left_depth_topic))

            rospy.Subscriber(
                self.img_right_depth_topic,
                Image,
                self.img_right_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)
            camera_info_topics.append(
                replace_last_segment(self.img_right_depth_topic))

            rospy.Subscriber(
                self.img_front_depth_topic,
                Image,
                self.img_front_depth_callback,
                queue_size=1000,
                tcp_nodelay=True)
            camera_info_topics.append(
                replace_last_segment(self.img_front_depth_topic))

        rospy.Subscriber(
            self.puppet_arm_left_topic,
            JointState,
            self.puppet_arm_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.puppet_arm_right_topic,
            JointState,
            self.puppet_arm_right_callback,
            queue_size=1000,
            tcp_nodelay=True)

        if self.puppet_ee_pose_left_topic is not None:
            rospy.Subscriber(
                self.puppet_ee_pose_left_topic,
                PoseStamped,
                self.puppet_ee_pose_left_callback,
                queue_size=1000,
                tcp_nodelay=True)
        elif self.puppet_franka_state_left_topic is not None:
            from franka_msgs.msg import FrankaState
            rospy.Subscriber(
                self.puppet_franka_state_left_topic,
                FrankaState,
                self.puppet_franka_state_left_callback,
                queue_size=1000,
                tcp_nodelay=True)

        if self.puppet_ee_pose_right_topic is not None:
            rospy.Subscriber(
                self.puppet_ee_pose_right_topic,
                PoseStamped,
                self.puppet_ee_pose_right_callback,
                queue_size=1000,
                tcp_nodelay=True)
        elif self.puppet_franka_state_right_topic is not None:
            from franka_msgs.msg import FrankaState
            rospy.Subscriber(
                self.puppet_franka_state_right_topic,
                FrankaState,
                self.puppet_franka_state_right_callback,
                queue_size=1000,
                tcp_nodelay=True)

        rospy.Subscriber(
            self.puppet_gripper_left_topic,
            JointState,
            self.puppet_gripper_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.puppet_gripper_right_topic,
            JointState,
            self.puppet_gripper_right_callback,
            queue_size=1000,
            tcp_nodelay=True)

        self.left_ee_pub = rospy.Publisher(
            self.cartesian_cmd_left_topic, PoseStamped, queue_size=10)
        self.right_ee_pub = rospy.Publisher(
            self.cartesian_cmd_right_topic, PoseStamped, queue_size=10)
        self.left_joint_pub = rospy.Publisher(
            self.joint_cmd_left_topic, JointState, queue_size=10)
        self.right_joint_pub = rospy.Publisher(
            self.joint_cmd_right_topic, JointState, queue_size=10)

        if self.gripper_action_left_name:
            try:
                import actionlib
                from franka_gripper.msg import MoveAction
                self.movegrip_left_client = actionlib.SimpleActionClient(
                    self.gripper_action_left_name, MoveAction)
                if not self.movegrip_left_client.wait_for_server(
                        rospy.Duration(2.0)):
                    rospy.logwarn(
                        'Left Franka gripper action server not ready')
            except Exception as exc:
                rospy.logwarn(
                    'Failed to initialize left Franka gripper action: %s', exc)
                self.movegrip_left_client = None

        if self.gripper_action_right_name:
            try:
                import actionlib
                from franka_gripper.msg import MoveAction
                self.movegrip_right_client = actionlib.SimpleActionClient(
                    self.gripper_action_right_name, MoveAction)
                if not self.movegrip_right_client.wait_for_server(
                        rospy.Duration(2.0)):
                    rospy.logwarn(
                        'Right Franka gripper action server not ready')
            except Exception as exc:
                rospy.logwarn(
                    'Failed to initialize right Franka gripper action: %s',
                    exc)
                self.movegrip_right_client = None

        for topic in camera_info_topics:
            try:
                camera_info = rospy.wait_for_message(
                    topic, CameraInfo, timeout=5)
            except rospy.ROSException:
                continue
            self.cam_info_dict[topic] = {
                'rostopic': topic,
                'height': camera_info.height,
                'width': camera_info.width,
                'distortion_model': camera_info.distortion_model,
                'D': camera_info.D,
                'K': camera_info.K,
                'R': camera_info.R,
                'P': camera_info.P,
                'binning_x': camera_info.binning_x,
                'binning_y': camera_info.binning_y
            }

    def get_current_joint_states(self):
        left_pos = None
        right_pos = None
        if len(self.puppet_arm_left_deque) > 0:
            left_pos = np.array(self.puppet_arm_left_deque[-1].position)
        if len(self.puppet_arm_right_deque) > 0:
            right_pos = np.array(self.puppet_arm_right_deque[-1].position)
        return left_pos, right_pos

    def _latest_joint_actual(self, side):
        if side == 'left':
            queue = self.puppet_arm_left_deque
        else:
            queue = self.puppet_arm_right_deque
        if len(queue) == 0:
            return None, None
        msg = queue[-1]
        return (np.asarray(msg.position[:7], dtype=np.float64).copy(),
                msg.header.stamp.to_sec())

    def _record_joint_tracking_sample(self,
                                      left_target,
                                      right_target,
                                      left_velocity=None,
                                      right_velocity=None):
        if not self.enable_joint_tracking_plot or self.command_mode != 'joint':
            return

        now = time.monotonic()
        if self._joint_tracking_start_time is None:
            self._joint_tracking_start_time = now

        left_actual, left_stamp = self._latest_joint_actual('left')
        right_actual, right_stamp = self._latest_joint_actual('right')
        self._joint_tracking_rows.append({
            'time':
            now - self._joint_tracking_start_time,
            'left_target':
            np.asarray(left_target[:7], dtype=np.float64).copy(),
            'right_target':
            np.asarray(right_target[:7], dtype=np.float64).copy(),
            'left_velocity':
            None if left_velocity is None else
            np.asarray(left_velocity[:7], dtype=np.float64).copy(),
            'right_velocity':
            None if right_velocity is None else
            np.asarray(right_velocity[:7], dtype=np.float64).copy(),
            'left_actual':
            left_actual,
            'right_actual':
            right_actual,
            'left_stamp':
            left_stamp,
            'right_stamp':
            right_stamp,
        })

    def _joint_tracking_output_base(self):
        if self._joint_tracking_base is None:
            output_dir = Path(self.joint_tracking_output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self._joint_tracking_base = output_dir / f'vla_qpos_tracking_{timestamp}'
        return self._joint_tracking_base

    def save_joint_tracking_outputs(self, final=True):
        if (not self.enable_joint_tracking_plot or not self._joint_tracking_rows
                or (final and self._joint_tracking_saved)):
            return

        base = self._joint_tracking_output_base()
        csv_path = base.with_suffix('.csv')
        png_path = base.with_suffix('.png')

        fieldnames = ['time', 'left_stamp', 'right_stamp']
        for side in ('left', 'right'):
            for kind in ('target', 'actual', 'velocity'):
                for joint_name in self.joint_names:
                    fieldnames.append(f'{side}_{kind}_{joint_name}')

        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._joint_tracking_rows:
                flat = {
                    'time': row['time'],
                    'left_stamp': row['left_stamp'],
                    'right_stamp': row['right_stamp'],
                }
                for side in ('left', 'right'):
                    for kind in ('target', 'actual', 'velocity'):
                        values = row[f'{side}_{kind}']
                        for idx, joint_name in enumerate(self.joint_names):
                            key = f'{side}_{kind}_{joint_name}'
                            flat[key] = (
                                np.nan if values is None else float(values[idx]))
                writer.writerow(flat)

        now = time.monotonic()
        should_plot = final or now - self._last_joint_tracking_plot_time > 10.0
        if not should_plot:
            print(f'Saved VLA joint tracking CSV to {csv_path}')
            return
        self._last_joint_tracking_plot_time = now

        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f'Saved VLA joint tracking CSV to {csv_path}; '
                  f'failed to plot: {exc}')
            return

        times = np.asarray(
            [row['time'] for row in self._joint_tracking_rows],
            dtype=np.float64)
        fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
        for col, side in enumerate(('left', 'right')):
            for joint_idx, joint_name in enumerate(self.joint_names):
                ax = axes[joint_idx, col]
                target = np.asarray([
                    row[f'{side}_target'][joint_idx]
                    for row in self._joint_tracking_rows
                ])
                actual = np.asarray([
                    np.nan if row[f'{side}_actual'] is None else
                    row[f'{side}_actual'][joint_idx]
                    for row in self._joint_tracking_rows
                ])
                ax.plot(times, target, label='target', linewidth=1.2)
                ax.plot(times, actual, label='actual', linewidth=1.0)
                ax.set_ylabel(joint_name)
                ax.grid(True, alpha=0.3)
                if joint_idx == 0:
                    ax.set_title(f'{side} arm')
                if joint_idx == 6:
                    ax.set_xlabel('time [s]')
                if joint_idx == 0 and col == 0:
                    ax.legend(loc='best')
        fig.suptitle('VLA Inference Target vs Actual Joint State')
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f'Saved VLA joint tracking CSV to {csv_path}')
        print(f'Saved VLA joint tracking plot to {png_path}')
        if final:
            self._joint_tracking_saved = True

    def _record_joint_action_rows(self, source, left_traj, right_traj, dt,
                                  base_time):
        if not self.enable_joint_tracking_plot or self.command_mode != 'joint':
            return

        left_traj = np.asarray(left_traj, dtype=np.float64)
        right_traj = np.asarray(right_traj, dtype=np.float64)
        for idx in range(len(left_traj)):
            left = left_traj[idx]
            right = right_traj[idx]
            self._joint_action_rows.append({
                'source':
                source,
                'time':
                base_time + idx * dt,
                'left':
                left[:7].copy(),
                'right':
                right[:7].copy(),
                'left_gripper':
                np.nan if left.shape[0] <= 7 else float(left[7]),
                'right_gripper':
                np.nan if right.shape[0] <= 7 else float(right[7]),
            })

    def record_joint_action_trajectories(self, raw_left, raw_right, sent_left,
                                         sent_right, raw_dt, sent_dt):
        if not self.enable_joint_tracking_plot or self.command_mode != 'joint':
            return

        now = time.monotonic()
        if self._joint_action_start_time is None:
            self._joint_action_start_time = now
        base_time = now - self._joint_action_start_time

        self._record_joint_action_rows('raw_action', raw_left, raw_right,
                                       raw_dt, base_time)
        self._record_joint_action_rows('sent_trajectory', sent_left,
                                       sent_right, sent_dt, base_time)

    def save_joint_action_outputs(self, final=True):
        if (not self.enable_joint_tracking_plot or not self._joint_action_rows
                or (final and self._joint_action_saved)):
            return

        base = self._joint_tracking_output_base()
        csv_path = base.with_name(f'{base.name}_actions.csv')
        png_path = base.with_name(f'{base.name}_actions.png')

        fieldnames = ['source', 'time']
        for side in ('left', 'right'):
            for joint_name in self.joint_names:
                fieldnames.append(f'{side}_{joint_name}')
            fieldnames.append(f'{side}_gripper')

        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._joint_action_rows:
                flat = {'source': row['source'], 'time': row['time']}
                for side in ('left', 'right'):
                    values = row[side]
                    for idx, joint_name in enumerate(self.joint_names):
                        flat[f'{side}_{joint_name}'] = float(values[idx])
                    flat[f'{side}_gripper'] = row[f'{side}_gripper']
                writer.writerow(flat)

        now = time.monotonic()
        should_plot = final or now - self._last_joint_action_plot_time > 10.0
        if not should_plot:
            print(f'Saved VLA action CSV to {csv_path}')
            return
        self._last_joint_action_plot_time = now

        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f'Saved VLA action CSV to {csv_path}; '
                  f'failed to plot: {exc}')
            return

        fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
        for col, side in enumerate(('left', 'right')):
            for joint_idx, joint_name in enumerate(self.joint_names):
                ax = axes[joint_idx, col]
                for source, style in (
                    ('raw_action', dict(marker='o', linestyle='None',
                                        markersize=2.0, alpha=0.7)),
                    ('sent_trajectory', dict(linewidth=1.2, alpha=0.9)),
                ):
                    rows = [
                        row for row in self._joint_action_rows
                        if row['source'] == source
                    ]
                    if not rows:
                        continue
                    times = np.asarray([row['time'] for row in rows],
                                       dtype=np.float64)
                    values = np.asarray(
                        [row[side][joint_idx] for row in rows],
                        dtype=np.float64)
                    ax.plot(times, values, label=source, **style)
                ax.set_ylabel(joint_name)
                ax.grid(True, alpha=0.3)
                if joint_idx == 0:
                    ax.set_title(f'{side} arm')
                if joint_idx == 6:
                    ax.set_xlabel('time [s]')
                if joint_idx == 0 and col == 0:
                    ax.legend(loc='best')
        fig.suptitle('VLA Raw Actions vs Sent Joint Trajectory')
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f'Saved VLA action CSV to {csv_path}')
        print(f'Saved VLA action plot to {png_path}')
        if final:
            self._joint_action_saved = True

    def _build_joint_state(self, qpos):
        import rospy
        from sensor_msgs.msg import JointState

        if len(qpos) < 7:
            raise ValueError('Joint command must contain at least 7 joints')

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        msg.position = [float(q) for q in qpos[:7]]
        return msg

    def _execute_joint_step(self, left_joint, right_joint):
        self.left_joint_pub.publish(self._build_joint_state(left_joint))
        self.right_joint_pub.publish(self._build_joint_state(right_joint))
        self._record_joint_tracking_sample(left_joint, right_joint)

        if len(left_joint) > 7:
            self._send_gripper_command('left', left_joint[7])
        if len(right_joint) > 7:
            self._send_gripper_command('right', right_joint[7])

    def hold_current_joints(self, repeats=3, sleep_dt=0.02):
        if self.command_mode != 'joint':
            return

        import rospy

        left_actual, _ = self._latest_joint_actual('left')
        right_actual, _ = self._latest_joint_actual('right')
        if left_actual is None and right_actual is None:
            return

        for _ in range(repeats):
            if rospy.is_shutdown():
                break
            if left_actual is not None:
                self.left_joint_pub.publish(self._build_joint_state(left_actual))
            if right_actual is not None:
                self.right_joint_pub.publish(
                    self._build_joint_state(right_actual))
            rospy.sleep(sleep_dt)

    def execute_step(self, left_eepose, right_eepose):
        if self.command_mode == 'joint':
            self._execute_joint_step(left_eepose, right_eepose)
            return

        import rospy
        from geometry_msgs.msg import Point, PoseStamped, Quaternion

        msg_left = PoseStamped()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.header.frame_id = self.base_frame_id
        msg_left.pose.position = Point(
            x=float(left_eepose[0]),
            y=float(left_eepose[1]),
            z=float(left_eepose[2]))
        msg_left.pose.orientation = Quaternion(
            x=float(left_eepose[3]),
            y=float(left_eepose[4]),
            z=float(left_eepose[5]),
            w=float(left_eepose[6]))
        self.left_ee_pub.publish(msg_left)

        msg_right = PoseStamped()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.header.frame_id = self.base_frame_id
        msg_right.pose.position = Point(
            x=float(right_eepose[0]),
            y=float(right_eepose[1]),
            z=float(right_eepose[2]))
        msg_right.pose.orientation = Quaternion(
            x=float(right_eepose[3]),
            y=float(right_eepose[4]),
            z=float(right_eepose[5]),
            w=float(right_eepose[6]))
        self.right_ee_pub.publish(msg_right)

        if len(left_eepose) > 7:
            self._send_gripper_command('left', left_eepose[7])
        if len(right_eepose) > 7:
            self._send_gripper_command('right', right_eepose[7])

    def _send_gripper_command(self,
                              side,
                              gripper_width,
                              force=False,
                              wait=False,
                              wait_timeout=2.0):
        if side == 'left':
            client = self.movegrip_left_client
            last_attr = '_last_left_gripper_cmd'
            last_time_attr = '_last_left_gripper_cmd_time'
        elif side == 'right':
            client = self.movegrip_right_client
            last_attr = '_last_right_gripper_cmd'
            last_time_attr = '_last_right_gripper_cmd_time'
        else:
            raise ValueError(f'Unknown gripper side: {side}')

        if client is None:
            return

        target_width = max(0.0, float(gripper_width))
        last_width = getattr(self, last_attr)
        now = time.monotonic()
        if (not force and last_width is not None
                and abs(target_width - last_width) < 1e-4
                and now - getattr(self, last_time_attr) < 0.5):
            return

        try:
            import rospy
            from franka_gripper.msg import MoveGoal
            goal = MoveGoal()
            goal.width = target_width
            goal.speed = float(self.gripper_speed)
            client.send_goal(goal)
            if wait:
                client.wait_for_result(rospy.Duration(wait_timeout))
            setattr(self, last_attr, target_width)
            setattr(self, last_time_attr, now)
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send %s gripper command: %s', side, exc)

    def open_grippers(self, wait=False):
        self._send_gripper_command(
            'left', self.gripper_open_width, force=True, wait=wait)
        self._send_gripper_command(
            'right', self.gripper_open_width, force=True, wait=wait)

    @staticmethod
    def _enforce_strict_monotone(times, min_gap=1e-4):
        times = np.asarray(times, dtype=np.float64).copy()
        for idx in range(1, times.shape[0]):
            if times[idx] <= times[idx - 1]:
                times[idx] = times[idx - 1] + min_gap
        return times

    @classmethod
    def _one_pole_lowpass(cls, values, times, tau):
        filtered = np.asarray(values, dtype=np.float64).copy()
        if filtered.shape[0] < 2 or tau <= 0.0:
            return filtered

        times = cls._enforce_strict_monotone(times, min_gap=1e-4)
        for idx in range(1, filtered.shape[0]):
            dt = max(float(times[idx] - times[idx - 1]), 1e-4)
            alpha = dt / (tau + dt)
            filtered[idx] = (
                filtered[idx - 1] + alpha *
                (filtered[idx] - filtered[idx - 1]))
        return filtered

    @classmethod
    def _smooth_qpos_waypoints(cls, qpos, schedule, tau):
        qpos = np.asarray(qpos, dtype=np.float64)
        if tau <= 0.0 or qpos.shape[0] < 3:
            return qpos.copy()

        times = cls._enforce_strict_monotone(schedule, min_gap=1e-4)
        forward = cls._one_pole_lowpass(qpos, times, tau)
        reverse_times = times[-1] - times[::-1]
        smoothed = cls._one_pole_lowpass(forward[::-1], reverse_times,
                                         tau)[::-1]
        smoothed[0] = qpos[0]
        smoothed[-1] = qpos[-1]
        return smoothed

    @staticmethod
    def _interp_trajectory(schedule, output_times, trajectory):
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.shape[0] <= 1:
            return trajectory.copy()

        interpolated = np.empty((len(output_times), trajectory.shape[1]),
                                dtype=np.float64)
        for dim in range(trajectory.shape[1]):
            interpolated[:, dim] = np.interp(output_times, schedule,
                                             trajectory[:, dim])
        return interpolated

    def _apply_joint_deadband(self, trajectory, side):
        trajectory = np.asarray(trajectory, dtype=np.float64).copy()
        if self.joint_deadband <= 0.0 or len(trajectory) == 0:
            return trajectory
        if trajectory.shape[1] < 7:
            return trajectory

        stable_attr = (
            '_last_stable_left_qpos'
            if side == 'left' else '_last_stable_right_qpos')
        stable = getattr(self, stable_attr)
        if stable is None:
            stable = trajectory[0, :7].copy()

        for idx in range(len(trajectory)):
            qpos = trajectory[idx, :7]
            if np.max(np.abs(qpos - stable)) < self.joint_deadband:
                trajectory[idx, :7] = stable
            else:
                stable = qpos.copy()

        setattr(self, stable_attr, stable)
        return trajectory

    def _prepare_joint_trajectory(self, left_traj, right_traj, dt):
        left_traj = np.asarray(left_traj, dtype=np.float64)
        right_traj = np.asarray(right_traj, dtype=np.float64)
        left_traj = self._apply_joint_deadband(left_traj, 'left')
        right_traj = self._apply_joint_deadband(right_traj, 'right')
        if len(left_traj) <= 1 or self.qpos_stream_rate <= 0.0:
            return left_traj, right_traj, dt

        schedule = np.arange(len(left_traj), dtype=np.float64) * float(dt)
        schedule = self._enforce_strict_monotone(schedule, min_gap=1e-4)

        left_q = self._smooth_qpos_waypoints(left_traj[:, :7], schedule,
                                             self.qpos_smoothing_tau)
        right_q = self._smooth_qpos_waypoints(right_traj[:, :7], schedule,
                                              self.qpos_smoothing_tau)
        left_processed = left_traj.copy()
        right_processed = right_traj.copy()
        left_processed[:, :7] = left_q
        right_processed[:, :7] = right_q

        stream_dt = 1.0 / self.qpos_stream_rate
        output_times = np.arange(0.0, schedule[-1] + 1e-9, stream_dt)
        if output_times.size == 0 or output_times[-1] < schedule[-1]:
            output_times = np.append(output_times, schedule[-1])

        return (
            self._interp_trajectory(schedule, output_times, left_processed),
            self._interp_trajectory(schedule, output_times, right_processed),
            stream_dt,
        )

    def execute_trajectory(self,
                           left_trajectory,
                           right_trajectory,
                           dt=0.1,
                           async_exec=False,
                           base_velocity=None):
        left_traj = np.asarray(left_trajectory)
        right_traj = np.asarray(right_trajectory)
        raw_left_traj = left_traj.copy()
        raw_right_traj = right_traj.copy()
        exec_dt = dt
        if self.command_mode == 'joint':
            left_traj, right_traj, exec_dt = self._prepare_joint_trajectory(
                left_traj, right_traj, dt)
            self.record_joint_action_trajectories(
                raw_left_traj, raw_right_traj, left_traj, right_traj, dt,
                exec_dt)

        self.stop_trajectory(join_timeout=0.2)
        self._traj_stop_event = threading.Event()

        stop_event = self._traj_stop_event
        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_trajectory,
                args=(left_traj, right_traj, exec_dt, stop_event),
                daemon=True)
            self._traj_thread.start()
        else:
            self._run_trajectory(left_traj, right_traj, exec_dt, stop_event)

    def _run_trajectory(self, left_traj, right_traj, dt, stop_event):
        import rospy
        rate = rospy.Rate(1.0 / dt)
        for i in range(len(left_traj)):
            if rospy.is_shutdown() or stop_event.is_set():
                break
            self.execute_step(left_traj[i], right_traj[i])
            rate.sleep()
        self.save_joint_tracking_outputs(final=False)
        self.save_joint_action_outputs(final=False)

    def stop_trajectory(self, join_timeout=None, hold_current=False):
        self._traj_stop_event.set()
        if (join_timeout is not None and self._traj_thread is not None
                and self._traj_thread.is_alive()):
            self._traj_thread.join(timeout=join_timeout)
        if hold_current:
            self.hold_current_joints()

    def is_trajectory_running(self):
        return (self._traj_thread is not None and self._traj_thread.is_alive())

    def move_to_joints(self, left_eepose, right_eepose):
        import rospy
        rate = rospy.Rate(self.publish_rate)
        for _ in range(10):
            if rospy.is_shutdown():
                break
            self.execute_step(left_eepose, right_eepose)
            rate.sleep()

    def home_both_arms(self):
        import rospy

        self.stop_trajectory(join_timeout=1.0)
        self.clear_observation_queues()

        if self.use_home_service:
            service_result = self._try_home_both_arms_via_services()
            if service_result is True:
                self.open_grippers(wait=True)
                self.clear_observation_queues()
                return

        rospy.loginfo(
            'Homing both Franka arms via controller_manager fallback')
        rospy.set_param('/target_joint_positions',
                        list(self.home_target_joint_positions))
        helpers = [
            self._make_home_helper('left', self.left_arm_ns,
                                   self.left_home_joint_state_topic),
            self._make_home_helper('right', self.right_arm_ns,
                                   self.right_home_joint_state_topic),
        ]
        for helper in helpers:
            helper.wait_for_services()

        errors = []
        lock = threading.Lock()

        def _home(helper):
            try:
                helper.home()
                helper.activate_controller(self.command_controller_name)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(f'{helper.arm_label}: {exc}')

        threads = [
            threading.Thread(target=_home, args=(helper, ), daemon=True)
            for helper in helpers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self.home_timeout + 5.0)

        for helper, thread in zip(helpers, threads):
            if thread.is_alive():
                errors.append(f'{helper.arm_label}: homing thread timed out')

        if errors:
            raise rospy.ROSException('; '.join(errors))

        self.open_grippers(wait=True)
        self.clear_observation_queues()
        rospy.loginfo('Both Franka arms reached the home pose')

    def _try_home_both_arms_via_services(self):
        import rospy
        from std_srvs.srv import Trigger

        service_names = [self.left_home_service, self.right_home_service]
        try:
            for service_name in service_names:
                rospy.wait_for_service(
                    service_name, timeout=self.home_service_wait_timeout)
        except rospy.ROSException:
            rospy.logwarn('Franka home service is not available; '
                          'using controller_manager fallback')
            return False

        rospy.loginfo(
            'Homing both Franka arms via vive_controller home services')
        errors = []
        lock = threading.Lock()

        def _call(service_name):
            try:
                response = rospy.ServiceProxy(service_name, Trigger)()
                if not response.success:
                    raise rospy.ROSException(response.message)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(f'{service_name}: {exc}')

        threads = [
            threading.Thread(target=_call, args=(service_name, ), daemon=True)
            for service_name in service_names
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=self.home_timeout + 5.0)

        for service_name, thread in zip(service_names, threads):
            if thread.is_alive():
                errors.append(f'{service_name}: service call timed out')

        if errors:
            raise rospy.ROSException('; '.join(errors))

        rospy.loginfo('Both Franka arms reached the home pose')
        return True

    def _make_home_helper(self, arm_label, arm_ns, joint_state_topic):
        import rospy
        from controller_manager_msgs.srv import (ListControllers,
                                                 LoadController,
                                                 SwitchController,
                                                 SwitchControllerRequest,
                                                 UnloadController)
        from sensor_msgs.msg import JointState

        return _ArmHomeHelper(
            rospy=rospy,
            arm_label=arm_label,
            arm_ns=arm_ns,
            joint_state_topic=joint_state_topic,
            home_joint_names=self.home_joint_names,
            home_target_joint_positions=self.home_target_joint_positions,
            home_joint_tolerance=self.home_joint_tolerance,
            home_joint_velocity_tolerance=self.home_joint_velocity_tolerance,
            home_min_duration=self.home_min_duration,
            home_settle_samples=self.home_settle_samples,
            home_timeout=self.home_timeout,
            list_controllers_cls=ListControllers,
            load_controller_cls=LoadController,
            switch_controller_cls=SwitchController,
            unload_controller_cls=UnloadController,
            switch_controller_request_cls=SwitchControllerRequest,
            joint_state_cls=JointState)


class _ArmHomeHelper:

    def __init__(self, rospy, arm_label, arm_ns, joint_state_topic,
                 home_joint_names, home_target_joint_positions,
                 home_joint_tolerance, home_joint_velocity_tolerance,
                 home_min_duration, home_settle_samples, home_timeout,
                 list_controllers_cls, load_controller_cls,
                 switch_controller_cls, unload_controller_cls,
                 switch_controller_request_cls, joint_state_cls):
        self.rospy = rospy
        self.arm_label = arm_label
        self.arm_ns = arm_ns
        self.joint_state_topic = joint_state_topic
        self.home_joint_names = list(home_joint_names)
        self.home_target_joint_positions = list(home_target_joint_positions)
        self.home_joint_tolerance = float(home_joint_tolerance)
        self.home_joint_velocity_tolerance = float(
            home_joint_velocity_tolerance)
        self.home_min_duration = float(home_min_duration)
        self.home_settle_samples = int(home_settle_samples)
        self.home_timeout = float(home_timeout)
        self.SwitchControllerRequest = switch_controller_request_cls
        self.JointState = joint_state_cls

        cm_ns = f'{arm_ns}/controller_manager'
        self.srv_list = rospy.ServiceProxy(f'{cm_ns}/list_controllers',
                                           list_controllers_cls)
        self.srv_load = rospy.ServiceProxy(f'{cm_ns}/load_controller',
                                           load_controller_cls)
        self.srv_switch = rospy.ServiceProxy(f'{cm_ns}/switch_controller',
                                             switch_controller_cls)
        self.srv_unload = rospy.ServiceProxy(f'{cm_ns}/unload_controller',
                                             unload_controller_cls)
        self._service_names = [
            f'{cm_ns}/list_controllers',
            f'{cm_ns}/load_controller',
            f'{cm_ns}/switch_controller',
            f'{cm_ns}/unload_controller',
        ]

    def wait_for_services(self):
        for service_name in self._service_names:
            self.rospy.wait_for_service(
                service_name, timeout=self.home_timeout)
        self.rospy.wait_for_message(
            self.joint_state_topic, self.JointState, timeout=self.home_timeout)

    def get_controller_states(self):
        response = self.srv_list()
        return {
            controller.name: controller.state
            for controller in response.controller
        }

    def ensure_loaded(self, controller_name):
        states = self.get_controller_states()
        if controller_name in states:
            return
        response = self.srv_load(controller_name)
        if not response.ok:
            raise self.rospy.ROSException(
                f'{self.arm_label}: failed to load controller '
                f'{controller_name}')

    def switch(self, start_controllers, stop_controllers):
        start_controllers = [name for name in start_controllers if name]
        stop_controllers = [name for name in stop_controllers if name]
        if not start_controllers and not stop_controllers:
            return
        request = self.SwitchControllerRequest()
        request.start_controllers = start_controllers
        request.stop_controllers = stop_controllers
        request.strictness = self.SwitchControllerRequest.STRICT
        request.start_asap = False
        request.timeout = 0.0
        response = self.srv_switch(request)
        if not response.ok:
            raise self.rospy.ROSException(
                f'{self.arm_label}: failed to switch controllers '
                f'(start={start_controllers}, stop={stop_controllers})')

    def activate_controller(self, target):
        self.ensure_loaded(target)
        states = self.get_controller_states()
        stop = [
            name for name in ALL_MANAGED_CONTROLLERS
            if name != target and states.get(name) == 'running'
        ]
        start = [] if states.get(target) == 'running' else [target]
        self.switch(start, stop)

    def home(self):
        self.activate_controller(HOMING_CONTROLLER)
        started_at = time.monotonic()
        deadline = started_at + self.home_timeout
        settled_samples = 0

        while time.monotonic() < deadline and not self.rospy.is_shutdown():
            joint_state = self.rospy.wait_for_message(
                self.joint_state_topic, self.JointState, timeout=1.0)
            if time.monotonic() - started_at < self.home_min_duration:
                settled_samples = 0
                continue
            if self.home_reached(joint_state):
                settled_samples += 1
                if settled_samples >= self.home_settle_samples:
                    self.rospy.sleep(0.3)
                    return
            else:
                settled_samples = 0

        raise self.rospy.ROSException(
            f'{self.arm_label}: timed out while moving to home pose')

    def home_reached(self, joint_state):
        positions = dict(zip(joint_state.name, joint_state.position))
        velocities = dict(zip(joint_state.name, joint_state.velocity))
        for joint_name, target in zip(self.home_joint_names,
                                      self.home_target_joint_positions):
            current = positions.get(joint_name)
            if current is None:
                return False
            if abs(current - target) > self.home_joint_tolerance:
                return False
            velocity = velocities.get(joint_name)
            if velocity is None:
                return False
            if abs(velocity) > self.home_joint_velocity_tolerance:
                return False
        return True
