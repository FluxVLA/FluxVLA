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

import threading
import time
from collections import deque

import numpy as np

from fluxvla.engines.utils.root import OPERATORS

DEFAULT_JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]

CARTESIAN_IMPEDANCE_CONTROLLER = 'cartesian_impedance_controller'
JOINT_RUCKIG_POSITION_CONTROLLER = 'joint_ruckig_position_controller'


def replace_last_segment(input_string, new_segment='camera_info'):
    last_slash_index = input_string.rfind('/')
    if last_slash_index != -1:
        return input_string[:last_slash_index + 1] + new_segment
    return new_segment


@OPERATORS.register_module()
class FrankaDualOperator:
    """Dual Franka operator backed by message_filters synchronized observation.

    The action interface is intentionally explicit:
    left/right arm trajectories are sent separately, and grippers are controlled
    by separate left/right width trajectories.
    """

    def __init__(
            self,
            img_left_topic,
            img_right_topic,
            img_front_topic,
            puppet_arm_left_topic,
            puppet_arm_right_topic,
            puppet_gripper_left_topic=None,  # kept for config compatibility
            puppet_gripper_right_topic=None,  # kept for config compatibility
            puppet_ee_pose_left_topic=None,
            puppet_ee_pose_right_topic=None,
            puppet_franka_state_left_topic=None,
            puppet_franka_state_right_topic=None,
            use_depth_image=False,
            img_left_depth_topic=None,
            img_right_depth_topic=None,
            img_front_depth_topic=None,
            base_frame_id='',
            sync_slop=0.04,
            sync_queue_size=30,
            synced_frame_queue_size=10,
            sync_warning_enabled=True,
            sync_warning_target_hz=30.0,
            sync_warning_window=2.0,
            sync_warning_min_hz_ratio=0.9,
            command_mode='joint',
            left_arm_ns='/left_arm',
            right_arm_ns='/right_arm',
            cartesian_cmd_left_topic=(
                '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
            cartesian_cmd_right_topic=(
                '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
            joint_cmd_left_topic=None,
            joint_cmd_right_topic=None,
            joint_names=None,
            gripper_action_left_name='/left_arm/franka_gripper/move',
            gripper_action_right_name='/right_arm/franka_gripper/move',
            gripper_speed=0.08,
            gripper_open_width=0.08,
            home_service='/cmd/home',
            home_service_wait_timeout=1.0,
            auto_switch_controller=True,
            controller_switch_strict=False,
            controller_switch_timeout=1.0,
            gripper_server_wait_timeout=2.0,
            **unused_kwargs):
        self.img_left_topic = img_left_topic
        self.img_right_topic = img_right_topic
        self.img_front_topic = img_front_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_arm_right_topic = puppet_arm_right_topic
        self.puppet_ee_pose_left_topic = puppet_ee_pose_left_topic
        self.puppet_ee_pose_right_topic = puppet_ee_pose_right_topic
        self.puppet_franka_state_left_topic = puppet_franka_state_left_topic
        self.puppet_franka_state_right_topic = puppet_franka_state_right_topic
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
        self.img_right_depth_topic = img_right_depth_topic
        self.img_front_depth_topic = img_front_depth_topic
        self.base_frame_id = base_frame_id
        self.sync_slop = float(sync_slop)
        self.sync_queue_size = int(sync_queue_size)
        self.synced_frame_queue_size = int(synced_frame_queue_size)
        self.sync_warning_enabled = bool(sync_warning_enabled)
        self.sync_warning_target_hz = float(sync_warning_target_hz)
        self.sync_warning_window = float(sync_warning_window)
        self.sync_warning_min_hz_ratio = float(sync_warning_min_hz_ratio)

        if command_mode not in {'joint', 'cartesian'}:
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
        self.joint_names = joint_names or DEFAULT_JOINT_NAMES
        self.gripper_action_left_name = gripper_action_left_name
        self.gripper_action_right_name = gripper_action_right_name
        self.gripper_speed = float(gripper_speed)
        self.gripper_open_width = float(gripper_open_width)
        self.home_service = home_service
        self.home_service_wait_timeout = float(home_service_wait_timeout)
        self.auto_switch_controller = bool(auto_switch_controller)
        self.controller_switch_strict = bool(controller_switch_strict)
        self.controller_switch_timeout = float(controller_switch_timeout)
        self.gripper_server_wait_timeout = float(gripper_server_wait_timeout)

        if self.use_depth_image and not all([
                img_left_depth_topic, img_right_depth_topic,
                img_front_depth_topic
        ]):
            raise ValueError(
                'When use_depth_image=True, all depth topics must be provided')
        if len(self.joint_names) != 7:
            raise ValueError('joint_names must contain exactly 7 joints')

        self._init_runtime()
        self._init_ros()

    def _init_runtime(self):
        from cv_bridge import CvBridge

        self.bridge = CvBridge()
        self.cam_info_dict = {}
        self._lock = threading.Lock()
        self._frames = deque(maxlen=self.synced_frame_queue_size)
        self._sync_names = []
        self._sync_subscribers = []
        self._sync = None
        self._sync_window_started_at = time.monotonic()
        self._sync_window_count = 0
        self.left_ee_pub = None
        self.right_ee_pub = None
        self.left_joint_pub = None
        self.right_joint_pub = None
        self.movegrip_left_client = None
        self.movegrip_right_client = None
        self._last_left_gripper_cmd = None
        self._last_right_gripper_cmd = None
        self._last_left_gripper_cmd_time = 0.0
        self._last_right_gripper_cmd_time = 0.0
        self._active_command_controller = None
        self._controller_switch_warning_shown = False
        self._traj_thread = None
        self._traj_stop_event = threading.Event()

    def _init_ros(self):
        import message_filters
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image, JointState

        rospy.init_node('franka_dual_operator', anonymous=True)

        camera_info_topics = self._setup_sync(
            message_filters=message_filters,
            rospy=rospy,
            Image=Image,
            JointState=JointState,
            PoseStamped=PoseStamped)
        self._setup_control(rospy, PoseStamped, JointState)
        self._load_camera_info(rospy, CameraInfo, camera_info_topics)

    def get_frame(self, slop=0.7):  # noqa: ARG002
        """Return the newest synchronized frame and discard stale frames."""
        with self._lock:
            if not self._frames:
                return False
            frame = self._frames[-1]
            self._frames.clear()
        return self._format_frame(frame)

    def clear_observation_queues(self):
        with self._lock:
            self._frames.clear()
            self._sync_window_started_at = time.monotonic()
            self._sync_window_count = 0

    def get_queue_status(self):
        return {'synced_frames': len(self._frames)}

    def _setup_sync(self, message_filters, rospy, Image, JointState,
                    PoseStamped):
        specs = [
            ('img_front', self.img_front_topic, Image),
            ('img_left', self.img_left_topic, Image),
            ('img_right', self.img_right_topic, Image),
            ('left_arm', self.puppet_arm_left_topic, JointState),
            ('right_arm', self.puppet_arm_right_topic, JointState),
        ]
        camera_info_topics = [
            replace_last_segment(self.img_front_topic),
            replace_last_segment(self.img_left_topic),
            replace_last_segment(self.img_right_topic),
        ]

        self._add_pose_specs(specs, PoseStamped)
        if self.use_depth_image:
            specs.extend([
                ('img_front_depth', self.img_front_depth_topic, Image),
                ('img_left_depth', self.img_left_depth_topic, Image),
                ('img_right_depth', self.img_right_depth_topic, Image),
            ])
            camera_info_topics.extend([
                replace_last_segment(self.img_front_depth_topic),
                replace_last_segment(self.img_left_depth_topic),
                replace_last_segment(self.img_right_depth_topic),
            ])

        self._sync_names = [name for name, _, _ in specs]
        self._sync_subscribers = [
            message_filters.Subscriber(topic, msg_cls)
            for _, topic, msg_cls in specs
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            self._sync_subscribers,
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False)
        self._sync.registerCallback(self._sync_callback)
        rospy.loginfo(
            'Franka observation sync: %d topics, slop=%.3fs',
            len(specs),
            self.sync_slop)
        return camera_info_topics

    def _add_pose_specs(self, specs, PoseStamped):
        if self.puppet_ee_pose_left_topic is not None:
            specs.append(
                ('left_pose', self.puppet_ee_pose_left_topic, PoseStamped))
        elif self.puppet_franka_state_left_topic is not None:
            from franka_msgs.msg import FrankaState
            specs.append(('left_franka_state',
                          self.puppet_franka_state_left_topic, FrankaState))

        if self.puppet_ee_pose_right_topic is not None:
            specs.append(
                ('right_pose', self.puppet_ee_pose_right_topic, PoseStamped))
        elif self.puppet_franka_state_right_topic is not None:
            from franka_msgs.msg import FrankaState
            specs.append(('right_franka_state',
                          self.puppet_franka_state_right_topic, FrankaState))

    def _setup_control(self, rospy, PoseStamped, JointState):
        self.left_ee_pub = rospy.Publisher(
            self.cartesian_cmd_left_topic, PoseStamped, queue_size=10)
        self.right_ee_pub = rospy.Publisher(
            self.cartesian_cmd_right_topic, PoseStamped, queue_size=10)
        self.left_joint_pub = rospy.Publisher(
            self.joint_cmd_left_topic, JointState, queue_size=10)
        self.right_joint_pub = rospy.Publisher(
            self.joint_cmd_right_topic, JointState, queue_size=10)

        self.movegrip_left_client = self._make_gripper_client(
            rospy, self.gripper_action_left_name, 'left')
        self.movegrip_right_client = self._make_gripper_client(
            rospy, self.gripper_action_right_name, 'right')

    def _make_gripper_client(self, rospy, action_name, side):
        if not action_name:
            return None
        try:
            import actionlib
            from franka_gripper.msg import MoveAction
            client = actionlib.SimpleActionClient(action_name, MoveAction)
            if not client.wait_for_server(
                    rospy.Duration(self.gripper_server_wait_timeout)):
                rospy.logwarn('%s Franka gripper action server not ready',
                              side.capitalize())
                return None
            return client
        except Exception as exc:
            rospy.logwarn('Failed to initialize %s gripper action: %s', side,
                          exc)
            return None

    def _sync_callback(self, *msgs):
        raw = dict(zip(self._sync_names, msgs))
        frame = self._build_frame(raw)
        with self._lock:
            self._frames.append(frame)
            self._record_sync_output()

    def _record_sync_output(self):
        if (not self.sync_warning_enabled or self.sync_warning_target_hz <= 0.0
                or self.sync_warning_window <= 0.0):
            return

        now = time.monotonic()
        self._sync_window_count += 1
        elapsed = now - self._sync_window_started_at
        if elapsed < self.sync_warning_window:
            return

        observed_hz = self._sync_window_count / elapsed
        min_hz = self.sync_warning_target_hz * self.sync_warning_min_hz_ratio
        if observed_hz < min_hz:
            self._log_sync_rate_warning(observed_hz, min_hz, elapsed)

        self._sync_window_started_at = now
        self._sync_window_count = 0

    def _log_sync_rate_warning(self, observed_hz, min_hz, elapsed):
        import rospy

        rospy.logwarn(
            'Franka message_filters sync rate low: observed=%.2fHz, '
            'minimum=%.2fHz over %.1fs',
            observed_hz,
            min_hz,
            elapsed)

    def _build_frame(self, raw):
        left_arm = raw['left_arm']
        right_arm = raw['right_arm']
        return {
            'img_front': raw['img_front'],
            'img_left': raw['img_left'],
            'img_right': raw['img_right'],
            'img_front_depth': raw.get('img_front_depth'),
            'img_left_depth': raw.get('img_left_depth'),
            'img_right_depth': raw.get('img_right_depth'),
            'left_arm': left_arm,
            'right_arm': right_arm,
            'left_pose': self._pose_from_raw(raw, 'left'),
            'right_pose': self._pose_from_raw(raw, 'right'),
            'left_gripper_width': self._gripper_width_from_joint_state(
                left_arm),
            'right_gripper_width': self._gripper_width_from_joint_state(
                right_arm),
        }

    def _format_frame(self, frame):
        return (
            self._to_cv_image(frame['img_front']),
            self._to_cv_image(frame['img_left']),
            self._to_cv_image(frame['img_right']),
            self._to_optional_cv_image(frame['img_front_depth']),
            self._to_optional_cv_image(frame['img_left_depth']),
            self._to_optional_cv_image(frame['img_right_depth']),
            frame['left_arm'],
            frame['right_arm'],
            frame['left_pose'],
            frame['right_pose'],
            frame['left_gripper_width'],
            frame['right_gripper_width'],
        )

    def _to_cv_image(self, msg):
        return self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def _to_optional_cv_image(self, msg):
        if msg is None:
            return None
        return self._to_cv_image(msg)

    def _pose_from_raw(self, raw, side):
        pose_key = f'{side}_pose'
        franka_state_key = f'{side}_franka_state'
        if pose_key in raw:
            return raw[pose_key]
        if franka_state_key in raw:
            return self._franka_state_to_pose_stamped(raw[franka_state_key])
        return None

    @staticmethod
    def _gripper_width_from_joint_state(msg):
        positions = dict(zip(msg.name, msg.position))
        finger1 = positions.get('panda_finger_joint1')
        finger2 = positions.get('panda_finger_joint2')
        if finger1 is None or finger2 is None:
            return None
        return float(finger1 + finger2)

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

    def _load_camera_info(self, rospy, CameraInfo, camera_info_topics):
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
                'binning_y': camera_info.binning_y,
            }

    def send_joints(self, left_qpos, right_qpos):
        self._ensure_command_controller(JOINT_RUCKIG_POSITION_CONTROLLER)
        self.left_joint_pub.publish(self._build_joint_state(left_qpos))
        self.right_joint_pub.publish(self._build_joint_state(right_qpos))

    def send_eepose(self, left_eepose, right_eepose):
        self._ensure_command_controller(CARTESIAN_IMPEDANCE_CONTROLLER)
        self.left_ee_pub.publish(self._build_pose_stamped(left_eepose))
        self.right_ee_pub.publish(self._build_pose_stamped(right_eepose))

    def send_gripper(self, left_width=None, right_width=None, wait=False):
        if left_width is None and right_width is None:
            raise ValueError('At least one gripper width must be provided')
        self._send_gripper_pair(
            left_width, right_width, force=True, wait=wait)

    def execute_trajectory(self,
                           left_arm_trajectory,
                           right_arm_trajectory,
                           left_gripper_trajectory,
                           right_gripper_trajectory,
                           head_trajectory=None,
                           dt: float = 0.1,
                           async_exec: bool = False):
        """Execute dual-arm trajectory in sync or async mode."""
        if dt <= 0.0:
            raise ValueError('dt must be positive')

        n = len(left_arm_trajectory)
        if len(right_arm_trajectory) != n:
            raise ValueError(
                'left_arm_trajectory and right_arm_trajectory length mismatch')
        if left_gripper_trajectory is not None and len(
                left_gripper_trajectory) != n:
            raise ValueError(
                'left_gripper_trajectory length must match arm trajectory')
        if right_gripper_trajectory is not None and len(
                right_gripper_trajectory) != n:
            raise ValueError(
                'right_gripper_trajectory length must match arm trajectory')

        self._traj_stop_event.set()
        self._traj_stop_event = threading.Event()
        stop_event = self._traj_stop_event
        args = (left_arm_trajectory, right_arm_trajectory,
                left_gripper_trajectory, right_gripper_trajectory,
                head_trajectory, dt, stop_event)

        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_trajectory, args=args, daemon=True)
            self._traj_thread.start()
        else:
            self._run_trajectory(*args)

    def _run_trajectory(self, left_arm_trajectory, right_arm_trajectory,
                        left_gripper_trajectory, right_gripper_trajectory,
                        head_trajectory, dt, stop_event):
        import rospy

        del head_trajectory
        rate = rospy.Rate(1.0 / dt)
        for idx in range(len(left_arm_trajectory)):
            if rospy.is_shutdown() or stop_event.is_set():
                return

            if self.command_mode == 'cartesian':
                self.send_eepose(left_arm_trajectory[idx],
                                 right_arm_trajectory[idx])
            else:
                self.send_joints(left_arm_trajectory[idx],
                                 right_arm_trajectory[idx])

            left_width = None
            right_width = None
            if left_gripper_trajectory is not None:
                left_width = float(left_gripper_trajectory[idx])
            if right_gripper_trajectory is not None:
                right_width = float(right_gripper_trajectory[idx])
            if left_width is not None or right_width is not None:
                self._send_gripper_pair(left_width, right_width)

            rate.sleep()

    def stop_trajectory(self):
        self._traj_stop_event.set()

    def is_trajectory_running(self):
        return (self._traj_thread is not None and self._traj_thread.is_alive())

    def home_both_arms(self):
        return self.gohome()

    def gohome(self):
        import rospy
        import rosservice

        self.clear_observation_queues()
        rospy.wait_for_service(
            self.home_service, timeout=self.home_service_wait_timeout)
        service_cls = rosservice.get_service_class_by_name(self.home_service)
        if service_cls is None:
            raise rospy.ROSException(
                f'Unable to resolve service type for {self.home_service}')

        rospy.loginfo('Homing both Franka arms via %s service',
                      self.home_service)
        response = rospy.ServiceProxy(self.home_service, service_cls)()
        if getattr(response, 'success', True) is False:
            message = getattr(response, 'message', '')
            raise rospy.ROSException(f'{self.home_service} failed: {message}')

        self._active_command_controller = None
        self.open_grippers(wait=True)
        self.clear_observation_queues()
        return response

    def open_grippers(self, wait=False):
        self._send_gripper_pair(
            self.gripper_open_width,
            self.gripper_open_width,
            force=True,
            wait=wait)

    def _build_joint_state(self, qpos):
        import rospy
        from sensor_msgs.msg import JointState

        if len(qpos) < 7:
            raise ValueError('Joint command must contain at least 7 values')

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        msg.position = [float(value) for value in qpos[:7]]
        return msg

    def _build_pose_stamped(self, eepose):
        import rospy
        from geometry_msgs.msg import Point, PoseStamped, Quaternion

        if len(eepose) < 7:
            raise ValueError('EE pose command must contain at least 7 values')

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame_id
        msg.pose.position = Point(
            x=float(eepose[0]), y=float(eepose[1]), z=float(eepose[2]))
        msg.pose.orientation = Quaternion(
            x=float(eepose[3]),
            y=float(eepose[4]),
            z=float(eepose[5]),
            w=float(eepose[6]))
        return msg

    def _send_gripper_pair(self,
                           left_width,
                           right_width,
                           force=False,
                           wait=False):
        if left_width is not None:
            self._send_gripper_command(
                'left', left_width, force=force, wait=wait)
        if right_width is not None:
            self._send_gripper_command(
                'right', right_width, force=force, wait=wait)

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
            goal.speed = self.gripper_speed
            client.send_goal(goal)
            if wait:
                client.wait_for_result(rospy.Duration(wait_timeout))
            setattr(self, last_attr, target_width)
            setattr(self, last_time_attr, now)
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send %s gripper command: %s', side, exc)

    def _ensure_command_controller(self, controller_name):
        if (not self.auto_switch_controller
                or self._active_command_controller == controller_name):
            return

        stop_controllers = self._controllers_to_stop(controller_name)
        for namespace in (self.left_arm_ns, self.right_arm_ns):
            if not self._switch_arm_controller(namespace, controller_name,
                                               stop_controllers):
                return
        self._active_command_controller = controller_name

    @staticmethod
    def _controllers_to_stop(controller_name):
        known_controllers = {
            CARTESIAN_IMPEDANCE_CONTROLLER,
            JOINT_RUCKIG_POSITION_CONTROLLER,
        }
        return sorted(known_controllers - {controller_name})

    def _switch_arm_controller(self, namespace, start_controller,
                               stop_controllers):
        import rospy

        service_name = self._controller_service_name(
            namespace, 'switch_controller')
        try:
            from controller_manager_msgs.srv import (
                SwitchController,
                SwitchControllerRequest,
            )
            rospy.wait_for_service(
                service_name, timeout=self.controller_switch_timeout)
            request = SwitchControllerRequest()
            request.start_controllers = [start_controller]
            request.stop_controllers = list(stop_controllers)
            request.strictness = getattr(SwitchControllerRequest,
                                         'BEST_EFFORT', 1)
            request.start_asap = True
            request.timeout = self.controller_switch_timeout
            response = rospy.ServiceProxy(service_name, SwitchController)(
                request)
            if not getattr(response, 'ok', False):
                raise rospy.ROSException(
                    f'{service_name} returned ok=False')
            return True
        except Exception as exc:
            if self.controller_switch_strict:
                raise
            self.auto_switch_controller = False
            if not self._controller_switch_warning_shown:
                rospy.logwarn(
                    'Failed to switch Franka controller via %s: %s. '
                    'Controller auto-switch is disabled for this operator.',
                    service_name,
                    exc)
                self._controller_switch_warning_shown = True
            return False

    @staticmethod
    def _controller_service_name(namespace, service):
        namespace = namespace.rstrip('/')
        return f'{namespace}/controller_manager/{service}'