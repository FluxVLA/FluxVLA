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
from collections import deque
from types import SimpleNamespace

import numpy as np

from fluxvla.engines.utils.root import OPERATORS


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

    def __init__(self,
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
                 cartesian_cmd_left_topic='/left_arm/cartesian_impedance_controller/equilibrium_pose',
                 cartesian_cmd_right_topic='/right_arm/cartesian_impedance_controller/equilibrium_pose',
                 gripper_action_left_name='/left_arm/franka_gripper/move',
                 gripper_action_right_name='/right_arm/franka_gripper/move',
                 use_depth_image=False,
                 img_left_depth_topic=None,
                 img_right_depth_topic=None,
                 img_front_depth_topic=None,
                 gripper_speed=0.08,
                 base_frame_id='',
                 publish_rate=30):
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
        self.cartesian_cmd_left_topic = cartesian_cmd_left_topic
        self.cartesian_cmd_right_topic = cartesian_cmd_right_topic
        self.gripper_action_left_name = gripper_action_left_name
        self.gripper_action_right_name = gripper_action_right_name
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
        self.img_right_depth_topic = img_right_depth_topic
        self.img_front_depth_topic = img_front_depth_topic
        self.gripper_speed = gripper_speed
        self.base_frame_id = base_frame_id
        self.publish_rate = publish_rate

        if self.use_depth_image:
            if not all([img_left_depth_topic, img_right_depth_topic, img_front_depth_topic]):
                raise ValueError('When use_depth_image=True, all depth topics must be provided')

        if (self.puppet_ee_pose_left_topic is None and self.puppet_franka_state_left_topic is None):
            raise ValueError('Either puppet_ee_pose_left_topic or puppet_franka_state_left_topic must be provided')

        if (self.puppet_ee_pose_right_topic is None and self.puppet_franka_state_right_topic is None):
            raise ValueError('Either puppet_ee_pose_right_topic or puppet_franka_state_right_topic must be provided')

        self._init()
        self._init_ros()

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
        self.cam_info_dict = {}
        self._traj_thread = None
        self._traj_stop_event = threading.Event()

    def get_frame(self, slop=0.7):
        """Get synchronized frame data from all sensors."""
        required_queues_empty = (
            len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or len(self.puppet_arm_left_deque) == 0
            or len(self.puppet_arm_right_deque) == 0
            or len(self.puppet_ee_pose_left_deque) == 0
            or len(self.puppet_ee_pose_right_deque) == 0
            or len(self.puppet_gripper_left_deque) == 0
            or len(self.puppet_gripper_right_deque) == 0)

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
            self.puppet_ee_pose_left_deque[-1].header.stamp.to_sec(),
            self.puppet_ee_pose_right_deque[-1].header.stamp.to_sec(),
            self.puppet_gripper_left_deque[-1].header.stamp.to_sec(),
            self.puppet_gripper_right_deque[-1].header.stamp.to_sec(),
        ]
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
            self.puppet_ee_pose_left_deque, self.puppet_ee_pose_right_deque,
            self.puppet_gripper_left_deque, self.puppet_gripper_right_deque
        ]
        for deque_obj in checks:
            if (len(deque_obj) == 0 or deque_obj[-1].header.stamp.to_sec() < frame_time):
                return False

        if self.use_depth_image:
            depth_checks = [
                self.img_left_depth_deque, self.img_right_depth_deque, self.img_front_depth_deque
            ]
            for deque_obj in depth_checks:
                if (len(deque_obj) == 0 or deque_obj[-1].header.stamp.to_sec() < frame_time):
                    return False
        return True

    def _synchronize_queues(self, frame_time):
        frame_time_max = 0
        queues_to_sync = [
            self.img_left_deque, self.img_right_deque, self.img_front_deque,
            self.puppet_arm_left_deque, self.puppet_arm_right_deque,
            self.puppet_ee_pose_left_deque, self.puppet_ee_pose_right_deque,
            self.puppet_gripper_left_deque, self.puppet_gripper_right_deque
        ]
        for queue in queues_to_sync:
            while queue[0].header.stamp.to_sec() < frame_time:
                queue.popleft()
            frame_time_max = max(frame_time_max, queue[0].header.stamp.to_sec())

        if self.use_depth_image:
            depth_queues = [
                self.img_left_depth_deque, self.img_right_depth_deque, self.img_front_depth_deque
            ]
            for queue in depth_queues:
                while queue[0].header.stamp.to_sec() < frame_time:
                    queue.popleft()
                frame_time_max = max(frame_time_max, queue[0].header.stamp.to_sec())
        return frame_time_max

    def _flush_outdated_data(self, frame_time):
        queues_to_flush = [
            self.img_left_deque, self.img_right_deque, self.img_front_deque,
            self.img_left_depth_deque, self.img_right_depth_deque, self.img_front_depth_deque,
            self.puppet_arm_left_deque, self.puppet_arm_right_deque,
            self.puppet_ee_pose_left_deque, self.puppet_ee_pose_right_deque,
            self.puppet_gripper_left_deque, self.puppet_gripper_right_deque
        ]
        for queue in queues_to_flush:
            while (len(queue) > 0 and queue[0].header.stamp.to_sec() <= frame_time):
                queue.popleft()

    def _extract_synchronized_data(self):
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        puppet_arm_left = self.puppet_arm_left_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()
        puppet_ee_pose_left = self.puppet_ee_pose_left_deque.popleft()
        puppet_ee_pose_right = self.puppet_ee_pose_right_deque.popleft()
        puppet_gripper_left = self.puppet_gripper_left_deque.popleft()
        puppet_gripper_right = self.puppet_gripper_right_deque.popleft()

        img_left_depth = None
        img_right_depth = None
        img_front_depth = None
        if self.use_depth_image:
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, puppet_ee_pose_left, puppet_ee_pose_right,
                puppet_gripper_left, puppet_gripper_right)

    def _append_with_limit(self, deque_obj, msg, max_len=2000):
        if len(deque_obj) >= max_len:
            deque_obj.popleft()
        deque_obj.append(msg)

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

        transform = np.array(msg.O_T_EE, dtype=np.float64).reshape((4, 4), order='F')
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
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image, JointState

        rospy.init_node('franka_dual_operator', anonymous=True)
        camera_info_topics = []

        rospy.Subscriber(self.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_left_topic))

        rospy.Subscriber(self.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_right_topic))

        rospy.Subscriber(self.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_front_topic))

        if self.use_depth_image:
            rospy.Subscriber(self.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            camera_info_topics.append(replace_last_segment(self.img_left_depth_topic))

            rospy.Subscriber(self.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            camera_info_topics.append(replace_last_segment(self.img_right_depth_topic))

            rospy.Subscriber(self.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
            camera_info_topics.append(replace_last_segment(self.img_front_depth_topic))

        rospy.Subscriber(self.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)

        if self.puppet_ee_pose_left_topic is not None:
            rospy.Subscriber(self.puppet_ee_pose_left_topic, PoseStamped, self.puppet_ee_pose_left_callback, queue_size=1000, tcp_nodelay=True)
        else:
            from franka_msgs.msg import FrankaState
            rospy.Subscriber(self.puppet_franka_state_left_topic, FrankaState, self.puppet_franka_state_left_callback, queue_size=1000, tcp_nodelay=True)

        if self.puppet_ee_pose_right_topic is not None:
            rospy.Subscriber(self.puppet_ee_pose_right_topic, PoseStamped, self.puppet_ee_pose_right_callback, queue_size=1000, tcp_nodelay=True)
        else:
            from franka_msgs.msg import FrankaState
            rospy.Subscriber(self.puppet_franka_state_right_topic, FrankaState, self.puppet_franka_state_right_callback, queue_size=1000, tcp_nodelay=True)

        rospy.Subscriber(self.puppet_gripper_left_topic, JointState, self.puppet_gripper_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.puppet_gripper_right_topic, JointState, self.puppet_gripper_right_callback, queue_size=1000, tcp_nodelay=True)

        self.left_ee_pub = rospy.Publisher(self.cartesian_cmd_left_topic, PoseStamped, queue_size=10)
        self.right_ee_pub = rospy.Publisher(self.cartesian_cmd_right_topic, PoseStamped, queue_size=10)

        if self.gripper_action_left_name:
            try:
                import actionlib
                from franka_gripper.msg import MoveAction
                self.movegrip_left_client = actionlib.SimpleActionClient(self.gripper_action_left_name, MoveAction)
                if not self.movegrip_left_client.wait_for_server(rospy.Duration(2.0)):
                    rospy.logwarn('Left Franka gripper action server not ready')
            except Exception as exc:
                rospy.logwarn('Failed to initialize left Franka gripper action: %s', exc)
                self.movegrip_left_client = None

        if self.gripper_action_right_name:
            try:
                import actionlib
                from franka_gripper.msg import MoveAction
                self.movegrip_right_client = actionlib.SimpleActionClient(self.gripper_action_right_name, MoveAction)
                if not self.movegrip_right_client.wait_for_server(rospy.Duration(2.0)):
                    rospy.logwarn('Right Franka gripper action server not ready')
            except Exception as exc:
                rospy.logwarn('Failed to initialize right Franka gripper action: %s', exc)
                self.movegrip_right_client = None

        for topic in camera_info_topics:
            try:
                camera_info = rospy.wait_for_message(topic, CameraInfo, timeout=5)
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

    def execute_step(self, left_eepose, right_eepose):
        import rospy
        from geometry_msgs.msg import Point, PoseStamped, Quaternion

        msg_left = PoseStamped()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.header.frame_id = self.base_frame_id
        msg_left.pose.position = Point(x=float(left_eepose[0]), y=float(left_eepose[1]), z=float(left_eepose[2]))
        msg_left.pose.orientation = Quaternion(x=float(left_eepose[3]), y=float(left_eepose[4]), z=float(left_eepose[5]), w=float(left_eepose[6]))
        self.left_ee_pub.publish(msg_left)

        msg_right = PoseStamped()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.header.frame_id = self.base_frame_id
        msg_right.pose.position = Point(x=float(right_eepose[0]), y=float(right_eepose[1]), z=float(right_eepose[2]))
        msg_right.pose.orientation = Quaternion(x=float(right_eepose[3]), y=float(right_eepose[4]), z=float(right_eepose[5]), w=float(right_eepose[6]))
        self.right_ee_pub.publish(msg_right)

    def execute_trajectory(self, left_trajectory, right_trajectory, dt=0.1, async_exec=False, base_velocity=None):
        left_traj = np.asarray(left_trajectory)
        right_traj = np.asarray(right_trajectory)

        self._traj_stop_event.set()
        self._traj_stop_event = threading.Event()

        stop_event = self._traj_stop_event
        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_trajectory,
                args=(left_traj, right_traj, dt, stop_event),
                daemon=True)
            self._traj_thread.start()
        else:
            self._run_trajectory(left_traj, right_traj, dt, stop_event)

    def _run_trajectory(self, left_traj, right_traj, dt, stop_event):
        import rospy
        rate = rospy.Rate(1.0 / dt)
        for i in range(len(left_traj)):
            if rospy.is_shutdown() or stop_event.is_set():
                break
            self.execute_step(left_traj[i], right_traj[i])
            rate.sleep()

    def stop_trajectory(self):
        self._traj_stop_event.set()

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
