import time
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
class FrankaOperator:
    """Franka operator for ROS-based arm control and observation sync.

    The operator mirrors the lightweight API exposed by `UROperator` while
    adapting command and state handling to Franka ROS controllers. End-effector
    pose commands are published to the SERL Cartesian impedance controller,
    joint targets can be sent to a configurable joint controller topic, and
    gripper commands use the `franka_gripper/move` action interface.
    """

    def __init__(self,
                 img_left_topic,
                 img_front_topic,
                 puppet_arm_left_topic,
                 puppet_gripper_left_topic,
                 puppet_franka_state_left_topic=None,
                 puppet_ee_pose_left_topic=None,
                 use_depth_image=False,
                 img_left_depth_topic=None,
                 img_front_depth_topic=None,
                 cartesian_cmd_topic=(
                     '/cartesian_impedance_controller/equilibrium_pose'),
                 joint_cmd_topic=(
                     '/joint_ruckig_position_controller/target_joint_state'),
                 gripper_action_name='/franka_gripper/move',
                 gripper_speed=0.08,
                 joint_names=None,
                 base_frame_id=''):
        self.img_left_topic = img_left_topic
        self.img_front_topic = img_front_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_gripper_left_topic = puppet_gripper_left_topic
        self.puppet_franka_state_left_topic = puppet_franka_state_left_topic
        self.puppet_ee_pose_left_topic = puppet_ee_pose_left_topic
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
        self.img_front_depth_topic = img_front_depth_topic
        self.cartesian_cmd_topic = cartesian_cmd_topic
        self.joint_cmd_topic = joint_cmd_topic
        self.gripper_action_name = gripper_action_name
        self.gripper_speed = gripper_speed
        self.base_frame_id = base_frame_id
        self.joint_names = joint_names or [
            f'panda_joint{i}' for i in range(1, 8)
        ]

        if self.use_depth_image:
            if not img_left_depth_topic or not img_front_depth_topic:
                raise ValueError(
                    'When use_depth_image=True, both img_left_depth_topic '
                    'and img_front_depth_topic must be provided')

        if (self.puppet_ee_pose_left_topic is None
                and self.puppet_franka_state_left_topic is None):
            raise ValueError(
                'Either puppet_ee_pose_left_topic or '
                'puppet_franka_state_left_topic must be provided')

        self._init_count()
        self._init()
        self._init_ros()

    def _init_count(self):
        self.rgb_left_count = 0
        self.rgb_front_count = 0
        self.depth_left_count = 0
        self.depth_front_count = 0

    def _init(self):
        from cv_bridge import CvBridge

        self.rgb_l = 0
        self.rgb_f = 0
        self.depth_l = 0
        self.depth_f = 0

        self.last_time_step = 0
        self.bridge = CvBridge()

        self.img_left_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_ee_pose_left_deque = deque()
        self.puppet_gripper_left_deque = deque()

        self.movegrip_client = None
        self.cam_info_dict = {}

    def get_frame(self, slop=0.7):
        required_queues_empty = (
            len(self.img_left_deque) == 0 or len(self.img_front_deque) == 0
            or len(self.puppet_arm_left_deque) == 0
            or len(self.puppet_ee_pose_left_deque) == 0
            or len(self.puppet_gripper_left_deque) == 0)

        depth_queues_empty = (
            self.use_depth_image and (len(self.img_left_depth_deque) == 0
                                      or len(self.img_front_depth_deque) == 0))

        if required_queues_empty or depth_queues_empty:
            self._handle_empty_queues()
            return False

        frame_time = self._calculate_frame_time()

        if not self._check_sensor_data_availability(frame_time):
            return False

        self.last_time_step = frame_time

        self.rgb_l = 0
        self.rgb_f = 0
        self.depth_l = 0
        self.depth_f = 0

        frame_time_max = self._synchronize_queues(frame_time)
        if abs(frame_time_max - frame_time) > slop:
            self._flush_outdated_data(frame_time)
            return False

        return self._extract_synchronized_data()

    def _handle_empty_queues(self):
        if len(self.img_left_deque) == 0:
            self.rgb_l += 1
            if self.rgb_l > 3:
                print('Error left RGB', str(time.time()))

        if len(self.img_front_deque) == 0:
            self.rgb_f += 1
            if self.rgb_f > 3:
                print('Error front RGB', str(time.time()))

        if self.use_depth_image:
            if len(self.img_left_depth_deque) == 0:
                self.depth_l += 1
                if self.depth_l > 3:
                    print('Error left Depth')

            if len(self.img_front_depth_deque) == 0:
                self.depth_f += 1
                if self.depth_f > 3:
                    print('Error front Depth')

    def _calculate_frame_time(self):
        timestamps = [
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec(),
            self.puppet_arm_left_deque[-1].header.stamp.to_sec(),
            self.puppet_ee_pose_left_deque[-1].header.stamp.to_sec(),
            self.puppet_gripper_left_deque[-1].header.stamp.to_sec(),
        ]

        if self.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec(),
            ])

        return min(timestamps)

    def _check_sensor_data_availability(self, frame_time):
        checks = [
            self.img_left_deque, self.img_front_deque, self.puppet_arm_left_deque,
            self.puppet_ee_pose_left_deque, self.puppet_gripper_left_deque
        ]

        for deque_obj in checks:
            if (len(deque_obj) == 0
                    or deque_obj[-1].header.stamp.to_sec() < frame_time):
                return False

        if self.use_depth_image:
            depth_checks = [
                self.img_left_depth_deque, self.img_front_depth_deque
            ]
            for deque_obj in depth_checks:
                if (len(deque_obj) == 0
                        or deque_obj[-1].header.stamp.to_sec() < frame_time):
                    return False

        return True

    def _synchronize_queues(self, frame_time):
        frame_time_max = 0

        queues_to_sync = [
            self.img_left_deque, self.img_front_deque,
            self.puppet_arm_left_deque, self.puppet_ee_pose_left_deque,
            self.puppet_gripper_left_deque
        ]

        for queue in queues_to_sync:
            while queue[0].header.stamp.to_sec() < frame_time:
                queue.popleft()
            frame_time_max = max(frame_time_max,
                                 queue[0].header.stamp.to_sec())

        if self.use_depth_image:
            depth_queues = [
                self.img_left_depth_deque, self.img_front_depth_deque
            ]
            for queue in depth_queues:
                while queue[0].header.stamp.to_sec() < frame_time:
                    queue.popleft()
                frame_time_max = max(frame_time_max,
                                     queue[0].header.stamp.to_sec())

        return frame_time_max

    def _flush_outdated_data(self, frame_time):
        queues_to_flush = [
            self.img_left_deque, self.img_front_deque,
            self.img_left_depth_deque, self.img_front_depth_deque,
            self.puppet_arm_left_deque, self.puppet_ee_pose_left_deque,
            self.puppet_gripper_left_deque
        ]

        for queue in queues_to_flush:
            while (len(queue) > 0
                   and queue[0].header.stamp.to_sec() <= frame_time):
                queue.popleft()

    def _extract_synchronized_data(self):
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(),
                                              'passthrough')
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(),
                                             'passthrough')

        puppet_arm_left = self.puppet_arm_left_deque.popleft()
        puppet_ee_pose_left = self.puppet_ee_pose_left_deque.popleft()
        puppet_gripper_left = self.puppet_gripper_left_deque.popleft()

        img_left_depth = None
        img_front_depth = None
        if self.use_depth_image:
            img_left_depth = self.bridge.imgmsg_to_cv2(
                self.img_left_depth_deque.popleft(), 'passthrough')
            img_front_depth = self.bridge.imgmsg_to_cv2(
                self.img_front_depth_deque.popleft(), 'passthrough')

        return (img_front, img_left, img_front_depth, img_left_depth,
                puppet_arm_left, puppet_ee_pose_left, puppet_gripper_left,
                self.last_time_step, self.last_time_step)

    def _append_with_limit(self, deque_obj, msg, max_len=20000):
        if len(deque_obj) >= max_len:
            deque_obj.popleft()
        deque_obj.append(msg)

    def img_left_callback(self, msg):
        self._append_with_limit(self.img_left_deque, msg)

    def img_front_callback(self, msg):
        self._append_with_limit(self.img_front_deque, msg)

    def img_left_depth_callback(self, msg):
        self._append_with_limit(self.img_left_depth_deque, msg)

    def img_front_depth_callback(self, msg):
        self._append_with_limit(self.img_front_depth_deque, msg)

    def puppet_arm_left_callback(self, msg):
        self._append_with_limit(self.puppet_arm_left_deque, msg)

    def puppet_ee_pose_left_callback(self, msg):
        self._append_with_limit(self.puppet_ee_pose_left_deque, msg)

    def puppet_gripper_left_callback(self, msg):
        stamped_width = self._joint_state_to_stamped_width(msg)
        self._append_with_limit(self.puppet_gripper_left_deque, stamped_width)

    def puppet_franka_state_left_callback(self, msg):
        pose_msg = self._franka_state_to_pose_stamped(msg)
        self._append_with_limit(self.puppet_ee_pose_left_deque, pose_msg)

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

        transform = np.array(msg.O_T_EE, dtype=np.float64).reshape(
            (4, 4), order='F')
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

        rospy.init_node('record_episodes', anonymous=True)
        camera_info_topics = []

        rospy.Subscriber(
            self.img_left_topic,
            Image,
            self.img_left_callback,
            queue_size=1000,
            tcp_nodelay=True)
        camera_info_topics.append(replace_last_segment(self.img_left_topic))

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

        if self.puppet_ee_pose_left_topic is not None:
            rospy.Subscriber(
                self.puppet_ee_pose_left_topic,
                PoseStamped,
                self.puppet_ee_pose_left_callback,
                queue_size=1000,
                tcp_nodelay=True)
        else:
            from franka_msgs.msg import FrankaState

            rospy.Subscriber(
                self.puppet_franka_state_left_topic,
                FrankaState,
                self.puppet_franka_state_left_callback,
                queue_size=1000,
                tcp_nodelay=True)

        rospy.Subscriber(
            self.puppet_gripper_left_topic,
            JointState,
            self.puppet_gripper_left_callback,
            queue_size=1000,
            tcp_nodelay=True)

        self.movel_pub = rospy.Publisher(
            self.cartesian_cmd_topic, PoseStamped, queue_size=10)
        self.servol_pub = rospy.Publisher(
            self.cartesian_cmd_topic, PoseStamped, queue_size=10)

        self.movej_pub = None
        self.servoj_pub = None
        if self.joint_cmd_topic:
            self.movej_pub = rospy.Publisher(
                self.joint_cmd_topic, JointState, queue_size=10)
            self.servoj_pub = rospy.Publisher(
                self.joint_cmd_topic, JointState, queue_size=10)

        if self.gripper_action_name:
            try:
                import actionlib
                from franka_gripper.msg import MoveAction

                self.movegrip_client = actionlib.SimpleActionClient(
                    self.gripper_action_name, MoveAction)
                if not self.movegrip_client.wait_for_server(
                        rospy.Duration(2.0)):
                    rospy.logwarn('Franka gripper action server %s not ready',
                                  self.gripper_action_name)
            except Exception as exc:  # pragma: no cover - ROS import/runtime
                rospy.logwarn('Failed to initialize Franka gripper action: %s',
                              exc)
                self.movegrip_client = None

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

    def _build_pose_stamped(self, eepose):
        import rospy
        from geometry_msgs.msg import Point, PoseStamped, Quaternion

        if len(eepose) != 7:
            raise ValueError('End-effector pose must contain exactly 7 '
                             'elements: [x, y, z, qx, qy, qz, qw]')

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

    def _build_joint_state(self, qpos):
        import rospy
        from sensor_msgs.msg import JointState

        if len(qpos) != len(self.joint_names):
            raise ValueError(f'Joint command must contain exactly '
                             f'{len(self.joint_names)} elements')

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        msg.position = [float(q) for q in qpos]
        return msg

    def movel(self, eepose):
        msg = self._build_pose_stamped(eepose)
        self.movel_pub.publish(msg)

    def movej(self, qpos):
        if self.movej_pub is None:
            raise RuntimeError('joint_cmd_topic is not configured')
        msg = self._build_joint_state(qpos)
        self.movej_pub.publish(msg)

    def servol(self, eepose):
        msg = self._build_pose_stamped(eepose)
        self.servol_pub.publish(msg)

    def servoj(self, qpos):
        if self.servoj_pub is None:
            raise RuntimeError('joint_cmd_topic is not configured')
        msg = self._build_joint_state(qpos)
        self.servoj_pub.publish(msg)

    def movegrip(self, gripper_position, speed=None, wait=False):
        from franka_gripper.msg import MoveGoal

        if self.movegrip_client is None:
            raise RuntimeError('Franka gripper action client is not available')

        goal = MoveGoal()
        goal.width = float(gripper_position)
        goal.speed = float(self.gripper_speed if speed is None else speed)

        self.movegrip_client.send_goal(goal)
        if wait:
            self.movegrip_client.wait_for_result()
