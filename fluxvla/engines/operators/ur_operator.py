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

from fluxvla.engines.operators.base_operator import BaseOperator
from fluxvla.engines.utils.root import OPERATORS

DEFAULT_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
DEFAULT_PREPARE_QPOS = [
    2.3911736011505127,
    -1.7057769934283655,
    2.1696739196777344,
    -0.5096147696124476,
    1.5789384841918945,
    -15.709390354140687,
]
SYNC_QUEUE_SIZE = 30
SYNCED_FRAME_QUEUE_SIZE = 10
SYNC_WARNING_WINDOW = 2.0
SYNC_WARNING_MIN_HZ_RATIO = 0.9
SYNC_WARNING_WARMUP = 3.0
HOME_COMMAND_DURATION = 2.0
GRIPPER_OPEN_POSITION = 0.085


@OPERATORS.register_module()
class UROperator(BaseOperator):
    """Single-arm UR operator using BaseOperator observation sync."""

    def __init__(self,
                 img_left_topic,
                 img_front_topic,
                 joint_state_topic=None,
                 cartesian_pose_topic=None,
                 gripper_state_topic=None,
                 use_depth_image=False,
                 img_left_depth_topic=None,
                 img_front_depth_topic=None,
                 sync_slop=0.04,
                 sync_warning_enabled=True,
                 sync_warning_target_hz=30.0,
                 command_mode='joint',
                 joint_cmd_topic='/cmd/servoj',
                 cartesian_cmd_topic='/cmd/servol',
                 gripper_cmd_topic='/cmd/gripper',
                 joint_names=None,
                 gripper_open_position=GRIPPER_OPEN_POSITION,
                 **unused_kwargs):
        """Configure UR observation topics, sync settings, and publishers."""
        self.img_left_topic = img_left_topic
        self.img_front_topic = img_front_topic
        self.joint_state_topic = joint_state_topic
        self.cartesian_pose_topic = cartesian_pose_topic
        self.gripper_state_topic = gripper_state_topic
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
        self.img_front_depth_topic = img_front_depth_topic

        super().__init__(
            sync_slop=sync_slop,
            sync_queue_size=SYNC_QUEUE_SIZE,
            synced_frame_queue_size=SYNCED_FRAME_QUEUE_SIZE,
            sync_warning_enabled=sync_warning_enabled,
            sync_warning_target_hz=sync_warning_target_hz,
            sync_warning_window=SYNC_WARNING_WINDOW,
            sync_warning_min_hz_ratio=SYNC_WARNING_MIN_HZ_RATIO,
            sync_warning_warmup=SYNC_WARNING_WARMUP)

        if (self.use_depth_image
                and (not img_left_depth_topic or not img_front_depth_topic)):
            raise ValueError(
                'When use_depth_image=True, both depth topics must be '
                'provided')
        if (not joint_state_topic or not cartesian_pose_topic
                or not gripper_state_topic):
            raise ValueError('joint_state_topic, cartesian_pose_topic, and '
                             'gripper_state_topic must be provided')
        if command_mode not in {'joint', 'cartesian'}:
            raise ValueError(f'Unsupported UR command_mode: {command_mode}')

        self.command_mode = command_mode
        self.joint_cmd_topic = joint_cmd_topic
        self.cartesian_cmd_topic = cartesian_cmd_topic
        self.gripper_cmd_topic = gripper_cmd_topic
        self.joint_names = joint_names or DEFAULT_JOINT_NAMES
        self.gripper_open_position = float(gripper_open_position)

        self.joint_pub = None
        self.cartesian_pub = None
        self.gripper_pub = None

        self._init_ros()

    def _init_ros(self):
        """Initialize ROS node, synchronized observations, and publishers."""
        import rospy
        from geometry_msgs.msg import Pose
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32

        rospy.init_node('ur_operator', anonymous=True)

        camera_info_topics = self.setup_observation_sync(
            self.build_observation_specs())
        self._setup_control(rospy, Pose, JointState, Float32)
        self.load_camera_info(camera_info_topics)

    def build_observation_specs(self):
        """Build synchronized image, joint, pose, and gripper specs."""
        from geometry_msgs.msg import PoseStamped
        from robotiq.msg import StampedFloat32
        from sensor_msgs.msg import Image, JointState

        specs = [
            {
                'name': 'img_front',
                'topic': self.img_front_topic,
                'msg_type': Image,
            },
            {
                'name': 'img_left',
                'topic': self.img_left_topic,
                'msg_type': Image,
            },
            {
                'name': 'left_arm',
                'topic': self.joint_state_topic,
                'msg_type': JointState,
            },
            {
                'name': 'left_pose',
                'topic': self.cartesian_pose_topic,
                'msg_type': PoseStamped,
            },
            {
                'name': 'left_gripper',
                'topic': self.gripper_state_topic,
                'msg_type': StampedFloat32,
            },
        ]
        if self.use_depth_image:
            specs.extend([
                {
                    'name': 'img_front_depth',
                    'topic': self.img_front_depth_topic,
                    'msg_type': Image,
                },
                {
                    'name': 'img_left_depth',
                    'topic': self.img_left_depth_topic,
                    'msg_type': Image,
                },
            ])
        return specs

    def _setup_control(self, rospy, Pose, JointState, Float32):
        """Initialize UR command publishers."""
        self.cartesian_pub = rospy.Publisher(
            self.cartesian_cmd_topic, Pose, queue_size=10)
        self.joint_pub = rospy.Publisher(
            self.joint_cmd_topic, JointState, queue_size=10)
        self.gripper_pub = rospy.Publisher(
            self.gripper_cmd_topic, Float32, queue_size=10)

    def get_frame(self, slop=0.7):
        """Return a UR observation tuple compatible with the runner."""
        frame = super().get_frame(slop=slop)
        if not frame:
            return False

        stamps = frame.get('stamps', {})
        frame_time_min = min(stamps.values()) if stamps else 0.0
        frame_time_max = max(stamps.values()) if stamps else 0.0

        return (
            frame['img_front'],
            frame['img_left'],
            frame.get('img_front_depth'),
            frame.get('img_left_depth'),
            frame['left_arm'],
            frame['left_pose'],
            frame['left_gripper'],
            frame_time_min,
            frame_time_max,
        )

    def send_joints(self, arm_targets):
        """Publish a left-arm servo joint command."""
        if isinstance(arm_targets, dict):
            arm_targets = next(iter(arm_targets.values()))
        self.servoj(arm_targets)

    def send_eepose(self, arm_targets):
        """Publish a left-arm servo Cartesian command."""
        if isinstance(arm_targets, dict):
            arm_targets = next(iter(arm_targets.values()))
        self.servol(arm_targets)

    def send_gripper(self, gripper_targets):
        """Publish a left gripper command."""
        if isinstance(gripper_targets, dict):
            gripper_targets = next(iter(gripper_targets.values()))
        self.movegrip(gripper_targets)

    def gohome(self, prepare_pose=None):
        """Move the UR arm to a joint- or cartesian-space prepare pose."""
        import rospy

        if prepare_pose is None:
            if self.command_mode == 'cartesian':
                raise ValueError(
                    'prepare_pose must be provided when command_mode is '
                    '"cartesian"')
            prepare_pose = DEFAULT_PREPARE_QPOS

        self.clear_observation_queues()
        try:
            if self.command_mode == 'cartesian':
                rospy.loginfo('Moving UR arm to provided prepare ee pose')
                self.servol(prepare_pose)
            else:
                rospy.loginfo('Moving UR arm to provided prepare joints')
                self.movej(prepare_pose)
            self.movegrip(self.gripper_open_position)
            rospy.sleep(HOME_COMMAND_DURATION)
            return prepare_pose
        finally:
            self.clear_observation_queues()

    def movel(self, eepose):
        """Publish a Cartesian pose command."""
        self._publish_pose(self.cartesian_pub, eepose)

    def servol(self, eepose):
        """Publish a Cartesian servo command."""
        self._publish_pose(self.cartesian_pub, eepose)

    def movej(self, qpos):
        """Publish a joint-space move command."""
        self._publish_joint_state(self.joint_pub, qpos)

    def servoj(self, qpos):
        """Publish a joint-space servo command."""
        self._publish_joint_state(self.joint_pub, qpos)

    def movegrip(self, gripper_position):
        """Publish a gripper position command."""
        from std_msgs.msg import Float32

        msg = Float32()
        msg.data = float(gripper_position)
        self.gripper_pub.publish(msg)

    def _publish_pose(self, publisher, eepose):
        from geometry_msgs.msg import Point, Pose, Quaternion

        msg = Pose()
        msg.position = Point(
            x=float(eepose[0]), y=float(eepose[1]), z=float(eepose[2]))
        msg.orientation = Quaternion(
            x=float(eepose[3]),
            y=float(eepose[4]),
            z=float(eepose[5]),
            w=float(eepose[6]))
        publisher.publish(msg)

    def _publish_joint_state(self, publisher, qpos):
        import rospy
        from sensor_msgs.msg import JointState

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(self.joint_names)
        msg.position = [float(value) for value in qpos[:len(self.joint_names)]]
        publisher.publish(msg)
