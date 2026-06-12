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
from collections.abc import Mapping

from fluxvla.engines.operators.base_operator import BaseOperator
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
DEFAULT_PREPARE_QPOS = [
    0.0,
    -0.7853981633974483,
    0.0,
    -2.356194490192345,
    0.0,
    1.5707963267948966,
    0.7853981633974483,
    0.08,
]
DEFAULT_PREPARE_EEPOSE = [
    0.3497205274359744,
    0.020354744470914822,
    0.47795087805191966,
    0.9990589457070866,
    -0.0032166606104491466,
    0.04248046116037038,
    0.008141653196007417,
    0.08,
]
HOME_COMMAND_DURATION = 2.0
BASE_FRAME_ID = ''
SYNC_QUEUE_SIZE = 30
SYNCED_FRAME_QUEUE_SIZE = 10
SYNC_WARNING_WINDOW = 2.0
SYNC_WARNING_MIN_HZ_RATIO = 0.9
SYNC_WARNING_WARMUP = 3.0
GRIPPER_SPEED = 1.0
GRIPPER_MAX_WIDTH = 0.08
GRIPPER_OPEN_WIDTH = 0.08


@OPERATORS.register_module()
class FrankaDualOperator(BaseOperator):
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
            puppet_ee_pose_left_topic=None,
            puppet_ee_pose_right_topic=None,
            puppet_franka_state_left_topic=None,
            puppet_franka_state_right_topic=None,
            use_depth_image=False,
            img_left_depth_topic=None,
            img_right_depth_topic=None,
            img_front_depth_topic=None,
            sync_slop=0.03,
            sync_warning_enabled=True,
            sync_warning_target_hz=30.0,
            command_mode='joint',
            cartesian_cmd_left_topic=(
                '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
            cartesian_cmd_right_topic=(
                '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
            joint_cmd_left_topic=(
                '/left_arm/ruckig_joint_impedance_controller/target_joint_state'
            ),
            joint_cmd_right_topic=(
                '/right_arm/ruckig_joint_impedance_controller/target_joint_state'
            ),
            joint_names=None,
            gripper_left_topic='/left_arm/franka_gripper/move/goal',
            gripper_right_topic='/right_arm/franka_gripper/move/goal',
            **unused_kwargs):
        """Configure dual-arm ROS topics, sync settings, and controllers.

        Args:
            img_left_topic (str): Left wrist camera image topic.
            img_right_topic (str): Right wrist camera image topic.
            img_front_topic (str): Front camera image topic.
            puppet_arm_left_topic (str): Left arm JointState observation topic.
            puppet_arm_right_topic (str): Right arm JointState observation
                topic.
            puppet_ee_pose_left_topic (str, optional): Left PoseStamped topic
                used directly as end-effector pose.
            puppet_ee_pose_right_topic (str, optional): Right PoseStamped topic
                used directly as end-effector pose.
            puppet_franka_state_left_topic (str, optional): Left FrankaState
                topic used to derive end-effector pose.
            puppet_franka_state_right_topic (str, optional): Right FrankaState
                topic used to derive end-effector pose.
            use_depth_image (bool): Whether to synchronize depth images.
            img_left_depth_topic (str, optional): Left wrist depth image topic.
            img_right_depth_topic (str, optional): Right wrist depth image
                topic.
            img_front_depth_topic (str, optional): Front depth image topic.
            sync_slop (float): Allowed timestamp difference for synchronized
                observations, in seconds.
            sync_warning_enabled (bool): Whether to warn about low sync rate.
            sync_warning_target_hz (float): Expected synchronized frame rate.
            command_mode (str): Command interface, either 'joint' or
                'cartesian'.
            cartesian_cmd_left_topic (str): Left Cartesian pose command topic.
            cartesian_cmd_right_topic (str): Right Cartesian pose command
                topic.
            joint_cmd_left_topic (str): Left joint command topic.
            joint_cmd_right_topic (str): Right joint command topic.
            joint_names (list[str], optional): Seven Franka joint names shared
                by both arms.
            gripper_left_topic (str): Left gripper move goal topic.
            gripper_right_topic (str): Right gripper move goal topic.
            **unused_kwargs: Extra config keys accepted for compatibility.
        """
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
        self.base_frame_id = BASE_FRAME_ID
        super().__init__(
            sync_slop=sync_slop,
            sync_queue_size=SYNC_QUEUE_SIZE,
            synced_frame_queue_size=SYNCED_FRAME_QUEUE_SIZE,
            sync_warning_enabled=sync_warning_enabled,
            sync_warning_target_hz=sync_warning_target_hz,
            sync_warning_window=SYNC_WARNING_WINDOW,
            sync_warning_min_hz_ratio=SYNC_WARNING_MIN_HZ_RATIO,
            sync_warning_warmup=SYNC_WARNING_WARMUP)

        if command_mode not in {'joint', 'cartesian'}:
            raise ValueError(
                f'Unsupported Franka command_mode: {command_mode}')
        self.command_mode = command_mode
        self.cartesian_cmd_left_topic = cartesian_cmd_left_topic
        self.cartesian_cmd_right_topic = cartesian_cmd_right_topic
        self.joint_cmd_left_topic = joint_cmd_left_topic
        self.joint_cmd_right_topic = joint_cmd_right_topic
        self.joint_names = joint_names or DEFAULT_JOINT_NAMES
        self.gripper_goal_left_topic = gripper_left_topic
        self.gripper_goal_right_topic = gripper_right_topic
        self.gripper_speed = GRIPPER_SPEED
        self.gripper_max_width = GRIPPER_MAX_WIDTH
        self.gripper_open_width = GRIPPER_OPEN_WIDTH
        self.left_ee_pub = None
        self.right_ee_pub = None
        self.left_joint_pub = None
        self.right_joint_pub = None
        self.left_gripper_pub = None
        self.right_gripper_pub = None
        self.MoveActionGoal = None

        if self.use_depth_image and not all([
                img_left_depth_topic, img_right_depth_topic,
                img_front_depth_topic
        ]):
            raise ValueError(
                'When use_depth_image=True, all depth topics must be provided')
        if len(self.joint_names) != 7:
            raise ValueError('joint_names must contain exactly 7 joints')

        self._init_ros()

    def _init_ros(self):
        """Initialize ROS node, observation sync, control publishers, cameras."""
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        rospy.init_node('franka_dual_operator', anonymous=True)

        camera_info_topics = self.setup_observation_sync(
            self.build_observation_specs())
        self._setup_control(rospy, PoseStamped, JointState)
        self.load_camera_info(camera_info_topics)

    def build_observation_specs(self):
        """Build synchronized image, joint, and end-effector observation specs."""
        from geometry_msgs.msg import PoseStamped
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
                'name': 'img_right',
                'topic': self.img_right_topic,
                'msg_type': Image,
            },
            {
                'name': 'left_arm',
                'topic': self.puppet_arm_left_topic,
                'msg_type': JointState,
            },
            {
                'name': 'right_arm',
                'topic': self.puppet_arm_right_topic,
                'msg_type': JointState,
            },
        ]

        self._add_pose_specs(specs, PoseStamped)
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
                {
                    'name': 'img_right_depth',
                    'topic': self.img_right_depth_topic,
                    'msg_type': Image,
                },
            ])
        return specs

    def _add_pose_specs(self, specs, PoseStamped):
        """Add configured left/right end-effector pose sources to specs."""
        if self.puppet_ee_pose_left_topic is not None:
            specs.append({
                'name': 'left_pose',
                'topic': self.puppet_ee_pose_left_topic,
                'msg_type': PoseStamped,
            })
        elif self.puppet_franka_state_left_topic is not None:
            from franka_msgs.msg import FrankaState
            specs.append({
                'name': 'left_franka_state',
                'topic': self.puppet_franka_state_left_topic,
                'msg_type': FrankaState,
            })

        if self.puppet_ee_pose_right_topic is not None:
            specs.append({
                'name': 'right_pose',
                'topic': self.puppet_ee_pose_right_topic,
                'msg_type': PoseStamped,
            })
        elif self.puppet_franka_state_right_topic is not None:
            from franka_msgs.msg import FrankaState
            specs.append({
                'name': 'right_franka_state',
                'topic': self.puppet_franka_state_right_topic,
                'msg_type': FrankaState,
            })

    def _setup_control(self, rospy, PoseStamped, JointState):
        """Initialize ROS publishers for both arms and grippers."""
        from franka_gripper.msg import MoveActionGoal

        self.MoveActionGoal = MoveActionGoal
        self.left_ee_pub = rospy.Publisher(
            self.cartesian_cmd_left_topic, PoseStamped, queue_size=10)
        self.right_ee_pub = rospy.Publisher(
            self.cartesian_cmd_right_topic, PoseStamped, queue_size=10)
        self.left_joint_pub = rospy.Publisher(
            self.joint_cmd_left_topic, JointState, queue_size=10)
        self.right_joint_pub = rospy.Publisher(
            self.joint_cmd_right_topic, JointState, queue_size=10)

        self.left_gripper_pub = rospy.Publisher(
            self.gripper_goal_left_topic, MoveActionGoal, queue_size=1)
        self.right_gripper_pub = rospy.Publisher(
            self.gripper_goal_right_topic, MoveActionGoal, queue_size=1)

    def send_joints(self, arm_targets):
        """Publish joint targets for one or both Franka arms."""
        self._validate_target_names(arm_targets, 'arm')
        if 'left' in arm_targets:
            self.left_joint_pub.publish(
                self._build_joint_state(arm_targets['left']))
        if 'right' in arm_targets:
            self.right_joint_pub.publish(
                self._build_joint_state(arm_targets['right']))

    def send_eepose(self, arm_targets):
        """Publish Cartesian end-effector pose targets for active arms."""
        self._validate_target_names(arm_targets, 'arm')
        if 'left' in arm_targets:
            self.left_ee_pub.publish(
                self._build_pose_stamped(arm_targets['left']))
        if 'right' in arm_targets:
            self.right_ee_pub.publish(
                self._build_pose_stamped(arm_targets['right']))

    def send_gripper(self, gripper_targets, wait=False):
        """Publish gripper width targets for one or both grippers."""
        del wait
        if not gripper_targets:
            raise ValueError('At least one gripper width must be provided')
        self._validate_target_names(gripper_targets, 'gripper')
        self._send_gripper_pair(
            gripper_targets.get('left'),
            gripper_targets.get('right'))

    def gohome(self, prepare_pose):
        """Move both Franka arms to prepare poses for the active command mode."""
        import rospy

        prepare_pose = self._normalize_prepare_pose(prepare_pose)
        arm_targets = {
            arm: pose[:7]
            for arm, pose in prepare_pose.items()
        }
        gripper_targets = {
            arm: pose[7]
            for arm, pose in prepare_pose.items()
        }

        self.clear_observation_queues()
        try:
            if self.command_mode == 'cartesian':
                rospy.loginfo(
                    'Moving Franka arms to provided prepare ee poses')
                self.send_eepose(arm_targets)
            else:
                rospy.loginfo('Moving Franka arms to provided prepare joints')
                self.send_joints(arm_targets)

            self.send_gripper(gripper_targets, wait=True)
            rospy.sleep(HOME_COMMAND_DURATION)
            return prepare_pose
        finally:
            self.clear_observation_queues()

    @staticmethod
    def _validate_target_names(targets, target_type):
        """Validate that target dictionaries only use left/right arm keys."""
        unsupported = set(targets) - {'left', 'right'}
        if unsupported:
            raise ValueError(
                f'Unsupported Franka {target_type} target(s): {unsupported}')

    def _normalize_prepare_pose(self, prepare_pose):
        """Normalize mapping or pair input into left/right prepare poses."""
        if prepare_pose is None:
            pose = self._default_prepare_pose()
            return {'left': pose.copy(), 'right': pose.copy()}

        if isinstance(prepare_pose, Mapping):
            poses = dict(prepare_pose)
            self._validate_target_names(poses, 'prepare pose')
            received = set(poses)
            if received != {'left', 'right'}:
                raise ValueError(
                    'Dual Franka prepare_pose must provide left and right '
                    f'poses, got {sorted(received)}')
        else:
            if len(prepare_pose) != 2:
                raise ValueError(
                    'Dual Franka prepare_pose must contain 2 arm poses, '
                    f'got {len(prepare_pose)}')
            poses = {
                'left': prepare_pose[0],
                'right': prepare_pose[1],
            }

        for arm, pose in poses.items():
            if len(pose) != 8:
                raise ValueError(
                    f'Prepare pose for {arm} must have 8 elements, '
                    f'got {len(pose)}')
        return poses

    def _default_prepare_pose(self):
        if self.command_mode == 'cartesian':
            return list(DEFAULT_PREPARE_EEPOSE)
        return list(DEFAULT_PREPARE_QPOS)

    def open_grippers(self, wait=False):
        """Open both grippers to the configured width."""
        del wait
        self._send_gripper_pair(
            self.gripper_open_width,
            self.gripper_open_width)

    def _build_joint_state(self, qpos):
        """Build a JointState command from the first seven joint values."""
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
        """Build a PoseStamped command from Cartesian pose values."""
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

    def _send_gripper_pair(self, left_width, right_width):
        """Publish optional left and right gripper width commands."""
        if left_width is not None:
            self._send_gripper_command('left', left_width)
        if right_width is not None:
            self._send_gripper_command('right', right_width)

    def _send_gripper_command(self, side, gripper_width):
        """Publish a gripper move command for one side."""
        if side == 'left':
            pub = self.left_gripper_pub
        elif side == 'right':
            pub = self.right_gripper_pub
        else:
            raise ValueError(f'Unknown gripper side: {side}')

        if pub is None:
            return

        try:
            pub.publish(self._build_gripper_goal(side, gripper_width))
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send %s gripper command: %s', side, exc)

    def _build_gripper_goal(self, side, gripper_width):
        """Build a Franka gripper move goal with width clamped to limits."""
        import rospy

        if self.MoveActionGoal is None:
            from franka_gripper.msg import MoveActionGoal
            self.MoveActionGoal = MoveActionGoal

        max_width = max(self.gripper_max_width, 0.0)
        target_width = min(max(float(gripper_width), 0.0), max_width)
        now = rospy.Time.now()
        msg = self.MoveActionGoal()
        msg.header.stamp = now
        msg.goal_id.stamp = now
        msg.goal_id.id = f'{side}_move_{time.time_ns()}'
        msg.goal.width = target_width
        msg.goal.speed = max(self.gripper_speed, 0.0)
        return msg