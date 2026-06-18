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
HOME_COMMAND_DURATION = 5.0
BASE_FRAME_ID = ''
SYNC_QUEUE_SIZE = 30
SYNCED_FRAME_QUEUE_SIZE = 10
SYNC_WARNING_WINDOW = 2.0
SYNC_WARNING_MIN_HZ_RATIO = 0.9
SYNC_WARNING_WARMUP = 3.0
GRIPPER_SPEED = 1.0
GRIPPER_MAX_WIDTH = 0.08
GRIPPER_OPEN_WIDTH = 0.08

# Binary ("grasp") gripper control mode tunables. These are intentionally
# module-level constants rather than __init__ args to keep the operator
# signature minimal; the only knob exposed to config is gripper_control_mode.
# Closing always grasps with max force / max speed.
GRIPPER_BINARY_THRESHOLD = 0.06  # width below this => intent to close
GRASP_WIDTH = 0.0  # fixed target width for a grasp (close) transition
GRASP_EPSILON_INNER = 0.08  # GraspGoal.epsilon.inner
GRASP_EPSILON_OUTER = 0.08  # GraspGoal.epsilon.outer
GRASP_FORCE = 70.0  # Franka Hand max continuous grasping force (N)
GRIPPER_ACTION_WAIT = 2.0  # seconds to wait for the action servers


def gripper_action_namespace(move_goal_topic):
    """Derive the franka_gripper action namespace from a move goal topic.

    '/left_arm/franka_gripper/move/goal' -> '/left_arm/franka_gripper', so the
    grasp/move action servers are addressed as '<ns>/grasp' and '<ns>/move'.
    """
    namespace = move_goal_topic
    for suffix in ('/goal', '/move'):
        if namespace.endswith(suffix):
            namespace = namespace[:-len(suffix)]
    return namespace


def build_grasp_goal():
    """Build a fixed-width franka_gripper GraspGoal (close) at max force."""
    from franka_gripper.msg import GraspGoal

    goal = GraspGoal()
    goal.width = GRASP_WIDTH
    goal.epsilon.inner = GRASP_EPSILON_INNER
    goal.epsilon.outer = GRASP_EPSILON_OUTER
    goal.speed = GRIPPER_SPEED
    goal.force = GRASP_FORCE
    return goal


def build_move_goal(width):
    """Build a franka_gripper MoveGoal (open) at the given width."""
    from franka_gripper.msg import MoveGoal

    goal = MoveGoal()
    goal.width = float(width)
    goal.speed = GRIPPER_SPEED
    return goal


@OPERATORS.register_module()
class FrankaOperator(BaseOperator):
    """Single Franka operator using BaseOperator observation sync."""

    def __init__(
            self,
            img_left_topic,
            img_front_topic,
            puppet_arm_left_topic,
            puppet_franka_state_left_topic=None,
            puppet_ee_pose_left_topic=None,
            use_depth_image=False,
            img_left_depth_topic=None,
            img_front_depth_topic=None,
            sync_slop=0.04,
            sync_warning_enabled=True,
            sync_warning_target_hz=30.0,
            command_mode='joint',
            cartesian_cmd_topic=(
                '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
            joint_cmd_topic=('/left_arm/ruckig_joint_impedance_controller'
                             '/target_joint_state'),
            joint_names=None,
            gripper_left_topic='/left_arm/franka_gripper/move/goal',
            gripper_control_mode='move',
            **unused_kwargs):
        """Configure single-arm ROS topics, sync settings, and controllers.

        Args:
            img_left_topic (str): Wrist camera image topic for the left arm.
            img_front_topic (str): Front camera image topic.
            puppet_arm_left_topic (str): Left arm JointState observation topic.
            puppet_franka_state_left_topic (str, optional): Left FrankaState
                topic used to derive end-effector pose.
            puppet_ee_pose_left_topic (str, optional): Left PoseStamped topic
                used directly as end-effector pose.
            use_depth_image (bool): Whether to synchronize depth images.
            img_left_depth_topic (str, optional): Left wrist depth image topic.
            img_front_depth_topic (str, optional): Front depth image topic.
            sync_slop (float): Allowed timestamp difference for synchronized
                observations, in seconds.
            sync_warning_enabled (bool): Whether to warn about low sync rate.
            sync_warning_target_hz (float): Expected synchronized frame rate.
            command_mode (str): Command interface, either 'joint' or
                'cartesian'.
            cartesian_cmd_topic (str): Cartesian pose command topic.
            joint_cmd_topic (str): Joint command topic.
            joint_names (list[str], optional): Seven Franka joint names.
            gripper_left_topic (str): Left gripper move goal topic.
            gripper_control_mode (str): Gripper interface, either 'move'
                (default; publish continuous width to the move goal topic every
                step) or 'grasp' (binarize width and only act on open/close
                transitions, using the franka_gripper grasp/move actions).
            **unused_kwargs: Extra config keys accepted for compatibility.
        """
        self.img_left_topic = img_left_topic
        self.img_front_topic = img_front_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_ee_pose_left_topic = puppet_ee_pose_left_topic
        self.puppet_franka_state_left_topic = puppet_franka_state_left_topic
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
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
        self.cartesian_cmd_topic = cartesian_cmd_topic
        self.joint_cmd_topic = joint_cmd_topic
        self.joint_names = joint_names or DEFAULT_JOINT_NAMES
        self.gripper_goal_left_topic = gripper_left_topic
        if gripper_control_mode not in {'move', 'grasp'}:
            raise ValueError(f'Unsupported Franka gripper_control_mode: '
                             f'{gripper_control_mode}')
        self.gripper_control_mode = gripper_control_mode
        self.gripper_speed = GRIPPER_SPEED
        self.gripper_max_width = GRIPPER_MAX_WIDTH
        self.gripper_open_width = GRIPPER_OPEN_WIDTH
        self.ee_pub = None
        self.joint_pub = None
        self.gripper_pub = None
        self.MoveActionGoal = None
        # Binary-mode action clients and remembered open/close state.
        self.grasp_client = None
        self.move_action_client = None
        self._gripper_binary_state = None

        depth_topics = [img_left_depth_topic, img_front_depth_topic]
        if self.use_depth_image and not all(depth_topics):
            raise ValueError(
                'When use_depth_image=True, both depth topics must be provided'
            )
        if (self.puppet_ee_pose_left_topic is None
                and self.puppet_franka_state_left_topic is None):
            raise ValueError('Either puppet_ee_pose_left_topic or '
                             'puppet_franka_state_left_topic must be provided')
        if len(self.joint_names) != 7:
            raise ValueError('joint_names must contain exactly 7 joints')

        self._init_ros()

    def _init_ros(self):
        """Initialize ROS node, observation sync, publishers, and cameras."""
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        rospy.init_node('franka_operator', anonymous=True)

        camera_info_topics = self.setup_observation_sync(
            self.build_observation_specs())
        self._setup_control(rospy, PoseStamped, JointState)
        self.load_camera_info(camera_info_topics)

    def build_observation_specs(self):
        """Build synchronized image, joint, and end-effector specs."""
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
                'name': 'left_arm',
                'topic': self.puppet_arm_left_topic,
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
            ])
        return specs

    def _add_pose_specs(self, specs, PoseStamped):
        """Add the configured left-arm pose source to observation specs."""
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

    def _setup_control(self, rospy, PoseStamped, JointState):
        """Initialize ROS publishers for arm and gripper commands."""
        from franka_gripper.msg import MoveActionGoal

        self.MoveActionGoal = MoveActionGoal
        self.ee_pub = rospy.Publisher(
            self.cartesian_cmd_topic, PoseStamped, queue_size=10)
        self.joint_pub = rospy.Publisher(
            self.joint_cmd_topic, JointState, queue_size=10)
        self.gripper_pub = rospy.Publisher(
            self.gripper_goal_left_topic, MoveActionGoal, queue_size=1)
        if self.gripper_control_mode == 'grasp':
            self._setup_gripper_action_clients()

    def _setup_gripper_action_clients(self):
        """Connect grasp/move action clients for binary gripper control.

        The action namespace is derived from the configured move goal topic
        ('/.../franka_gripper/move/goal' -> '/.../franka_gripper'); a missing
        server is logged and left as None so sending falls back to the move
        publisher instead of raising.
        """
        import actionlib
        import rospy
        from franka_gripper.msg import GraspAction, MoveAction

        namespace = gripper_action_namespace(self.gripper_goal_left_topic)
        grasp_client = actionlib.SimpleActionClient(f'{namespace}/grasp',
                                                    GraspAction)
        move_client = actionlib.SimpleActionClient(f'{namespace}/move',
                                                   MoveAction)
        timeout = rospy.Duration(GRIPPER_ACTION_WAIT)
        if grasp_client.wait_for_server(timeout):
            self.grasp_client = grasp_client
        else:
            rospy.logwarn(
                'Gripper grasp action server %s/grasp unavailable; '
                'falling back to move publisher', namespace)
        if move_client.wait_for_server(timeout):
            self.move_action_client = move_client
        else:
            rospy.logwarn(
                'Gripper move action server %s/move unavailable; '
                'falling back to move publisher', namespace)

    def send_joints(self, arm_targets):
        """Publish a left-arm joint target."""
        self._validate_single_left_target(arm_targets, 'arm')
        self.joint_pub.publish(self._build_joint_state(arm_targets['left']))

    def send_eepose(self, arm_targets):
        """Publish a left-arm Cartesian end-effector pose target."""
        self._validate_single_left_target(arm_targets, 'arm')
        self.ee_pub.publish(self._build_pose_stamped(arm_targets['left']))

    def send_gripper(self, gripper_targets, wait=False):
        """Publish a left gripper width target."""
        del wait
        if not gripper_targets:
            raise ValueError('A left gripper width must be provided')
        self._validate_single_left_target(gripper_targets, 'gripper')
        self._send_gripper_command(gripper_targets['left'])

    @staticmethod
    def _validate_single_left_target(targets, target_type):
        """Validate that single-arm commands only target the left arm."""
        if set(targets) != {'left'}:
            raise ValueError(
                f'Single Franka {target_type} target must use only "left"; '
                f'got {sorted(targets)}')

    def gohome(self, prepare_pose):
        """Move the Franka arm to a prepare pose for the command mode."""
        import rospy

        prepare_pose = self._normalize_prepare_pose(prepare_pose)
        arm_targets = {arm: pose[:7] for arm, pose in prepare_pose.items()}
        gripper_targets = {arm: pose[7] for arm, pose in prepare_pose.items()}

        self.clear_observation_queues()
        try:
            if self.command_mode == 'cartesian':
                rospy.loginfo('Moving Franka arm to provided prepare ee pose')
                self.send_eepose(arm_targets)
            else:
                rospy.loginfo('Moving Franka arm to provided prepare joints')
                self.send_joints(arm_targets)

            self.send_gripper(gripper_targets, wait=True)
            rospy.sleep(HOME_COMMAND_DURATION)
            return prepare_pose
        finally:
            self.clear_observation_queues()

    def _normalize_prepare_pose(self, prepare_pose):
        """Normalize flat or mapping prepare_pose into a left-arm dict."""
        if prepare_pose is None:
            return {'left': self._default_prepare_pose()}

        if isinstance(prepare_pose, Mapping):
            poses = dict(prepare_pose)
            self._validate_single_left_target(poses, 'prepare pose')
            pose = poses['left']
        else:
            pose = prepare_pose
            if self._is_wrapped_single_pose(pose):
                pose = pose[0]
            poses = {'left': pose}

        if len(pose) != 8:
            raise ValueError(
                f'Prepare pose for left must have 8 elements, got {len(pose)}')
        return poses

    def _default_prepare_pose(self):
        if self.command_mode == 'cartesian':
            return list(DEFAULT_PREPARE_EEPOSE)
        return list(DEFAULT_PREPARE_QPOS)

    @staticmethod
    def _is_wrapped_single_pose(pose):
        """Return True when a single-arm pose is wrapped in one list level."""
        return (len(pose) == 1 and hasattr(pose[0], '__len__')
                and not isinstance(pose[0], (str, bytes)))

    def open_gripper(self, wait=False):
        """Open the left gripper to the configured width."""
        del wait
        # Force the next binary command to be sent even if we believe the
        # gripper is already open, so resets always issue an open.
        self._gripper_binary_state = None
        self._send_gripper_command(self.gripper_open_width)

    def open_grippers(self, wait=False):
        """Compatibility wrapper for callers shared with dual-arm operators."""
        self.open_gripper(wait=wait)

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

    def _send_gripper_command(self, gripper_width):
        """Publish a gripper move command if the gripper publisher exists."""
        if self.gripper_control_mode == 'grasp':
            self._send_gripper_binary(gripper_width)
            return

        if self.gripper_pub is None:
            return

        try:
            self.gripper_pub.publish(self._build_gripper_goal(gripper_width))
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send gripper command: %s', exc)

    def _build_gripper_goal(self, gripper_width):
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
        msg.goal_id.id = f'left_move_{time.time_ns()}'
        msg.goal.width = target_width
        msg.goal.speed = max(self.gripper_speed, 0.0)
        return msg

    def _send_gripper_binary(self, gripper_width):
        """Send a grasp/open only when the binary gripper state changes.

        Continuous widths are thresholded into close/open. A grasp (close)
        sends the franka_gripper grasp action at a fixed width with max force;
        an open sends the move action to the open width. Commands are skipped
        while the state is unchanged, so the slow blocking actions fire only on
        transitions.
        """
        should_close = float(gripper_width) < GRIPPER_BINARY_THRESHOLD
        if self._gripper_binary_state == should_close:
            return
        if should_close:
            sent = self._send_grasp(self.grasp_client)
        else:
            sent = self._send_open(self.move_action_client,
                                   self.gripper_open_width)
        if sent:
            self._gripper_binary_state = should_close

    def _send_grasp(self, grasp_client):
        """Send a blocking grasp goal; fall back to the move publisher."""
        if grasp_client is None:
            self._fallback_move_publish(GRASP_WIDTH)
            return True
        try:
            grasp_client.send_goal(build_grasp_goal())
            grasp_client.wait_for_result()
            return True
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send gripper grasp goal: %s', exc)
            return False

    def _send_open(self, move_client, open_width):
        """Send a blocking move-open goal; fall back to the move publisher."""
        if move_client is None:
            self._fallback_move_publish(open_width)
            return True
        try:
            move_client.send_goal(build_move_goal(open_width))
            move_client.wait_for_result()
            return True
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send gripper open goal: %s', exc)
            return False

    def _fallback_move_publish(self, width):
        """Publish a move goal when an action server is unavailable."""
        if self.gripper_pub is None:
            return
        try:
            self.gripper_pub.publish(self._build_gripper_goal(width))
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send gripper command: %s', exc)
