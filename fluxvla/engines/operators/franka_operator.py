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
DEFAULT_HOME_JOINT_POSITIONS = [
    0.0,
    -0.7853981633974483,
    0.0,
    -2.356194490192345,
    0.0,
    1.5707963267948966,
    0.7853981633974483,
]

CARTESIAN_IMPEDANCE_CONTROLLER = 'cartesian_impedance_controller'
JOINT_RUCKIG_IMPEDANCE_CONTROLLER = 'ruckig_joint_impedance_controller'
CONTROLLER_CHECK_TIMEOUT = 1.0
HOME_COMMAND_DURATION = 5.0


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
            base_frame_id='',
            sync_slop=0.04,
            sync_queue_size=30,
            synced_frame_queue_size=10,
            sync_warning_enabled=True,
            sync_warning_target_hz=30.0,
            sync_warning_window=2.0,
            sync_warning_min_hz_ratio=0.9,
            sync_warning_warmup=3.0,
            command_mode='joint',
            arm_ns='',
            cartesian_cmd_topic=None,
            joint_cmd_topic=None,
            joint_names=None,
            gripper_left_topic=None,
            gripper_speed=1.0,
            gripper_max_width=0.098,
            gripper_open_width=0.08,
            **unused_kwargs):
        self.img_left_topic = img_left_topic
        self.img_front_topic = img_front_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_ee_pose_left_topic = puppet_ee_pose_left_topic
        self.puppet_franka_state_left_topic = puppet_franka_state_left_topic
        self.use_depth_image = use_depth_image
        self.img_left_depth_topic = img_left_depth_topic
        self.img_front_depth_topic = img_front_depth_topic
        self.base_frame_id = base_frame_id

        super().__init__(
            sync_slop=sync_slop,
            sync_queue_size=sync_queue_size,
            synced_frame_queue_size=synced_frame_queue_size,
            sync_warning_enabled=sync_warning_enabled,
            sync_warning_target_hz=sync_warning_target_hz,
            sync_warning_window=sync_warning_window,
            sync_warning_min_hz_ratio=sync_warning_min_hz_ratio,
            sync_warning_warmup=sync_warning_warmup)

        if command_mode not in {'joint', 'cartesian'}:
            raise ValueError(
                f'Unsupported Franka command_mode: {command_mode}')
        self.command_mode = command_mode
        self.arm_ns = arm_ns
        self.cartesian_cmd_topic = (
            cartesian_cmd_topic or self._namespaced_default_topic(
                self.arm_ns,
                f'{CARTESIAN_IMPEDANCE_CONTROLLER}/equilibrium_pose'))
        self.joint_cmd_topic = (
            joint_cmd_topic or self._namespaced_default_topic(
                self.arm_ns,
                f'{JOINT_RUCKIG_IMPEDANCE_CONTROLLER}/target_joint_state'))
        self.joint_names = joint_names or DEFAULT_JOINT_NAMES
        self.gripper_goal_left_topic = (
            gripper_left_topic or self._namespaced_default_topic(
                self.arm_ns, 'franka_gripper/move/goal'))
        self.gripper_speed = float(gripper_speed)
        self.gripper_max_width = float(gripper_max_width)
        self.gripper_open_width = float(gripper_open_width)
        self.controller_check_timeout = CONTROLLER_CHECK_TIMEOUT
        self.ee_pub = None
        self.joint_pub = None
        self.gripper_pub = None
        self.MoveActionGoal = None
        self._controller_checks_done = set()

        if self.use_depth_image and not all(
            [img_left_depth_topic, img_front_depth_topic]):
            raise ValueError(
                'When use_depth_image=True, both depth topics must be provided')
        if (self.puppet_ee_pose_left_topic is None
                and self.puppet_franka_state_left_topic is None):
            raise ValueError(
                'Either puppet_ee_pose_left_topic or '
                'puppet_franka_state_left_topic must be provided')
        if len(self.joint_names) != 7:
            raise ValueError('joint_names must contain exactly 7 joints')

        self._init_ros()

    @staticmethod
    def _namespaced_default_topic(namespace, relative_topic):
        namespace = namespace.rstrip('/')
        if not namespace:
            return f'/{relative_topic}'
        return f'{namespace}/{relative_topic}'

    def _init_ros(self):
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        rospy.init_node('franka_operator', anonymous=True)

        camera_info_topics = self.setup_observation_sync(
            self.build_observation_specs())
        self._setup_control(rospy, PoseStamped, JointState)
        self.load_camera_info(camera_info_topics)

    def build_observation_specs(self):
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
        from franka_gripper.msg import MoveActionGoal

        self.MoveActionGoal = MoveActionGoal
        self.ee_pub = rospy.Publisher(
            self.cartesian_cmd_topic, PoseStamped, queue_size=10)
        self.joint_pub = rospy.Publisher(
            self.joint_cmd_topic, JointState, queue_size=10)
        self.gripper_pub = rospy.Publisher(
            self.gripper_goal_left_topic, MoveActionGoal, queue_size=1)

    def send_joints(self, arm_targets):
        self._check_command_controller(JOINT_RUCKIG_IMPEDANCE_CONTROLLER,
                                       'joint')
        self._validate_single_left_target(arm_targets, 'arm')
        self.joint_pub.publish(self._build_joint_state(arm_targets['left']))

    def send_eepose(self, arm_targets):
        self._check_command_controller(CARTESIAN_IMPEDANCE_CONTROLLER,
                                       'cartesian')
        self._validate_single_left_target(arm_targets, 'arm')
        self.ee_pub.publish(self._build_pose_stamped(arm_targets['left']))

    def send_gripper(self, gripper_targets, wait=False):
        del wait
        if not gripper_targets:
            raise ValueError('A left gripper width must be provided')
        self._validate_single_left_target(gripper_targets, 'gripper')
        self._send_gripper_command(gripper_targets['left'])

    @staticmethod
    def _validate_single_left_target(targets, target_type):
        if set(targets) != {'left'}:
            raise ValueError(
                f'Single Franka {target_type} target must use only "left"; '
                f'got {sorted(targets)}')

    def gohome(self):
        import rospy

        self.clear_observation_queues()
        restore_cartesian = self.command_mode == 'cartesian'
        if restore_cartesian:
            self._switch_command_controller(
                JOINT_RUCKIG_IMPEDANCE_CONTROLLER,
                CARTESIAN_IMPEDANCE_CONTROLLER)

        home_targets = {'left': DEFAULT_HOME_JOINT_POSITIONS}
        try:
            rospy.loginfo('Homing Franka arm via joint command')
            self.send_joints(home_targets)
            self.open_gripper(wait=True)
            rospy.sleep(HOME_COMMAND_DURATION)
            return home_targets
        finally:
            if restore_cartesian:
                self._switch_command_controller(
                    CARTESIAN_IMPEDANCE_CONTROLLER,
                    JOINT_RUCKIG_IMPEDANCE_CONTROLLER)
            self.clear_observation_queues()

    def open_gripper(self, wait=False):
        del wait
        self._send_gripper_command(self.gripper_open_width)

    def open_grippers(self, wait=False):
        self.open_gripper(wait=wait)

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

    def _send_gripper_command(self, gripper_width):
        if self.gripper_pub is None:
            return

        try:
            self.gripper_pub.publish(self._build_gripper_goal(gripper_width))
        except Exception as exc:
            import rospy
            rospy.logwarn('Failed to send gripper command: %s', exc)

    def _build_gripper_goal(self, gripper_width):
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

    def _check_command_controller(self, controller_name, command_mode):
        self._check_arm_controller(self.arm_ns, controller_name, command_mode)

    def _switch_command_controller(self, start_controller, stop_controller):
        if self._switch_arm_controller(self.arm_ns, start_controller,
                                       stop_controller):
            self._controller_checks_done.clear()

    def _switch_arm_controller(self, namespace, start_controller,
                               stop_controller):
        import rospy

        service_name = self._controller_service_name(namespace,
                                                     'switch_controller')
        try:
            from controller_manager_msgs.srv import (
                SwitchController,
                SwitchControllerRequest,
            )
            rospy.wait_for_service(
                service_name, timeout=self.controller_check_timeout)
            request = SwitchControllerRequest()
            request.start_controllers = [start_controller]
            request.stop_controllers = [stop_controller]
            request.strictness = getattr(SwitchControllerRequest,
                                         'BEST_EFFORT', 1)
            request.start_asap = True
            request.timeout = self.controller_check_timeout
            response = rospy.ServiceProxy(service_name, SwitchController)(
                request)
            if not getattr(response, 'ok', False):
                raise rospy.ROSException(
                    f'{service_name} returned ok=False')
            return True
        except Exception as exc:
            rospy.logwarn(
                'Failed to switch Franka controller on %s: start=%s, stop=%s, '
                'service=%s, error=%s',
                namespace,
                start_controller,
                stop_controller,
                service_name,
                exc)
            return False

    def _check_arm_controller(self, namespace, controller_name, command_mode):
        import rospy

        service_name = self._controller_service_name(namespace,
                                                     'list_controllers')
        check_key = (service_name, controller_name)
        if check_key in self._controller_checks_done:
            return

        try:
            from controller_manager_msgs.srv import ListControllers
            rospy.wait_for_service(
                service_name, timeout=self.controller_check_timeout)
            request = ListControllers._request_class()
            response = rospy.ServiceProxy(service_name, ListControllers)(
                request)
        except Exception as exc:
            rospy.logwarn('Unable to check Franka %s controller via %s: %s',
                          command_mode, service_name, exc)
            self._controller_checks_done.add(check_key)
            return

        controller = self._find_controller(response.controller,
                                           controller_name)
        if controller is None:
            rospy.logwarn(
                'Franka %s command expects controller "%s" on %s, but it is '
                'not loaded. Start the matching launch/controller before '
                'executing.',
                command_mode,
                controller_name,
                namespace or '/',
            )
            self._controller_checks_done.add(check_key)
            return

        if controller.state != 'running':
            rospy.logwarn(
                'Franka %s command expects controller "%s" on %s to be '
                'running, but state is "%s".',
                command_mode,
                controller_name,
                namespace or '/',
                controller.state)
        self._controller_checks_done.add(check_key)

    @staticmethod
    def _find_controller(controllers, name):
        for controller in controllers:
            if controller.name == name:
                return controller
        return None

    @staticmethod
    def _controller_service_name(namespace, service):
        namespace = namespace.rstrip('/')
        if not namespace:
            return f'/controller_manager/{service}'
        return f'{namespace}/controller_manager/{service}'
