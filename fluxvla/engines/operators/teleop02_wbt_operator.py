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

import cv2
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation

from fluxvla.engines.utils.root import OPERATORS

# 31 joint names in teleop_cmd_WBT order
STATE_JOINT_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'head_yaw_joint',
    'head_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_yaw_joint',
    'left_wrist_pitch_joint',
    'left_wrist_roll_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_yaw_joint',
    'right_wrist_pitch_joint',
    'right_wrist_roll_joint',
]

# Default joint stiffness / damping
DEFAULT_KP = 140.0
DEFAULT_KD = 4.0


def _rot6d_to_rotation(rot6d):
    """Convert 6D rotation (Zhou et al.) to scipy Rotation.

    Uses numpy Gram-Schmidt, consistent with the data collection
    pipeline in create_wbt_lerobot_dataset.py.

    Args:
        rot6d (np.ndarray): (6,) array of 6D rotation.

    Returns:
        Rotation: scipy Rotation object.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[:3], rot6d[3:6]

    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)

    mat = np.stack([b1, b2, b3], axis=-2)  # (3, 3)
    return Rotation.from_matrix(mat)


def _rot6d_to_quat_xyzw(rot6d):
    """Convert 6D rotation (Zhou et al.) to quaternion [qx,qy,qz,qw]."""
    return _rot6d_to_rotation(rot6d).as_quat()


def _wrap_to_pi(angle):
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def interpolate_wbt_actions(actions,
                            source_hz=30,
                            target_hz=50,
                            method='cubic',
                            init_base_pos=None,
                            init_base_yaw=0.0):
    """Interpolate WBT actions from source_hz to target_hz.

    Handles mixed delta/absolute semantics:
        [0:31]  joint q — absolute, cubic/linear interpolate
        [31:33] base x/y — body-frame delta, integrate then interpolate
        [33]    base z — absolute, interpolate
        [34:40] base rot6d — yaw delta + abs pitch/roll, integrate then
                interpolate
        [40:]   discrete tail (hands and optional done), nearest neighbor

    Args:
        actions (np.ndarray): (T, D) actions at source_hz. D must be at
            least 42. D=42 is the legacy 2-hand layout, while D=53 is the
            raw 12-finger + done layout.
        source_hz (int): Source frequency.
        target_hz (int): Target frequency.
        method (str): 'cubic' or 'linear'.
        init_base_pos (np.ndarray): (3,) initial world base position.
            Defaults to [0, 0, 0.9].
        init_base_yaw (float): Initial accumulated yaw in radians.

    Returns:
        tuple: (actions_interp, base_pos_interp, base_quat_interp)
            - actions_interp: (T_out, D) interpolated actions (joints +
              discrete tail filled, base dims zeroed as they are in
              absolute outputs)
            - base_pos_interp: (T_out, 3) absolute world positions
            - base_quat_interp: (T_out, 4) absolute quaternions (xyzw)
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 42:
        raise ValueError(
            f'actions must have shape (T, D>=42), got {actions.shape}')
    T = len(actions)
    if T < 2:
        # Not enough points to interpolate
        pos = (
            init_base_pos.copy()
            if init_base_pos is not None else np.array([0.0, 0.0, 0.9]))
        quat = Rotation.identity().as_quat()
        return (actions.copy(), pos[np.newaxis], quat[np.newaxis])

    if init_base_pos is None:
        init_base_pos = np.array([0.0, 0.0, 0.9], dtype=np.float64)

    t_src = np.arange(T) / float(source_hz)
    duration = (T - 1) / float(source_hz)
    T_out = int(duration * target_hz) + 1
    t_tgt = np.arange(T_out) / float(target_hz)

    # ---- (1) Integrate base delta -> absolute world trajectory ----
    yaw_abs = np.zeros(T, dtype=np.float64)
    pitch_abs = np.zeros(T, dtype=np.float64)
    roll_abs = np.zeros(T, dtype=np.float64)
    world_xy = np.zeros((T, 2), dtype=np.float64)
    world_xy[0] = init_base_pos[:2]

    for i in range(T):
        euler = _rot6d_to_rotation(actions[i, 34:40]).as_euler('ZYX')
        if i == 0:
            yaw_abs[i] = _wrap_to_pi(init_base_yaw + euler[0])
        else:
            yaw_abs[i] = _wrap_to_pi(yaw_abs[i - 1] + euler[0])
        pitch_abs[i] = euler[1]
        roll_abs[i] = euler[2]

        if i > 0:
            dx_body = actions[i, 31]
            dy_body = actions[i, 32]
            yaw_prev = yaw_abs[i - 1]
            cos_y = np.cos(yaw_prev)
            sin_y = np.sin(yaw_prev)
            world_xy[i, 0] = (
                world_xy[i - 1, 0] + cos_y * dx_body - sin_y * dy_body)
            world_xy[i, 1] = (
                world_xy[i - 1, 1] + sin_y * dx_body + cos_y * dy_body)

    base_z = actions[:, 33].copy()

    # ---- (2) Interpolate in absolute space ----
    # Joint positions [0:31] — absolute
    if method == 'cubic':
        cs_joints = CubicSpline(t_src, actions[:, :31], axis=0)
        joints_out = cs_joints(t_tgt)
    else:
        f_joints = interp1d(t_src, actions[:, :31], axis=0, kind='linear')
        joints_out = f_joints(t_tgt)

    # World xy
    if method == 'cubic':
        xy_out = CubicSpline(t_src, world_xy, axis=0)(t_tgt)
    else:
        xy_out = interp1d(t_src, world_xy, axis=0, kind='linear')(t_tgt)

    # Base z — absolute
    if method == 'cubic':
        z_out = CubicSpline(t_src, base_z)(t_tgt)
    else:
        z_out = interp1d(t_src, base_z, kind='linear')(t_tgt)

    # Yaw (unwrap to avoid 2pi jumps), pitch, roll
    yaw_unwrapped = np.unwrap(yaw_abs)
    if method == 'cubic':
        yaw_out = CubicSpline(t_src, yaw_unwrapped)(t_tgt)
        pitch_out = CubicSpline(t_src, pitch_abs)(t_tgt)
        roll_out = CubicSpline(t_src, roll_abs)(t_tgt)
    else:
        yaw_out = interp1d(t_src, yaw_unwrapped, kind='linear')(t_tgt)
        pitch_out = interp1d(t_src, pitch_abs, kind='linear')(t_tgt)
        roll_out = interp1d(t_src, roll_abs, kind='linear')(t_tgt)

    # Discrete tail [40:] — nearest neighbor. This preserves both the
    # legacy 2-dim hand flags and newer 12-finger + done action layouts.
    tail_dim = actions.shape[1] - 40
    tail_out = np.zeros((T_out, tail_dim), dtype=np.float64)
    for i in range(T_out):
        src_idx = min(round(t_tgt[i] * source_hz), T - 1)
        tail_out[i] = actions[src_idx, 40:]

    # ---- (3) Assemble outputs ----
    actions_out = np.zeros((T_out, actions.shape[1]), dtype=np.float64)
    actions_out[:, :31] = joints_out
    # base dims [31:40] zeroed — caller uses absolute pos/quat instead
    actions_out[:, 40:] = tail_out

    base_pos_out = np.zeros((T_out, 3), dtype=np.float64)
    base_pos_out[:, :2] = xy_out
    base_pos_out[:, 2] = z_out

    base_quat_out = np.zeros((T_out, 4), dtype=np.float64)
    for i in range(T_out):
        rot = Rotation.from_euler('ZYX',
                                  [yaw_out[i], pitch_out[i], roll_out[i]])
        base_quat_out[i] = rot.as_quat()

    return actions_out, base_pos_out, base_quat_out


@OPERATORS.register_module()
class Teleop02WbtOperator:
    """Teleop02 WBT (whole-body tracking) operator using mros middleware.

    Sends joint-level commands + base_link pose via /teleop_cmd_WBT,
    matching the message format recorded in the WBT water task bags.

    The default robot layout has 31 joint positions, 2 hand closed flags
    (state = 33d), and 42-dim actions:
        [0:31]  joint position commands (q)
        [31:34] base_link position (body-frame delta_x/delta_y, absolute_z)
        [34:40] base_link rotation as rot6d from ZYX Euler
                (delta_yaw, absolute_pitch, absolute_roll)
        [40]    left_hand_closed
        [41]    right_hand_closed

    Set hand_state_dim=12 and hand_action_dim=12 for datasets that use raw
    BrainCo 12-finger state/action tails. In that mode action [52] may carry
    done and is ignored by the operator.
    """

    def __init__(self,
                 head_rgb_topic='/head/color/image_raw/compressed',
                 left_wrist_rgb_topic=(
                     '/left_wrist_camera/color/image_raw/compressed'),
                 joint_state_topic='/joint/state',
                 finger_state_topic='/brainco1/hand/state',
                 finger_cmd_topic='/brainco1/hand/cmd_vla',
                 teleop_wbt_topic='/teleop_cmd_WBT',
                 cmd_vel_topic='/sdk_cmd_vel_vla',
                 hand_state_dim=2,
                 hand_action_dim=2,
                 hand_command_scale=100.0,
                 binarize_hand_action=True,
                 finger_force_levels=None):
        """Initialize Teleop02WbtOperator with mros topics.

        Args:
            head_rgb_topic (str): Topic for head camera compressed image.
            left_wrist_rgb_topic (str): Topic for left wrist camera
                compressed image.
            joint_state_topic (str): Topic for joint state feedback.
            finger_state_topic (str): Topic for finger state feedback.
            finger_cmd_topic (str): Topic for finger commands.
            teleop_wbt_topic (str): Topic for WBT teleop commands
                (TeleopMsg with joint_cmd + base_link anchor).
            cmd_vel_topic (str): Topic for base velocity commands.
            hand_state_dim (int): Number of hand dims appended to the
                31-dim joint state. Use 2 for legacy left/right closed flags
                or 12 for raw BrainCo finger state.
            hand_action_dim (int): Number of hand action dims at action
                indices [40:40+hand_action_dim]. Use 2 for legacy
                left/right closed flags or 12 for raw BrainCo finger cmds.
            hand_command_scale (float): Scale applied to raw 0/1 hand action
                values before publishing to the BrainCo command topic.
            binarize_hand_action (bool): If True, threshold raw hand actions
                at 0.5 before applying hand_command_scale.
            finger_force_levels (tuple or None): Optional left/right force
                levels for indices [12], [13]. If None, existing send_action
                defaults are preserved.
        """
        self.head_rgb_topic = head_rgb_topic
        self.left_wrist_rgb_topic = left_wrist_rgb_topic
        self.joint_state_topic = joint_state_topic
        self.finger_state_topic = finger_state_topic
        self.finger_cmd_topic = finger_cmd_topic
        self.teleop_wbt_topic = teleop_wbt_topic
        self.cmd_vel_topic = cmd_vel_topic
        self.hand_state_dim = int(hand_state_dim)
        self.hand_action_dim = int(hand_action_dim)
        self.hand_command_scale = float(hand_command_scale)
        self.binarize_hand_action = bool(binarize_hand_action)
        self.finger_force_levels = (None
                                    if finger_force_levels is None else tuple(
                                        float(v) for v in finger_force_levels))

        self.last_finger_state = np.zeros(12, dtype=np.float32)
        self.last_finger_cmd = np.zeros(14, dtype=np.float32)

        # Accumulated base pose for delta action integration
        self._accum_base_pos = np.zeros(3, dtype=np.float64)
        self._accum_base_pos[2] = 0.9
        self._accum_base_yaw = 0.0
        self._accum_base_rot = Rotation.identity()

        # Trajectory thread state for async execution
        self._traj_thread = None
        self._traj_stop_event = threading.Event()
        self._last_get_frame_warning_time = {}

        self._init_mros()

    def _init_mros(self):
        """Initialize mros node, subscribers, and publishers."""
        import mros
        from mros.controller_msgs.msg import JointState
        from mros.sensor_msgs.msg import CompressedImage
        from mros.std_msgs.msg import Float32Array
        from mros.teleop_msgs.msg import TeleopMsg

        mros.init('FluxVLATeleop02WbtNode')

        # Subscribers
        self.color_subscriber = mros.subscribe(self.head_rgb_topic,
                                               CompressedImage, None)
        self.left_wrist_color_subscriber = mros.subscribe(
            self.left_wrist_rgb_topic, CompressedImage, None)
        self.joint_state_subscriber = mros.subscribe(self.joint_state_topic,
                                                     JointState, None)
        self.finger_state_subscriber = mros.subscribe(self.finger_state_topic,
                                                      Float32Array, None)

        # Publishers
        self.teleop_wbt_publisher = mros.advertise(self.teleop_wbt_topic,
                                                   TeleopMsg, None)
        self.finger_publisher = mros.advertise(
            self.finger_cmd_topic, Float32Array, queue_size=10)
        print(
            '[mros] Teleop02WbtOperator subscribed to '
            f'head={self.head_rgb_topic}, '
            f'left_wrist={self.left_wrist_rgb_topic}, '
            f'joint_state={self.joint_state_topic}, '
            f'finger_state={self.finger_state_topic}',
            flush=True)

    def _observation_hand_state(self):
        hand_state_dim = int(getattr(self, 'hand_state_dim', 2))
        if hand_state_dim == 12:
            return self.last_finger_state[:12].astype(np.float32, copy=True)
        if hand_state_dim != 2:
            raise ValueError(
                f'hand_state_dim must be 2 or 12, got {hand_state_dim}')

        # Compute hand closed flags from last finger cmd.
        left_cmd_avg = float(np.mean(self.last_finger_cmd[0:12:2]))
        right_cmd_avg = float(np.mean(self.last_finger_cmd[1:12:2]))
        left_hand_closed = 1.0 if left_cmd_avg > 20 else 0.0
        right_hand_closed = 1.0 if right_cmd_avg > 20 else 0.0
        return np.array([left_hand_closed, right_hand_closed],
                        dtype=np.float32)

    def _resolve_finger_force_levels(self, default_levels):
        configured = getattr(self, 'finger_force_levels', None)
        if configured is None:
            return tuple(float(v) for v in default_levels)
        if len(configured) != 2:
            raise ValueError(f'finger_force_levels must contain 2 values, got '
                             f'{configured}')
        return tuple(float(v) for v in configured)

    def _build_finger_cmd(self, action, default_force_levels):
        hand_action_dim = int(getattr(self, 'hand_action_dim', 2))
        force_levels = self._resolve_finger_force_levels(default_force_levels)

        if hand_action_dim == 12:
            if action.shape[0] < 52:
                raise ValueError(
                    f'raw 12-dim hand action requires at least 52 dims, '
                    f'got {action.shape[0]}')
            hand_action = np.asarray(action[40:52], dtype=np.float64)
            if getattr(self, 'binarize_hand_action', True):
                hand_action = (hand_action >= 0.5).astype(np.float64)
            else:
                hand_action = hand_action.copy()
            finger_cmd = [0.0] * 14
            scale = float(getattr(self, 'hand_command_scale', 100.0))
            finger_cmd[:12] = (hand_action * scale).tolist()
            finger_cmd[12] = force_levels[0]
            finger_cmd[13] = force_levels[1]
            return finger_cmd

        if hand_action_dim != 2:
            raise ValueError(
                f'hand_action_dim must be 2 or 12, got {hand_action_dim}')

        # Legacy 2-dim closed/open action expands to all BrainCo fingers.
        left_closed = float(action[40])
        right_closed = float(action[41])
        left_val = 100.0 if left_closed >= 0.5 else 0.0
        right_val = 100.0 if right_closed >= 0.5 else 0.0
        finger_cmd = [0.0] * 14
        for i in range(0, 12, 2):
            finger_cmd[i] = left_val
        for i in range(1, 12, 2):
            finger_cmd[i] = right_val

        # Thumb aux always closed for better grasping.
        finger_cmd[2] = 100.0
        finger_cmd[3] = 100.0

        finger_cmd[12] = force_levels[0]
        finger_cmd[13] = force_levels[1]
        return finger_cmd

    def _publish_finger_cmd(self, finger_cmd):
        from mros.std_msgs.msg import Float32Array

        self.last_finger_cmd = np.array(finger_cmd, dtype=np.float32)
        finger_msg = Float32Array()
        finger_msg.data = finger_cmd
        # print('finger_cmd:', finger_cmd, flush=True)
        self.finger_publisher.publish(finger_msg)

    def _compressed_msg_to_numpy(self, msg):
        """Decode CompressedImage (jpeg/png) to numpy BGR image.

        Args:
            msg: mros CompressedImage message.

        Returns:
            np.ndarray or None: BGR image array, or None on failure.
        """
        try:
            data = msg.data
            if isinstance(data, (bytes, bytearray)):
                buf = data
            elif isinstance(data, np.ndarray):
                buf = data.tobytes()
            else:
                buf = bytes(data)
            np_arr = np.frombuffer(buf, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f'Failed to decode compressed image: {e}')
            return None

    def _read_compressed_rgb_image(self, subscriber, timeout):
        """Read a compressed image message and return an RGB image."""
        msg = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = subscriber.readMsgRT()
            if msg is not None:
                break
            time.sleep(0.005)
        if msg is None:
            return None

        image = self._compressed_msg_to_numpy(msg)
        if image is None:
            return None
        return image[:, :, ::-1].copy()

    def _warn_get_frame_missing(self, key, message, interval=1.0):
        now = time.monotonic()
        last = self._last_get_frame_warning_time.get(key, 0.0)
        if now - last >= interval:
            print(message, flush=True)
            self._last_get_frame_warning_time[key] = now

    def get_frame(self, timeout=0.05):
        """Get synchronized observation from all sensors.

        Reads head camera image, left wrist camera image, joint states, and
        finger states.
        Returns combined 33-dim state vector (31 joints + 2 hand_closed) or
        43-dim state vector (31 joints + 12 raw BrainCo finger states).

        Args:
            timeout (float): Maximum wait time in seconds for each
                sensor reading. Defaults to 0.05.

        Returns:
            tuple or False: If successful, returns
                (head_img_rgb, left_wrist_img_rgb, joint_state). If failed,
                returns False.
        """
        # (1) Read head image
        head_img = self._read_compressed_rgb_image(self.color_subscriber,
                                                   timeout)
        if head_img is None:
            self._warn_get_frame_missing(
                'head_img',
                f'No head image received from {self.head_rgb_topic}')
            return False

        # (2) Read left wrist image
        left_wrist_img = self._read_compressed_rgb_image(
            self.left_wrist_color_subscriber, timeout)
        if left_wrist_img is None:
            self._warn_get_frame_missing(
                'left_wrist_img',
                'No left wrist image received from '
                f'{self.left_wrist_rgb_topic}')
            return False

        # (3) Read joint state
        joint_state_msg = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            joint_state_msg = self.joint_state_subscriber.readMsgRT()
            if joint_state_msg is not None:
                break
            time.sleep(0.005)
        if joint_state_msg is None:
            self._warn_get_frame_missing(
                'joint_state',
                f'No joint state message received from '
                f'{self.joint_state_topic}')
            return False

        joint_state = np.asarray(joint_state_msg.q, dtype=np.float32)
        if joint_state.size < 31:
            self._warn_get_frame_missing(
                'joint_state_size',
                f'Joint state size {joint_state.size} < 31 from '
                f'{self.joint_state_topic}')
            return False
        joint_state = joint_state[:31]

        # (4) Read finger state
        finger_state_msg = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            finger_state_msg = self.finger_state_subscriber.readMsgRT()
            if finger_state_msg is not None:
                break
            time.sleep(0.005)

        if finger_state_msg is not None:
            finger_state = np.asarray(finger_state_msg.data, dtype=np.float32)
            if finger_state.size >= 12:
                self.last_finger_state = finger_state[:12].copy()

        # Combine into 33-dim legacy state or 43-dim raw-hand state.
        state = np.concatenate([joint_state, self._observation_hand_state()])

        return (head_img, left_wrist_img, state)

    def _make_keypoint(self,
                       name,
                       pos=(0.0, 0.0, 0.0),
                       quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
        """Construct a TeleopMsg KeyPoint.

        Args:
            name (str): Keypoint name.
            pos (tuple): (x, y, z) position.
            quat_xyzw (tuple): (qx, qy, qz, qw) quaternion.

        Returns:
            KeyPoint: Constructed keypoint message.
        """
        from mros.teleop_msgs.msg import KeyPoint

        kp = KeyPoint()
        kp.name = name
        kp.pose.position.x = float(pos[0])
        kp.pose.position.y = float(pos[1])
        kp.pose.position.z = float(pos[2])
        kp.pose.orientation.x = float(quat_xyzw[0])
        kp.pose.orientation.y = float(quat_xyzw[1])
        kp.pose.orientation.z = float(quat_xyzw[2])
        kp.pose.orientation.w = float(quat_xyzw[3])
        return kp

    def send_action(self, action):
        """Send a WBT action to the robot.

        Action layout:
            [0:31]  joint position commands (q) -> joint_cmd
            [31:34] base_link position:
                    x/y are body-frame i->i+1 deltas,
                    z is normalized absolute value
            [34:40] base_link rotation as rot6d:
                    yaw is i->i+1 delta,
                    pitch/roll are normalized absolute values
            [40:42] legacy left/right hand_closed, or
            [40:52] raw BrainCo finger commands when hand_action_dim=12.
            [52]    optional done dim, ignored by deployment.

        Publishes a TeleopMsg to /teleop_cmd_WBT containing:
            - joint_cmd: JointCmd with 31 joint position targets
            - anchors[0]: base_link KeyPoint (xyz + quaternion)
        Also publishes finger commands to the finger_cmd_topic.

        Args:
            action (np.ndarray): 42/43/53-dim action vector.
        """
        from mros.controller_msgs.msg import JointCmd
        from mros.teleop_msgs.msg import TeleopMsg

        action = np.asarray(action, dtype=np.float64)

        # Parse WBT action.
        joint_cmd_q = action[0:31]
        base_pos_action = action[31:34]
        base_rot6d_action = action[34:40]

        if not hasattr(self, '_accum_base_yaw'):
            self._accum_base_yaw = self._accum_base_rot.as_euler('ZYX')[0]

        # Hybrid delta semantics from create_wbt_lerobot_dataset.py:
        # x/y/yaw are i -> i+1 frame deltas. x/y are expressed in the
        # current frame's body frame, so rotate them by current yaw before
        # integrating into world position. z/pitch/roll are absolute values.
        cos_yaw = np.cos(self._accum_base_yaw)
        sin_yaw = np.sin(self._accum_base_yaw)
        delta_x_body, delta_y_body = base_pos_action[:2]
        self._accum_base_pos[0] += (
            cos_yaw * delta_x_body - sin_yaw * delta_y_body)
        self._accum_base_pos[1] += (
            sin_yaw * delta_x_body + cos_yaw * delta_y_body)
        self._accum_base_pos[2] = base_pos_action[2]

        base_action_euler = _rot6d_to_rotation(base_rot6d_action).as_euler(
            'ZYX')
        self._accum_base_yaw = _wrap_to_pi(self._accum_base_yaw +
                                           base_action_euler[0])
        self._accum_base_rot = Rotation.from_euler('ZYX', [
            self._accum_base_yaw,
            base_action_euler[1],
            base_action_euler[2],
        ])

        base_pos = self._accum_base_pos
        base_quat_xyzw = self._accum_base_rot.as_quat()

        # Construct TeleopMsg
        teleop_msg = TeleopMsg()
        teleop_msg.header.frame_id = 'world'
        teleop_msg.world.orientation.w = 1.0

        # Joint command
        jcmd = JointCmd()
        jcmd.names = list(STATE_JOINT_NAMES)
        jcmd.q = joint_cmd_q.astype(np.float32).tolist()
        jcmd.v = [0.0] * 31
        jcmd.tau = [0.0] * 31
        jcmd.kp = [DEFAULT_KP] * 31
        jcmd.kd = [DEFAULT_KD] * 31
        jcmd.mode = [0] * 31
        jcmd.na = 31
        teleop_msg.joint_cmd = jcmd

        # Base link anchor (anchors[0])
        teleop_msg.anchors = [
            self._make_keypoint('base_link', base_pos, base_quat_xyzw),
        ]
        # print('teleop_wbt_msg:', teleop_msg, flush=True)
        self.teleop_wbt_publisher.publish(teleop_msg)

        finger_cmd = self._build_finger_cmd(
            action, default_force_levels=(2.0, 2.0))
        self._publish_finger_cmd(finger_cmd)

    def send_action_absolute(self, action, base_pos, base_quat_xyzw):
        """Send action with pre-computed absolute base pose.

        Skips delta integration — the caller provides the world-frame
        base position and orientation directly.

        Args:
            action (np.ndarray): 42/43/53-dim action. Joints [0:31] and
                the configured hand action tail are used; base dims ignored.
            base_pos (np.ndarray): (3,) absolute world position.
            base_quat_xyzw (np.ndarray): (4,) quaternion [qx,qy,qz,qw].
        """
        from mros.controller_msgs.msg import JointCmd
        from mros.teleop_msgs.msg import TeleopMsg

        action = np.asarray(action, dtype=np.float64)
        base_pos = np.asarray(base_pos, dtype=np.float64)
        base_quat_xyzw = np.asarray(base_quat_xyzw, dtype=np.float64)

        # Construct TeleopMsg
        teleop_msg = TeleopMsg()
        teleop_msg.header.frame_id = 'world'
        teleop_msg.world.orientation.w = 1.0

        # Joint command
        jcmd = JointCmd()
        jcmd.names = list(STATE_JOINT_NAMES)
        jcmd.q = action[0:31].astype(np.float32).tolist()
        jcmd.v = [0.0] * 31
        jcmd.tau = [0.0] * 31
        jcmd.kp = [DEFAULT_KP] * 31
        jcmd.kd = [DEFAULT_KD] * 31
        jcmd.mode = [0] * 31
        jcmd.na = 31
        teleop_msg.joint_cmd = jcmd

        # Base link anchor — absolute pose
        teleop_msg.anchors = [
            self._make_keypoint('base_link', base_pos, base_quat_xyzw),
        ]
        # print('teleop_wbt_msg:', teleop_msg, flush=True)
        self.teleop_wbt_publisher.publish(teleop_msg)

        # Sync accumulated state for potential fallback to delta mode
        self._accum_base_pos = base_pos.copy()
        self._accum_base_rot = Rotation.from_quat(base_quat_xyzw)
        self._accum_base_yaw = self._accum_base_rot.as_euler('ZYX')[0]

        finger_cmd = self._build_finger_cmd(
            action, default_force_levels=(1.0, 1.0))
        self._publish_finger_cmd(finger_cmd)

    def execute_trajectory_interpolated(self,
                                        actions,
                                        source_hz,
                                        target_hz,
                                        method='cubic',
                                        async_exec=False,
                                        running_flag_fn=None):
        """Execute trajectory with frequency interpolation.

        Interpolates 42-dim WBT actions from source_hz to target_hz,
        handling delta-to-absolute conversion for base pose, then sends
        each step via send_action_absolute.

        Args:
            actions (np.ndarray): (N, 42) actions at source_hz.
            source_hz (int): Model output frequency (e.g. 30).
            target_hz (int): Robot command frequency (e.g. 50).
            method (str): Interpolation method ('cubic' or 'linear').
            async_exec (bool): If True, run in background thread.
            running_flag_fn (callable, optional): Checked each step.
        """
        execute_start = time.perf_counter()
        # print(
        #     f'[timing] operator_execute_trajectory_interpolated_start '
        #     f'num_actions={len(actions)} '
        #     f'source_hz={source_hz} target_hz={target_hz}',
        #     flush=True)

        self.stop_trajectory()
        if self._traj_thread is not None:
            self._traj_thread.join(timeout=2.0)
        self._traj_stop_event = threading.Event()

        # Interpolate
        interp_start = time.perf_counter()
        actions_interp, base_pos_interp, base_quat_interp = \
            interpolate_wbt_actions(
                actions, source_hz, target_hz, method=method,
                init_base_pos=self._accum_base_pos.copy(),
                init_base_yaw=self._accum_base_yaw)
        # print(
        #     f'[timing] interpolation='
        #     f'{(time.perf_counter() - interp_start) * 1000.0:.3f} ms '
        #     f'output_steps={len(actions_interp)}',
        #     flush=True)

        dt_target = 1.0 / target_hz

        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_trajectory_absolute,
                args=(actions_interp, base_pos_interp, base_quat_interp,
                      dt_target, self._traj_stop_event, running_flag_fn),
                daemon=True)
            self._traj_thread.start()
        else:
            self._run_trajectory_absolute(actions_interp, base_pos_interp,
                                          base_quat_interp, dt_target,
                                          self._traj_stop_event,
                                          running_flag_fn)

        # print(
        #     f'[timing] operator_execute_trajectory_interpolated_total='
        #     f'{(time.perf_counter() - execute_start) * 1000.0:.3f} ms',
        #     flush=True)

    def _run_trajectory_absolute(self, actions, base_pos, base_quat, dt,
                                 stop_event, running_flag_fn):
        """Execute interpolated trajectory with absolute base poses.

        Args:
            actions (np.ndarray): (N, 42) interpolated actions.
            base_pos (np.ndarray): (N, 3) absolute world positions.
            base_quat (np.ndarray): (N, 4) absolute quaternions (xyzw).
            dt (float): Sleep duration between steps.
            stop_event (threading.Event): Set to abort early.
            running_flag_fn (callable or None): Checked each step.
        """
        trajectory_start = time.perf_counter()
        for i in range(len(actions)):
            if stop_event.is_set():
                break
            if running_flag_fn is not None and not running_flag_fn():
                break
            self.send_action_absolute(actions[i], base_pos[i], base_quat[i])
            time.sleep(dt)
        # print(
        #     f'[timing] operator_run_trajectory_absolute_total='
        #     f'{(time.perf_counter() - trajectory_start) * 1000.0:.3f} ms '
        #     f'num_actions={len(actions)}',
        #     flush=True)

    def execute_trajectory(self,
                           actions,
                           dt,
                           async_exec=False,
                           running_flag_fn=None):
        """Execute a trajectory of 42-dim actions.

        Args:
            actions (np.ndarray): Trajectory array of shape (N, 42).
            dt (float): Time step between trajectory points in seconds.
            async_exec (bool): If True, execute in a background thread.
            running_flag_fn (callable, optional): Returns bool; checked
                each step for graceful shutdown.
        """
        execute_start = time.perf_counter()
        # print(
        #     f'[timing] operator_execute_trajectory_start '
        #     f'num_actions={len(actions)} dt={dt:.6f} '
        #     f'async_exec={async_exec}',
        #     flush=True)

        stop_start = time.perf_counter()
        self.stop_trajectory()
        # print(
        #     f'[timing] operator_stop_trajectory='
        #     f'{(time.perf_counter() - stop_start) * 1000.0:.3f} ms',
        #     flush=True)
        if self._traj_thread is not None:
            join_start = time.perf_counter()
            self._traj_thread.join(timeout=2.0)
            # print(
            #     f'[timing] operator_join_previous_thread='
            #     f'{(time.perf_counter() - join_start) * 1000.0:.3f} ms '
            #     f'alive_after_join={self._traj_thread.is_alive()}',
            #     flush=True)
        self._traj_stop_event = threading.Event()

        if async_exec:
            start_thread_start = time.perf_counter()
            self._traj_thread = threading.Thread(
                target=self._run_trajectory,
                args=(actions, dt, self._traj_stop_event, running_flag_fn),
                daemon=True)
            self._traj_thread.start()
            start_thread_ms = (time.perf_counter() -
                               start_thread_start) * 1000.0
            # print(
            #     f'[timing] operator_start_trajectory_thread='
            #     f'{start_thread_ms:.3f} ms',
            #     flush=True)
        else:
            self._run_trajectory(actions, dt, self._traj_stop_event,
                                 running_flag_fn)
        # print(
        #     f'[timing] operator_execute_trajectory_total='
        #     f'{(time.perf_counter() - execute_start) * 1000.0:.3f} ms',
        #     flush=True)

    def _run_trajectory(self, actions, dt, stop_event, running_flag_fn):
        """Execute actions sequentially with rate control.

        Args:
            actions (np.ndarray): Trajectory array of shape (N, 42).
            dt (float): Sleep duration between steps.
            stop_event (threading.Event): Set to abort early.
            running_flag_fn (callable or None): Checked each step.
        """
        trajectory_start = time.perf_counter()
        first_action_sent = False
        for action_idx, action in enumerate(actions):
            if stop_event.is_set():
                break
            if running_flag_fn is not None and not running_flag_fn():
                break
            send_start = time.perf_counter()
            self.send_action(action)
            if not first_action_sent:
                first_action_sent = True
                since_start_ms = (time.perf_counter() -
                                  trajectory_start) * 1000.0
                # print(
                #     f'[timing] operator_first_send_action='
                #     f'{(time.perf_counter() - send_start) * 1000.0:.3f} ms '
                #     f'since_trajectory_start='
                #     f'{since_start_ms:.3f} ms',
                #     flush=True)
            time.sleep(dt)
        # print(
        #     f'[timing] operator_run_trajectory_total='
        #     f'{(time.perf_counter() - trajectory_start) * 1000.0:.3f} ms '
        #     f'num_actions={len(actions)}',
        #     flush=True)

    def stop_trajectory(self):
        """Signal the background trajectory thread to stop."""
        self._traj_stop_event.set()
