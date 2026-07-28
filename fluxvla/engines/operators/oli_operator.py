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

import json
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fluxvla.engines.operators.base_operator import BaseOperator
from fluxvla.engines.utils.root import OPERATORS

# 31 joint names in whole-body command order.
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

# Default joint stiffness / damping.
DEFAULT_KP = 140.0
DEFAULT_KD = 4.0


class NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that tolerates numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _rotmat_to_quat_xyzw(mat):
    """Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw]."""
    m = np.asarray(mat, dtype=np.float64)
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-8)


def _is_degenerate_rot6d(rot6d):
    """True if the 6D rotation basis is too degenerate to orthonormalize."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[:3], rot6d[3:6]
    if np.linalg.norm(a1) < 1e-6:
        return True
    b1 = a1 / np.linalg.norm(a1)
    residual = a2 - np.dot(b1, a2) * b1
    return bool(np.linalg.norm(residual) < 1e-6)


def _rot6d_to_quat_xyzw(rot6d):
    """Convert a 6D rotation (Zhou et al.) to a quaternion [qx,qy,qz,qw].

    Uses numpy Gram-Schmidt, consistent with the data collection pipeline.

    Args:
        rot6d (np.ndarray): (6,) array of 6D rotation.

    Returns:
        np.ndarray: (4,) quaternion in [qx, qy, qz, qw] order.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[:3], rot6d[3:6]

    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)

    mat = np.stack([b1, b2, b3], axis=-2)  # (3, 3)
    return _rotmat_to_quat_xyzw(mat)


def _wrap_to_pi(angle):
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@OPERATORS.register_module()
class OliOperator(BaseOperator):
    """Oli whole-body (loco-manipulation) operator.

    The legacy ``websocket`` backend keeps the original ROS observation and
    LimX WebSocket command path.  The ``mros`` backend consumes the head and
    optional wrist compressed-image topics and publishes whole-body commands
    on ``/teleop_cmd_WBT``.  Both modes share BaseOperator lifecycle and
    trajectory state.

    state (33-dim): 31 joint positions + 2 hand-closed flags.
    action (42-dim):
        [0:31]  joint position commands (q)
        [31:34] base_link position (xyz, absolute)
        [34:40] base_link rotation (rot6d)
        [40]    left_hand_closed
        [41]    right_hand_closed
    """

    def __init__(self,
                 head_rgb_topic='/head/color/image_raw/compressed',
                 joint_state_topic='/joint/state',
                 robot_ip='10.192.1.2',
                 ws_port=5000,
                 ws_accid=None,
                 control_backend='websocket',
                 left_wrist_rgb_topic=None,
                 finger_state_topic='/brainco1/hand/state',
                 finger_cmd_topic='/brainco1/hand/cmd_vla',
                 teleop_wbt_topic='/teleop_cmd_WBT',
                 finger_force_levels=None):
        """Initialize OliOperator.

        Args:
            head_rgb_topic (str): ROS topic for head compressed RGB image.
            joint_state_topic (str): ROS topic for joint state feedback.
            robot_ip (str): Robot IP for the WebSocket control channel.
            ws_port (int): WebSocket port. Defaults to 5000.
            ws_accid (str): WebSocket account id; None means auto-detect.
            control_backend (str): ``websocket`` for the legacy transport or
                ``mros`` for the WBT topic transport.
            left_wrist_rgb_topic (str, optional): Optional second camera used
                by the HUD04 checkpoint in MROS mode.
            finger_state_topic (str): BrainCo hand-state topic in MROS mode.
            finger_cmd_topic (str): BrainCo hand-command topic in MROS mode.
            teleop_wbt_topic (str): Whole-body command topic in MROS mode.
            finger_force_levels (tuple, optional): Left/right hand force
                levels. Defaults to (3, 3) for WebSocket and (2, 2) for MROS.
        """
        if control_backend not in {'websocket', 'mros'}:
            raise ValueError(
                f'Unsupported Oli control_backend: {control_backend}')

        self.head_rgb_topic = head_rgb_topic
        self.left_wrist_rgb_topic = left_wrist_rgb_topic
        self.joint_state_topic = joint_state_topic
        self.finger_state_topic = finger_state_topic
        self.finger_cmd_topic = finger_cmd_topic
        self.teleop_wbt_topic = teleop_wbt_topic
        self.control_backend = control_backend
        self.command_mode = 'joint'

        self.robot_ip = robot_ip
        self.ws_port = ws_port
        self.ws_accid = ws_accid
        self.ws_client = None
        self.ws_connected = False
        self.ws_lock = threading.Lock()
        self.json_encoder = NumpySafeEncoder

        self.last_finger_state = np.zeros(12, dtype=np.float32)
        self.last_finger_cmd = np.zeros(14, dtype=np.float32)
        self.finger_force_levels = finger_force_levels
        self._accum_base_pos = np.array([0.0, 0.0, 0.9], dtype=np.float64)
        self._accum_base_yaw = 0.0
        self._accum_base_rot = Rotation.identity()
        self._last_get_frame_warning_time = {}

        super().__init__(sync_warning_enabled=False)

        if self.control_backend == 'mros':
            self._init_mros()
        else:
            self._init_ros()
            self._init_websocket()

    # ========== ROS sensor input ==========

    def _init_ros(self):
        """Initialize ROS node, subscribers, and buffers (lazy import)."""
        import rospy
        from sensor_msgs.msg import CompressedImage, JointState

        self.head_img_deque = deque(maxlen=5)
        self.joint_state_deque = deque(maxlen=5)

        if rospy.get_name() == '/unnamed':
            rospy.init_node('oli_operator_node', anonymous=True)

        rospy.Subscriber(
            self.head_rgb_topic,
            CompressedImage,
            self._head_img_callback,
            queue_size=1000,
            tcp_nodelay=True)
        rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_state_callback,
            queue_size=1000,
            tcp_nodelay=True)

    def _init_mros(self):
        """Initialize MROS subscriptions and WBT command publishers."""
        import mros
        from mros.controller_msgs.msg import JointState
        from mros.sensor_msgs.msg import CompressedImage
        from mros.std_msgs.msg import Float32Array
        from mros.teleop_msgs.msg import TeleopMsg

        mros.init('FluxVLAOliNode')
        self.color_subscriber = mros.subscribe(self.head_rgb_topic,
                                               CompressedImage, None)
        self.left_wrist_color_subscriber = None
        if self.left_wrist_rgb_topic:
            self.left_wrist_color_subscriber = mros.subscribe(
                self.left_wrist_rgb_topic, CompressedImage, None)
        self.joint_state_subscriber = mros.subscribe(self.joint_state_topic,
                                                     JointState, None)
        self.finger_state_subscriber = mros.subscribe(self.finger_state_topic,
                                                      Float32Array, None)

        self.teleop_wbt_publisher = mros.advertise(self.teleop_wbt_topic,
                                                   TeleopMsg, None)
        self.finger_publisher = mros.advertise(
            self.finger_cmd_topic, Float32Array, queue_size=10)
        print(
            '[mros] OliOperator subscribed to '
            f'head={self.head_rgb_topic}, '
            f'left_wrist={self.left_wrist_rgb_topic or "disabled"}, '
            f'joint_state={self.joint_state_topic}, '
            f'finger_state={self.finger_state_topic}',
            flush=True)

    def _head_img_callback(self, msg):
        """Buffer the latest head image message."""
        self.head_img_deque.append(msg)

    def _joint_state_callback(self, msg):
        """Buffer the latest joint state message."""
        self.joint_state_deque.append(msg)

    def _decode_compressed(self, msg):
        """Decode a CompressedImage message to a BGR numpy image."""
        try:
            import cv2
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f'Failed to decode compressed image: {e}')
            return None

    def get_frame(self, timeout=0.05):
        """Return the latest observation for the configured backend."""
        if self.control_backend == 'mros':
            return self._get_mros_frame(timeout=timeout)
        return self._get_ros_frame()

    def _get_ros_frame(self):
        """Get the latest available observation (latest-only polling).

        Returns the most recent head image and joint state WITHOUT timestamp
        synchronization — this is a latest-only poll, not a synchronized read.

        Returns:
            tuple or False: ``(head_img_rgb, state_33d)`` on success, where
                state is 31 joints + 2 hand-closed flags; ``False`` if the
                head image or joint state is not yet available.
        """
        if (len(self.head_img_deque) == 0 or len(self.joint_state_deque) == 0):
            return False

        head_bgr = self._decode_compressed(self.head_img_deque[-1])
        if head_bgr is None:
            return False
        head_img = head_bgr[:, :, ::-1].copy()  # BGR -> RGB

        joint_msg = self.joint_state_deque[-1]
        names = list(joint_msg.name) if getattr(joint_msg, 'name', None) \
            else []
        if names:
            positions = list(joint_msg.position)
            if len(names) != len(positions):
                print('Joint name/position length mismatch; '
                      'dropping frame')
                return False
            name_to_pos = dict(zip(names, positions))
            if len(name_to_pos) != len(names):
                print('Duplicate joint names in joint state; '
                      'dropping frame')
                return False
            missing = [n for n in STATE_JOINT_NAMES if n not in name_to_pos]
            if missing:
                print(f'Joints {missing} missing from joint state; '
                      f'cannot assemble Oli state')
                return False
            joint_state = np.array([name_to_pos[n] for n in STATE_JOINT_NAMES],
                                   dtype=np.float32)
        else:
            # No joint names published; assume canonical STATE_JOINT_NAMES
            # order.
            joint_state = np.asarray(joint_msg.position, dtype=np.float32)
            if joint_state.size < 31:
                print(f'Joint state size {joint_state.size} < 31')
                return False
            joint_state = joint_state[:31]
        if not np.all(np.isfinite(joint_state)):
            print('Non-finite joint positions; dropping frame')
            return False

        # Hand-closed flags mirror the last sent finger command
        # (data-collection convention), not a direct sensor reading; both are
        # 0 before the first send_action call.
        left_cmd_avg = float(np.mean(self.last_finger_cmd[0:12:2]))
        right_cmd_avg = float(np.mean(self.last_finger_cmd[1:12:2]))
        left_hand_closed = 1.0 if left_cmd_avg > 20 else 0.0
        right_hand_closed = 1.0 if right_cmd_avg > 20 else 0.0

        state = np.concatenate([
            joint_state,
            np.array([left_hand_closed, right_hand_closed], dtype=np.float32)
        ])
        return (head_img, state)

    def _warn_get_frame_missing(self, key, message, interval=1.0):
        now = time.monotonic()
        last = self._last_get_frame_warning_time.get(key, 0.0)
        if now - last >= interval:
            print(message, flush=True)
            self._last_get_frame_warning_time[key] = now

    def _read_mros_compressed_rgb(self, subscriber, timeout):
        if subscriber is None:
            return None
        deadline = time.monotonic() + float(timeout)
        msg = None
        while time.monotonic() < deadline:
            msg = subscriber.readMsgRT()
            if msg is not None:
                break
            time.sleep(0.005)
        if msg is None:
            return None

        try:
            data = msg.data
            if isinstance(data, np.ndarray):
                data = data.tobytes()
            elif not isinstance(data, (bytes, bytearray)):
                data = bytes(data)
            encoded = np.frombuffer(data, dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except Exception as exc:
            print(f'Failed to decode MROS compressed image: {exc}')
            return None
        if bgr is None:
            return None
        return bgr[:, :, ::-1].copy()

    def _read_mros_message(self, subscriber, timeout):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            msg = subscriber.readMsgRT()
            if msg is not None:
                return msg
            time.sleep(0.005)
        return None

    def _hand_closed_state(self):
        left_avg = float(np.mean(self.last_finger_cmd[0:12:2]))
        right_avg = float(np.mean(self.last_finger_cmd[1:12:2]))
        return np.array([float(left_avg > 20.0),
                         float(right_avg > 20.0)],
                        dtype=np.float32)

    def _get_mros_frame(self, timeout=0.05):
        """Read head/wrist images and the 33-dim WBT state from MROS."""
        head_img = self._read_mros_compressed_rgb(self.color_subscriber,
                                                  timeout)
        if head_img is None:
            self._warn_get_frame_missing(
                'head_img',
                f'No head image received from {self.head_rgb_topic}')
            return False

        left_wrist_img = None
        if self.left_wrist_color_subscriber is not None:
            left_wrist_img = self._read_mros_compressed_rgb(
                self.left_wrist_color_subscriber, timeout)
            if left_wrist_img is None:
                self._warn_get_frame_missing(
                    'left_wrist_img', 'No left wrist image received from '
                    f'{self.left_wrist_rgb_topic}')
                return False

        joint_msg = self._read_mros_message(self.joint_state_subscriber,
                                            timeout)
        if joint_msg is None:
            self._warn_get_frame_missing(
                'joint_state',
                f'No joint state received from {self.joint_state_topic}')
            return False
        joint_state = np.asarray(joint_msg.q, dtype=np.float32)
        if joint_state.size < len(STATE_JOINT_NAMES):
            self._warn_get_frame_missing(
                'joint_state_size', f'Joint state size {joint_state.size} < '
                f'{len(STATE_JOINT_NAMES)}')
            return False
        joint_state = joint_state[:len(STATE_JOINT_NAMES)]
        if not np.all(np.isfinite(joint_state)):
            self._warn_get_frame_missing(
                'joint_state_finite',
                'Non-finite joint positions received; dropping frame')
            return False

        finger_msg = self._read_mros_message(self.finger_state_subscriber,
                                             timeout)
        if finger_msg is not None:
            finger_state = np.asarray(finger_msg.data, dtype=np.float32)
            if finger_state.size >= 12:
                self.last_finger_state = finger_state[:12].copy()

        state = np.concatenate([joint_state, self._hand_closed_state()])
        if left_wrist_img is None:
            return (head_img, state)
        return (head_img, left_wrist_img, state)

    # ========== WebSocket control output ==========

    def _init_websocket(self):
        """Initialize the WebSocket control channel (lazy import)."""
        try:
            import websocket
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'websocket-client is required for Oli robot control. '
                'Install it with: pip install websocket-client') from exc

        self.ws_url = f'ws://{self.robot_ip}:{self.ws_port}'
        self.ws_client = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._ws_on_open,
            on_message=self._ws_on_message,
            on_close=self._ws_on_close,
            on_error=self._ws_on_error)

        self._ws_thread = threading.Thread(
            target=self._ws_run_forever, daemon=True)
        self._ws_thread.start()

        timeout = 5.0
        start_time = time.time()
        while (not self.ws_connected and (time.time() - start_time) < timeout):
            time.sleep(0.1)

        if self.ws_connected:
            print(f'OliOperator WebSocket connected to {self.ws_url}')
        else:
            raise ConnectionError(
                f'OliOperator WebSocket connection timeout to {self.ws_url}')

        if self.ws_accid is None:
            accid_timeout = 5.0
            start_time = time.time()
            while (self.ws_accid is None
                   and (time.time() - start_time) < accid_timeout):
                time.sleep(0.1)
            if self.ws_accid is not None:
                print(f'OliOperator auto-detected ws_accid: {self.ws_accid}')
            else:
                print('OliOperator WARNING: ws_accid auto-detection timed '
                      'out, control commands may not work')

    def _ws_run_forever(self):
        """Run the WebSocket client loop in a background thread."""
        try:
            self.ws_client.run_forever()
        except Exception as e:
            print(f'WebSocket run_forever error: {e}')

    def _ws_on_open(self, ws):
        """WebSocket on_open callback."""
        self.ws_connected = True

    def _ws_on_message(self, ws, message):
        """WebSocket on_message callback; auto-detects accid and logs
        failures."""
        try:
            response = json.loads(message)
            if not isinstance(response, dict):
                return
            recv_accid = response.get('accid', None)
            if self.ws_accid is None and recv_accid is not None:
                self.ws_accid = recv_accid

            if recv_accid != self.ws_accid:
                return
            if response.get('title', '') == 'notify_robot_info':
                return

            title = response.get('title', '')
            resp_data = response.get('data', {})
            if title == 'notify_invalid_request':
                print(f'WebSocket invalid request: {resp_data}')
                return
            if isinstance(resp_data, dict) and 'result' in resp_data:
                if resp_data['result'] != 'success':
                    print(f'WebSocket command failed [{title}]: '
                          f"{resp_data['result']}")
        except json.JSONDecodeError:
            print(f'WebSocket invalid JSON: {message}')

    def _ws_on_close(self, ws, close_status_code, close_msg):
        """WebSocket on_close callback."""
        self.ws_connected = False
        print(f'WebSocket closed: {close_status_code} - {close_msg}')

    def _ws_on_error(self, ws, error):
        """WebSocket on_error callback."""
        print(f'WebSocket error: {error}')

    def _ws_send_request(self, title, data=None):
        """Send a WebSocket request to the robot (non-blocking)."""
        if data is None:
            data = {}
        message = {
            'accid': self.ws_accid,
            'title': title,
            'timestamp': int(time.time() * 1000),
            'guid': str(uuid.uuid4()),
            'data': data,
        }
        with self.ws_lock:
            try:
                if self.ws_client and self.ws_connected:
                    self.ws_client.send(
                        json.dumps(message, cls=self.json_encoder))
            except Exception as e:
                print(f'WebSocket send error: {e}')

    # ========== Command helpers ==========

    def send_action(self, action):
        """Send a whole-body action through the configured transport.

        Args:
            action (np.ndarray): Action vector with at least the 42 WBT dims.
        """
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1 or action.size < 42:
            raise ValueError(
                f'OliOperator expects a (D>=42,) action, got {action.shape}')
        if not np.all(np.isfinite(action)):
            raise ValueError('OliOperator received a non-finite action')

        if self.control_backend == 'mros':
            self._send_action_mros(action)
        else:
            self._send_action_websocket(action)

    def _send_action_websocket(self, action):
        """Send one action using the legacy WebSocket requests."""

        joint_cmd_q = action[0:31]
        base_pos = action[31:34]
        base_rot6d = action[34:40]
        left_closed = float(action[40])
        right_closed = float(action[41])

        self._send_joints(joint_cmd_q)
        if _is_degenerate_rot6d(base_rot6d):
            print('OliOperator: degenerate base rot6d; skipping base pose')
        else:
            base_quat_xyzw = _rot6d_to_quat_xyzw(base_rot6d)
            self._send_base_pose(base_pos, base_quat_xyzw)
        self._send_hands(left_closed, right_closed)

    def _make_keypoint(self,
                       name,
                       pos=(0.0, 0.0, 0.0),
                       quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
        from mros.teleop_msgs.msg import KeyPoint

        keypoint = KeyPoint()
        keypoint.name = name
        keypoint.pose.position.x = float(pos[0])
        keypoint.pose.position.y = float(pos[1])
        keypoint.pose.position.z = float(pos[2])
        keypoint.pose.orientation.x = float(quat_xyzw[0])
        keypoint.pose.orientation.y = float(quat_xyzw[1])
        keypoint.pose.orientation.z = float(quat_xyzw[2])
        keypoint.pose.orientation.w = float(quat_xyzw[3])
        return keypoint

    def _integrate_base_action(self, base_pos_action, base_rot6d_action):
        """Integrate body-frame x/y/yaw deltas into an absolute pose."""
        cos_yaw = np.cos(self._accum_base_yaw)
        sin_yaw = np.sin(self._accum_base_yaw)
        delta_x, delta_y = base_pos_action[:2]
        self._accum_base_pos[0] += cos_yaw * delta_x - sin_yaw * delta_y
        self._accum_base_pos[1] += sin_yaw * delta_x + cos_yaw * delta_y
        self._accum_base_pos[2] = base_pos_action[2]

        if not _is_degenerate_rot6d(base_rot6d_action):
            action_quat = _rot6d_to_quat_xyzw(base_rot6d_action)
            action_euler = Rotation.from_quat(action_quat).as_euler('ZYX')
            self._accum_base_yaw = _wrap_to_pi(self._accum_base_yaw +
                                               action_euler[0])
            self._accum_base_rot = Rotation.from_euler('ZYX', [
                self._accum_base_yaw,
                action_euler[1],
                action_euler[2],
            ])

        return self._accum_base_pos.copy(), self._accum_base_rot.as_quat()

    def _send_action_mros(self, action):
        """Publish one WBT joint/base/hand command through MROS."""
        from mros.controller_msgs.msg import JointCmd
        from mros.teleop_msgs.msg import TeleopMsg

        base_pos, base_quat = self._integrate_base_action(
            action[31:34], action[34:40])

        joint_cmd = JointCmd()
        joint_cmd.names = list(STATE_JOINT_NAMES)
        joint_cmd.q = action[:31].astype(np.float32).tolist()
        joint_cmd.v = [0.0] * 31
        joint_cmd.tau = [0.0] * 31
        joint_cmd.kp = [DEFAULT_KP] * 31
        joint_cmd.kd = [DEFAULT_KD] * 31
        joint_cmd.mode = [0] * 31
        joint_cmd.na = 31

        teleop_msg = TeleopMsg()
        teleop_msg.header.frame_id = 'world'
        teleop_msg.world.orientation.w = 1.0
        teleop_msg.joint_cmd = joint_cmd
        teleop_msg.anchors = [
            self._make_keypoint('base_link', base_pos, base_quat)
        ]
        self.teleop_wbt_publisher.publish(teleop_msg)
        self._send_hands(float(action[40]), float(action[41]))

    def _send_joints(self, q):
        """Send 31 joint position targets via ``request_servoj``."""
        q = [float(v) for v in np.asarray(q, dtype=np.float64)]
        n = len(q)
        self._ws_send_request(
            'request_servoj', {
                'q': q,
                'v': [0.0] * n,
                'kp': [DEFAULT_KP] * n,
                'kd': [DEFAULT_KD] * n,
                'tau': [0.0] * n,
                'mode': [0] * n,
                'na': 0,
            })

    def _send_base_pose(self, pos, quat_xyzw):
        """Send the base_link target pose.

        NOTE (hardware integration point): the whole-body base-pose request
        title is robot-SDK specific and is not part of the public LimX
        WebSocket protocol. Adapt ``request_base_pose`` and its payload to
        your robot's controller.
        """
        self._ws_send_request(
            'request_base_pose', {
                'position': [float(pos[0]),
                             float(pos[1]),
                             float(pos[2])],
                'orientation': [
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                    float(quat_xyzw[3]),
                ],
            })

    def _send_hands(self, left_closed, right_closed):
        """Send dexterous-hand open/close command.

        NOTE (hardware integration point): the hand-command request title is
        robot-SDK specific. The 14-dim payload (12 finger + 2 force levels)
        matches the data-collection convention.
        """
        left_val = 100.0 if left_closed >= 0.5 else 0.0
        right_val = 100.0 if right_closed >= 0.5 else 0.0
        finger_cmd = [0.0] * 14
        for i in range(0, 12, 2):
            finger_cmd[i] = left_val
        for i in range(1, 12, 2):
            finger_cmd[i] = right_val
        # Indices 2/3: left/right thumb-aux fingers; always closed for grasp.
        finger_cmd[2] = 100.0
        finger_cmd[3] = 100.0
        force_levels = self.finger_force_levels
        if force_levels is None:
            force_levels = ((2.0, 2.0) if self.control_backend == 'mros' else
                            (3.0, 3.0))
        if len(force_levels) != 2:
            raise ValueError('finger_force_levels must contain 2 values')
        finger_cmd[12] = float(force_levels[0])
        finger_cmd[13] = float(force_levels[1])

        self.last_finger_cmd = np.array(finger_cmd, dtype=np.float32)
        if self.control_backend == 'mros':
            from mros.std_msgs.msg import Float32Array
            finger_msg = Float32Array()
            finger_msg.data = finger_cmd
            self.finger_publisher.publish(finger_msg)
        else:
            self._ws_send_request('request_hand_cmd', {'cmd': finger_cmd})

    def build_observation_specs(self):
        """Oli uses backend-native observation readers, not ROS sync specs."""
        return []

    def send_joints(self, arm_targets):
        """Implement the BaseOperator joint-command interface."""
        if self.control_backend != 'websocket':
            raise NotImplementedError(
                'MROS Oli joint commands require a complete WBT action')
        if isinstance(arm_targets, dict):
            arm_targets = next(iter(arm_targets.values()))
        self._send_joints(arm_targets)

    def send_eepose(self, arm_targets):
        raise NotImplementedError('Oli uses whole-body WBT actions')

    def send_gripper(self, gripper_targets, wait=False):
        del wait
        if isinstance(gripper_targets, dict):
            values = list(gripper_targets.values())
        else:
            values = list(np.asarray(gripper_targets).reshape(-1))
        if len(values) != 2:
            raise ValueError('Oli gripper target must contain left and right')
        self._send_hands(float(values[0]), float(values[1]))

    def gohome(self,
               initial_state=None,
               duration_sec=5.0,
               publish_rate=30.0,
               running_flag_fn=None):
        """Smoothly move the MROS robot to a configured 33-dim state.

        The reset state contains 31 joint targets and two hand-closed flags.
        Base x/y/yaw are held at the operator's current accumulated target;
        they are not part of ``/joint/state`` and therefore are not guessed.
        """
        if initial_state is None:
            raise ValueError('Oli gohome requires a 33-dim initial_state')
        if self.control_backend != 'mros':
            raise NotImplementedError(
                'Oli configured-state reset currently requires MROS')

        target = np.asarray(initial_state, dtype=np.float64)
        if target.shape != (33, ) or not np.all(np.isfinite(target)):
            raise ValueError(
                'Oli initial_state must be a finite 33-dim vector')
        duration_sec = float(duration_sec)
        publish_rate = float(publish_rate)
        if duration_sec <= 0 or publish_rate <= 0:
            raise ValueError(
                'Reset duration and publish rate must be positive')

        joint_msg = self._read_mros_message(
            self.joint_state_subscriber, timeout=1.0)
        if joint_msg is None:
            raise RuntimeError(
                f'No joint state received from {self.joint_state_topic}; '
                'reset was not started')
        start_q = np.asarray(joint_msg.q, dtype=np.float64)
        if start_q.size < 31 or not np.all(np.isfinite(start_q[:31])):
            raise RuntimeError('Invalid joint state; reset was not started')
        start_q = start_q[:31].copy()

        steps = max(1, int(round(duration_sec * publish_rate)))
        period = 1.0 / publish_rate
        next_publish_time = time.monotonic()
        completed = False
        for step in range(1, steps + 1):
            if running_flag_fn is not None and not running_flag_fn():
                break
            ratio = step / steps
            blend = ratio**3 * (10.0 + ratio * (-15.0 + 6.0 * ratio))
            action = np.zeros(42, dtype=np.float64)
            action[:31] = start_q + blend * (target[:31] - start_q)
            action[31:34] = [0.0, 0.0, self._accum_base_pos[2]]
            action[34:40] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            action[40:42] = target[31:33]
            self.send_action(action)

            next_publish_time += period
            time.sleep(max(0.0, next_publish_time - time.monotonic()))
        else:
            completed = True

        if not completed:
            raise RuntimeError('Oli initial-state reset was interrupted')
        print(
            f'[reset] Reached configured Oli initial state in '
            f'{duration_sec:.1f}s ({steps} steps).',
            flush=True)
        return target.copy()

    def close(self):
        """Stop local execution and close the legacy WebSocket if present."""
        self.stop_trajectory()
        if self.ws_client:
            self.ws_client.close()
            self.ws_connected = False
