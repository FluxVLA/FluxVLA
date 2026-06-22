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

import numpy as np

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


@OPERATORS.register_module()
class OliOperator:
    """Oli whole-body (loco-manipulation) operator.

    Sensor input is read over ROS (rospy); whole-body control commands are
    sent to the robot over the LimX WebSocket JSON protocol, mirroring the
    ``Tron2Operator`` transport split. Importing this module requires no
    middleware: ``rospy`` and ``websocket`` are imported lazily and only
    when the operator is instantiated.

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
                 ws_accid=None):
        """Initialize OliOperator.

        Args:
            head_rgb_topic (str): ROS topic for head compressed RGB image.
            joint_state_topic (str): ROS topic for joint state feedback.
            robot_ip (str): Robot IP for the WebSocket control channel.
            ws_port (int): WebSocket port. Defaults to 5000.
            ws_accid (str): WebSocket account id; None means auto-detect.
        """
        self.head_rgb_topic = head_rgb_topic
        self.joint_state_topic = joint_state_topic

        self.robot_ip = robot_ip
        self.ws_port = ws_port
        self.ws_accid = ws_accid
        self.ws_client = None
        self.ws_connected = False
        self.ws_lock = threading.Lock()
        self.json_encoder = NumpySafeEncoder

        self.last_finger_cmd = np.zeros(14, dtype=np.float32)

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

    def get_frame(self):
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
        """Send a 42-dim whole-body action to the robot.

        Args:
            action (np.ndarray): 42-dim action vector (see class docstring).
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (42, ):
            raise ValueError(
                f'OliOperator expects a (42,) action, got {action.shape}')
        if not np.all(np.isfinite(action)):
            raise ValueError('OliOperator received a non-finite action')

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
        # Force level (last two dims).
        finger_cmd[12] = 3.0
        finger_cmd[13] = 3.0

        self.last_finger_cmd = np.array(finger_cmd, dtype=np.float32)
        self._ws_send_request('request_hand_cmd', {'cmd': finger_cmd})

    def close(self):
        """Close the WebSocket connection."""
        if self.ws_client:
            self.ws_client.close()
            self.ws_connected = False
