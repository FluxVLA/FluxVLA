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


def replace_last_segment(input_string, new_segment='camera_info'):
    last_slash_index = input_string.rfind('/')
    if last_slash_index != -1:
        return input_string[:last_slash_index + 1] + new_segment
    return new_segment


class BaseOperator:
    """Common operator base for timestamp sync and trajectory execution."""

    def __init__(self,
                 sync_slop=0.04,
                 sync_queue_size=30,
                 synced_frame_queue_size=10,
                 sync_warning_enabled=True,
                 sync_warning_target_hz=30.0,
                 sync_warning_window=2.0,
                 sync_warning_min_hz_ratio=0.9,
                 sync_warning_warmup=3.0):
        self.sync_slop = float(sync_slop)
        self.sync_queue_size = int(sync_queue_size)
        self.synced_frame_queue_size = int(synced_frame_queue_size)
        self.sync_warning_enabled = bool(sync_warning_enabled)
        self.sync_warning_target_hz = float(sync_warning_target_hz)
        self.sync_warning_window = float(sync_warning_window)
        self.sync_warning_min_hz_ratio = float(sync_warning_min_hz_ratio)
        self.sync_warning_warmup = float(sync_warning_warmup)

        self._init_base_runtime()

    def _init_base_runtime(self):
        from cv_bridge import CvBridge

        self.bridge = CvBridge()
        self.cam_info_dict = {}
        self._lock = threading.Lock()
        self._frames = deque(maxlen=self.synced_frame_queue_size)
        self._sync_names = []
        self._sync_image_names = set()
        self._sync_topic_by_name = {}
        self._sync_input_counts = {}
        self._sync_last_arrival_at = {}
        self._sync_subscribers = []
        self._sync = None
        self._sync_warning_started_at = time.monotonic()
        self._sync_window_started_at = time.monotonic()
        self._sync_window_count = 0
        self._last_sync_output_at = None
        self._last_empty_frame_warning_at = 0.0
        self._traj_thread = None
        self._traj_stop_event = threading.Event()

    def build_observation_specs(self):
        """Return topic specs to synchronize.

        Each spec should contain name, topic, and msg_type.
        A dict is preferred:
        {'name': 'img_front', 'topic': '/camera/...', 'msg_type': Image}
        """
        raise NotImplementedError

    def setup_observation_sync(self, specs):
        import message_filters
        import rospy

        normalized_specs = [self._normalize_sync_spec(spec) for spec in specs]
        self._sync_names = [spec['name'] for spec in normalized_specs]
        self._sync_topic_by_name = {
            spec['name']: spec['topic']
            for spec in normalized_specs
        }
        self._sync_input_counts = {
            spec['name']: 0
            for spec in normalized_specs
        }
        self._sync_last_arrival_at = {
            spec['name']: None
            for spec in normalized_specs
        }
        self._sync_image_names = {
            spec['name']
            for spec in normalized_specs
            if self._is_image_msg_type(spec['msg_type'])
        }
        self._sync_subscribers = [
            message_filters.Subscriber(spec['topic'], spec['msg_type'])
            for spec in normalized_specs
        ]
        for subscriber, spec in zip(self._sync_subscribers, normalized_specs):
            subscriber.registerCallback(self._record_sync_input, spec['name'])
        self._sync = message_filters.ApproximateTimeSynchronizer(
            self._sync_subscribers,
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            allow_headerless=False)
        self._sync.registerCallback(self._sync_callback)
        rospy.loginfo('%s observation sync: %d topics, slop=%.3fs',
                      self.__class__.__name__, len(normalized_specs),
                      self.sync_slop)
        return [
            replace_last_segment(spec['topic']) for spec in normalized_specs
            if spec['name'] in self._sync_image_names
        ]

    @staticmethod
    def _normalize_sync_spec(spec):
        if isinstance(spec, dict):
            return spec
        name, topic, msg_type = spec
        return {'name': name, 'topic': topic, 'msg_type': msg_type}

    @staticmethod
    def _is_image_msg_type(msg_type):
        return getattr(msg_type, '__name__', '') == 'Image'

    def get_frame(self, slop=0.7):  # noqa: ARG002
        """Return the newest synchronized frame and discard stale frames."""
        with self._lock:
            if not self._frames:
                self._maybe_log_empty_frame_warning()
                return False
            frame = self._frames[-1]
            self._frames.clear()
        return self._format_frame(frame)

    def clear_observation_queues(self):
        with self._lock:
            self._frames.clear()
            self._sync_warning_started_at = time.monotonic()
            self._sync_window_started_at = time.monotonic()
            self._sync_window_count = 0
            self._last_sync_output_at = None
            self._last_empty_frame_warning_at = 0.0

    def get_queue_status(self):
        return {
            'synced_frames': len(self._frames),
            'sync_inputs': dict(self._sync_input_counts),
            'sync_last_arrival_at': dict(self._sync_last_arrival_at),
        }

    def _record_sync_input(self, msg, name):
        del msg
        with self._lock:
            if name in self._sync_input_counts:
                self._sync_input_counts[name] += 1
                self._sync_last_arrival_at[name] = time.monotonic()

    def _sync_callback(self, *msgs):
        raw = dict(zip(self._sync_names, msgs))
        frame = dict(raw)
        frame['stamps'] = {
            name: msg.header.stamp.to_sec()
            for name, msg in raw.items() if hasattr(msg, 'header')
        }
        with self._lock:
            self._frames.append(frame)
            self._last_sync_output_at = time.monotonic()
            self._record_sync_output()

    def _format_frame(self, frame):
        formatted = {}
        for name, value in frame.items():
            if name == 'stamps':
                formatted[name] = value
            elif name in self._sync_image_names:
                formatted[name] = self._to_optional_cv_image(value)
            else:
                formatted[name] = value
        return formatted

    def _to_cv_image(self, msg):
        return self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def _to_optional_cv_image(self, msg):
        if msg is None:
            return None
        return self._to_cv_image(msg)

    def _record_sync_output(self):
        if (not self.sync_warning_enabled or self.sync_warning_target_hz <= 0.0
                or self.sync_warning_window <= 0.0):
            return

        now = time.monotonic()
        if now - self._sync_warning_started_at < self.sync_warning_warmup:
            self._sync_window_started_at = now
            self._sync_window_count = 0
            return

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

    def _maybe_log_empty_frame_warning(self):
        if (not self.sync_warning_enabled or self.sync_warning_window <= 0.0
                or not self._sync_names):
            return

        now = time.monotonic()
        if now - self._sync_warning_started_at < self.sync_warning_warmup:
            return
        if now - self._last_empty_frame_warning_at < self.sync_warning_window:
            return

        self._last_empty_frame_warning_at = now
        self._log_empty_frame_warning(now)

    def _log_empty_frame_warning(self, now):
        import rospy

        if self._last_sync_output_at is None:
            no_sync_for = now - self._sync_warning_started_at
        else:
            no_sync_for = now - self._last_sync_output_at

        missing_names = [
            name for name, count in self._sync_input_counts.items()
            if count == 0
        ]
        missing_topics = [
            self._sync_topic_by_name.get(name, name) for name in missing_names
        ]
        stale_topics = [
            self._sync_topic_by_name.get(name, name)
            for name, last_arrival_at in self._sync_last_arrival_at.items()
            if (last_arrival_at is not None and now -
                last_arrival_at > self.sync_warning_window)
        ]

        rospy.logwarn(
            '%s has no synchronized frame for %.1fs '
            '(slop=%.3fs, queue_size=%d). Missing topics: %s. '
            'Stale topics: %s. If none are missing or stale, check topic '
            'header timestamps.',
            self.__class__.__name__,
            no_sync_for,
            self.sync_slop,
            self.sync_queue_size,
            ', '.join(missing_topics) if missing_topics else 'none',
            ', '.join(stale_topics) if stale_topics else 'none',
        )

    def _log_sync_rate_warning(self, observed_hz, min_hz, elapsed):
        import rospy

        rospy.logwarn(
            '%s sync rate low: observed=%.2fHz, minimum=%.2fHz over %.1fs',
            self.__class__.__name__, observed_hz, min_hz, elapsed)

    def load_camera_info(self, camera_info_topics):
        import rospy
        from sensor_msgs.msg import CameraInfo

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

    def execute_trajectory(self,
                           arm_trajectories,
                           gripper_trajectories=None,
                           head_trajectory=None,
                           dt: float = 0.1,
                           async_exec: bool = False):
        """Execute arm/gripper trajectories in sync or async mode."""
        if dt <= 0.0:
            raise ValueError('dt must be positive')

        gripper_trajectories = gripper_trajectories or {}
        n = self._validate_trajectory_lengths(arm_trajectories,
                                              gripper_trajectories)

        self._traj_stop_event.set()
        self._traj_stop_event = threading.Event()
        stop_event = self._traj_stop_event
        args = (arm_trajectories, gripper_trajectories, head_trajectory, n, dt,
                stop_event)

        if async_exec:
            self._traj_thread = threading.Thread(
                target=self._run_trajectory, args=args, daemon=True)
            self._traj_thread.start()
        else:
            self._run_trajectory(*args)

    @staticmethod
    def _validate_trajectory_lengths(arm_trajectories, gripper_trajectories):
        if not arm_trajectories:
            raise ValueError('arm_trajectories must not be empty')

        lengths = {
            f'arm:{name}': len(trajectory)
            for name, trajectory in arm_trajectories.items()
        }
        lengths.update({
            f'gripper:{name}': len(trajectory)
            for name, trajectory in gripper_trajectories.items()
            if trajectory is not None
        })
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(f'Trajectory length mismatch: {lengths}')
        return unique_lengths.pop()

    def _run_trajectory(self, arm_trajectories, gripper_trajectories,
                        head_trajectory, num_steps, dt, stop_event):
        import rospy

        del head_trajectory
        rate = rospy.Rate(1.0 / dt)
        for idx in range(num_steps):
            if rospy.is_shutdown() or stop_event.is_set():
                return

            arm_targets = {
                name: trajectory[idx]
                for name, trajectory in arm_trajectories.items()
            }
            if self.command_mode == 'cartesian':
                self.send_eepose(arm_targets)
            else:
                self.send_joints(arm_targets)

            gripper_targets = {
                name: float(trajectory[idx])
                for name, trajectory in gripper_trajectories.items()
                if trajectory is not None
            }
            if gripper_targets:
                self.send_gripper(gripper_targets)

            rate.sleep()

    def stop_trajectory(self):
        self._traj_stop_event.set()

    def is_trajectory_running(self):
        return (self._traj_thread is not None and self._traj_thread.is_alive())

    def send_joints(self, arm_targets):
        raise NotImplementedError

    def send_eepose(self, arm_targets):
        raise NotImplementedError

    def send_gripper(self, gripper_targets, wait=False):
        raise NotImplementedError

    def gohome(self):
        raise NotImplementedError
