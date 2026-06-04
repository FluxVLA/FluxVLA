#!/usr/bin/env python3
"""Benchmark Franka observation synchronization strategies.

Run one method at a time to avoid doubling ROS subscription bandwidth:

    python scripts/benchmark_franka_sync.py --method manual
    python scripts/benchmark_franka_sync.py --method message_filters

The benchmark reports synchronized frame rate, timestamp spread within each
synced frame, long frame gaps, approximate skipped frames, and process CPU.
"""

import argparse
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class TopicSpec:
    name: str
    topic: str
    msg_cls: object = None


def stamp_seconds(msg):
    return msg.header.stamp.to_sec()


def percentile(values, pct):
    if not values:
        return float('nan')
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = min(max(idx, 0), len(ordered) - 1)
    return ordered[idx]


class SyncMetrics:

    def __init__(self, target_hz, max_frame_gap, warmup_s):
        self.target_hz = float(target_hz)
        self.max_frame_gap = float(max_frame_gap)
        self.warmup_s = float(warmup_s)
        self.started_at = None
        self.cpu_started_at = None
        self.first_sample_at = None
        self.last_wall = None
        self.last_ref_stamp = None
        self.last_seq_by_name = {}
        self.sync_count = 0
        self.frame_spans = []
        self.wall_intervals = []
        self.stamp_intervals = []
        self.long_gap_count = 0
        self.estimated_skipped_frames = 0
        self.seq_jump_count = 0
        self.seq_jump_frames = 0
        self.manual_slop_failures = 0
        self.manual_dropped_messages = 0

    def start(self):
        self.started_at = time.monotonic()
        self.cpu_started_at = time.process_time()

    def in_warmup(self):
        return time.monotonic() - self.started_at < self.warmup_s

    def record_sync(self, named_msgs):
        if self.in_warmup():
            return

        now = time.monotonic()
        stamps = [stamp_seconds(msg) for msg in named_msgs.values()]
        ref_stamp = min(stamps)
        frame_span = max(stamps) - min(stamps)

        if self.first_sample_at is None:
            self.first_sample_at = now
        if self.last_wall is not None:
            self.wall_intervals.append(now - self.last_wall)
        if self.last_ref_stamp is not None:
            stamp_dt = ref_stamp - self.last_ref_stamp
            self.stamp_intervals.append(stamp_dt)
            if stamp_dt > self.max_frame_gap:
                self.long_gap_count += 1
            expected_steps = int(round(stamp_dt * self.target_hz))
            if expected_steps > 1:
                self.estimated_skipped_frames += expected_steps - 1

        frame_had_seq_jump = False
        for name, msg in named_msgs.items():
            seq = getattr(msg.header, 'seq', None)
            last_seq = self.last_seq_by_name.get(name)
            if seq is not None and last_seq is not None and seq > last_seq + 1:
                self.seq_jump_count += seq - last_seq - 1
                frame_had_seq_jump = True
            if seq is not None:
                self.last_seq_by_name[name] = seq

        if frame_had_seq_jump:
            self.seq_jump_frames += 1

        self.sync_count += 1
        self.frame_spans.append(frame_span)
        self.last_wall = now
        self.last_ref_stamp = ref_stamp

    def record_manual_drop(self, dropped):
        if not self.in_warmup():
            self.manual_dropped_messages += dropped

    def record_manual_slop_failure(self):
        if not self.in_warmup():
            self.manual_slop_failures += 1

    def summary(self):
        now = time.monotonic()
        measured_duration = max(0.0, now - self.started_at - self.warmup_s)
        cpu_duration = max(0.0, time.process_time() - self.cpu_started_at)
        sync_hz = (
            self.sync_count / measured_duration
            if measured_duration > 0.0 else 0.0)
        gap_denominator = max(1, len(self.stamp_intervals))
        long_gap_ratio = self.long_gap_count / gap_denominator
        return {
            'measured_duration_s': measured_duration,
            'sync_count': self.sync_count,
            'sync_hz': sync_hz,
            'target_hz': self.target_hz,
            'target_ok': sync_hz >= self.target_hz * 0.95,
            'cpu_percent': (
                cpu_duration / measured_duration * 100.0
                if measured_duration > 0.0 else float('nan')),
            'span_ms_mean': (
                statistics.mean(self.frame_spans) * 1000.0
                if self.frame_spans else float('nan')),
            'span_ms_p50': percentile(self.frame_spans, 50) * 1000.0,
            'span_ms_p95': percentile(self.frame_spans, 95) * 1000.0,
            'span_ms_max': (
                max(self.frame_spans) * 1000.0
                if self.frame_spans else float('nan')),
            'wall_dt_ms_p95': percentile(self.wall_intervals, 95) * 1000.0,
            'wall_dt_ms_max': (
                max(self.wall_intervals) * 1000.0
                if self.wall_intervals else float('nan')),
            'stamp_dt_ms_p95': percentile(self.stamp_intervals, 95) * 1000.0,
            'stamp_dt_ms_max': (
                max(self.stamp_intervals) * 1000.0
                if self.stamp_intervals else float('nan')),
            'long_gap_count': self.long_gap_count,
            'long_gap_ratio': long_gap_ratio,
            'estimated_skipped_frames': self.estimated_skipped_frames,
            'seq_jump_count': self.seq_jump_count,
            'seq_jump_frames': self.seq_jump_frames,
            'manual_slop_failures': self.manual_slop_failures,
            'manual_dropped_messages': self.manual_dropped_messages,
        }


class PreviewWindow:
    """Display one synchronized image stream without adding subscriptions."""

    def __init__(self, topic_name=None, every=1):
        self.topic_name = topic_name
        self.every = max(1, int(every))
        self.enabled = topic_name is not None
        self.bridge = None
        self.latest_msg = None
        self.has_new_frame = False
        self.lock = threading.Lock()
        self.submitted = 0
        self.displayed = 0
        self.first_display_at = None
        self.warned_missing = False
        self.warned_not_image = False
        self.window_name = (
            f'sync preview: {topic_name}' if topic_name else 'sync preview')

    def submit(self, named_msgs):
        if not self.enabled:
            return
        msg = named_msgs.get(self.topic_name)
        if msg is None:
            if not self.warned_missing:
                print(f'Preview topic name "{self.topic_name}" is not in '
                      'the synchronized frame.')
                self.warned_missing = True
            return
        if getattr(msg, '_type', '') != 'sensor_msgs/Image':
            if not self.warned_not_image:
                print(f'Preview topic name "{self.topic_name}" is '
                      f'{getattr(msg, "_type", type(msg))}, not Image.')
                self.warned_not_image = True
            return

        self.submitted += 1
        if self.submitted % self.every != 0:
            return
        with self.lock:
            self.latest_msg = msg
            self.has_new_frame = True

    def render(self, metrics):
        if not self.enabled:
            return

        with self.lock:
            if not self.has_new_frame:
                return
            msg = self.latest_msg
            self.has_new_frame = False

        try:
            import cv2
            import rospy
            from cv_bridge import CvBridge
        except Exception as exc:
            print(f'Preview disabled: failed to import cv2/cv_bridge: {exc}')
            self.enabled = False
            return

        if self.bridge is None:
            self.bridge = CvBridge()

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            image = self._to_display_image(image)

        now = time.monotonic()
        if self.first_display_at is None:
            self.first_display_at = now
        self.displayed += 1
        preview_hz = (
            self.displayed / max(now - self.first_display_at, 1e-6))
        seq = getattr(msg.header, 'seq', 0)
        stamp = msg.header.stamp.to_sec()
        cv2.putText(
            image,
            (f'{self.topic_name} seq={seq} stamp={stamp:.3f} '
             f'sync={metrics.sync_count} preview_hz={preview_hz:.1f}'),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA)
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            rospy.signal_shutdown('preview window closed')

    @staticmethod
    def _to_display_image(image):
        import cv2
        import numpy as np

        if image.ndim == 2:
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                image = image.astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        return image

    def close(self):
        if not self.enabled:
            return
        try:
            import cv2

            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class ManualDequeBenchmark:

    def __init__(self, specs, slop, queue_limit, poll_hz, metrics,
                 preview=None):
        self.specs = specs
        self.slop = float(slop)
        self.queue_limit = int(queue_limit)
        self.poll_hz = float(poll_hz)
        self.metrics = metrics
        self.preview = preview
        self.lock = threading.Lock()
        self.queues = {spec.name: deque() for spec in specs}
        self.subscribers = []

    def start(self):
        import rospy

        for spec in self.specs:
            callback = self._make_callback(spec.name)
            self.subscribers.append(
                rospy.Subscriber(
                    spec.topic,
                    spec.msg_cls,
                    callback,
                    queue_size=1,
                    tcp_nodelay=True))

    def _make_callback(self, name):

        def _callback(msg):
            queue = self.queues[name]
            with self.lock:
                if len(queue) >= self.queue_limit:
                    queue.popleft()
                queue.append(msg)

        return _callback

    def spin_for(self, duration_s):
        import rospy

        rate = rospy.Rate(self.poll_hz)
        deadline = time.monotonic() + duration_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            named_msgs = self._try_sync()
            if named_msgs is not None:
                self.metrics.record_sync(named_msgs)
                if self.preview is not None:
                    self.preview.submit(named_msgs)
            if self.preview is not None:
                self.preview.render(self.metrics)
            rate.sleep()

    def _try_sync(self):
        with self.lock:
            if any(len(queue) == 0 for queue in self.queues.values()):
                return None

            frame_time = min(
                stamp_seconds(queue[-1]) for queue in self.queues.values())
            frame_time_max = 0.0
            dropped = 0

            for queue in self.queues.values():
                while len(queue) > 0 and stamp_seconds(queue[0]) < frame_time:
                    queue.popleft()
                    dropped += 1
                if len(queue) == 0:
                    self.metrics.record_manual_drop(dropped)
                    return None
                frame_time_max = max(frame_time_max, stamp_seconds(queue[0]))

            if abs(frame_time_max - frame_time) > self.slop:
                self.metrics.record_manual_slop_failure()
                for queue in self.queues.values():
                    while (len(queue) > 0
                           and stamp_seconds(queue[0]) <= frame_time):
                        queue.popleft()
                        dropped += 1
                self.metrics.record_manual_drop(dropped)
                return None

            named_msgs = {
                name: queue.popleft()
                for name, queue in self.queues.items()
            }
            self.metrics.record_manual_drop(dropped)
            return named_msgs


class MessageFiltersBenchmark:

    def __init__(self, specs, slop, ats_queue_size, metrics, preview=None):
        self.specs = specs
        self.slop = float(slop)
        self.ats_queue_size = int(ats_queue_size)
        self.metrics = metrics
        self.preview = preview
        self.subscribers = []
        self.synchronizer = None

    def start(self):
        import message_filters

        self.subscribers = [
            message_filters.Subscriber(spec.topic, spec.msg_cls)
            for spec in self.specs
        ]
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=self.ats_queue_size,
            slop=self.slop,
            allow_headerless=False)
        self.synchronizer.registerCallback(self._callback)

    def _callback(self, *msgs):
        named_msgs = {
            spec.name: msg
            for spec, msg in zip(self.specs, msgs)
        }
        self.metrics.record_sync(named_msgs)
        if self.preview is not None:
            self.preview.submit(named_msgs)

    def spin_for(self, duration_s):
        import rospy

        deadline = time.monotonic() + duration_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.preview is not None:
                self.preview.render(self.metrics)
            rospy.sleep(0.05)


def resolve_topic_types(specs):
    import rostopic

    resolved = []
    for spec in specs:
        msg_cls, real_topic, _ = rostopic.get_topic_class(
            spec.topic, blocking=True)
        if msg_cls is None:
            raise RuntimeError(f'Unable to resolve message type: {spec.topic}')
        resolved.append(TopicSpec(spec.name, real_topic or spec.topic, msg_cls))
    return resolved


def build_topic_specs(args):
    specs = [
        TopicSpec('img_front', args.img_front_topic),
        TopicSpec('img_left', args.img_left_topic),
        TopicSpec('img_right', args.img_right_topic),
        TopicSpec('left_joint', args.left_joint_topic),
        TopicSpec('right_joint', args.right_joint_topic),
    ]

    if args.include_gripper:
        specs.extend([
            TopicSpec('left_gripper', args.left_gripper_topic),
            TopicSpec('right_gripper', args.right_gripper_topic),
        ])

    if args.command_mode == 'cartesian':
        if args.left_pose_topic and args.right_pose_topic:
            specs.extend([
                TopicSpec('left_pose', args.left_pose_topic),
                TopicSpec('right_pose', args.right_pose_topic),
            ])
        elif args.left_franka_state_topic and args.right_franka_state_topic:
            specs.extend([
                TopicSpec('left_franka_state', args.left_franka_state_topic),
                TopicSpec('right_franka_state', args.right_franka_state_topic),
            ])
        else:
            raise ValueError(
                'cartesian mode requires either pose topics or franka_state '
                'topics for both arms')

    if args.include_depth:
        specs.extend([
            TopicSpec('img_front_depth', args.img_front_depth_topic),
            TopicSpec('img_left_depth', args.img_left_depth_topic),
            TopicSpec('img_right_depth', args.img_right_depth_topic),
        ])

    for extra in args.extra_topic:
        name, sep, topic = extra.partition('=')
        if not sep or not name or not topic:
            raise ValueError(
                f'Invalid --extra-topic "{extra}", expected name=/topic')
        specs.append(TopicSpec(name, topic))

    return specs


def print_summary(method, specs, summary):
    print('\n=== Sync benchmark summary ===')
    print(f'method: {method}')
    print('topics:')
    for spec in specs:
        print(f'  {spec.name}: {spec.topic} ({spec.msg_cls._type})')
    print(f"duration_s: {summary['measured_duration_s']:.2f}")
    print(f"sync_count: {summary['sync_count']}")
    print(
        f"sync_hz: {summary['sync_hz']:.2f} "
        f"(target {summary['target_hz']:.2f}, "
        f"ok={summary['target_ok']})")
    print(f"cpu_percent: {summary['cpu_percent']:.1f}")
    print(
        'frame_span_ms: '
        f"mean={summary['span_ms_mean']:.2f}, "
        f"p50={summary['span_ms_p50']:.2f}, "
        f"p95={summary['span_ms_p95']:.2f}, "
        f"max={summary['span_ms_max']:.2f}")
    print(
        'wall_interval_ms: '
        f"p95={summary['wall_dt_ms_p95']:.2f}, "
        f"max={summary['wall_dt_ms_max']:.2f}")
    print(
        'stamp_interval_ms: '
        f"p95={summary['stamp_dt_ms_p95']:.2f}, "
        f"max={summary['stamp_dt_ms_max']:.2f}")
    print(
        f"long_gaps: {summary['long_gap_count']} "
        f"({summary['long_gap_ratio'] * 100.0:.2f}%)")
    print(
        f"estimated_skipped_frames: "
        f"{summary['estimated_skipped_frames']}")
    print(
        f"header_seq_jumps: {summary['seq_jump_count']} "
        f"in {summary['seq_jump_frames']} synced frames")
    if method == 'manual':
        print(f"manual_slop_failures: {summary['manual_slop_failures']}")
        print(
            f"manual_dropped_messages: "
            f"{summary['manual_dropped_messages']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark manual deque sync vs ROS message_filters sync.')
    parser.add_argument(
        '--method',
        choices=('manual', 'message_filters'),
        required=True,
        help='Run exactly one synchronization method per process.')
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--warmup', type=float, default=3.0)
    parser.add_argument('--target-hz', type=float, default=30.0)
    parser.add_argument(
        '--max-frame-gap',
        type=float,
        default=None,
        help='Long-gap threshold in seconds. Defaults to 2 / target_hz.')
    parser.add_argument('--slop', type=float, default=0.02)
    parser.add_argument('--manual-poll-hz', type=float, default=60.0)
    parser.add_argument('--manual-queue-limit', type=int, default=2000)
    parser.add_argument('--ats-queue-size', type=int, default=30)
    parser.add_argument(
        '--command-mode', choices=('joint', 'cartesian'), default='joint')
    parser.add_argument('--include-depth', action='store_true')
    parser.add_argument(
        '--include-gripper',
        action='store_true',
        help='Include gripper joint_states in the strict sync set.')
    parser.add_argument(
        '--preview-name',
        default=None,
        help='Display this synchronized Image by topic name, e.g. img_front.')
    parser.add_argument(
        '--preview-every',
        type=int,
        default=1,
        help='Display every Nth synchronized preview frame.')

    parser.add_argument(
        '--img-front-topic', default='/camera_front/color/image_raw')
    parser.add_argument(
        '--img-left-topic', default='/camera_left_wrist/color/image_raw')
    parser.add_argument(
        '--img-right-topic', default='/camera_right_wrist/color/image_raw')
    parser.add_argument(
        '--img-front-depth-topic', default='/camera_front/depth/image_raw')
    parser.add_argument(
        '--img-left-depth-topic',
        default='/camera_left_wrist/depth/image_raw')
    parser.add_argument(
        '--img-right-depth-topic',
        default='/camera_right_wrist/depth/image_raw')
    parser.add_argument(
        '--left-joint-topic', default='/left_arm/joint_states')
    parser.add_argument(
        '--right-joint-topic', default='/right_arm/joint_states')
    parser.add_argument(
        '--left-gripper-topic',
        default='/left_arm/franka_gripper/joint_states')
    parser.add_argument(
        '--right-gripper-topic',
        default='/right_arm/franka_gripper/joint_states')
    parser.add_argument('--left-pose-topic', default=None)
    parser.add_argument('--right-pose-topic', default=None)
    parser.add_argument(
        '--left-franka-state-topic',
        default='/left_arm/franka_state_controller/franka_states')
    parser.add_argument(
        '--right-franka-state-topic',
        default='/right_arm/franka_state_controller/franka_states')
    parser.add_argument(
        '--extra-topic',
        action='append',
        default=[],
        help='Additional stamped topic as name=/topic. Can be repeated.')
    return parser.parse_args()


def main():
    args = parse_args()

    import rospy

    rospy.init_node(
        f'franka_sync_benchmark_{args.method}', anonymous=True)

    specs = resolve_topic_types(build_topic_specs(args))
    if args.preview_name is not None:
        names = {spec.name for spec in specs}
        if args.preview_name not in names:
            raise ValueError(
                f'--preview-name must be one of {sorted(names)}, '
                f'got {args.preview_name}')
    preview = PreviewWindow(args.preview_name, args.preview_every)
    max_frame_gap = (
        args.max_frame_gap
        if args.max_frame_gap is not None else 2.0 / args.target_hz)
    metrics = SyncMetrics(
        target_hz=args.target_hz,
        max_frame_gap=max_frame_gap,
        warmup_s=args.warmup)

    if args.method == 'manual':
        benchmark = ManualDequeBenchmark(
            specs=specs,
            slop=args.slop,
            queue_limit=args.manual_queue_limit,
            poll_hz=args.manual_poll_hz,
            metrics=metrics,
            preview=preview)
    else:
        benchmark = MessageFiltersBenchmark(
            specs=specs,
            slop=args.slop,
            ats_queue_size=args.ats_queue_size,
            metrics=metrics,
            preview=preview)

    print(f'Starting {args.method} benchmark...')
    print(
        f'duration={args.duration}s, warmup={args.warmup}s, '
        f'slop={args.slop}s, target_hz={args.target_hz}Hz')
    benchmark.start()
    metrics.start()
    try:
        benchmark.spin_for(args.duration + args.warmup)
    finally:
        preview.close()
    print_summary(args.method, specs, metrics.summary())


if __name__ == '__main__':
    main()
