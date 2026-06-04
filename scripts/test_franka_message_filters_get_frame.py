#!/usr/bin/env python3
"""Continuously test FrankaDualOperator.get_frame().

The script only reads synchronized observations. It warns when get_frame stalls,
when synchronized frame timestamps jump too far, or when arm JointState seq
numbers skip.
"""

import argparse
import time

from fluxvla.engines.operators.franka_dual_operator import FrankaDualOperator

LEFT_ARM_IDX = 6
RIGHT_ARM_IDX = 7
LEFT_GRIPPER_IDX = 10
RIGHT_GRIPPER_IDX = 11


def frame_stamp(frame):
    return min(
        frame[LEFT_ARM_IDX].header.stamp.to_sec(),
        frame[RIGHT_ARM_IDX].header.stamp.to_sec(),
    )


def stamp_span_ms(frame):
    left_stamp = frame[LEFT_ARM_IDX].header.stamp.to_sec()
    right_stamp = frame[RIGHT_ARM_IDX].header.stamp.to_sec()
    return abs(right_stamp - left_stamp) * 1000.0


def warn(rospy, message):
    rospy.logwarn(message)


def check_seq_jump(frame, last_seq, rospy):
    for key, index in (('left_arm', LEFT_ARM_IDX),
                       ('right_arm', RIGHT_ARM_IDX)):
        seq = getattr(frame[index].header, 'seq', None)
        if seq is None:
            continue
        previous = last_seq.get(key)
        if previous is not None and seq > previous + 1:
            warn(
                rospy,
                f'{key} header.seq jumped from {previous} to {seq} '
                f'(missed {seq - previous - 1})',
            )
        last_seq[key] = seq


def build_operator(args):
    return FrankaDualOperator(
        img_front_topic=args.img_front_topic,
        img_left_topic=args.img_left_topic,
        img_right_topic=args.img_right_topic,
        puppet_arm_left_topic=args.left_joint_topic,
        puppet_arm_right_topic=args.right_joint_topic,
        puppet_ee_pose_left_topic=args.left_pose_topic,
        puppet_ee_pose_right_topic=args.right_pose_topic,
        puppet_franka_state_left_topic=args.left_franka_state_topic,
        puppet_franka_state_right_topic=args.right_franka_state_topic,
        use_depth_image=args.include_depth,
        img_front_depth_topic=args.img_front_depth_topic,
        img_left_depth_topic=args.img_left_depth_topic,
        img_right_depth_topic=args.img_right_depth_topic,
        sync_slop=args.slop,
        sync_queue_size=args.sync_queue_size,
        synced_frame_queue_size=args.frame_queue_size,
        base_frame_id=args.base_frame_id,
    )


def run(args):
    import rospy

    operator = build_operator(args)
    poll_dt = 1.0 / args.poll_hz
    expected_dt = 1.0 / args.target_hz
    max_frame_gap = (
        args.max_frame_gap
        if args.max_frame_gap is not None else args.gap_factor * expected_dt)
    no_frame_warn_interval = args.no_frame_warn_interval

    started_at = time.monotonic()
    last_frame_wall = started_at
    last_warn_wall = 0.0
    last_stamp = None
    last_seq = {}
    frames = 0
    deadline = (
        started_at + args.duration if args.duration > 0.0 else float('inf'))

    rospy.loginfo(
        'Testing get_frame: target=%.2fHz, poll=%.2fHz, slop=%.3fs, '
        'max_frame_gap=%.3fs',
        args.target_hz,
        args.poll_hz,
        args.slop,
        max_frame_gap,
    )

    while not rospy.is_shutdown() and time.monotonic() < deadline:
        frame = operator.get_frame()
        now = time.monotonic()

        if not frame:
            if (now - last_frame_wall > no_frame_warn_interval
                    and now - last_warn_wall > no_frame_warn_interval):
                status = operator.get_queue_status()
                warn(
                    rospy,
                    f'no synchronized frame for '
                    f'{now - last_frame_wall:.2f}s; status={status}',
                )
                last_warn_wall = now
            time.sleep(poll_dt)
            continue

        frames += 1
        wall_gap = now - last_frame_wall
        if wall_gap > max_frame_gap:
            warn(
                rospy,
                f'wall-clock get_frame gap {wall_gap * 1000:.1f}ms '
                f'exceeded {max_frame_gap * 1000:.1f}ms',
            )
        last_frame_wall = now
        current_stamp = frame_stamp(frame)
        if last_stamp is not None:
            stamp_gap = current_stamp - last_stamp
            if stamp_gap > max_frame_gap:
                skipped = max(0, round(stamp_gap / expected_dt) - 1)
                warn(
                    rospy,
                    f'synchronized frame timestamp gap {stamp_gap * 1000:.1f}ms '
                    f'(estimated skipped frames: {skipped})',
                )
        last_stamp = current_stamp

        if args.warn_seq_jumps:
            check_seq_jump(frame, last_seq, rospy)

        if args.log_every > 0 and frames % args.log_every == 0:
            elapsed = max(now - started_at, 1e-6)
            rospy.loginfo(
                'frames=%d, hz=%.2f, span=%.2fms, '
                'left_gripper=%s, right_gripper=%s',
                frames,
                frames / elapsed,
                stamp_span_ms(frame),
                frame[LEFT_GRIPPER_IDX],
                frame[RIGHT_GRIPPER_IDX],
            )

        time.sleep(poll_dt)

    elapsed = max(time.monotonic() - started_at, 1e-6)
    rospy.loginfo('Finished: frames=%d, get_frame_hz=%.2f', frames,
                  frames / elapsed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Continuously call Franka message_filters get_frame().')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Seconds to run. 0 means run until Ctrl-C.')
    parser.add_argument('--target-hz', type=float, default=30.0)
    parser.add_argument('--poll-hz', type=float, default=120.0)
    parser.add_argument('--slop', type=float, default=0.04)
    parser.add_argument('--sync-queue-size', type=int, default=30)
    parser.add_argument('--frame-queue-size', type=int, default=10)
    parser.add_argument('--gap-factor', type=float, default=2.0)
    parser.add_argument('--max-frame-gap', type=float, default=None)
    parser.add_argument('--no-frame-warn-interval', type=float, default=1.0)
    parser.add_argument('--log-every', type=int, default=30)
    parser.add_argument(
        '--warn-seq-jumps',
        action='store_true',
        help='Warn when JointState header.seq skips. Off by default because '
        'high-rate joint topics naturally skip seq when sampled at 30 Hz.')
    parser.add_argument('--include-depth', action='store_true')
    parser.add_argument('--base-frame-id', default='')

    parser.add_argument('--img-front-topic',
                        default='/camera_front/color/image_raw')
    parser.add_argument('--img-left-topic',
                        default='/camera_left_wrist/color/image_raw')
    parser.add_argument('--img-right-topic',
                        default='/camera_right_wrist/color/image_raw')
    parser.add_argument('--img-front-depth-topic',
                        default='/camera_front/depth/image_raw')
    parser.add_argument('--img-left-depth-topic',
                        default='/camera_left_wrist/depth/image_raw')
    parser.add_argument('--img-right-depth-topic',
                        default='/camera_right_wrist/depth/image_raw')
    parser.add_argument('--left-joint-topic', default='/left_arm/joint_states')
    parser.add_argument('--right-joint-topic',
                        default='/right_arm/joint_states')
    parser.add_argument('--left-pose-topic', default=None)
    parser.add_argument('--right-pose-topic', default=None)
    parser.add_argument(
        '--left-franka-state-topic',
        default='/left_arm/franka_state_controller/franka_states',
    )
    parser.add_argument(
        '--right-franka-state-topic',
        default='/right_arm/franka_state_controller/franka_states',
    )
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
