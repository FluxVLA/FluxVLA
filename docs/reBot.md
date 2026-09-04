# reBot Bimanual Real-Robot Deployment Guide

This guide defines the FluxVLA deployment contract for two reBot 102 leader
arms, two B601-DM follower arms, one front RealSense D455, and two wrist
RealSense D405 cameras. It covers the LeRobot v3 data contract and PI0.5
real-robot inference over ROS1.

______________________________________________________________________

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Workspace Setup](#2-workspace-setup)
3. [Hardware Mapping](#3-hardware-mapping)
4. [LeRobot Data Contract](#4-lerobot-data-contract)
5. [Data Collection](#5-data-collection)
6. [Validation and Upload](#6-validation-and-upload)
7. [FluxVLA Inference](#7-fluxvla-inference)
8. [Safety](#8-safety)

______________________________________________________________________

## 1. System Requirements

- Ubuntu 20.04
- ROS1 Noetic
- Conda environment: `~/miniconda3/envs/lerobot_bimanual`
- Two reBot 102 leader arms
- Two reBot B601-DM follower arms
- Intel RealSense D455 front camera and two D405 wrist cameras
- Stable `/dev/rebot_*` links for all four arm serial devices

Unlike the Franka workflow, reBot collection writes LeRobot v3 directly. ROS
Parquet collection and a separate conversion pass are not required.

## 2. Workspace Setup

```bash
cd ~/reBot/rebot_ros1_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

Each new terminal must source ROS and the workspace:

```bash
source /opt/ros/noetic/setup.bash
source ~/reBot/rebot_ros1_ws/devel/setup.bash
```

## 3. Hardware Mapping

Arm serial devices:

```text
/dev/rebot_follower_left
/dev/rebot_follower_right
/dev/rebot_leader_left
/dev/rebot_leader_right
```

Camera mapping:

```text
cam_front        D455  <FRONT_CAMERA_SERIAL>
cam_wrist_left   D405  <LEFT_WRIST_CAMERA_SERIAL>
cam_wrist_right  D405  <RIGHT_WRIST_CAMERA_SERIAL>
```

Validate all devices before collection:

```bash
rosrun rebot_bimanual_ros rebot-check-devices --active
```

## 4. LeRobot Data Contract

Each frame contains:

```text
observation.state                    float32[14]
action                               float32[14]
observation.images.cam_front         video[480,640,3]
observation.images.cam_wrist_left    video[480,640,3]
observation.images.cam_wrist_right   video[480,640,3]
timestamp                            float32[1]
frame_index                          int64[1]
episode_index                        int64[1]
index                                int64[1]
task_index                           int64[1]
```

State, action, timestamps, and indices are stored in `data/**/*.parquet`.
Images are AV1 MP4 streams under `videos/`, and task text is indexed through
`meta/tasks.parquet`.

State and action values follow the ROS `JointState` convention and are recorded
in radians in this order:

```text
left_shoulder_pan.pos
left_shoulder_lift.pos
left_elbow_flex.pos
left_wrist_flex.pos
left_wrist_yaw.pos
left_wrist_roll.pos
left_gripper.pos
right_shoulder_pan.pos
right_shoulder_lift.pos
right_elbow_flex.pos
right_wrist_flex.pos
right_wrist_yaw.pos
right_wrist_roll.pos
right_gripper.pos
```

This 14-D contract is embodiment-specific. It intentionally does not pad reBot
to Franka's 16-D dual-arm state.

## 5. Data Collection

```bash
roslaunch rebot_bimanual_ros dual_arm.launch \
  enable_leaders:=true \
  enable_teleop:=true \
  enable_cameras:=true \
  enable_camera_preview:=true \
  launch_data_collection:=true \
  dataset_repo_id:=rebot_dual/20260811 \
  task_description:="Pick up the object and place it in the box" \
  data_frame_rate:=30 \
  num_episodes:=0 \
  auto_record_on_leader_motion:=true \
  motion_start_threshold_rad:=0.20 \
  home_stop_threshold_rad:=0.15 \
  home_dwell_sec:=0.8 \
  home_capture_sec:=1.0
```

The date can be used as the dataset name. A missing directory is created, while
an existing directory is resumed automatically after FPS/schema validation.
The default local root for the command above is:

```text
~/reBot/Data/lerobot/rebot_dual/20260811
```

The driver publishes follower states and leader states as `JointState`; ROS
relays turn the leader streams into follower command topics. The recorder uses
approximate timestamp synchronization for the two follower states and three
images, then associates the latest command before each synchronized frame.

At startup, hold both leaders at the desired reset pose until the recorder logs
`Captured the dual-leader reset pose`. Recording starts when any of the twelve
arm joints moves at least `0.20 rad` from reset. It saves when all twelve joints
remain within `0.15 rad` for `0.8 s`, then rearms for the next episode. Gripper
motion is excluded from boundary detection. Automatic arming is blocked until
the camera and robot topics have produced a synchronized frame.

To use the same manual topic contract as the Franka collector, set
`auto_record_on_leader_motion:=false` and publish:

```bash
rostopic pub -1 /data_collection/record_cmd std_msgs/String "data: 'start'"
rostopic pub -1 /data_collection/record_cmd std_msgs/String "data: 'stop'"
rostopic pub -1 /data_collection/record_cmd std_msgs/String "data: 'cancel'"
```

`stop` saves and `cancel` discards the active episode. Neither command stops
the driver or disables torque. Stopping the full launch disconnects the two
followers and disables their torque.

The three RealSense pipelines are opened sequentially by one Conda
`pyrealsense2` process and retain the canonical ROS image topics. This avoids
the USB-claim race seen with three concurrent ROS librealsense 2.50 nodelets.

Datasets created by older wrappers may use `main`, `left_wrist`, and
`right_wrist` camera keys. Do not resume those datasets with this canonical
schema; start a new dataset ID instead.

## 6. Validation and Upload

Validate the local dataset before training or upload:

```bash
rosrun rebot_bimanual_ros rebot-dataset validate \
  --repo-id rebot_dual/20260811
```

Upload the validated dataset to a remote server over SSH. Replace the port,
user, host, and destination directory for the deployment environment:

```bash
rosrun rebot_bimanual_ros rebot-dataset upload \
  --repo-id rebot_dual/20260811 \
  --destination <REMOTE_USER>@<REMOTE_IP>:/path/to/RealRobot_rebot_dual_lerobotv3/ \
  --ssh-port <SSH_PORT>
```

The command always validates the schema first, then runs `rsync` with archive,
compression, progress, and partial-file preservation enabled. It copies the
dataset directory itself into the remote base directory. Add `--dry-run` to
check the source and destination without writing remote files. `rsync` must be
installed on both the local and remote hosts; use `scp -r -P <SSH_PORT>` if the
remote server cannot install it.

## 7. FluxVLA Inference

FluxVLA reads this output through `ParquetDatasetV3`, which supports LeRobot
v3 metadata (`tasks.parquet`, `meta/episodes/*.parquet`, and `stats.json`). A
reBot PI0.5 training recipe must preserve this contract:

```text
dataset type: ParquetDatasetV3
state key: observation.state
action key: action
images: cam_front, cam_wrist_left, cam_wrist_right
state/action dimension: 14 (padded internally to the model maximum)
```

The standalone inference config is
`configs/pi05/pi05_rebot_dual_inference.py`. It does not inherit another
robot config. Before launch, provide:

```text
checkpoints/pi05_base/                    tokenizer assets
<run-directory>/
  checkpoints/<checkpoint>.safetensors   fine-tuned model
  dataset_statistics.json                transformed statistics
```

The statistics file follows the standard FluxVLA layout: `private.proprio`
contains state statistics and `private.action` contains transformed action
statistics. Joint actions are state-relative in dimensions 0-5 and 7-12;
grippers in dimensions 6 and 13 remain absolute.

Start the reBot ROS drivers and cameras, verify the configured topics, then
run:

```bash
python scripts/inference_real_robot.py \
  --config configs/pi05/pi05_rebot_dual_inference.py \
  --ckpt-path <run-directory>/checkpoints/<checkpoint>.safetensors
```

`RebotDualOperator` receives three RGB streams and two 7-D joint states.
`RebotDualInferenceRunner` uses the common FluxVLA operator and runner bases;
it does not inherit ALOHA hardware behavior. Continuous reBot gripper angles
remain part of each 7-D arm command. Configure and validate
`inference.prepare_pose` before using task ID `0` for automatic reset.

## 8. Safety

- Support both followers before stopping collection.
- Final disconnect disables follower torque; episode transitions do not.
- Do not run calibration, teleoperation, and recording processes concurrently.
- Validate stable serial mappings after moving USB connections.
- Stop immediately if a leader stream stalls or a follower reports a CAN error.
