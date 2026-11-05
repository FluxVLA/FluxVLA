# reBot Dual-Arm Training and Inference

The reBot integration uses the same FluxVLA inference entry point and
synchronous/asynchronous execution pattern as ALOHA, with reBot-specific
ROS topics, joint names, and continuous gripper angles.

## Installation

From the FluxVLA checkout, use the standard real-robot installation path:

```bash
conda create -n fluxvla python=3.10 -y
conda activate fluxvla
bash scripts/install_env.sh real-only
```

ROS1 Noetic and a running robot interface that provides the topics below
are prerequisites. Source the ROS environment in the inference terminal:

```bash
source /opt/ros/noetic/setup.bash
conda activate fluxvla
```

## Data and ROS Interface

The configuration uses `ParquetDatasetV3` with the following data keys:

```text
observation.state                    float32[14]
action                               float32[14]
observation.images.cam_front
observation.images.cam_wrist_left
observation.images.cam_wrist_right
```

State and action values are joint angles in radians, ordered as the left
arm's seven values followed by the right arm's seven values. Each arm uses:

```text
shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_yaw, wrist_roll, gripper
```

The default operator topics are:

| Role                 | ROS topic                                              | Message type             |
| -------------------- | ------------------------------------------------------ | ------------------------ |
| Front camera         | `/camera_front/color/image_raw`                        | `sensor_msgs/Image`      |
| Left wrist camera    | `/camera_left_wrist/color/image_raw`                   | `sensor_msgs/Image`      |
| Right wrist camera   | `/camera_right_wrist/color/image_raw`                  | `sensor_msgs/Image`      |
| Left joint feedback  | `/left_arm/joint_states`                               | `sensor_msgs/JointState` |
| Right joint feedback | `/right_arm/joint_states`                              | `sensor_msgs/JointState` |
| Left joint commands  | `/left_arm/rebot_joint_controller/target_joint_state`  | `sensor_msgs/JointState` |
| Right joint commands | `/right_arm/rebot_joint_controller/target_joint_state` | `sensor_msgs/JointState` |

## Training Configuration

`configs/pi05/pi05_rebot_dual_inference.py` contains both the PI0.5
training recipe and inference settings. Replace the dataset placeholder
`./datasets/RealRobot_reBot_lerobot_v3/rebot_example` with the actual dataset
path and use the standard `scripts/train.py` workflow.

Arm actions are relative to the observed state in dimensions 0–5 and 7–12;
grippers in dimensions 6 and 13 remain absolute. The matching statistics
file uses the standard `private.proprio` and `private.action` keys.

## Inference

Prepare the model assets and matching training statistics:

```text
checkpoints/pi05_base/                    base model/tokenizer assets
<run-directory>/
  checkpoints/<checkpoint>.safetensors   fine-tuned model
  dataset_statistics.json                matching statistics
  tokenizer/                            saved tokenizer
```

With camera and joint feedback topics available, run:

```bash
python scripts/inference_real_robot.py \
  --config configs/pi05/pi05_rebot_dual_inference.py \
  --ckpt-path <run-directory>/checkpoints/<checkpoint>.safetensors
```

Set `inference.task_descriptions` to the training task text. The interactive
prompt selects a task ID and a number of consecutive inference chunks;
the number is not a count of complete task episodes.

As in ALOHA, `inference.async_execution` selects synchronous or background
trajectory execution. `inference.action_chunk` limits the predicted chunk
used by the runner, and `inference.execute_horizon` controls the execution
interval. Subsequent asynchronous chunks use `resample_remaining` to
account for elapsed inference time. Choose chunk coverage appropriate to
the measured inference latency.

To run the synchronous comparison, append:

```bash
--cfg-options inference.async_execution=False
```

Configure `inference.prepare_pose` explicitly before using task ID `0` to
reset. Unlike ALOHA's gripper threshold processing, reBot preserves the
continuous gripper joint angles. Do not run competing command publishers
such as teleoperation during inference. Keep the swept workspace clear
and the hardware stop accessible; interrupting inference is not a
verified hardware hold.

## Local RTC Inference

`configs/pi05/pi05_rebot_dual_rtc_inference.py` adds local RTC inference
through `RebotDualRTCInferenceRunner`, following the existing ALOHA RTC timing
and model-input contract.

- `guidance` (default) works with checkpoints trained without RTC prefix
  conditioning.
- `prefix` should be used with a checkpoint trained with
  `model.rtc_training_config`; see [RTC](rtc.md).

Run guidance with the same model assets and statistics as standard reBot
inference:

```bash
python scripts/inference_real_robot.py \
  --config configs/pi05/pi05_rebot_dual_rtc_inference.py \
  --ckpt-path <run-directory>/checkpoints/<checkpoint>.safetensors
```

Use these overrides for the ordinary asynchronous baseline or prefix mode:

```bash
--cfg-options inference.rtc_config.enabled=False
--cfg-options inference.rtc_config.method=prefix
```

The example predicts 50 actions, executes in intervals of 10, and estimates
`prefix_len` from the previous inference latency. reBot arm actions are
relative to the current observation, so the runner rebases the normalized
prefix to preserve its absolute joint targets; absolute gripper angles and
padding remain unchanged.

Like `OliRTCInferenceRunner`, reBot RTC is currently local-only: the remote
protocol does not carry `prev_actions`, `prefix_len`, or `rtc_config`. Empty
prefixes fall back to ordinary prediction. Check for a positive logged prefix
length after the first chunk to confirm conditioning is active.

The real-robot check covered a short ordinary-async/guidance comparison only.
It did not establish task success, uniformly smoother motion, prefix-mode
hardware behavior, VJP behavior, or gripper correctness.
