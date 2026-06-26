# Oli Whole-Body (Loco-Manipulation) Operator

`OliOperator` + `OliInferenceRunner` provide a minimal whole-body inference
path for the Oli humanoid: a head camera and 33-dim proprioceptive state in,
a 42-dim whole-body action out.

## Spaces

- **State (33-dim):** 31 joint positions + 2 hand-closed flags
  (`left`, `right`).
- **Action (42-dim):**
  - `[0:31]` joint position commands (`q`)
  - `[31:34]` `base_link` position `xyz` (absolute)
  - `[34:40]` `base_link` rotation as 6D (Zhou et al.)
  - `[40]` `left_hand_closed`, `[41]` `right_hand_closed`

### Canonical 31-joint order

`/joint/state` messages are reordered to this order by joint name (positional
fallback when names are absent). A model trained in a different joint order
will command the wrong joints. This same `STATE_JOINT_NAMES` order is also the
order of the 31-element `q` vector sent in the WebSocket `request_servoj`
command; the LimX controller must interpret servoj `q` in this order.

```
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
head_yaw_joint
head_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_yaw_joint
left_wrist_pitch_joint
left_wrist_roll_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_yaw_joint
right_wrist_pitch_joint
right_wrist_roll_joint
```

## Transport

Sensor input is read over **ROS (rospy)**; control output is sent over the
**LimX WebSocket JSON protocol** (`request_servoj` for joints), mirroring
`Tron2Operator`. Both `rospy` and `websocket-client` are imported lazily, so
the modules import without any middleware installed.

### ROS topics (defaults)

| Purpose     | Topic                              | Type                          |
| ----------- | ---------------------------------- | ----------------------------- |
| Head RGB    | `/head/color/image_raw/compressed` | `sensor_msgs/CompressedImage` |
| Joint state | `/joint/state`                     | `sensor_msgs/JointState`      |

The two hand-closed state dims are derived from the last sent hand command
(command echo), not a hand-state sensor subscription. `get_frame` returns the
latest available image and joint state without timestamp synchronization
(latest-only polling).

### Hardware integration points

The base-pose (`request_base_pose`) and hand (`request_hand_cmd`) WebSocket
request titles are **robot-SDK specific** and are not part of the public LimX
protocol. Adapt their titles/payloads in
`fluxvla/engines/operators/oli_operator.py` (`_send_base_pose`, `_send_hands`)
to your controller.

Note: `disable_puppet_arm=True` only makes the runner skip sending actions; it
does NOT make initialization hardware-free — the operator still connects ROS
and WebSocket on construction.

## Run

```bash
python scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_eagle_3b_oli_full_finetune.py \
  --ckpt-path /path/to/oli_checkpoint.safetensors
```

`dataset_statistics.json` must sit two directories above the checkpoint, per
`BaseInferenceRunner`. The config's model dims (`state_dim`, `action_dim`,
`ori_action_dim`) and `embodiment_id` are example values — align them with
your trained checkpoint.
