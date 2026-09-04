# Oli Whole-Body (Loco-Manipulation) Operator

`OliOperator` + `OliInferenceRunner` provide a whole-body inference path for
the Oli humanoid. The default layout uses a head camera, a 33-dim
proprioceptive state, and a 42-dim whole-body action.

## Spaces

- **State (33-dim):** 31 joint positions + 2 hand-closed flags
  (`left`, `right`).
- **Action (42-dim):**
  - `[0:31]` joint position commands (`q`)
  - `[31:34]` `base_link` position `xyz` (absolute)
  - `[34:40]` `base_link` rotation as 6D (Zhou et al.)
  - `[40]` `left_hand_closed`, `[41]` `right_hand_closed`

### Full dexterous-hand layout (MROS)

Set `hand_mode='finger'` for a checkpoint trained on individual BrainCo
finger values. In this mode, the state is **43-dim** (31 joints + 12 finger
state values) and the action is **52-dim**. The first 40 action dimensions
are unchanged; `[40:52]` supplies the finger values. The operator appends
`finger_force_levels` and publishes the resulting 14-dimensional command to
`finger_cmd_topic`.

```python
operator=dict(
    type='OliOperator',
    control_backend='mros',
    left_wrist_rgb_topic='/left_wrist_camera/color/image_raw/compressed',
    finger_state_topic='/brainco1/hand/state',
    finger_cmd_topic='/brainco1/hand/cmd',
    hand_mode='finger',
)
```

`hand_mode='finger'` is MROS-only. Use `hand_mode='binary'` for existing
42-dim open/close-hand checkpoints.

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

`OliOperator` supports two transport backends:

- `control_backend='websocket'`: observations arrive through ROS (`rospy`),
  while commands use the LimX WebSocket JSON protocol (`request_servoj` for
  joints).
- `control_backend='mros'`: compressed images, joint state, and finger state
  arrive through MROS; whole-body commands are published as `TeleopMsg` on
  `teleop_wbt_topic`, and hand commands as `Float32Array` on
  `finger_cmd_topic`.

Middleware imports remain lazy, so importing the module itself does not
require a running robot stack.

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
`fluxvla/engines/operators/oli_operator.py` (`_send_base_pose`,
`_send_hand_action`) to your controller.

Note: `disable_puppet_arm=True` only makes the runner skip sending actions; it
does NOT make initialization hardware-free — the operator still connects ROS
and WebSocket on construction.

## Prompt, prepare pose, and pause

`OliInferenceRunner` treats one execution as one predicted action chunk,
optionally truncated by `execute_horizon`. In interactive mode it first asks
for a prompt ID and then for the number of chunks to execute.

Set `prepare_pose` to a 33-dimensional binary-hand state or a 43-dimensional
finger-hand state. When `prepare_pose_prompt_id` is configured, entering that
ID smoothly moves the MROS robot to the pose over
`prepare_pose_duration_sec`, then returns to prompt selection.

With `interactive=True`, type `p` and press Enter while actions are running to
pause after the current chunk and return to prompt selection. The configured
`default_execution_count` is the default number of chunks for one interactive
selection.

With `interactive=False`, the runner never reads stdin. It continuously uses
`default_prompt_id`, preserves cross-chunk history, and ignores
`default_execution_count`. The RTC runner also keeps one scheduler and one
producer/actor pair alive, preserving its action prefix across chunks. Use
Ctrl+C to stop either runner cleanly. When nonzero, `max_publish_step` limits
the action steps in a continuous run or one interactive prompt selection; zero
disables the limit.

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
