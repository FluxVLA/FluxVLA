# GPT-6 OLI keypoint control

This integration has two independent layers.

## Keypoint MROS deployment

`OliOperator(command_mode='keypoint')` adds keypoint command transport to the
existing FluxVLA OLI operator. It subscribes to:

- `/head/color/image_raw/compressed`
- `/left_wrist_camera/color/image_raw/compressed`
- `/right_wrist_camera/color/image_raw/compressed`
- `/joint/state`
- `/brainco1/hand/state`

The operator reads five named poses from `/cur_keybody` and reads base state
separately from `/current_base_height` and `/curr_base_quat`. Its command API
accepts five absolute key-body poses. The 9-D base action uses the same
contract as joint mode: body-frame x/y deltas, absolute height, and a hybrid
rotation action encoded as rot6d. After conversion to ZYX Euler angles, yaw is
a per-step delta and is accumulated, while pitch and roll are absolute
world-frame targets. Identity rot6d therefore requests zero pitch/roll; it is
not a hold command. A degenerate all-zero rot6d is reserved as the hold
sentinel. The Operator converts the action into the absolute pose required by
the bag-compatible nine-anchor `TeleopMsg` published on `/teleop_cmd`.

State input is selected independently: `state_mode='joint'` returns 31 joints
plus the configured hand representation, while `state_mode='keypoint'` returns
the named head, feet, and wrist xyz+rot6d features, plus a complete base
xyz+rot6d pose. Until WBT exposes measured base x/y, that pose combines the
Operator's accumulated commanded x/y with measured height and rotation. It
also synchronizes the base integrator before inference and returns the
configured hand representation. Checkpoint preprocessing selects the fields
declared by its modality. The final twelve keypoint action values are published
in their original order, followed by the configured force levels.

The GPT policy builds every action from the current observation. Unchanged
key-body poses are echoed from that frame. Its base tool uses total body-frame
`dx/dy`, relative height `dz`, relative `yaw_delta_deg`, and absolute
world-frame `roll_deg/pitch_deg` for one action chunk. The policy divides
x/y/yaw across the 50 published points because the Operator integrates those
fields per point. Height, roll, and pitch remain absolute targets and are
linearly interpolated. Omitted base motion emits zero x/y and the all-zero
rotation sentinel together with the current measured height.

The Operator accepts `upper_only` and `full` keypoint action layouts. With raw
fingers they are 45-D and 66-D; with binary hands they are 35-D and 56-D.
`upper_only` supplies left/right wrist, head, base rotation delta, and hands,
while feet and base height are filled from the current keypoint state. `full`
supplies all five key-body poses, the full base action, and hands.

## GPT-6 policy

`OpenAIResponsesOliVLA` reuses main's Responses API implementation. It sends
the available RGB views, all five key-body poses, base pose, and a summary of
the 12-D finger state to GPT-6. The `control_oli` tool can request base-frame
xyz deltas for wrists, feet, and head; local relative RPY deltas for wrists and
feet; base translation, height, and yaw changes; and all six native channels
of either hand. The policy converts relative keypoint requests into absolute
poses, so the 66-D Operator/WBT contract remains unchanged.

Every GPT command is constrained before it reaches MROS:

- wrist displacement is limited to 8 cm per call;
- head displacement is limited to 3 cm per call;
- foot displacement is limited to 8 cm per call;
- base xy displacement is limited to 10 cm and height change to 8 cm per call;
- keypoint rotation and base yaw are limited to 15 degrees per call;
- absolute base roll and pitch are limited to 15 degrees;
- wrist and head xyz are clipped to the configured workspace;
- keypoint poses and hands are linearly interpolated to the target over all 50
  steps.

## Running

Set credentials without writing them into the config:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # optional gateway
```

Observation-only mode defaults to one action chunk and publishes nothing:

```bash
/root/miniconda3/envs/fluxvla/bin/python scripts/inference.py \
  --config configs/openai/gpt6_astra_oli_keypoint_inference.py
```

Live mode must be enabled explicitly. The runner's execution count controls
the number of GPT action chunks:

```bash
/root/miniconda3/envs/fluxvla/bin/python scripts/inference.py \
  --config configs/openai/gpt6_astra_oli_keypoint_inference.py \
  --cfg-options inference.disable_puppet_arm=False \
  inference.default_execution_count=20
```

Type `p` and press Enter while a chunk is running to pause after that chunk.
Only one process may publish robot commands. First validate the behavior
against Mujoco. This layer does not provide collision
checking, force/torque limiting, inverse-kinematics feasibility checks, or an
emergency-stop implementation; those remain controller/operator obligations.
