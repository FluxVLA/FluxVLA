# GPT-6 OLI four-point control

This integration has two independent layers.

## Four-point MROS deployment

`OliOperator(command_mode='four_point')` ports the essential command transport
from `deploy/candy0811-0812-20260814-sim` into the existing FluxVLA OLI
operator. It subscribes to:

- `/head/color/image_raw/compressed`
- `/left_wrist_camera/color/image_raw/compressed`
- `/right_wrist_camera/color/image_raw/compressed`
- `/joint/state`
- `/brainco1/hand/state`

Before sending a command it captures `/cur_keybody`, `/current_base_height`,
and `/curr_base_quat`. The captured base and feet remain fixed. Each body
action is published as the bag-compatible nine-anchor `TeleopMsg` on
`/teleop_cmd`.

The existing `hand_mode` selects the hand contract:

- `binary`: 33D state and 35D action. Two open/closed values are expanded to
  the established BrainCo command.
- `finger`: 43D state and 45D action. The final twelve values are published in
  their original order, followed by the configured force levels.

## GPT-6 policy

`OpenAIResponsesOliVLA` reuses main's Responses API implementation. It sends
the three RGB views, current left/right wrist and head poses, and binary hand
state to GPT-6. The `control_oli` tool can request absolute xyz targets and
open/close either hand. Wrist/head orientation, base, and feet cannot be
changed by GPT.

Every GPT command is constrained before it reaches MROS:

- wrist displacement is limited to 8 cm per call;
- head displacement is limited to 3 cm per call;
- xyz is clipped to the configured workspace;
- the target is converted to a 50-step minimum-jerk trajectory;
- hand changes occur only in the final settling steps;
- every live trajectory requires an explicit operator `y` confirmation.

`n` rejects the action and takes a new observation. `q`, EOF, or `Ctrl-C`
stops without publishing the proposed action. The decision is recorded in the
Responses history so GPT is not told that a rejected action ran.

## Running

Set credentials without writing them into the config:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # optional gateway
```

Observation-only mode performs one API call and publishes nothing:

```bash
/root/miniconda3/envs/fluxvla/bin/python scripts/inference.py \
  --config configs/openai/gpt6_astra_oli_four_point_inference.py
```

Live mode must be enabled explicitly. Each proposed action still pauses for
manual confirmation:

```bash
/root/miniconda3/envs/fluxvla/bin/python scripts/inference.py \
  --config configs/openai/gpt6_astra_oli_four_point_inference.py \
  --cfg-options inference.disable_puppet_arm=False inference.max_iterations=20
```

Only one process may publish robot commands. First validate the observation
and action summary against Mujoco. This layer does not provide collision
checking, force/torque limiting, inverse-kinematics feasibility checks, or an
emergency-stop implementation; those remain controller/operator obligations.
