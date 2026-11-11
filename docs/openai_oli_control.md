# OLI keypoint control through the Responses API

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
xyz+rot6d pose. Until WBT exposes measured planar odometry, that pose combines
the Operator's accumulated commanded x/y/yaw with measured height, roll, and
pitch. Keeping x/y/yaw in one command-space integrator preserves their shared
delta semantics. The configured hand representation is included in the same
state. Checkpoint preprocessing selects the fields declared by its modality.
The final twelve keypoint action values are published in their original order,
followed by the configured force levels.

The OLI policy requests absolute `xyz_m` positions and pairs of semantic
direction vectors. Key-body targets use the robot-base frame; base targets use
the fixed command-world frame. The adapter converts a base target to per-step
body-frame x/y and yaw increments, absolute height, and absolute pitch/roll,
while preserving the Operator wire. Omitted base motion emits zero x/y and the
all-zero rotation sentinel together with the current measured height.

The Operator accepts `upper_only` and `full` keypoint action layouts. With raw
fingers they are 45-D and 66-D; with binary hands they are 35-D and 56-D.
`upper_only` supplies left/right wrist, head, base rotation delta, and hands,
while feet and base height are filled from the current keypoint state. `full`
supplies all five key-body poses, the full base action, and hands.

## Hosted OLI policy

Saved configs and direct callers should follow the
[hosted VLA migration guide](hosted_vla_migration.md).

`OpenAIVLA(embodiment='oli')` combines the Responses API transport with the
shared OLI policy. It sends
the available RGB views, all five key-body poses, base pose, and a summary of
the 12-D finger state to the configured model (GPT-6 by default). The base input combines accumulated command
x/y/yaw with measured height/pitch/roll; head, wrist, and foot input poses are
absolute in the robot-base frame. Orientation uses semantic direction pairs:
wrist finger direction and palm-facing normal, head gaze and up directions,
foot toe and top-normal directions, and base forward and up directions.
Local frames are defined by the same canonical pose: the
robot stands upright, its arms hang at its sides, and both palms face inward.
In that pose every local frame is aligned with robot-base `+x` forward, `+y`
left, and `+z` up. Each frame is rigidly attached to its target. For both
neutral wrists, local `-z` points from wrist to fingers and local `+x` points
approximately toward the thumb. The left palm faces local `-y`, while the right
palm faces local `+y`.

For head, wrist, and foot outputs, `xyz_m` is an absolute target in the
robot-base frame. Base `xyz_m` is absolute in the command-world frame.
Orientation fields are `finger_direction_in_base` and
`palm_facing_direction_in_base` for wrists, `gaze_direction_in_base` and
`head_up_direction_in_base` for head, `toe_direction_in_base` and
`foot_top_normal_in_base` for feet, and `forward_direction_in_world` and
`up_direction_in_world` for base. Supply both directions when requesting an
orientation; omit both to hold it. Hands use six native channels per side.
The policy converts these absolute requests into the existing 66-D contract.

Every model command is constrained before it reaches MROS:

- wrist displacement is limited to 8 cm per call;
- head displacement is limited to 5 cm per call by the provided config
  (the model constructor default is 3 cm);
- foot displacement is limited to 8 cm per call;
- base xy displacement is limited to 10 cm and height change to 8 cm per call;
- relative pose rotation is limited to 15 degrees per call;
- absolute base roll and pitch are limited to 15 degrees;
- wrist and head targets are clipped to the configured workspace. If a
  requested key-body's current position is outside it, inference fails before
  generating actions rather than moving farther than the per-call limit.
  Omitted targets remain held, including poses outside the workspace;
- keypoint positions and hands are linearly interpolated over 50 steps;
  keypoint rotations follow the relative rotation vector.

Native finger observations require valid feedback. Missing first feedback or
a cache older than `finger_state_max_age_sec` (default 1 second) drops the
observation. Invalid shape, non-finite values, or values outside `[0,100]`
raise an error. Age is measured since a successful local message read, not a
sensor timestamp; deployment must verify whether MROS can return cached
messages repeatedly. Binary hand observations retain main's command-based
behavior.

Hand targets are checked before interpolation, and the runner preflights all
points it will execute before publishing any of them. Both MROS body and hand
messages are built and validated before sending. After a publisher exception,
further commands through that operator are blocked. A body call returning
success updates the accumulated command pose, but is not an execution receipt;
hand publication may still fail. Check controller state and reinitialize the
operator before resuming; do not automatically replay incremental actions.

## Running

Set credentials without writing them into the config:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # optional gateway
```

Observation-only mode defaults to one action chunk and publishes nothing:

```bash
python scripts/inference.py \
  --config configs/openai/gpt6_astra_oli_keypoint_inference.py \
  --cfg-options inference.disable_puppet_arm=True
```

Live mode must be enabled explicitly. The runner's execution count controls
the number of model action chunks:

```bash
python scripts/inference.py \
  --config configs/openai/gpt6_astra_oli_keypoint_inference.py \
  --cfg-options inference.disable_puppet_arm=False \
  inference.default_execution_count=20
```

Run these commands in the configured FluxVLA environment. The default config
now sets `disable_puppet_arm=True`; observation-only still calls the model API.

### Switching to MiniMax

`OpenAIVLA` is the Responses protocol entry point, not a restriction to OpenAI
models. MiniMax's China endpoint, `https://api.minimax.cn/v1`, with
`MiniMax-M3.1-Flash` was tested on 2026-11-10 using this same OLI adapter.
Change the **endpoint, API key and model name** together; no new model class
or config file is needed for this tested path. A provider advertising only
Chat Completions compatibility is not sufficient: this transport calls
`/responses` and requires image inputs, function calls and multi-turn tool
history.

Load the MiniMax China key into `MINIMAX_CN_API_KEY` from your local secret
store, then reuse the existing config:

```bash
export OPENAI_BASE_URL='https://api.minimax.cn/v1'
export OPENAI_API_KEY_ENV='MINIMAX_CN_API_KEY'
python scripts/inference.py \
  --config configs/openai/gpt6_astra_oli_keypoint_inference.py \
  --cfg-options inference_model.model=MiniMax-M3.1-Flash \
  inference_model.max_output_tokens=4096 \
  inference.disable_puppet_arm=True
```

The key variable's name is configurable; using `OPENAI_API_KEY` with a
MiniMax key also works when the endpoint points to the matching MiniMax
service. Keys issued for different services or regions are not
interchangeable. The 4096-token cap is the tested output budget, not an OLI
protocol requirement. Keep `type='OpenAIVLA'` and `embodiment='oli'` for this
example; the separate `MiniMaxVLA` entry point currently registers only
LIBERO and RoboCasa policies.

Validation covered two real API turns with changing synthetic images,
`control_oli` calls, preserved history, trace output and valid 90-by-66 hold
actions. Robot runs also exercised this endpoint, but an incomplete head
orientation later caused target validation to fail. API compatibility does
not establish that every generated motion target is valid or that a full
robot task has passed. Supply both orientation directions together or omit
both; the tool schema currently leaves them optional individually, while
runtime validation rejects an incomplete pair before returning any action
chunk. The observation-only example above still requires complete live
observations; it disables action publication, not sensor collection.

## Diagnosing a stopped run

| Log marker                                 | Meaning and next check                                                                                                  |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `OLI_HAND_STATE_UNAVAILABLE`               | Waiting for the first valid or fresh hand message; inspect the hand bridge, topic and local read age.                   |
| `OLI_HAND_STATE_INVALID`                   | Invalid finger feedback; check its layout, finite values and native `[0,100]` units.                                    |
| `OLI_TARGET_INVALID`                       | Invalid model target; inspect the tool arguments and target value range.                                                |
| `OLI_WORKSPACE_STATE`                      | A requested key-body starts outside workspace; check coordinate frames and bounds.                                      |
| `OLI_ACTION_INVALID` / `OLI_CHUNK_INVALID` | An action point fails validation; chunk errors identify a zero-based step. No prefix is sent by this preflight.         |
| `OLI_FORCE_INVALID`                        | Force configuration must have two finite values.                                                                        |
| `OLI_PUBLISH_FAILED`                       | Delivery is uncertain or partial; the message identifies body or hands. Check the robot before recreating the operator. |
| `OLI_PUBLISH_LATCHED`                      | A previous publish failure blocks subsequent commands; no automatic recovery or replay.                                 |
| `OLI_EXECUTION_FAILED`                     | The zero-based failing step and completed local send calls; these counts are not robot execution receipts.              |
| `OLI_CLEANUP_FAILED`                       | Cleanup failed; the original inference/publish error remains the primary exception.                                     |

Shape/non-finite target errors also include the field name (for example
`OLI target.left_hand`). Keep the complete traceback, config overrides, model
trace, and exact code revision when recording a hardware failure.

## Trace files for successful and failed calls

With `trace_dir` configured, each logical model call creates `call_0001`,
`call_0002`, and so on. The request is saved before calling the API, and any
JSON response is saved before tool parsing or target validation. The HTML
report includes pending, received, generated and failed states.

| File                         | Meaning                                                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `request.json` and `image_*` | Prepared request and its extracted input images; saved before transport. No HTTP authorization headers are recorded. |
| `response.json`              | Received JSON, including invalid tool responses. Absent if no JSON response was obtained.                            |
| `response.txt`               | Raw response text when JSON decoding fails in transport, such as an HTML gateway reply.                              |
| `arguments.json`             | Parsed tool arguments, saved before motion validation.                                                               |
| `metadata.json`              | Response ID/model, latency and token metadata when a JSON response was received.                                     |
| `status.json`                | Call start/update time, status, stage, and `execution_status=not_recorded`.                                          |
| `error.json`                 | Exception type/message, failing stage, traceback and chained causes, including HTTP status when available.           |
| `action_chunk.json`          | Generated actions; absent when generation failed before this stage. This is not proof of publication or execution.   |

Stages identify `request`, `response_parse`, `target_validation`,
`interpolation`, and trace preparation/save/finalization. A transport timeout,
HTTP failure, invalid tool arguments or rejected target retains its original
exception. If recording that exception also fails, the console reports
`OLI_TRACE_WRITE_FAILED` and preserves the primary error. A trace write failure
on a successful generation prevents returning actions to the runner.

HTTP retries inside one protocol call share a single trace directory. Error
responses that the protocol raises instead of returning are represented by
the exception chain and available error text; the trace does not fabricate a
JSON response. Budget exhaustion or invalid observations before request
preparation do not create a new API-call directory. Publication failures are
reported separately by the runner/operator log markers above.

Type `p` and press Enter while a chunk is running to pause after that chunk.
Only one process may publish robot commands. First validate the behavior
against Mujoco. This layer does not provide collision
checking, force/torque limiting, inverse-kinematics feasibility checks, or an
emergency-stop implementation; those remain controller/operator obligations.
