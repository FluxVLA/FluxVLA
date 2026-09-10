# GPT LIBERO control contract

`configs/openai/gpt6_astra_libero_inference.py` and its named LIBERO suite
configs use `OpenAIResponsesVLA`. All tasks remain selected by default.

## Position, rotation, and gripper

Example `move_to` arguments:

```json
{
  "targets": {
    "z": 0.85,
    "rotation_delta": [0, 0, 0.2],
    "gripper": 1
  },
  "note": "Rotate about world z to align the fingertips before descending."
}
```

- `x`, `y`, `z`: absolute world position in meters, clipped to configured
  workspace bounds. Omitted coordinates retain their current value.
- `rotation_delta`: optional **relative world-frame axis-angle vector**, in
  radians. Its direction is the rotation axis, and its length is the total
  requested angle **for the whole call**, using the right-hand rule. It is
  neither Euler angles nor a rotation repeated in full at each simulator step.
  `[0, 0, 0.2]` requests about 11.5 degrees about world z. Omit it to preserve
  orientation. The standard LIBERO Panda base axes are world-aligned.
- `gripper`: `0` means closed, `1` means open (native LIBERO actions `+1` and
  `-1`, respectively). Omission preserves the previous gripper command.

The adapter emits normalized native `[dx, dy, dz, drx, dry, drz, gripper]`
chunks; `IdentityLiberoAction` must not apply dataset-stat denormalization.
Translation and rotation share the chunk duration, but have separate speed
limits. Rotation clipping preserves the requested axis by limiting the
vector's norm, rather than clipping its coordinates independently.

The default `rotation_action_scale=0.1` estimates achieved radians per unit
action per simulator step. It is **not** OSC's raw 0.5-radian goal offset.
With `max_rotation_speed_fraction=0.25` and `action_horizon=10`, one call
requests at most approximately 0.25 radians. Actual motion depends on pose,
controller dynamics, and contact. This is an open-loop chunk between GPT
observations, not an exact pose servo: inspect the next observation and
correct rather than assuming the requested pose was reached.

The old adapter exposed no rotation parameter, instructed GPT to keep a
fixed orientation, and filled all three rotation channels with zero. Old
rollouts and metrics therefore do not validate the rotation-enabled policy.

The fix is confined to the API policy and its config. The existing shared
LIBERO runner already forwards all seven action channels, including rotation;
it does not need modification. Neither shared evaluation runner is changed.

## Offline simulator diagnostic (no API key or paid requests)

From the repository root in the FluxVLA environment:

```bash
MUJOCO_GL=egl python tools/check_openai_libero_control.py
```

This feeds scripted Responses tool outputs through the actual dataset,
policy parser, chunk adapter, action transform, and real LIBERO physics. It
checks positive/negative rotation about all three axes, XYZ translation,
gripper signs, and a zero-rotation baseline on task 0 / trial 0. Each case
resets the same initial state. The fresh `work_dirs/diagnostics/libero-control-*`
directory contains `report.json` and `rotation_smoke.mp4` (external + wrist
views). This is a controller smoke test, **not a GPT task-success evaluation**.

Regression tests also cover malformed/non-finite commands, rotation omitted
or combined with translation, per-call angular limits, and delivery of the
rotation channels through the unchanged LIBERO run loop:

```bash
python -m pytest -q \
  test/test_models/test_openai_libero_rotation.py \
  test/test_models/test_openai_responses_vla.py \
  test/test_runners/test_libero_openai_rotation_compatibility.py \
  test/test_models/test_openai_robocasa_wrist.py \
  test/test_configs/test_gpt6_astra_libero_config.py
```

Invalid LIBERO tool fields are rejected rather than silently ignored. The
prompt's planning milestones follow `max_llm_calls`.

## RoboCasa is a different action interface

`OpenAIResponsesRobocasaVLA` uses `control_gr1`, not `move_to`. Each arm already
exposes seven joint deltas, ordered as shoulder pitch/roll/yaw, elbow pitch,
then wrist yaw/roll/pitch. The last three entries of `left_arm_delta` and
`right_arm_delta` therefore control the wrists; they are not zero-filled.
The policy converts each delta to an absolute joint target and holds that
target over its action chunk. The 29D native action layout is left arm,
right arm, left hand, right hand, waist. The RoboCasa identity transform
retains those wrist values without LIBERO-style clipping or normalization.

The regression test exercises both arms, all three wrist joints and both
signs through the configured policy, transform, and runner action layout.
No LIBERO `rotation_delta` parameter should be added to the RoboCasa config.

## Separate runner issues (not changed here)

Both runners check the episode step budget outside the inner action loop,
so a final chunk can exceed the requested horizon. This is independent of
rotation support and is deliberately left for a separate runner change.
The default GPT RoboCasa policy always emits eight steps and its 720-step
budget is divisible by eight; non-divisible overrides can expose the issue.
The LIBERO successful-step counter also remains unchanged. The earlier
runner budget/validation patch was removed to avoid mixing a shared behavior
change with this API-specific rotation fix.

To measure GPT task success again, use the normal evaluation command in a
fresh run after setting `OPENAI_BASE_URL` and `OPENAI_API_KEY` in the shell;
do not combine old fixed-orientation episodes with new results.
