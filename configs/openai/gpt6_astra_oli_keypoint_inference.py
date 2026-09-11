# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Checkpoint-free GPT-6 control through OLI's keypoint MROS interface."""

from os import environ as _environ

_OPENAI_BASE_URL = (_environ.get('OPENAI_BASE_URL')
                    or 'https://api.openai.com/v1').rstrip('/')
_OPENAI_API_KEY_ENV = _environ.get('OPENAI_API_KEY_ENV') or 'OPENAI_API_KEY'

inference_model = dict(
    type='OpenAIResponsesOliVLA',
    model='gpt-6-astra',
    base_url=_OPENAI_BASE_URL,
    api_key_env=_OPENAI_API_KEY_ENV,
    reasoning_effort='medium',
    request_timeout=120.0,
    max_retries=2,
    retry_backoff=2.0,
    image_detail='high',
    image_format='JPEG',
    jpeg_quality=95,
    image_horizon=2,
    max_llm_calls=50,
    action_horizon=50,
    max_wrist_delta=0.08,
    max_head_delta=0.03,
    max_foot_delta=0.08,
    max_base_xy_delta=0.10,
    max_base_height_delta=0.08,
    max_rotation_delta_deg=15.0,
    max_base_tilt_deg=15.0,
    workspace_bounds=[[-0.30, 0.80], [-0.65, 0.65], [-0.30, 1.50]],
)

del _environ, _OPENAI_BASE_URL, _OPENAI_API_KEY_ENV

inference = dict(
    type='OliInferenceRunner',
    policy_mode='openai',
    interactive=True,
    default_prompt_id='1',
    default_execution_count=1,
    task_descriptions={
        '1': ('pick up the white candy and place it in the left section of '
              'the snack tray with left arm'),
        '2': ('pick up the purple candy and place it in the right section of '
              'the snack tray with left arm'),
        '3': ('pick up the red candy and place it in the middle section of '
              'the snack tray with left arm'),
    },
    # Observation-only and one API call by default. Live control must be
    # enabled explicitly by setting disable_puppet_arm=False.
    disable_puppet_arm=True,
    publish_rate=30.0,
    camera_names=['head', 'left_wrist', 'right_wrist'],
    operator=dict(
        type='OliOperator',
        control_backend='mros',
        command_mode='keypoint',
        hand_mode='finger',
        state_mode='keypoint',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic=('/left_wrist_camera/color/image_raw/compressed'),
        right_wrist_rgb_topic=(
            '/right_wrist_camera/color/image_raw/compressed'),
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        teleop_command_topic='/teleop_cmd',
        finger_cmd_topic='/brainco1/hand/cmd',
        keybody_state_topic='/cur_keybody',
        base_height_topic='/current_base_height',
        base_quat_topic='/curr_base_quat',
    ),
)
