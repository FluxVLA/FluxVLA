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
    max_head_delta=0.05,
    max_foot_delta=0.08,
    max_base_xy_delta=0.10,
    max_base_height_delta=0.08,
    max_rotation_delta_deg=15.0,
    max_base_tilt_deg=15.0,
    trace_dir='work_dirs/oli_gpt_traces',
    trace_console=True,
    workspace_bounds=[[-0.30, 0.80], [-0.65, 0.65], [-0.30, 1.50]],
)

del _environ, _OPENAI_BASE_URL, _OPENAI_API_KEY_ENV

_CANDY_GRASP = (
    ' Use a thumb-index pinch. Choose a wrist pre-grasp pose that brings the '
    'thumb tip and index fingertip to opposite sides of the candy along their '
    'natural closing paths. Keep the middle, ring, and little fingers closed '
    'and clear of the object.')

inference = dict(
    type='OliInferenceRunner',
    interactive=True,
    default_prompt_id='1',
    default_execution_count=1,
    allow_custom_instruction=True,
    task_descriptions={
        '0': ('stand upright in a comfortable ready pose with the feet '
              'planted and the torso straight. Tilt the head down by about '
              '40 degrees, and hold both hands naturally in front of the '
              'lower chest with the elbows bent, the fingers pointing '
              'forward, and the palms facing each other. Keep the fingers '
              'relaxed and open, '
              'with both hands safely above and clear of the table'),
        '1': ('pick up the white candy and place it in the left section of '
              'the snack tray with left arm.' + _CANDY_GRASP),
        '2': ('pick up the purple candy and place it in the right section of '
              'the snack tray with left arm.' + _CANDY_GRASP),
        '3': ('pick up the red candy and place it in the middle section of '
              'the snack tray with left arm.' + _CANDY_GRASP),
        '5': ('wave both wrists freely in the air using small position and '
              'rotation changes while keeping the hands clear of the table '
              'and other objects. Keep the fingers open. Before each wrist '
              'rotation, predict the visible effect of the requested thumb '
              'and palm directions; on the next observation, compare the '
              'images and reported directions with that prediction, then '
              'correct the target if needed. Keep the head, base, and both '
              'feet still'),
    },
    disable_puppet_arm=False,
    publish_rate=30.0,
    enable_mixed_precision=False,
    camera_names=['head', 'left_wrist', 'right_wrist'],
    dataset=dict(
        type='OpenAIOliInferenceDataset',
        image_names=['head', 'left_wrist', 'right_wrist'],
    ),
    denormalize_action=dict(
        type='IdentityAction',
        action_dim=66,
    ),
    action_chunk=50,
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
