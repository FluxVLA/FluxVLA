# Copyright 2026 Limx Dynamics
"""Online GPT-6 planning and completion monitoring, with PI05 execution.

Uses the same Responses API and environment settings as the existing
configs/openai/gpt6_astra_* examples. The planner uses the online GPT service.
Pass task_description and goal_conditions at runtime. Preview still calls
the ONLINE GPT service, but never loads a VLA or sends robot actions.
"""

from os import environ as _environ

_base_ = ['../pi05/pi05_paligemma_aloha_full_finetune.py']

executor = _base_.inference.copy()
executor.update(
    async_execution=False, execute_horizon=10, use_robot_base=False)

_client = dict(
    model='gpt-6-astra',
    base_url=(_environ.get('OPENAI_BASE_URL')
              or 'https://api.openai.com/v1').rstrip('/'),
    api_key_env=(_environ.get('OPENAI_API_KEY_ENV') or 'OPENAI_API_KEY'),
    # Keep online calls short; do not relax observation freshness for latency.
    reasoning_effort='low',
    max_output_tokens=4096,
    request_timeout=10.0,
    max_retries=0,
    image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    image_detail='high',
)

inference = dict(
    _delete_=True,
    type='HierarchicalInferenceRunner',
    executor=executor,
    dry_run=True,
    max_steps=100,
    max_observation_age_s=10.0,
    stable_observations=3,
    skill_catalog=dict(
        entities=dict(cup='red cup', tray='tray'),
        skills=[
            dict(
                name='put_into',
                parameters=dict(object=['cup'], target=['tray']),
                instruction='Put the {object} into the {target}.',
                preconditions={
                    'visible:{object}': True,
                    'visible:{target}': True
                },
                completion={'placed:{object}:{target}': True},
                max_chunks=40),
        ]),
    planner=dict(type='OpenAISubtaskPlanner', **_client),
    monitor=dict(
        type='OpenAIPredicateMonitor',
        descriptions={
            'visible:cup':
            'The red cup is clearly visible.',
            'visible:tray':
            'The target tray is clearly visible.',
            'placed:cup:tray':
            'The red cup is inside the tray, resting on '
            'it and released from the gripper.',
        },
        **_client),
)

del _client, _environ
