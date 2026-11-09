# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checkpoint-free MiniMax-M3.1-Flash full-suite evaluation on RoboCasa GR1.

The config evaluates all 24 standard RoboCasa tasks once by default. Override
``eval.num_trials_per_task`` to increase the number of episodes per task.

The policy uses MiniMax's Anthropic-compatible Messages API by default; set
``inference_model.api_protocol`` to ``'chat_completions'`` or ``'responses'``
to use the OpenAI-compatible endpoints instead.

Both MiniMax platforms are supported, each with its own API key::

    # China (default): https://api.minimax.cn
    export MINIMAX_CN_API_KEY='...'
    # International: https://api.minimax.io
    export MINIMAX_REGION=intl
    export MINIMAX_INTL_API_KEY='...'

``--cfg-options inference_model.region=intl`` switches the region as well.
The endpoints and key variable follow the region unless overridden, e.g. for
a gateway, with ``MINIMAX_BASE_URL``, ``MINIMAX_ANTHROPIC_BASE_URL`` or
``MINIMAX_API_KEY_ENV``. The generic ``OPENAI_BASE_URL`` and
``ANTHROPIC_BASE_URL`` variables are deliberately ignored so that other SDK
settings cannot redirect the key.
"""

# from os import environ as _environ
_environ = __import__('os').environ

_MINIMAX_REGION = _environ.get('MINIMAX_REGION') or 'cn'
# None lets the policy resolve region defaults, so a region override via
# --cfg-options also switches the endpoints and the key variable.
_MINIMAX_BASE_URL = _environ.get('MINIMAX_BASE_URL') or None
_MINIMAX_ANTHROPIC_BASE_URL = (
    _environ.get('MINIMAX_ANTHROPIC_BASE_URL') or None)
_MINIMAX_API_KEY_ENV = _environ.get('MINIMAX_API_KEY_ENV') or None

inference_model = dict(
    type='MiniMaxRobocasaVLA',
    model='MiniMax-M3.1-Flash',
    region=_MINIMAX_REGION,
    api_protocol='anthropic',
    base_url=_MINIMAX_BASE_URL,
    anthropic_base_url=_MINIMAX_ANTHROPIC_BASE_URL,
    api_key_env=_MINIMAX_API_KEY_ENV,
    # MiniMax's Messages and Responses APIs only accept 'auto'; the system
    # prompt already demands exactly one tool call per turn.
    tool_choice='auto',
    # M3.1-Flash always thinks; low effort bounds closed-loop latency.
    reasoning_effort='low',
    # Thinking tokens count towards this cap, so keep ample headroom.
    max_output_tokens=16384,
    request_timeout=180.0,
    max_retries=2,
    retry_backoff=2.0,
    image_detail='default',
    image_format='JPEG',
    jpeg_quality=95,
    image_horizon=2,
    # 90 calls x 8 environment steps covers RoboCasa's 720-step horizon.
    max_llm_calls=90,
    action_horizon=8,
    max_arm_joint_delta=0.12,
    max_waist_joint_delta=0.05,
)

del _environ, _MINIMAX_REGION, _MINIMAX_BASE_URL
del _MINIMAX_ANTHROPIC_BASE_URL, _MINIMAX_API_KEY_ENV

_ROBOCASA_TASK_NAMES = [
    'PnPBottleToCabinetClose',
    'PnPCanToDrawerClose',
    'PnPCupToDrawerClose',
    'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose',
    'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA',
    'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA',
    'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
    'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA',
    'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
    'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA',
    'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA',
    'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA',
    'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA',
    'PosttrainPnPNovelFromTrayToTieredshelfSplitA',
]

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='minimax-m3.1-flash',
    task_list=[
        f'gr1_unified/{task_name}_GR1ArmsAndWaistFourierHands_Env'
        for task_name in _ROBOCASA_TASK_NAMES
    ],
    total_tasks=len(_ROBOCASA_TASK_NAMES),
    task_ids=None,
    eval_chunk_size=8,
    max_episode_steps=720,
    num_trials_per_task=1,
    seed=7,
    unnorm_key='robocasa_gr1_api_native',
    action_order='n15',
    action_keys={
        'action.left_arm': (0, 7),
        'action.right_arm': (7, 14),
        'action.left_hand': (14, 20),
        'action.right_hand': (20, 26),
        'action.waist': (26, 29),
    },
    requires_dataset_stats=False,
    dataset=dict(
        type='OpenAIRobocasaEvalDataset',
        img_keys=['video.ego_view_bg_crop_pad_res256_freq20'],
        resize_size=512,
    ),
    denormalize_action=dict(
        type='IdentityRobocasaAction', action_dim=29, clip=False),
    denormalize_action_chunk=True,
    model_build_device='cpu',
    enable_mixed_precision_training=False,
    deterministic_env=True,
    deterministic_action_sampling=True,
    save_video=True,
    rollout_video_key='video.ego_view_bg_crop_pad_res256_freq20',
    output_dir='work_dirs/minimax_m31_flash_robocasa',
)

del _ROBOCASA_TASK_NAMES
