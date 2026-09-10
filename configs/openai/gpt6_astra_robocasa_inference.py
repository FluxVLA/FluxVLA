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
"""Checkpoint-free GPT-6 Astra full-suite evaluation on RoboCasa GR1.

The config evaluates all 24 standard RoboCasa tasks once by default. Override
``eval.num_trials_per_task`` to increase the number of episodes per task.

Connection settings are read from the environment::

    export OPENAI_API_KEY='...'
    export OPENAI_BASE_URL='https://api.openai.com/v1'  # optional

Set ``OPENAI_API_KEY_ENV`` when an OpenAI-compatible provider exposes its key
under another environment-variable name.
"""

from os import environ as _environ

_DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'
_OPENAI_BASE_URL = (_environ.get('OPENAI_BASE_URL')
                    or _DEFAULT_OPENAI_BASE_URL).rstrip('/')
_OPENAI_API_KEY_ENV = (_environ.get('OPENAI_API_KEY_ENV') or 'OPENAI_API_KEY')

inference_model = dict(
    type='OpenAIResponsesRobocasaVLA',
    model='gpt-6-astra',
    base_url=_OPENAI_BASE_URL,
    api_key_env=_OPENAI_API_KEY_ENV,
    reasoning_effort='medium',
    max_output_tokens=None,
    request_timeout=120.0,
    max_retries=2,
    retry_backoff=2.0,
    image_detail='high',
    image_format='JPEG',
    jpeg_quality=95,
    image_horizon=2,
    # 90 calls x 8 environment steps covers RoboCasa's 720-step horizon.
    max_llm_calls=90,
    action_horizon=8,
    max_arm_joint_delta=0.12,
    max_waist_joint_delta=0.05,
)

del _environ, _DEFAULT_OPENAI_BASE_URL, _OPENAI_BASE_URL, _OPENAI_API_KEY_ENV

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
    model_family='gpt6-astra',
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
    output_dir='work_dirs/gpt6_astra_robocasa',
)

del _ROBOCASA_TASK_NAMES
