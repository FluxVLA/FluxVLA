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
"""Checkpoint-free MiniMax-M3.1-Flash evaluation on LIBERO.

The default evaluates every task in the suite once. Override
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
    type='MiniMaxLiberoVLA',
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
    max_llm_calls=50,
    # One API move is converted to at most ten LIBERO control steps.
    action_horizon=10,
    max_speed_fraction=0.5,
    position_action_scale=0.01,
    rotation_action_scale=0.1,
    max_rotation_speed_fraction=0.25,
    gripper_settle_steps=8,
    workspace_bounds=[[-0.45, 0.45], [-0.45, 0.45], [-0.05, 1.40]],
    # Appearance-level grounding only, never coordinates or simulator state.
    task_visual_hints={
        'pick up the alphabet soup and place it in the basket':
        ('The alphabet soup is the blue-and-orange cylindrical can with '
         'large yellow ALPHABET SOUP lettering. It is not the red-and-green '
         'tomato sauce can.'),
    },
)

# Keep environment helpers out of MMEngine's serialized config namespace.
del _environ, _MINIMAX_REGION, _MINIMAX_BASE_URL
del _MINIMAX_ANTHROPIC_BASE_URL, _MINIMAX_API_KEY_ENV

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_object',
    task_ids=None,
    model_family='minimax-m3.1-flash',
    eval_chunk_size=10,
    resize_size=512,
    num_trials_per_task=1,
    num_steps_wait=10,
    # Fifty ten-step API action chunks plus settling/reset headroom.
    max_steps=550,
    seed=7,
    norm_stats_path=None,
    requires_dataset_stats=False,
    dataset=dict(
        type='OpenAILiberoEvalDataset',
        img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
        resize_size=512,
    ),
    denormalize_action=dict(type='IdentityLiberoAction', action_dim=7),
    model_build_device='cpu',
    enable_mixed_precision_training=False,
    preprocess_every_step=False,
    save_rollout_videos=True,
    save_failed_rollout_videos=False,
    save_multi_view_rollout_videos=True,
    output_dir='work_dirs/minimax_m31_flash_libero',
)
