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
"""Checkpoint-free GPT-6 Astra evaluation on LIBERO.

The default is intentionally a one-episode smoke evaluation to limit API
usage. Override ``eval.task_ids`` and ``eval.num_trials_per_task`` for a full
benchmark after validating the controller on the local simulator.
"""

inference_model = dict(
    type='OpenAIResponsesVLA',
    model='gpt-6-astra',
    # The company LiteLLM gateway exposes GPT-6 Astra through the standard
    # Responses API and is reachable from the LIBERO evaluation host.
    base_url='https://litellm.limx.cn/v1',
    api_key_env='OPENAI_API_KEY',
    # Match the reference run's reasoning setting; the larger call budget and
    # higher-resolution images are more important for closed-loop control.
    reasoning_effort='medium',
    max_output_tokens=None,
    request_timeout=120.0,
    max_retries=2,
    retry_backoff=2.0,
    image_detail='high',
    image_format='JPEG',
    jpeg_quality=95,
    image_horizon=2,
    # Allow enough observe/act cycles to finish grasp-and-place instead of
    # exhausting the budget during visual servoing toward the source object.
    max_llm_calls=50,
    # One high-level API move is converted to at most ten LIBERO control
    # steps, matching the deliberate move/re-observe cadence of the reference.
    action_horizon=10,
    # Ten simulator steps at 0.5 action magnitude move roughly 5 cm, which
    # keeps motions controlled while avoiding repeated 2.5 cm approach calls.
    max_speed_fraction=0.5,
    # Empirical end-effector displacement per unit action and simulator step
    # under LIBERO's OSC_POSE controller.
    position_action_scale=0.01,
    gripper_settle_steps=8,
    workspace_bounds=[[-0.45, 0.45], [-0.45, 0.45], [-0.05, 1.40]],
    # GPT does not share LIBERO's HOPE-object visual vocabulary. Supply only
    # an appearance-level catalog hint (never coordinates or simulator state)
    # so the benchmark noun can be grounded in the rendered scene.
    task_visual_hints={
        'pick up the alphabet soup and place it in the basket':
        ('The alphabet soup is the blue-and-orange cylindrical can with '
         'large yellow ALPHABET SOUP lettering. It is not the red-and-green '
         'tomato sauce can.'),
    },
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_object',
    task_ids=[0],
    model_family='gpt6-astra',
    eval_chunk_size=10,
    resize_size=512,
    num_trials_per_task=1,
    num_steps_wait=10,
    # Fifty ten-step GPT action chunks plus settling/reset headroom.
    max_steps=550,
    seed=7,
    dataset_stats_path=None,
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
    result_output_dir='work_dirs/gpt6_astra_libero',
)
