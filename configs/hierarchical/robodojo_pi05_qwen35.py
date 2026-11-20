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
"""Local subtask-trained Qwen3.5 + PI05 with unmodified Themis main.

Only the planner backend changes; the PI05 checkpoint, original training
candidate vocabulary and full benchmark schedule are inherited unchanged.
Both models run in the VLA server process. No GPT endpoint or API key is used.
Use the matching launch.json entries for four tasks x five episodes; output
paths remain launch-time arguments, not checkpoint-relative constants.
"""

_base_ = ['./robodojo_pi05_gpt6.py']

from os import environ as _environ  # noqa: E402

_planner_dir = _environ.get(
    'ROBODOJO_VLM_PLANNER_DIR',
    '/mnt/data/oss/users/liyinhao/projects/FluxVLA/work_dirs/'
    'qwen35_4b_robodojo_subtask_d78c5a778_4node_bs64_old_'
    '20261116_dlcf6osrr2ymq9yd')

policy = dict(
    planner=dict(
        # Do not merge GPT model/base_url/retry options into the local backend.
        _delete_=True,
        type='LocalVLMSubtaskPlanner',
        checkpoint_path=_environ.get('ROBODOJO_VLM_PLANNER_CHECKPOINT')
        or f'{_planner_dir}/checkpoints/'
        'step-007311-epoch-03-loss=0.0678.safetensors',
        assets_path=_environ.get('ROBODOJO_VLM_PLANNER_ASSETS')
        or f'{_planner_dir}/vlm_assets',
        # None selects the current model-worker device, not another worker.
        device=_environ.get('ROBODOJO_VLM_PLANNER_DEVICE') or None,
        torch_dtype='bf16',
        attn_implementation='sdpa',
        processor_type='qwen3_5',
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        image_max_pixels=262144,
        max_input_tokens=4096,
        max_new_tokens=256,
        constrained_decoding=True,
        completion_mode='disabled'))

del _environ, _planner_dir
