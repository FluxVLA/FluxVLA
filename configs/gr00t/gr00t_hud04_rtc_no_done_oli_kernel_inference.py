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
"""HUD04 kernel inference through OliInferenceRunner and OliOperator."""

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

_base_globals = {}
_base_candidates = [
    _Path('configs/gr00t/gr00t_hud04_rtc_no_done_oli_full_finetune.py'),
    _Path('/workspace/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_no_done_oli_full_finetune.py'),
]
_base_path = next(path for path in _base_candidates if path.exists())
exec(compile(_base_path.read_text(), str(_base_path), 'exec'), _base_globals)

model = _deepcopy(_base_globals['model'])
train_dataloader = _deepcopy(_base_globals['train_dataloader'])
runner = _deepcopy(_base_globals['runner'])
inference_model = _deepcopy(_base_globals['inference_model'])
inference = _deepcopy(_base_globals['inference'])

inference_model['vlm_backbone']['type'] = 'EagleInferenceBackbone'
inference_model['vla_head']['type'] = 'FlowMatchingInferenceHead'
inference_model['vla_head']['max_input_seq_len'] = 580
inference_model['vla_head']['diffusion_model_cfg'] = dict(
    attention_head_dim=48,
    cross_attention_dim=2048,
    dropout=0.2,
    final_dropout=True,
    interleave_self_attention=True,
    norm_type='ada_norm',
    num_attention_heads=32,
    num_layers=16,
    output_dim=1024,
    positional_embeddings=None,
)
inference_model['vla_head']['num_inference_timesteps'] = 2

# Remove controls owned by the legacy Teleop02 RTC runner.  The accelerated
# inference head is retained, while execution is intentionally synchronous and
# selected through prompt ID + execution count.
for _key in (
        'use_done_state_machine',
        'async_execution',
        'async_remaining_actions_threshold',
        'target_hz',
        'interpolation_method',
        'rtc_config',
):
    inference.pop(_key, None)

inference.update(
    type='OliInferenceRunner',
    interactive=True,
    default_prompt_id='0',
    default_execution_count=1,
    execute_horizon=16,
    publish_rate=30,
    camera_names=['head', 'left_wrist'],
    apply_jpeg_compression=True,
)
inference['operator'] = dict(
    type='OliOperator',
    control_backend='mros',
    head_rgb_topic='/head/color/image_raw/compressed',
    left_wrist_rgb_topic='/left_wrist_camera/color/image_raw/compressed',
    joint_state_topic='/joint/state',
    finger_state_topic='/brainco1/hand/state',
    finger_cmd_topic='/brainco1/hand/cmd',
    teleop_wbt_topic='/teleop_cmd_WBT',
    finger_force_levels=(2.0, 2.0),
)

del _base_candidates, _base_globals, _base_path, _deepcopy, _key, _Path
