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

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

_base_globals = {}
_base_candidates = [
    _Path('configs/gr00t/gr00t_hud04_rtc_no_done_full_finetune.py'),
    _Path('/workspace/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_no_done_full_finetune.py'),
    _Path('/home/limx/sober/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_no_done_full_finetune.py'),
]
_base_path = next(path for path in _base_candidates if path.exists())
exec(compile(_base_path.read_text(), str(_base_path), 'exec'), _base_globals)

WBT_DISCRETE_ACTION_DIMS = _base_globals['WBT_DISCRETE_ACTION_DIMS']
WBT_DISCRETE_STATE_DIMS = _base_globals['WBT_DISCRETE_STATE_DIMS']
WBT_DISCRETE_NORM_TYPE = _base_globals['WBT_DISCRETE_NORM_TYPE']
WBT_CONTINUOUS_NORM_TYPE = _base_globals['WBT_CONTINUOUS_NORM_TYPE']
WBT_NORM_KW = _deepcopy(_base_globals['WBT_NORM_KW'])
WBT_DENORM_KW = _deepcopy(_base_globals['WBT_DENORM_KW'])

model = _deepcopy(_base_globals['model'])
train_dataloader = _deepcopy(_base_globals['train_dataloader'])
runner = _deepcopy(_base_globals['runner'])
inference = _deepcopy(_base_globals['inference'])

# RTC specific configs
inference['publish_rate'] = 30
inference['async_remaining_actions_threshold'] = 8
inference['execute_horizon'] = 16
inference['target_hz'] = 50
inference['rtc_config']['prefix_len'] = 7

inference_model = _deepcopy(_base_globals['inference_model'])
inference_model['vlm_backbone']['type'] = 'EagleInferenceBackbone'
inference_model['vla_head']['type'] = 'ResidualFlowMatchingInferenceHead'
# v3 distilled residual: bounded continuous-velocity correction added on
# top of the frozen direct-2-step base. Must match the trained head.
inference_model['vla_head']['residual_hidden_dim'] = 256
inference_model['vla_head']['residual_max_abs'] = 0.1
inference_model['vla_head']['continuous_action_dim'] = 40
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
# Adjust denoise step for faster inference, setting num_inference_timesteps to 2.
# Compare to Origin num_inference_timesteps is 4.
inference_model['vla_head']['num_inference_timesteps'] = 2

del _base_candidates, _base_globals, _base_path, _deepcopy, _Path
