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
"""HUD04 kernel inference through OliRTCInferenceRunner and OliOperator.

This keeps Oli's prompt-ID/execution-count interaction and adds asynchronous
RTC prefix conditioning. One requested execution means one generated action
chunk; overlapping RTC prefix steps are not executed twice.
"""

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

_base_globals = {}
_base_candidates = [
    _Path('configs/gr00t/gr00t_hud04_rtc_no_done_oli_kernel_inference.py'),
    _Path('/workspace/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_no_done_oli_kernel_inference.py'),
]
_base_path = next(path for path in _base_candidates if path.exists())
exec(compile(_base_path.read_text(), str(_base_path), 'exec'), _base_globals)

model = _deepcopy(_base_globals['model'])
train_dataloader = _deepcopy(_base_globals['train_dataloader'])
runner = _deepcopy(_base_globals['runner'])
inference_model = _deepcopy(_base_globals['inference_model'])
inference = _deepcopy(_base_globals['inference'])

_task_descriptions = dict(inference['task_descriptions'])
if '0' in _task_descriptions:
    if '1' in _task_descriptions:
        raise ValueError('Cannot reserve prompt 0: prompt 1 already exists')
    _task_descriptions['1'] = _task_descriptions.pop('0')
inference['task_descriptions'] = _task_descriptions

# Captured from /joint/state at 2026-07-22 09:03 CST using the median of 20
# samples (maximum per-joint standard deviation: 4.96e-5 rad). The final two
# values are the data-pipeline hand-closed flags; both hands were open.
_initial_state = [
    -0.138400,
    -0.021600,
    -0.046600,
    0.123400,
    -0.057983,
    -0.000304,
    -0.083100,
    -0.082700,
    -0.138400,
    0.131600,
    -0.093793,
    0.081237,
    -0.001800,
    -0.009472,
    -0.090990,
    -0.077700,
    0.177800,
    0.124100,
    0.239197,
    -0.315030,
    -0.380800,
    0.152800,
    0.192900,
    0.055200,
    0.520800,
    -0.165896,
    -0.092670,
    -0.668100,
    -0.232800,
    -0.143500,
    0.066000,
    0.0,
    0.0,
]

inference.update(
    type='OliRTCInferenceRunner',
    default_prompt_id='1',
    zero_prompt_resets=True,
    initial_state=_initial_state,
    reset_duration_sec=5.0,
    async_remaining_actions_threshold=8,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=7,
    ),
)

del (_base_candidates, _base_globals, _base_path, _deepcopy, _initial_state,
     _task_descriptions, _Path)
