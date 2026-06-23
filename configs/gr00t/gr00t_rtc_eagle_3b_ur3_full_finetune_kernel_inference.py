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

_base_ = './gr00t_eagle_3b_ur3_rtc_kernel_inference.py'

inference = dict(
    ckpt_path='/home/ur3/sober/ur_checkpoint/gr00t_rtc_eagle_3b_ur3_full_finetune_orin/checkpoints/step-006000-epoch-00-loss=0.0946.safetensors',
)
