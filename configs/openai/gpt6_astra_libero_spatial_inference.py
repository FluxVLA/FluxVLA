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
"""Checkpoint-free GPT-6 Astra smoke evaluation on LIBERO-Spatial."""

_base_ = './gpt6_astra_libero_inference.py'

inference_model = dict(task_visual_hints=dict(_delete_=True))

eval = dict(
    task_suite_name='libero_spatial',
    task_ids=[0],
    output_dir='work_dirs/gpt6_astra_libero_spatial',
)
