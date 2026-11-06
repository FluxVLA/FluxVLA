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

_base_ = './xvla_libero_full_finetune_base.py'

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            data_root_path=('./datasets/libero_xvla_lerobot/'
                            'libero_goal_no_noops_lerobot'), ), ), )

eval = dict(task_suite_name='libero_goal')
