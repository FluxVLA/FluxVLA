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
"""Local PI0.5 RTC inference config for a dual-arm reBot."""

_base_ = './pi05_rebot_dual_inference.py'

inference = dict(
    type='RebotDualRTCInferenceRunner',
    async_execution=True,
    action_chunk=50,
    execute_horizon=10,
    rtc_config=dict(
        enabled=True,
        method='guidance',
        prefix_len=None,
        decay_end=10,
        schedule='exp',
        max_guidance_weight=5.0,
        use_vjp=False))
