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
"""GPT candidate planner + episode memory + unchanged PI05 action inputs.

Retains the GPT base's annotation coverage checks and candidate limits. Unknown
goals still fail explicitly; the harness does not invent absent robot skills.
"""

_base_ = ['./robodojo_pi05_gpt6.py']

policy = dict(
    type='AgentHarnessPolicy',
    harness=dict(
        min_subtask_chunks=2,
        switch_confirmations=2,
        max_hold_chunks=8,
        no_change_patience=4,
        max_stall_replans=2,
        stop_on_stall=False,
        keyframe_interval_chunks=8,
        memory=dict(trace_dir=None)))

themis = dict(
    evaluation=dict(policy_client=dict(send_policy_context=True)),
    server=dict(mode='stateful', runner=dict(policy_execution='episode')))
