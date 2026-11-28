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
"""Episode-memory harness + existing subtask Qwen3.5 + PI05 checkpoints.

Full scheduling is inherited; launch.json selects small debug batches.
Requires a Themis client supporting send_policy_context and one model worker.
Local Qwen keeps its SFT image+goal prompt and cannot declare task completion.
Output/trace locations are supplied by launch arguments, not checkpoint paths.
"""

_base_ = ['./robodojo_pi05_qwen35.py']

policy = dict(
    type='AgentHarnessPolicy',
    harness=dict(
        min_subtask_chunks=2,
        switch_confirmations=2,
        max_hold_chunks=8,
        no_change_patience=4,
        max_stall_replans=2,
        # Visual change is NOT task progress; calibrate before enabling stops.
        stop_on_stall=False,
        keyframe_interval_chunks=8,
        observer=dict(threshold=0.01, thumbnail_size=32),
        memory=dict(
            recent_events=8,
            summary_subtasks=32,
            recent_keyframes=8,
            saved_keyframes=64,
            trace_dir=None,
            trace_fsync=True)))

themis = dict(
    evaluation=dict(policy_client=dict(send_policy_context=True)),
    server=dict(mode='stateful', runner=dict(policy_execution='episode')))
