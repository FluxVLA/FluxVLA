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
"""Episode-scoped harness composition; runners only consume policy outputs."""

from collections.abc import Mapping

from fluxvla.engines.utils.root import POLICIES
from fluxvla.harness import AgentHarness
from .hierarchical_policy import HierarchicalPolicy
from .vla_policy import PolicyOutput


class _HarnessPlanner:

    def __init__(self, planner, harness):
        self.planner, self.harness = planner, harness
        self.completion_mode = 'disabled'

    def reset(self):
        self.planner.reset()

    def choose(self, **request):
        return self.harness.choose(self.planner, **request)


@POLICIES.register_module()
class AgentHarnessPolicy(HierarchicalPolicy):
    """Literal skills and execution memory, without a success oracle.

    API planners receive bounded execution memory; existing subtask-only VLMs
    keep their original image+goal prompt. Tensor preprocessing and action
    postprocessing remain the original HierarchicalPolicy/VLA path. Native
    sync and RTC runners retain their existing execution barriers and ACKs.
    """

    def __init__(self, harness=None, **kwargs):
        if (kwargs.get('completion_mode') != 'disabled'
                or kwargs.get('subtask_source') is None):
            raise ValueError('AgentHarnessPolicy requires literal subtasks '
                             'and completion_mode=disabled')
        if harness is not None and not isinstance(harness, Mapping):
            raise ValueError('harness must be a settings mapping')
        super().__init__(**kwargs)
        self.harness = AgentHarness(
            image_keys=self.controller.planner_image_keys,
            **dict(harness or {}))
        self.controller.planner = _HarnessPlanner(self.controller.planner,
                                                  self.harness)

    @property
    def supports_per_request_inference(self):
        return False

    def set_episode_context(self, episode_id):
        self.harness.memory.bind_episode(episode_id)

    def reset(self, goal, *, output_processor=None):
        super().reset(goal, output_processor=output_processor)
        self.harness.reset(goal)

    def step(self, observation, observation_id, *, policy_seed=None):
        self._check_ready()
        self.harness.observe(observation, observation_id)
        if self.harness.should_stop:
            event = self.controller.finish('failed',
                                           'harness_no_observed_change')
            self.harness.memory.record('stopped', reason=event['reason'])
            return PolicyOutput(
                status='failed',
                metadata=dict(planning=event, harness=self.harness.snapshot()))
        try:
            result = super().step(
                observation, observation_id, policy_seed=policy_seed)
        except Exception as exc:
            # Persist only a type: provider errors may contain credentials.
            self.harness.memory.record(
                'inference_error', error_type=type(exc).__name__)
            raise
        if result.status == 'ready':
            self.harness.issue(self.controller.active.describe(),
                               self.controller.version, len(result.actions),
                               observation_id)
        else:
            self.harness.memory.record('policy_status', status=result.status)
        result.metadata['harness'] = self.harness.snapshot()
        return result

    def acknowledge(self, executed_steps):
        # Parent validates the real horizon/version before any memory changes.
        super().acknowledge(executed_steps)
        self.harness.acknowledge(executed_steps)
