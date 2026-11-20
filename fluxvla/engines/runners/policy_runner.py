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
"""Compatibility policy lifecycle for synchronous deployment and serving."""

from typing import Dict, Optional

from fluxvla.engines.utils import (build_observation_adapter_from_cfg,
                                   build_policy_from_cfg)
# Compatibility exports for synchronous/serving callers. Native evaluation
# constructs predictors through the external evaluation factory instead.
from fluxvla.policies.runtime import POLICY_UNSET, resolve_policy_options


class PolicyModelAdapter:
    """Reuse an executor's model hooks without teaching a policy about ROS."""

    def __init__(self, predict_action, reset_inference_history=None):
        self.predict_action = predict_action
        self._reset = reset_inference_history

    def reset_inference_history(self):
        if self._reset is not None:
            self._reset()


class PolicyRunnerMixin:
    """One lifecycle, independent of environment and action representation.

    Native evaluation uses externally assembled ActionPredictor instances.
    This mixin preserves synchronous deployment and serving compatibility;
    robot subclasses need not repeat configuration resolution.
    """

    def configure_policy(self,
                         policy: Optional[Dict] = POLICY_UNSET,
                         observation_adapter: Optional[Dict] = POLICY_UNSET,
                         *,
                         cfg=None,
                         section=None):
        options = resolve_policy_options(
            cfg,
            section,
            overrides=dict(
                policy=policy, observation_adapter=observation_adapter))
        self.policy_config = options['policy']
        self.observation_adapter_config = options['observation_adapter']
        self.policy = None
        self.observation_adapter = None
        self.last_policy_output = None

    @property
    def policy_requested(self):
        return (self.policy_config is not None
                or self.observation_adapter_config is not None)

    def setup_policy(self, *, default=False, **components):
        """Build once, after checkpoint/device/input processor setup."""
        if self.policy is not None:
            return self.policy
        if not self.policy_requested and not default:
            return None
        self.policy = build_policy_from_cfg(
            self.policy_config if self.policy_config is not None else dict(
                type='VLAPolicy'),
            default_args=components)
        self.observation_adapter = (
            build_observation_adapter_from_cfg(self.observation_adapter_config)
            if self.observation_adapter_config is not None else None)
        return self.policy

    def reset_policy(self, goal, **kwargs):
        self.last_policy_output = None
        if self.policy is not None:
            self.policy.reset(goal, **kwargs)

    def step_policy(self,
                    observation,
                    observation_id,
                    *,
                    environment=None,
                    policy_seed=None):
        if self.policy is None:
            raise RuntimeError('Set up the policy before requesting actions')
        if self.observation_adapter is not None:
            observation = self.observation_adapter(observation, environment)
        self.last_policy_output = self.policy.step(
            observation, observation_id, policy_seed=policy_seed)
        return self.last_policy_output

    def acknowledge_policy(self, executed_steps):
        self.policy.acknowledge(executed_steps)

    def release_policy(self):
        """Drop references to loaded models when a runner is cleaned up."""
        self.policy = None
        self.observation_adapter = None
        self.last_policy_output = None
