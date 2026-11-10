# Copyright 2026 Limx Dynamics
"""Shared policy lifecycle for evaluation and synchronous inference runners."""

from collections.abc import Mapping
from typing import Dict, Optional

from fluxvla.engines.utils import (build_observation_adapter_from_cfg,
                                   build_policy_from_cfg)


def _copy_options(value):
    """Copy config containers, not already loaded planner/model objects."""
    if isinstance(value, Mapping):
        return {key: _copy_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_options(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_options(item) for item in value)
    return value


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

    Concrete runners only supply input/output processors and execute native
    actions. Configuration is accepted by BaseEvalRunner/BaseInferenceRunner,
    not repeated by every environment/robot subclass. A legacy runner can
    leave policy support opt-in while migrating its old inference path.
    """

    def configure_policy(self,
                         policy: Optional[Dict] = None,
                         observation_adapter: Optional[Dict] = None):
        self.policy_config = _copy_options(policy)
        self.observation_adapter_config = _copy_options(observation_adapter)
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
