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
"""Action predictors: one execution API with or without a planner.

Configuration and composition belong here, not in an environment's rollout
loop. Model training continues to use the original tensor-based VLA API.
"""

import hashlib
import json
from collections.abc import Mapping

POLICY_UNSET = object()


def copy_options(value):
    if isinstance(value, Mapping):
        return {key: copy_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_options(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_options(item) for item in value)
    return value


def _config_value(config, key):
    if isinstance(config, Mapping):
        return config.get(key, POLICY_UNSET)
    return getattr(config, key, POLICY_UNSET)


def resolve_policy_options(cfg=None, section=None, overrides=None):
    """Explicit options > section.runner > section > root; None disables."""
    if section not in (None, 'eval', 'inference'):
        raise ValueError('Policy config section must be eval or inference')
    section_cfg = _config_value(cfg, section) if section else None
    layers = (cfg, section_cfg, _config_value(section_cfg,
                                              'runner'), overrides)
    result = {}
    for key in ('policy', 'observation_adapter'):
        value = None
        for layer in layers:
            candidate = _config_value(layer, key)
            if candidate is not POLICY_UNSET:
                value = candidate
        result[key] = copy_options(value)
    return result


def predictor_fingerprint(options):
    values = (options.get('policy'), options.get('observation_adapter'))
    if values == (None, None):
        return None
    payload = json.dumps(values, sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()


class ActionPredictor:
    """Raw observation -> native actions; runners never inspect the backend."""

    def __init__(self, backend, io, adapter=None, fingerprint=None):
        self.backend = backend
        self.io = io
        self.adapter = adapter
        self.fingerprint = fingerprint
        self.last_output = None

    @property
    def replay_image(self):
        return getattr(self.io, 'replay_image', None)

    def reset(self, goal, **context):
        self.last_output = None
        self.io.reset(goal, **context)
        self.backend.unnorm_key = self.io.unnorm_key
        self.backend.reset(goal)

    def predict_action(self,
                       observation,
                       observation_id,
                       *,
                       environment=None,
                       policy_seed=None):
        self.io.before_predict(environment)
        if self.adapter is not None:
            observation = self.adapter(observation, environment)
        self.last_output = self.backend.step(
            observation, observation_id, policy_seed=policy_seed)
        return self.last_output

    def observe(self,
                observation,
                observation_id,
                *,
                action_index=None,
                terminal=False):
        return self.io.observe(
            observation,
            observation_id,
            action_index=action_index,
            terminal=terminal)

    def acknowledge(self, executed_steps):
        self.backend.acknowledge(executed_steps)

    def close(self):
        self.backend = None
        self.adapter = None
        self.io = None
        self.last_output = None


def build_action_predictor(io, *, policy=None, observation_adapter=None):
    """Assemble once, outside the runner; a plain VLA uses the same wrapper."""
    from fluxvla.engines.utils.builder import (
        build_observation_adapter_from_cfg, build_policy_from_cfg)

    options = dict(policy=policy, observation_adapter=observation_adapter)
    backend = build_policy_from_cfg(
        policy if policy is not None else dict(type='VLAPolicy'),
        default_args=dict(
            vla=io.vla,
            input_processor=io,
            output_processor=io.process_actions,
            unnorm_key=io.unnorm_key,
            mixed_precision_dtype=io.mixed_precision_dtype,
            enable_mixed_precision=io.enable_mixed_precision))
    adapter = (
        build_observation_adapter_from_cfg(observation_adapter)
        if observation_adapter is not None else None)
    return ActionPredictor(backend, io, adapter,
                           predictor_fingerprint(options))


class DeferredPredictor:
    """Delay GPU construction until the simulator worker selects its device."""

    def __init__(self, factory, fingerprint=None):
        self.factory = factory
        self.fingerprint = fingerprint
        self.instance = None

    def prepare(self, device=None):
        if self.instance is None:
            self.instance = self.factory(device)
        return self.instance

    def close(self):
        if self.instance is not None:
            self.instance.close()
            self.instance = None
