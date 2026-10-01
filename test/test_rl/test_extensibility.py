"""Model declarations and flow mechanics must not assume either backbone."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from torch import nn

from fluxvla.rl.models.flow.policy import FlowPPOPolicyMixin
from fluxvla.rl.models.registry import (POLICY_SPECS, checkpoint_resolver_for,
                                        get_policy_spec)
from fluxvla.rl.rlinf_registry import build_registered_policy, register


def test_third_declaration_builds_without_dispatch_edits(
        model_cfg, monkeypatch):
    # Reuse a tiny native backbone under a third registration name. No edits
    # to builder/driver/worker dispatch should be necessary.
    spec = replace(get_policy_spec('fluxvla_pi05'), model_type='fluxvla_test')
    monkeypatch.setitem(POLICY_SPECS, spec.model_type, spec)
    model_cfg.model_type = spec.model_type
    built = build_registered_policy(model_cfg)
    assert built.model_action_horizon == 3
    seen = {}
    monkeypatch.setattr(
        'rlinf.models.register_model',
        lambda name, builder, **kwargs: seen.update({name: builder}))
    register()
    assert seen[spec.model_type] is build_registered_policy


def test_registration_metadata_is_lazy():
    code = '''
import sys
import fluxvla.rl.models.registry
import fluxvla.rl.rlinf_registry
assert 'rlinf' not in sys.modules
assert 'fluxvla.rl.models.pi05.policy' not in sys.modules
assert 'fluxvla.rl.models.smolvla.policy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True,
                            text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_model_and_wrong_adapter_rejected():
    with pytest.raises(ValueError, match='Unknown FluxVLA'):
        get_policy_spec('not_registered')
    with pytest.raises(ValueError, match='supports adapters'):
        get_policy_spec('fluxvla_smolvla').validate_adapter('robotwin')


def test_checkpoint_aliases_are_model_specific(policy):
    assert checkpoint_resolver_for(policy) is not None
    assert checkpoint_resolver_for(nn.Linear(2, 2)) is None


class ToyAdapter:

    def __call__(self, env_obs):
        return env_obs

    def env_actions(self, actions, action_chunk, action_dim, *, env_obs):
        return actions[:, :action_chunk, :action_dim]


class ToyNativeModel(nn.Module):
    """Deliberately no PI0.5/SmolVLA horizon or backbone attribute names."""

    def __init__(self):
        super().__init__()
        self.context_encoder = nn.Linear(5, 6)
        self.flow_expert = nn.Linear(4, 4)
        self.num_steps = 3

    @property
    def device(self):
        return self.flow_expert.weight.device

    def predict_action(self, noise, **obs):
        actions = noise.clone()
        for _ in range(self.num_steps):
            actions = actions - self.flow_expert(actions) / self.num_steps
        return actions


class ToyFlowPolicy(FlowPPOPolicyMixin, ToyNativeModel):
    model_action_horizon = 6
    model_action_dim = 4
    critic_hidden_size = 6
    rl_trainable_modules = ('flow_expert', )
    rl_frozen_modules = ('context_encoder', )

    def _prefix(self, obs):
        with torch.no_grad():
            hidden = self.context_encoder(obs['states'])[:, None]
        return hidden, torch.ones(hidden.shape[:2], dtype=torch.bool), None

    def _velocity(self, obs, mask, cache, latent, timestep):
        return self.flow_expert(latent).float()


def test_flow_contract_uses_model_neutral_dimensions():
    policy = ToyFlowPolicy()
    policy.configure_rl(ToyAdapter(), action_chunk=2, action_dim=3)
    obs = {'states': torch.randn(2, 5)}
    for key in ('images', 'img_masks', 'lang_tokens', 'lang_masks'):
        obs[key] = torch.zeros(2, 1)
    noise = torch.randn(2, 6, 4)
    actions, old = policy.predict_action_batch(obs, mode='train', noise=noise)
    assert actions.shape == (2, 2, 3)
    assert old['forward_inputs']['chains'].shape == (2, 4, 6, 4)
    policy.train()
    replay = policy(forward_inputs=old['forward_inputs'])
    torch.testing.assert_close(replay['logprobs'], old['prev_logprobs'])
    torch.testing.assert_close(replay['values'], old['prev_values'][:, 0])
    assert not policy.context_encoder.training
    assert not any(p.requires_grad
                   for p in policy.context_encoder.parameters())
    assert all(p.requires_grad for p in policy.flow_expert.parameters())
    replay['logprobs'].sum().backward()
    assert policy.flow_expert.weight.grad is not None
    actual, _ = policy.predict_action_batch(obs, mode='eval', noise=noise)
    expected = policy.predict_action(noise)[:, :2, :3]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('options', [
    {
        'action_chunk': 7
    },
    {
        'action_dim': 5
    },
    {
        'noise_level': 0
    },
    {
        'compute_dtype': torch.float16
    },
    {
        'rollout_micro_batch_size': 0
    },
])
def test_shared_configuration_validation(options):
    kwargs = dict(action_chunk=2, action_dim=3)
    kwargs.update(options)
    with pytest.raises(ValueError):
        ToyFlowPolicy().configure_rl(ToyAdapter(), **kwargs)
