"""Shared flow-SDE rollout and PPO replay contract for Flux flow policies."""

from contextlib import nullcontext

import torch
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead

from fluxvla.rl.models.flow.sampler import (gaussian_entropy, gaussian_logprob,
                                            timesteps, transition)


class FlowPPOPolicyMixin:
    """Flow-only PPO mechanics; native model keys are never renamed.

    Subclasses declare dimensions/module paths and implement _prefix,
    _velocity and native predict_action. Environment conversion stays in
    observation_adapter. Non-flow policies must implement their own replay.
    """

    rl_trainable_modules = ()
    rl_frozen_modules = ()

    @property
    def model_action_horizon(self):
        raise NotImplementedError

    @property
    def model_action_dim(self):
        raise NotImplementedError

    @property
    def critic_hidden_size(self):
        raise NotImplementedError

    def _configure_model_rl(self):
        """Optional model-specific numeric alignment before critic creation."""

    def configure_rl(self,
                     observation_adapter,
                     *,
                     action_chunk=10,
                     action_dim=7,
                     noise_level=0.5,
                     compute_dtype=torch.float32,
                     rollout_micro_batch_size=None,
                     value_hidden_sizes=(512, 128)):
        if not 0 < action_chunk <= self.model_action_horizon:
            raise ValueError('action_chunk must be within the model horizon')
        if not 0 < action_dim <= self.model_action_dim:
            raise ValueError('action_dim must be within the model dimension')
        if noise_level <= 0 or not torch.isfinite(torch.tensor(noise_level)):
            raise ValueError('PPO requires a finite, positive noise_level')
        if compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('compute_dtype must be FP32 or BF16')
        if (rollout_micro_batch_size is not None
                and rollout_micro_batch_size < 1):
            raise ValueError('rollout_micro_batch_size must be positive')
        timesteps(self.num_steps, 'cpu')
        self.observation_adapter = observation_adapter
        self.action_chunk = int(action_chunk)
        self.action_env_dim = int(action_dim)
        self.noise_level = float(noise_level)
        self.compute_dtype = compute_dtype
        self.rollout_micro_batch_size = rollout_micro_batch_size
        self._configure_model_rl()
        self.value_head = ValueHead(
            self.critic_hidden_size,
            hidden_sizes=value_hidden_sizes,
            activation='relu')
        self.requires_grad_(False)
        for name in (*self.rl_trainable_modules, 'value_head'):
            self.get_submodule(name).requires_grad_(True)
        for name in self.rl_frozen_modules:
            self.get_submodule(name).requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for name in self.rl_frozen_modules:
            # Native constructors may call eval before all modules exist.
            try:
                module = self.get_submodule(name)
            except AttributeError:
                continue
            module.eval()
        return self

    def _network_context(self):
        # FSDP2 preserves chain/image input tensors. Autocast is local to
        # network evaluation; probability arithmetic is outside this context.
        if self.compute_dtype == torch.bfloat16:
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _prepare_obs(self, env_obs):
        return {
            key: tensor.to(self.device).contiguous()
            for key, tensor in self.observation_adapter(env_obs).items()
        }

    def _value(self, hidden, mask):
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden *
                  weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        with self._network_context():
            return self.value_head(pooled)[:, 0].float()

    @torch.no_grad()
    def get_values(self, env_obs):
        """Bootstrap values without advancing the stochastic sampler's RNG."""
        micro = self.rollout_micro_batch_size
        if micro and len(env_obs['states']) > micro:
            return torch.cat([
                self.get_values({
                    key: None if value is None else value[start:start + micro]
                    for key, value in env_obs.items()
                }) for start in range(0, len(env_obs['states']), micro)
            ])
        hidden, mask, _ = self._prefix(self._prepare_obs(env_obs))
        return self._value(hidden, mask)[:, None]

    @torch.no_grad()
    def predict_action_batch(self,
                             env_obs,
                             mode='eval',
                             *,
                             noise=None,
                             rng=None,
                             **kwargs):
        if mode not in ('train', 'eval'):
            raise ValueError(f'Unknown rollout mode: {mode}')
        micro = self.rollout_micro_batch_size
        if mode == 'train' and micro and len(env_obs['states']) > micro:
            # BF16 GEMM/convolution reductions can depend on batch shape. Use
            # the actor's microbatch shape during training sampling too, while
            # leaving environment parallelism and paired ODE evaluation intact.
            parts = [
                self.predict_action_batch(
                    {
                        key:
                        None if value is None else value[start:start + micro]
                        for key, value in env_obs.items()
                    },
                    mode=mode,
                    noise=None if noise is None else noise[start:start +
                                                           micro],
                    rng=rng)
                for start in range(0, len(env_obs['states']), micro)
            ]
            actions = torch.cat([item[0] for item in parts])
            result = {
                key: torch.cat([item[1][key] for item in parts])
                for key in ('prev_logprobs', 'prev_values')
            }
            result['forward_inputs'] = {
                key:
                torch.cat([item[1]['forward_inputs'][key] for item in parts])
                for key in parts[0][1]['forward_inputs']
            }
            return actions, result
        obs = self._prepare_obs(env_obs)
        batch = obs['states'].shape[0]
        shape = (batch, self.model_action_horizon, self.model_action_dim)
        if noise is None:
            noise = torch.randn(
                shape, device=self.device, dtype=torch.float32, generator=rng)
        if tuple(noise.shape) != shape:
            raise ValueError(
                f'Expected noise shape {shape}, got {tuple(noise.shape)}')
        noise = noise.to(device=self.device, dtype=torch.float32).clone()
        hidden, mask, cache = self._prefix(obs)
        values = self._value(hidden, mask)[:, None]
        if mode == 'eval':
            with self._network_context():
                model_actions = super().predict_action(**obs, noise=noise)
            logprob = torch.zeros(
                (batch, self.action_chunk, self.action_env_dim),
                device=self.device,
                dtype=torch.float32)
            forward_inputs = dict(obs)
        else:
            chosen = int(
                torch.randint(
                    self.num_steps, (), device=self.device, generator=rng))
            indices = torch.full((batch, self.num_steps),
                                 chosen,
                                 dtype=torch.long,
                                 device=self.device)
            times = timesteps(self.num_steps, self.device)
            x = noise
            chain = [x.clone()]
            for index in range(self.num_steps):
                idx = torch.full((batch, ),
                                 index,
                                 device=self.device,
                                 dtype=torch.long)
                velocity = self._velocity(obs, mask, cache, x, times[idx])
                if index == chosen:
                    mean, std = transition(
                        x,
                        velocity,
                        idx,
                        num_steps=self.num_steps,
                        noise_level=self.noise_level)
                    x = mean + torch.randn(
                        shape, device=self.device, generator=rng) * std
                    logprob = gaussian_logprob(x, mean, std)
                else:
                    x = x + (times[index + 1] - times[index]) * velocity
                chain.append(x.clone())
            model_actions = x
            logprob = logprob[:, :self.action_chunk, :self.
                              action_env_dim].contiguous()
            forward_inputs = dict(
                obs, chains=torch.stack(chain, dim=1), denoise_inds=indices)
        actions = self.observation_adapter.env_actions(
            model_actions,
            self.action_chunk,
            self.action_env_dim,
            env_obs=env_obs).to(self.device)
        forward_inputs.update(
            action=actions.reshape(batch, -1),
            model_action=model_actions.reshape(batch, -1))
        return actions, {
            'prev_logprobs': logprob.float(),
            'prev_values': values.float(),
            'forward_inputs': {
                key: value.detach().clone().contiguous()
                for key, value in forward_inputs.items()
            }
        }

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                'FluxVLA RL v1 supports PPO default forward only')
        return self.default_forward(**kwargs)

    def default_forward(self,
                        forward_inputs,
                        compute_logprobs=True,
                        compute_values=True,
                        compute_entropy=False,
                        **kwargs):
        obs = {
            key: forward_inputs[key]
            for key in ('images', 'img_masks', 'lang_tokens', 'lang_masks',
                        'states')
        }
        hidden, mask, cache = self._prefix(obs)
        batch_size = obs['states'].shape[0]
        device = obs['states'].device
        logprob = torch.zeros(
            (batch_size, self.action_chunk, self.action_env_dim),
            device=device)
        entropy = torch.zeros((batch_size, 1), device=device)
        if compute_logprobs or compute_entropy:
            chain = forward_inputs['chains']
            # Reject stored samples that the backend downcast before this call.
            if chain.dtype != torch.float32:
                raise ValueError('chains must stay FP32; '
                                 'use FSDP2 cast_forward_inputs=False')
            indices = forward_inputs['denoise_inds'][:, 0].long()
            batch = torch.arange(batch_size, device=device)
            before, after = chain[batch, indices], chain[batch, indices + 1]
            velocity = self._velocity(
                obs, mask, cache, before,
                timesteps(self.num_steps, device)[indices])
            mean, std = transition(
                before,
                velocity,
                indices,
                num_steps=self.num_steps,
                noise_level=self.noise_level)
            if compute_logprobs:
                logprob = gaussian_logprob(
                    after, mean,
                    std)[:, :self.action_chunk, :self.action_env_dim]
            if compute_entropy:
                entropy = (
                    gaussian_entropy(std)
                    [:, :self.action_chunk, :self.action_env_dim].mean(
                        dim=(1, 2))[:, None])
        values = self._value(hidden, mask) if compute_values else torch.zeros(
            batch_size, device=device)
        return {
            'logprobs': logprob.contiguous(),
            'values': values,
            'entropy': entropy
        }
