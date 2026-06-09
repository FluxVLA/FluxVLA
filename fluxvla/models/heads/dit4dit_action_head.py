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
#
# Adapted from Mondo-Robotics/DiT4DiT:
# DiT4DiT/model/modules/action_model/ActionDiT.py

from __future__ import annotations
from functools import partial
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.distributions import Beta

from fluxvla.engines import HEADS
from fluxvla.models.blocks.cross_attention_dit import (BasicTransformerBlock,
                                                       DiT)
from fluxvla.models.heads.flow_matching_head import (
    SinusoidalPositionalEncoding, swish)

DIT_CONFIGS = {
    'DiT-B': {
        'attention_head_dim': 64,
        'num_attention_heads': 12,
    },
    'DiT-L': {
        'attention_head_dim': 48,
        'num_attention_heads': 32,
    },
}


class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        bsz, horizon, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == bsz:
            timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        elif timesteps.dim() != 2 or timesteps.shape != (bsz, horizon):
            raise ValueError(
                'Expected `timesteps` to have shape [B] or [B, T], got '
                f'{tuple(timesteps.shape)}.')

        action_emb = self.layer1(actions)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        hidden = torch.cat([action_emb, time_emb], dim=-1)
        hidden = swish(self.layer2(hidden))
        return self.layer3(hidden)


@HEADS.register_module()
class DiT4DiTActionHead(nn.Module):
    """DiT4DiT flow-matching action head.

    This head consumes Cosmos hidden tokens directly:
    ``vl_embs: Tensor[B, S, D] -> actions: Tensor[B, T, action_dim]``.
    It intentionally omits the token-compression and multi-embodiment layers
    used by FluxVLA's generic ``FlowMatchingHead`` because DiT4DiT trains
    the action DiT against the selected Cosmos transformer layer features.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int = 2560,
        state_dim: int = 0,
        action_model_type: str = 'DiT-B',
        diffusion_model_cfg: Optional[Dict] = None,
        action_horizon: Optional[int] = None,
        future_action_window_size: Optional[int] = None,
        num_inference_timesteps: int = 4,
        add_pos_embed: bool = True,
        max_seq_len: int = 1024,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        ori_action_dim: Optional[int] = None,
        output_action_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if action_model_type not in DIT_CONFIGS:
            raise ValueError(
                f'Unsupported action_model_type={action_model_type}. '
                f'Available: {sorted(DIT_CONFIGS)}')

        if action_horizon is None:
            if future_action_window_size is None:
                raise ValueError(
                    '`action_horizon` or `future_action_window_size` is '
                    'required for DiT4DiTActionHead.')
            action_horizon = int(future_action_window_size) + 1

        diffusion_model_cfg = dict(diffusion_model_cfg or {})
        diffusion_model_cfg = {
            **DIT_CONFIGS[action_model_type],
            **diffusion_model_cfg,
        }

        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.state_dim = int(state_dim or 0)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.add_pos_embed = bool(add_pos_embed)
        self.num_timestep_buckets = int(num_timestep_buckets)
        self.noise_s = float(noise_s)
        self.ori_action_dim = output_action_dim or ori_action_dim

        self.model = DiT(**diffusion_model_cfg)
        self.input_embedding_dim = (
            self.model.config.num_attention_heads *
            self.model.config.attention_head_dim)

        self.state_encoder = (
            MLP(
                input_dim=self.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            ) if self.state_dim > 0 else None)
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len,
                                                   self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, 0.0, 0.02)

        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return BasicTransformerBlock

    def sample_time(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(
            device=device, dtype=dtype)
        return sample / self.noise_s

    def enable_gradient_checkpointing(self) -> None:
        fn = getattr(self.model, 'enable_gradient_checkpointing', None)
        if callable(fn):
            fn()

    def _prepare_state(self, state: Optional[torch.Tensor],
                       batch_size: int) -> Optional[torch.Tensor]:
        if self.state_encoder is None:
            return None
        if state is None:
            raise ValueError(
                'DiT4DiTActionHead was configured with state_dim > 0, '
                'but `state` is None.')
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(f'Expected state shape [B, D] or [B, N, D], got '
                             f'{tuple(state.shape)}.')
        if state.shape[0] != batch_size:
            raise ValueError(
                f'State batch {state.shape[0]} != action batch {batch_size}.')
        if state.shape[-1] < self.state_dim:
            state = F.pad(state, (0, self.state_dim - state.shape[-1]))
        elif state.shape[-1] > self.state_dim:
            state = state[..., :self.state_dim]
        return state

    def _add_position_embedding(self,
                                action_features: torch.Tensor) -> torch.Tensor:
        if not self.add_pos_embed:
            return action_features
        pos_ids = torch.arange(
            action_features.shape[1],
            dtype=torch.long,
            device=action_features.device)
        return action_features + self.position_embedding(pos_ids).unsqueeze(0)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        device = vl_embs.device
        actions = actions.to(device=device, dtype=vl_embs.dtype)
        action_mask = action_mask.to(device=device, dtype=vl_embs.dtype)

        noise = torch.randn_like(actions)
        t = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1.0 - t) * actions + t * noise
        velocity = noise - actions

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        action_features = self._add_position_embedding(action_features)

        state = self._prepare_state(state, actions.shape[0])
        state_features = (
            self.state_encoder(state.to(device=device, dtype=vl_embs.dtype))
            if state is not None else None)
        model_inputs = (
            torch.cat((state_features, action_features), dim=1)
            if state_features is not None else action_features)

        model_output = self.model(
            hidden_states=model_inputs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]
        loss = ((pred_actions.float() - velocity.float())**2 *
                action_mask.float())
        return loss.sum() / action_mask.float().sum().clamp_min(1.0)

    @torch.no_grad()
    def predict_action(self,
                       vl_embs: torch.Tensor,
                       state: Optional[torch.Tensor] = None,
                       **kwargs) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            dtype=vl_embs.dtype,
            device=device,
        )
        dt = 1.0 / self.num_inference_timesteps

        state = self._prepare_state(state, batch_size)
        state_features = (
            self.state_encoder(state.to(device=device, dtype=vl_embs.dtype))
            if state is not None else None)

        for step in range(self.num_inference_timesteps):
            t_cont = 1.0 - step / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((batch_size, ),
                                   t_discretized,
                                   dtype=torch.long,
                                   device=device)

            action_features = self.action_encoder(actions, timesteps)
            action_features = self._add_position_embedding(action_features)
            model_inputs = (
                torch.cat((state_features, action_features), dim=1)
                if state_features is not None else action_features)

            model_output = self.model(
                hidden_states=model_inputs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions - dt * pred_velocity

        if self.ori_action_dim is not None:
            actions = actions[..., :self.ori_action_dim]
        return actions

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(
            _module_wrap_policy,
            module_classes={self.transformer_layer_cls},
        )


__all__ = ['DiT4DiTActionHead']
