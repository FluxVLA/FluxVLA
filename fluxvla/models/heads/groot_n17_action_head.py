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
"""FluxVLA-native GR00T N1.7 action head."""

from __future__ import annotations
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from fluxvla.engines import HEADS
from fluxvla.engines.utils.fsdp_wrapping import build_module_wrap_policy
from fluxvla.models.blocks import cross_attention_dit
from fluxvla.models.heads.flow_matching_head import (
    CategorySpecificMLP, MultiEmbodimentActionEncoder)


class AlternateVLDiT(cross_attention_dit.DiT):
    """N1.7-specific DiT with alternating text/image cross-attention."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ):
        del encoder_attention_mask
        if image_mask is None or backbone_attention_mask is None:
            raise ValueError('AlternateVLDiT requires image and backbone '
                             'attention masks.')

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        image_attention_mask = image_mask & backbone_attention_mask
        text_attention_mask = (~image_mask) & backbone_attention_mask
        all_hidden_states = [hidden_states]

        if not self.config.interleave_self_attention:
            raise ValueError('AlternateVLDiT requires interleaved self '
                             'attention blocks.')

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                attention_mask = (
                    text_attention_mask if idx %
                    (2 * self.attend_text_every_n_blocks) == 0 else
                    image_attention_mask)
                # FluxVLA's shared BasicTransformerBlock consumes the mask
                # through ``attention_mask``. Keep that model-specific detail
                # here instead of changing the shared DiT implementation.
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) +
            shift[:, None])
        output = self.proj_out_2(hidden_states)
        if return_all_hidden_states:
            return output, all_hidden_states
        return output


@HEADS.register_module()
class GrootN17ActionHead(nn.Module):
    """Native equivalent of official ``Gr00tN1d7ActionHead``."""

    supports_gradient_checkpointing = True

    def __init__(self, config, **config_overrides):
        super().__init__()
        if config_overrides:
            config_dict = (
                dict(config) if isinstance(config, dict) else vars(config))
            valid_overrides = {
                key: value
                for key, value in config_overrides.items() if value is not None
            }
            config_dict.update(valid_overrides)
            config = SimpleNamespace(**config_dict)
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        diffusion_model_cfg = dict(config.diffusion_model_cfg)
        diffusion_model_cfg['cross_attention_dim'] = (
            config.backbone_embedding_dim)
        if config.use_alternate_vl_dit:
            diffusion_model_cfg['attend_text_every_n_blocks'] = (
                config.attend_text_every_n_blocks)
            self.model = AlternateVLDiT(**diffusion_model_cfg)
        else:
            self.model = cross_attention_dit.DiT(**diffusion_model_cfg)

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.num_timestep_buckets = config.num_timestep_buckets

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln else nn.Identity())
        self.vl_self_attention = (
            cross_attention_dit.SelfAttentionTransformer(
                **config.vl_self_attention_cfg)
            if config.use_vlln else nn.Identity())
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len,
                                                   self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # The head can be constructed inside a meta-device context when an
        # evaluation checkpoint is materialized. Beta is not an nn.Module, so
        # keep its scalar parameters explicitly on CPU and move samples later.
        self.beta_dist = Beta(
            torch.tensor(config.noise_beta_alpha, device='cpu'),
            torch.tensor(config.noise_beta_beta, device='cpu'),
        )
        self.state_dropout_prob = config.state_dropout_prob
        self.rtc_training_config = getattr(config, 'rtc_training_config', None)

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return the wrapping policy for N1.7 action-head modules."""
        return build_module_wrap_policy({
            cross_attention_dit.SelfAttentionTransformer,
            cross_attention_dit.DiT,
        })

    @staticmethod
    def _sample_initial_actions(size, dtype, device, seed: int | None = None):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        return torch.randn(
            size=size,
            dtype=dtype,
            device=device,
            generator=generator,
        )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_vl_features(self,
                            input_features: torch.Tensor) -> torch.Tensor:
        input_features = self.vlln(input_features)
        return self.vl_self_attention(input_features)

    def encode_state_features(
        self,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert states.shape[1] == self.config.state_history_length
        states = states.view(states.shape[0], 1, -1)
        return self.state_encoder(states, embodiment_ids)

    def encode_features(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vl_embeds = self.process_vl_features(input_features)
        state_features = self.encode_state_features(states, embodiment_ids)
        return vl_embeds, state_features

    def forward(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del sample_weight
        vl_embeds, state_features = self.encode_features(
            input_features, states, embodiment_ids)
        device = vl_embeds.device

        if self.training and self.state_dropout_prob > 0:
            dropout_sample = torch.rand(
                state_features.shape[0], device=state_features.device)
            do_dropout = dropout_sample < self.state_dropout_prob
            do_dropout = do_dropout[:, None,
                                    None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        noise = torch.randn(
            actions.shape, device=actions.device, dtype=actions.dtype)
        t_scalar = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype)
        action_horizon = actions.shape[1]
        if (self.rtc_training_config
                and self.rtc_training_config.get('enabled', False)):
            from fluxvla.engines.utils.rtc_training import (
                apply_rtc_time_conditioning, sample_training_delay)
            delays = sample_training_delay(
                batch_size=actions.shape[0],
                max_delay=self.rtc_training_config.get('max_delay', 5),
                distribution=self.rtc_training_config.get(
                    'distribution', 'exponential'),
                temperature=self.rtc_training_config.get('temperature', 1.0),
                device=actions.device,
            )
            action_time, action_masks = apply_rtc_time_conditioning(
                t_scalar,
                action_masks,
                delays,
                action_horizon,
            )
        else:
            action_time = t_scalar[:, None].expand(-1, action_horizon)
        noisy_trajectory = ((1 - action_time.unsqueeze(-1)) * noise +
                            action_time.unsqueeze(-1) * actions)
        velocity = actions - noise

        action_timestep = (action_time * self.num_timestep_buckets).long()
        global_timestep = (t_scalar * self.num_timestep_buckets).long()
        action_features = self.action_encoder(
            noisy_trajectory,
            action_timestep,
            embodiment_ids,
        )
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = attention_mask
        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=global_timestep,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=global_timestep,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_ids)
        pred_actions = pred[:, -actions.shape[1]:]
        action_loss = (
            F.mse_loss(pred_actions, velocity, reduction='none') *
            action_masks)
        loss = action_loss.sum() / (action_masks.sum() + 1e-6)
        return {
            'loss': loss,
            'action_loss': action_loss,
            'action_mask': action_masks,
            'backbone_features': vl_embeds,
            'state_features': state_features,
        }

    @torch.no_grad()
    def get_action_from_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,
        prefix_len: int = 0,
        rtc_config: dict | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = self._sample_initial_actions(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
            seed=seed,
        )
        dt = 1.0 / self.num_inference_timesteps

        use_prefix_rtc = (
            prev_actions is not None and prefix_len > 0
            and rtc_config is not None)
        if use_prefix_rtc:
            method = rtc_config.get('method', 'prefix')
            if method != 'prefix':
                raise NotImplementedError(
                    'GrootN17ActionHead currently supports only RTC prefix '
                    f'conditioning, got method={method!r}.')
            if prev_actions.ndim == 2:
                prev_actions = prev_actions.unsqueeze(0)
            if prev_actions.ndim != 3:
                raise ValueError('RTC prev_actions must have shape [B, T, D], '
                                 f'got {tuple(prev_actions.shape)}.')
            if prev_actions.shape[0] != batch_size:
                raise ValueError('RTC prev_actions batch size does not match '
                                 f'input batch: {prev_actions.shape[0]} != '
                                 f'{batch_size}.')
            if prev_actions.shape[-1] < self.action_dim:
                prev_actions = F.pad(
                    prev_actions,
                    (0, self.action_dim - prev_actions.shape[-1]),
                    value=0.0,
                )
            elif prev_actions.shape[-1] > self.action_dim:
                prev_actions = prev_actions[..., :self.action_dim]
            prev_actions = prev_actions.to(device=device, dtype=actions.dtype)
            if prefix_len > self.action_horizon:
                raise ValueError(
                    f'RTC prefix_len={prefix_len} exceeds action horizon '
                    f'{self.action_horizon}.')
            if prefix_len > prev_actions.shape[1]:
                raise ValueError(
                    f'RTC prefix_len={prefix_len} exceeds prev_actions '
                    f'horizon {prev_actions.shape[1]}.')

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size, ),
                fill_value=t_discretized,
                device=device,
            )
            action_timesteps = timesteps_tensor
            if use_prefix_rtc:
                actions[:, :prefix_len] = prev_actions[:, :prefix_len]
                action_timesteps = torch.full(
                    (batch_size, self.action_horizon),
                    fill_value=t_discretized,
                    dtype=torch.long,
                    device=device,
                )
                # RTC training marks the already-known prefix as fully
                # denoised with the terminal action-encoder timestep.
                action_timesteps[:, :prefix_len] = self.num_timestep_buckets
            action_features = self.action_encoder(actions, action_timesteps,
                                                  embodiment_ids)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=image_mask,
                    backbone_attention_mask=attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_ids)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity

        if use_prefix_rtc:
            actions[:, :prefix_len] = prev_actions[:, :prefix_len]

        return {
            'action_pred': actions,
            'backbone_features': vl_embeds,
            'state_features': state_features,
        }

    @torch.no_grad()
    def get_action(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,
        prefix_len: int = 0,
        rtc_config: dict | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        vl_embeds, state_features = self.encode_features(
            input_features, states, embodiment_ids)
        return self.get_action_from_features(
            backbone_features=vl_embeds,
            state_features=state_features,
            embodiment_ids=embodiment_ids,
            attention_mask=attention_mask,
            image_mask=image_mask,
            prev_actions=prev_actions,
            prefix_len=prefix_len,
            rtc_config=rtc_config,
            seed=seed,
        )

    @torch.no_grad()
    def predict_action(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        prefix_len: int = 0,
        rtc_config: dict | None = None,
        image_mask: torch.Tensor | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        return self.get_action(
            input_features=input_features,
            states=states,
            attention_mask=attention_mask,
            embodiment_ids=embodiment_ids,
            image_mask=image_mask,
            prev_actions=prev_actions,
            prefix_len=prefix_len,
            rtc_config=rtc_config,
            seed=seed,
        )['action_pred']
