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

import torch
import torch.nn.functional as F

from fluxvla.engines import HEADS
from fluxvla.models.blocks import AlternateVLDiT, DiT
from fluxvla.models.heads.flow_matching_head import FlowMatchingHead


@HEADS.register_module()
class GrootN17ActionHead(FlowMatchingHead):
    """Native equivalent of official ``Gr00tN1d7ActionHead``."""

    supports_gradient_checkpointing = True

    def __init__(self, config, **config_overrides):
        if config_overrides:
            config_dict = (
                dict(config) if isinstance(config, dict) else vars(config))
            valid_overrides = {
                key: value
                for key, value in config_overrides.items() if value is not None
            }
            config_dict.update(valid_overrides)
            config = SimpleNamespace(**config_dict)
        model_cls = AlternateVLDiT if config.use_alternate_vl_dit else DiT
        model_extra_kwargs = {
            'cross_attention_dim': config.backbone_embedding_dim,
        }
        if config.use_alternate_vl_dit:
            model_extra_kwargs['attend_text_every_n_blocks'] = (
                config.attend_text_every_n_blocks)
        super().__init__(
            hidden_size=config.hidden_size,
            state_dim=config.max_state_dim * config.state_history_length,
            input_embedding_dim=config.input_embedding_dim,
            action_dim=config.max_action_dim,
            num_inference_timesteps=config.num_inference_timesteps,
            max_num_embodiments=config.max_num_embodiments,
            use_vlln=config.use_vlln,
            backbone_embedding_dim=config.backbone_embedding_dim,
            vl_self_attention_cfg=config.vl_self_attention_cfg,
            add_positional_embeddings=config.add_pos_embed,
            max_seq_len=config.max_seq_len,
            num_timestep_buckets=config.num_timestep_buckets,
            noise_s=config.noise_s,
            noise_beta_alpha=config.noise_beta_alpha,
            noise_beta_beta=config.noise_beta_beta,
            num_steps=config.action_horizon,
            zero_padded_action_dims=False,
            clamp_sample_time=False,
            diffusion_model_cls=model_cls,
            diffusion_model_cfg=config.diffusion_model_cfg,
            diffusion_model_extra_kwargs=model_extra_kwargs,
            use_future_tokens=False,
        )
        self.config = config
        self.action_horizon = config.action_horizon
        self.state_dropout_prob = config.state_dropout_prob

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
            do_dropout = (
                torch.rand(
                    state_features.shape[0], device=state_features.device) <
                self.state_dropout_prob)
            do_dropout = do_dropout[:, None,
                                    None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        noise = torch.randn(
            actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(
            noisy_trajectory,
            t_discretized,
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
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
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

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size, ),
                fill_value=t_discretized,
                device=device,
            )
            action_features = self.action_encoder(actions, timesteps_tensor,
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
            seed=seed,
        )

    @torch.no_grad()
    def predict_action(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        prefix_len: int = 0,
        image_mask: torch.Tensor | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del prefix_len, kwargs
        return self.get_action(
            input_features=input_features,
            states=states,
            attention_mask=attention_mask,
            embodiment_ids=embodiment_ids,
            image_mask=image_mask,
            seed=seed,
        )['action_pred']
