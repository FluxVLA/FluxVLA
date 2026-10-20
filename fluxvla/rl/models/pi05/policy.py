"""Flux PI0.5 with an RLinf-compatible, differentiable PPO interface."""

import torch
import torch.nn.functional as F
from rlinf.models.embodiment.base_policy import BasePolicy

from fluxvla.engines.utils.model_utils import (create_sinusoidal_pos_embedding,
                                               make_att_2d_masks)
from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching
from fluxvla.rl.models.flow.policy import FlowPPOPolicyMixin


class FluxPI05RLPolicy(FlowPPOPolicyMixin, PI05FlowMatching, BasePolicy):
    """Preserves all Flux SFT parameter names; adds only ``value_head.*``."""

    _no_split_modules = ('GemmaDecoderLayer', 'SiglipEncoderLayer')
    rl_trainable_modules = ('llm_expert', 'action_in_proj', 'action_out_proj',
                            'time_mlp_in', 'time_mlp_out')
    rl_frozen_modules = ('vision_backbone', 'llm_backbone', 'projector',
                         'llm_expert.embed_tokens')

    @property
    def model_action_horizon(self):
        return self.n_action_steps

    @property
    def model_action_dim(self):
        return self.max_action_dim

    @property
    def critic_hidden_size(self):
        return self.llm_backbone.config.hidden_size

    @property
    def config(self):
        # ConditionGemmaModel is already the backbone, without an .llm wrapper.
        return self.llm_backbone.config

    def _configure_model_rl(self):
        if self.openpi_fp32_flow:
            from fluxvla.rl.models.pi05.precision import (
                align_openpi_rope, align_openpi_vision,
                keep_adaptive_norm_fp32)
            keep_adaptive_norm_fp32(self.llm_expert)
            align_openpi_vision(self.vision_backbone)
            align_openpi_rope(self.llm_backbone)
            align_openpi_rope(self.llm_expert)

    def embed_suffix(self, states, noisy_actions, timestep):
        if not self.openpi_fp32_flow:
            return super().embed_suffix(states, noisy_actions, timestep)
        with self._disable_autocast(noisy_actions):
            # OpenPI constructs sinusoidal frequencies in FP64, then casts
            # the embedding to FP32 before the two timestep projections.
            time = create_sinusoidal_pos_embedding(
                timestep,
                self.proj_width,
                min_period=4e-3,
                max_period=4.0,
                device=timestep.device,
                dtype=torch.float64).float()
            action = self.action_in_proj(noisy_actions.float())
            time = F.silu(self.time_mlp_in(time))
            time = F.silu(self.time_mlp_out(time))
        mask = torch.ones(
            action.shape[:2], dtype=torch.bool, device=action.device)
        attention = torch.zeros_like(mask)
        attention[:, 0] = True
        return action, mask, attention, time

    def _cast_gemma_input(self, tensor):
        if tensor is not None and self.openpi_fp32_flow:
            return tensor.to(self.compute_dtype)
        return super()._cast_gemma_input(tensor)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        mask = super()._prepare_attention_masks_4d(att_2d_masks)
        # Original Flux assumes BF16 inference. SDPA also accepts FP32 bias,
        # which is required when explicitly testing/using an FP32 network.
        return mask.float() if self.compute_dtype == torch.float32 else mask

    def _prefix(self, obs):
        # The VLM is frozen. A fresh cache is made for each actor forward;
        # rollout caches/embeddings never enter forward_inputs.
        with torch.no_grad(), self._network_context():
            prefix, mask, attention = self.embed_prefix(
                obs['images'], obs['lang_tokens'], obs['img_masks'],
                obs['lang_masks'])
            self.llm_backbone.config._attn_implementation = (
                self.attention_implementation)
            outputs = self.llm_backbone(
                inputs_embeds=self._cast_gemma_input(prefix),
                attention_mask=self._prepare_attention_masks_4d(
                    make_att_2d_masks(mask, attention)),
                position_ids=mask.cumsum(dim=1) - 1,
                use_cache=True,
                past_key_values=None)
        return outputs.last_hidden_state, mask, outputs.past_key_values

    def _velocity(self, obs, mask, cache, x, time):
        with self._network_context():
            return self.denoise_step(obs['states'], mask, cache, x,
                                     time).float()
