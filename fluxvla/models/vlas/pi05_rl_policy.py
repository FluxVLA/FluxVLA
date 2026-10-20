"""Flux PI0.5 with an RLinf-compatible, differentiable PPO interface."""

import torch
from rlinf.models.embodiment.base_policy import BasePolicy

from fluxvla.engines import VLAS
from fluxvla.engines.utils.model_utils import make_att_2d_masks
from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching
from .flow_rl_policy import FlowRLPolicyMixin


@VLAS.register_module()
class FluxPI05RLPolicy(FlowRLPolicyMixin, PI05FlowMatching, BasePolicy):
    """Preserve Flux SFT keys and add the shared PPO/value interface."""

    _no_split_modules = ('GemmaDecoderLayer', 'SiglipEncoderLayer')
    rl_trainable_modules = ('llm_expert', 'action_in_proj', 'action_out_proj',
                            'time_mlp_in', 'time_mlp_out')
    rl_frozen_modules = ('vision_backbone', 'llm_backbone', 'projector')

    @property
    def config(self):
        return self.llm_backbone.config

    @property
    def model_action_horizon(self):
        return self.n_action_steps

    @property
    def model_action_dim(self):
        return self.max_action_dim

    @property
    def critic_hidden_size(self):
        return self.llm_backbone.config.hidden_size

    def _configure_model_rl(self):
        # The expert consumes inputs_embeds; its token embedding is unused.
        self.llm_expert.embed_tokens.requires_grad_(False)

    def _cast_gemma_input(self, tensor):
        if tensor is not None and self.openpi_fp32_flow:
            return tensor.to(self.compute_dtype)
        return super()._cast_gemma_input(tensor)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        mask = super()._prepare_attention_masks_4d(att_2d_masks)
        return mask.float() if self.compute_dtype == torch.float32 else mask

    def _prefix(self, obs):
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
