"""SmolVLA frontend with the shared RLinf flow-SDE PPO contract."""

import torch
from rlinf.models.embodiment.base_policy import BasePolicy

from fluxvla.engines.utils.model_utils import make_att_2d_masks
from fluxvla.models.vlas.smolvla_flowmatching import SmolVLAFlowMatching
from fluxvla.rl.models.flow.policy import FlowPPOPolicyMixin


class FluxSmolVLARLPolicy(FlowPPOPolicyMixin, SmolVLAFlowMatching, BasePolicy):
    """Preserve SFT keys; freeze VLM and its state projection, add a critic.

    Text/expert decoder forwards are bypassed by SmolVLA's interleaving.
    Never FSDP-wrap LlamaDecoderLayer: its unshard hooks would not execute.
    The recipe leaves these parameters under the root FSDP2 forward.
    """

    _no_split_modules = ('SmolVLMEncoderLayer', )
    rl_trainable_modules = ('llm_expert', 'action_in_proj', 'action_out_proj',
                            'action_time_mlp_in', 'action_time_mlp_out')
    rl_frozen_modules = ('vlm_backbone', 'state_proj')

    @property
    def n_action_steps(self):
        # Keep the previously published compatibility property.
        return self.chunk_size

    @property
    def model_action_horizon(self):
        return self.chunk_size

    @property
    def model_action_dim(self):
        return self.max_action_dim

    @property
    def critic_hidden_size(self):
        return self.vlm_backbone.hidden_size

    def _prefix(self, obs):
        # Same helpers and ordering as native forward_model's prefix-only
        # path, retaining its final hidden states for the value head as well.
        # No cache/hidden states are persisted in the rollout trajectory.
        with torch.no_grad(), self._network_context():
            hidden, mask, attention = self.embed_prefix(
                obs['images'], obs['img_masks'], obs['lang_tokens'],
                obs['lang_masks'], obs['states'])
            attention = make_att_2d_masks(mask, attention)
            positions = mask.cumsum(dim=1) - 1
            layers, self_attention = self._get_model_layers()
            cache = [None] * self.num_vlm_layers
            for index, layer in enumerate(layers[0]):
                forward = (
                    self._forward_attn_layer if self_attention[index] else
                    self._forward_cross_attn_layer)
                output, _ = forward(
                    layer,
                    layers[1][index],
                    hidden,
                    None,
                    positions,
                    attention,
                    use_cache=True,
                    past_key_values=cache,
                    layer_idx=index)
                hidden = self._apply_residual_ffn(layer, output, hidden)
            hidden = self.vlm_backbone.norm(hidden)
        return hidden, mask, cache

    def _velocity(self, obs, mask, cache, x, time):
        with self._network_context():
            return self._denoise_step(x, mask, cache, time).float()
