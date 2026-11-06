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

import contextlib
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from fluxvla_export.components.pi_policy import PI0Computation
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.cache_utils import Cache

from fluxvla.engines import (VLAS, build_llm_backbone_from_cfg,
                             build_projector_from_cfg)
from fluxvla.engines.losses import reduce_action_bc_loss
from fluxvla.engines.utils.fsdp_wrapping import build_combined_wrap_policy
from fluxvla.engines.utils.model_utils import (apply_rotary_pos_emb,
                                               gated_residual,
                                               make_att_2d_masks, sample_beta)
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI0FlowMatching(PI0Computation, BaseVLA):
    """PI0 Flow Matching Model for Vision-Language Alignment.
    Implemented based on https://arxiv.org/abs/2410.24164

    This model is designed to handle vision-language alignment tasks
    using flow matching techniques, leveraging a vision backbone,
    language model backbone, projector, and a VLA head.

    Args:
        state_proj (Dict): Configuration dictionary for the state
            projector.
        action_in_proj (Dict): Configuration dictionary for the action
            input projector.
        action_out_proj (Dict): Configuration dictionary for the action
            output projector.
        action_time_mlp_in (Dict): Configuration dictionary for the action
            time MLP input.
        action_time_mlp_out (Dict): Configuration dictionary for the action
            time MLP output.
        vlm_backbone (str): Identifier for the vision-language model backbone.
        vla_head (str): Identifier for the vision-language alignment head.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
        freeze_vision_backbone (bool): Whether to freeze the vision backbone.
        freeze_llm_backbone (bool): Whether to freeze the language model
            backbone.
        freeze_projector (bool): Whether to freeze the projector.
        vision_backbone_fp32 (bool): Whether to use FP32 for the vision
            backbone.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer
            of the model.
        ignore_index (int): Index to ignore in loss calculations.
        norm_stats (Dict, optional): Normalization statistics for the model.
        time_sampler (str): ``beta`` matches OpenPI; the default
            ``legacy_power_ratio`` preserves existing FluxVLA recipes.
        loss_action_dim (int, optional): Number of padded action dimensions
            supervised by flow matching. Defaults to ``ori_action_dim``.
        openpi_fp32_flow (bool): Keep noise, actions, timestep projections,
            and the velocity head in FP32, matching OpenPI JAX. Gemma inputs
            are cast to BF16 at the model boundary.
        preserve_fp32_residuals (bool): Keep Gemma inputs in FP32 when its
            parameters are FP32. Defaults to False for legacy recipe parity.
        **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(self,
                 action_in_proj: Dict,
                 action_out_proj: Dict,
                 llm_expert: Dict,
                 proj_width: int,
                 time_mlp_in: Dict = None,
                 time_mlp_out: Dict = None,
                 action_time_mlp_in: Dict = None,
                 action_time_mlp_out: Dict = None,
                 n_action_steps: int = 50,
                 state_proj: Dict = None,
                 llm_backbone: Dict = None,
                 vision_backbone: Dict = None,
                 vlm_backbone: str = None,
                 projector: Dict = None,
                 vla_head: str = None,
                 enable_mixed_precision_training: bool = True,
                 freeze_vision_backbone=True,
                 freeze_llm_backbone=True,
                 freeze_vlm_backbone=True,
                 freeze_llm_expert=False,
                 freeze_projector=False,
                 vision_backbone_fp32: bool = False,
                 unfreeze_last_layer: bool = False,
                 ignore_index: int = -100,
                 norm_stats: Dict = None,
                 pretrained_name_or_path: Optional[str] = None,
                 name_mapping: Optional[Dict[str, str]] = None,
                 strict_mapping: Optional[bool] = False,
                 attention_implementation: Optional[str] = 'sdpa',
                 params_to_change_dtype: Optional[List[str]] = [],
                 max_action_dim: int = 7,
                 ori_action_dim: int = None,
                 loss_action_dim: int = None,
                 num_steps: int = 10,
                 time_sampler: str = 'legacy_power_ratio',
                 time_beta_alpha: float = 1.5,
                 time_beta_beta: float = 1.0,
                 openpi_fp32_flow: bool = False,
                 rtc_training_config: Optional[Dict] = None,
                 preserve_fp32_residuals: bool = False,
                 **kwargs):
        super(PI0FlowMatching, self).__init__(
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            vlm_backbone=vlm_backbone,
            projector=projector,
            vla_head=vla_head,
            pretrained_name_or_path=pretrained_name_or_path,
            enable_mixed_precision_training=enable_mixed_precision_training,
            freeze_vision_backbone=freeze_vision_backbone,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_vlm_backbone=freeze_vlm_backbone,
            freeze_projector=freeze_projector,
            vision_backbone_fp32=vision_backbone_fp32,
            unfreeze_last_layer=unfreeze_last_layer,
            ignore_index=ignore_index,
            norm_stats=norm_stats)
        self.all_module_keys = [
            'vlm_backbone', 'vision_backbone', 'llm_backbone', 'projector',
            'state_proj', 'action_in_proj', 'action_out_proj',
            'action_time_mlp_in', 'action_time_mlp_out', 'head', 'llm_expert'
        ]
        if state_proj is not None:
            self.state_proj = build_projector_from_cfg(state_proj)
        else:
            self.state_proj = None
        self.freeze_llm_expert = freeze_llm_expert
        self.action_in_proj = build_projector_from_cfg(action_in_proj)
        self.action_out_proj = build_projector_from_cfg(action_out_proj)
        if time_mlp_in is not None:
            self.time_mlp_in = build_projector_from_cfg(time_mlp_in)
        else:
            self.time_mlp_in = None
        if time_mlp_out is not None:
            self.time_mlp_out = build_projector_from_cfg(time_mlp_out)
        else:
            self.time_mlp_out = None
        if action_time_mlp_in is not None:
            self.action_time_mlp_in = build_projector_from_cfg(
                action_time_mlp_in)
        else:
            self.action_time_mlp_in = None
        if action_time_mlp_out is not None:
            self.action_time_mlp_out = build_projector_from_cfg(
                action_time_mlp_out)
        else:
            self.action_time_mlp_out = None
        self.proj_width = proj_width
        self.n_action_steps = n_action_steps
        self.llm_expert = build_llm_backbone_from_cfg(llm_expert)

        self.attention_implementation = attention_implementation
        self.attention_interface = self.get_attention_interface()
        self.params_to_change_dtype = params_to_change_dtype
        self.name_mapping = name_mapping
        self.strict_mapping = strict_mapping
        self.max_action_dim = max_action_dim
        self.ori_action_dim = ori_action_dim
        self.loss_action_dim = (
            ori_action_dim
            if loss_action_dim is None else int(loss_action_dim))
        if (self.loss_action_dim is not None
                and not 0 < self.loss_action_dim <= self.max_action_dim):
            raise ValueError(
                'loss_action_dim must be in [1, max_action_dim], got '
                f'{self.loss_action_dim} with max_action_dim='
                f'{self.max_action_dim}.')
        self.num_steps = num_steps
        if time_sampler not in ('beta', 'legacy_power_ratio'):
            raise ValueError(f'Unsupported time_sampler={time_sampler!r}.')
        if time_beta_alpha <= 0 or time_beta_beta <= 0:
            raise ValueError('time_beta_alpha and time_beta_beta must be '
                             'positive.')
        self.time_sampler = time_sampler
        self.time_beta_alpha = float(time_beta_alpha)
        self.time_beta_beta = float(time_beta_beta)
        self.openpi_fp32_flow = bool(openpi_fp32_flow)
        self.preserve_fp32_residuals = bool(preserve_fp32_residuals)
        self.rtc_training_config = rtc_training_config

    def to_bfloat16(self):
        for name, param in self.named_parameters():
            if any(selector in name
                   for selector in self.params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.time_beta_alpha,
            self.time_beta_beta,
            bsize,
            device,
            sampler=self.time_sampler)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def _select_flow_loss_dimensions(self, prediction, target):
        """Select only the configured action dimensions for supervision."""
        if self.loss_action_dim is None:
            return prediction, target
        if (prediction.shape[-1] < self.loss_action_dim
                or target.shape[-1] < self.loss_action_dim):
            raise ValueError(
                'Flow loss tensors have fewer dimensions than configured: '
                f'prediction={prediction.shape[-1]}, '
                f'target={target.shape[-1]}, '
                f'loss_action_dim={self.loss_action_dim}.')
        return (prediction[..., :self.loss_action_dim],
                target[..., :self.loss_action_dim])

    def _forward_transformer_layers(
        self,
        inputs_embeds: List[torch.Tensor],
        attention_masks: torch.Tensor,
        position_ids: torch.Tensor,
        models: List,
        num_layers: int,
        adarms_cond: List[Optional[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Forward pass through transformer layers.

        Args:
            inputs_embeds (List[torch.Tensor]): List of input embeddings.
            attention_masks (torch.Tensor): Attention masks.
            position_ids (torch.Tensor): Position IDs.
            models (List): List of model instances (VLM and LLM expert).
            num_layers (int): Number of layers to process.
            adarms_cond (List[Optional[torch.Tensor]]): ADARMS
                condition tensors.

        Returns:
            List[torch.Tensor]: Output embeddings after processing all layers.
        """
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                layer = models[i].layers[layer_idx]
                hidden_states, gate = layer.input_layernorm(
                    hidden_states, cond=adarms_cond[i])
                gates.append(gate)
                hidden_shape = (*hidden_states.shape[:-1], -1,
                                layer.self_attn.head_dim)

                compute_fp32 = getattr(layer.self_attn, 'compute_fp32', False)
                context = (
                    self._disable_autocast(hidden_states)
                    if compute_fp32 else contextlib.nullcontext())
                with context:
                    if compute_fp32:
                        hidden_states = hidden_states.float()
                    query_state = layer.self_attn.q_proj(hidden_states).view(
                        hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(
                        hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(
                        hidden_shape).transpose(1, 2)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            dummy_tensor = torch.zeros(
                query_states.shape[0],
                query_states.shape[2],
                query_states.shape[-1],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            cos, sin = (
                self.llm_backbone.rotary_emb(dummy_tensor, position_ids))
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)

            scaling = self.llm_backbone.layers[layer_idx].self_attn.scaling

            att_output, _ = self.attention_interface(
                self.llm_backbone.layers[layer_idx].self_attn,
                query_states,
                key_states,
                value_states,
                attention_masks,
                scaling,
            )
            # Attention helpers return [B, L, H, D].  Flatten the actual
            # number of query heads instead of assuming the eight-head
            # PaliGemma configuration, so alternate Gemma widths retain the
            # correct input size for each model's output projection.
            att_output = att_output.flatten(start_dim=2)

            outputs_embeds = []
            start_pos = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]

                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(
                        layer.self_attn.o_proj.weight.dtype)
                compute_fp32 = getattr(layer.self_attn, 'compute_fp32', False)
                context = (
                    self._disable_autocast(att_output)
                    if compute_fp32 else contextlib.nullcontext())
                with context:
                    out_emb = layer.self_attn.o_proj(
                        att_output[:, start_pos:end_pos])

                out_emb = gated_residual(hidden_states, out_emb, gates[i])
                after_first_residual = out_emb.clone()
                out_emb, gate = layer.post_attention_layernorm(
                    out_emb, cond=adarms_cond[i])
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)

                out_emb = layer.mlp(out_emb)
                out_emb = gated_residual(after_first_residual, out_emb, gate)
                outputs_embeds.append(out_emb)
                start_pos = end_pos

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds

    def forward(self,
                images: List[torch.Tensor],
                lang_tokens: torch.Tensor,
                states: torch.Tensor,
                actions: torch.Tensor,
                action_masks: Optional[torch.Tensor] = None,
                img_masks: Optional[torch.Tensor] = None,
                lang_masks: Optional[torch.Tensor] = None,
                past_key_values: Optional[Union[List[torch.FloatTensor],
                                                Cache]] = None,
                use_cache: Optional[bool] = None,
                fill_kv_cache: Optional[bool] = None,
                noise=None,
                time=None,
                *args,
                **kwarg):
        """
        Forward pass for training the PI0 Flow Matching model.

        Args:
            inputs_embeds (List[torch.Tensor]): List of input
                embeddings, including image and language embeddings.
            attention_masks (List[torch.Tensor]): List of attention masks
                for the inputs.
            position_ids (torch.Tensor): Position IDs for the inputs.
            past_key_values (Optional[Union[List[torch.FloatTensor],
                Cache]]): Past key values for caching.
            use_cache (Optional[bool]): Whether to use cache for
                faster inference.
            fill_kv_cache (Optional[bool]): Whether to fill the
                key-value cache.
            noise (Optional[torch.Tensor]): Noise tensor for
                flow matching.
            time (Optional[torch.Tensor]): Time tensor for
                flow matching.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        if self.openpi_fp32_flow:
            actions = actions.float()
            noise = noise.float()
            time = time.float()

        # `time` is the sampled scalar flow-matching time, shape (B,).
        # Below we derive `t` which is passed to embed_suffix as the timestep:
        #   - RTC:     t is (B, T), per-position time (delay positions get
        #              0.0, meaning clean in PI0 convention).
        #   - vanilla: t keeps (B,), same time for all positions.
        T = actions.shape[1]
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
                device=actions.device)
            t, action_masks = apply_rtc_time_conditioning(
                time, action_masks, delays, T, clean_time=0.0)  # (B, T)
            x_t = t.unsqueeze(-1) * noise + (1 - t.unsqueeze(-1)) * actions
        else:
            t = time  # (B,)
            x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions

        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            lang_tokens=lang_tokens,
            img_masks=img_masks,
            lang_masks=lang_masks,
            past_key_values=past_key_values)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(states, x_t, t))
        prefix_embs = self._cast_gemma_input(prefix_embs)
        suffix_embs = self._cast_gemma_input(suffix_embs)
        inputs_embeds = [prefix_embs, suffix_embs]
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        attention_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(attention_masks)

        suffix_out, _ = self.forward_model(
            inputs_embeds=inputs_embeds,
            attention_masks=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            adarms_cond=[None, adarms_cond],
            time=time)
        action_time_dim = x_t.shape[1]
        suffix_out = suffix_out[:, -action_time_dim:]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self._project_action_output(suffix_out)
        v_t, u_t = self._select_flow_loss_dimensions(v_t, u_t)
        losses = F.mse_loss(u_t, v_t, reduction='none')
        sample_weight = kwarg.get('sample_weight')
        loss = reduce_action_bc_loss(
            losses, action_mask=action_masks, sample_weight=sample_weight)

        return_dict = dict(
            predictions=v_t,
            loss=loss,
        )
        return return_dict

    def _predict_action_prefix_rtc(self, x_t, denoise, bsize, dt, device,
                                   prev_actions, prefix_len):
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        prefix_mask = torch.zeros(
            bsize, self.n_action_steps, dtype=torch.bool, device=device)
        prefix_mask[:, :prefix_len] = True

        while time >= -dt / 2:
            t_batch = time.expand(bsize)
            x_t[:, :prefix_len] = prev_actions[:, :prefix_len]
            t_pos = t_batch.unsqueeze(1)
            t_pos = t_pos.expand(-1, self.n_action_steps).clone()
            t_pos[prefix_mask] = 0.0
            v_t = denoise(x_t, t_pos)
            x_t += dt * v_t
            time += dt

        x_t[:, :prefix_len] = prev_actions[:, :prefix_len]
        return x_t

    def _predict_action_guidance_rtc(self, x_t, denoise, bsize, dt, device,
                                     prev_actions, prefix_len, rtc_config):
        from fluxvla.engines.utils.rtc_guidance import (apply_rtc_guidance,
                                                        compute_prefix_weights)

        prefix_weights = compute_prefix_weights(
            self.n_action_steps,
            prefix_len,
            rtc_config.get('decay_end', prefix_len * 2),
            rtc_config.get('schedule', 'exp'),
            device=device)
        max_gw = rtc_config.get('max_guidance_weight', 5.0)
        use_vjp = rtc_config.get('use_vjp', False)

        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            t_batch = time.expand(bsize)
            # PI0 uses opposite velocity sign.
            # rtc_guidance expects: 0=noise, 1=clean.
            v_fm = apply_rtc_guidance(
                x_t,
                lambda x: -denoise(x, t_batch),
                prev_actions,
                prefix_weights,
                1.0 - time.item(),
                max_gw,
                use_vjp=use_vjp)
            x_t += dt * (-v_fm)
            time += dt
        return x_t

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns the FSDP auto wrapping policy for the model.

        This policy wraps low-level modules (e.g., nn.Linear, nn.LayerNorm),
        and combines with VLM's existing policy. It explicitly avoids wrapping
        nn.Embedding to prevent errors during sharding.
        """
        wrapping_policies = []
        if self.vlm_backbone is not None:
            vlm_wrapping_policy = self.vlm_backbone.get_fsdp_wrapping_policy()
            wrapping_policies.append(vlm_wrapping_policy)
        if self.llm_backbone is not None:
            llm_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()
            wrapping_policies.append(llm_wrapping_policy)

        if self.vision_backbone is not None:
            vision_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            wrapping_policies.append(vision_wrapping_policy)

        def match_module(module, *args, **kwargs):
            import torch.nn as nn

            # Wrap Linear and LayerNorm
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                return True
            # Explicit opt-in
            if hasattr(module, 'is_fsdp_wrap') and module.is_fsdp_wrap:
                return True
            return False

        return build_combined_wrap_policy([*wrapping_policies, match_module])

    def get_fsdp_execution_block_wrapping_policy(self) -> Callable:
        """Wrap modules whose ``forward`` methods PI0 actually executes.

        The legacy policy wraps individual linear and normalization modules,
        producing hundreds of tiny collectives on a multi-node global process
        group. PI0 also bypasses ``GemmaDecoderLayer.forward`` in its fused
        prefix/suffix path, so wrapping full decoder layers would leave their
        sharded parameters inaccessible. Gemma MLPs and SigLIP encoder layers
        are real execution boundaries and are safe, coarse FSDP units.
        """
        try:
            vision_layers = (
                self.vision_backbone.vision.vision_model.encoder.layers)
        except AttributeError as exc:
            raise ValueError(
                'Execution-block FSDP requires a SigLIP-style vision '
                'encoder with an encoder.layers module list.') from exc
        if not vision_layers:
            raise ValueError('The vision encoder has no execution blocks.')

        vision_layer_classes = {type(layer) for layer in vision_layers}
        if len(vision_layer_classes) != 1:
            names = sorted(cls.__name__ for cls in vision_layer_classes)
            raise ValueError(
                'Vision execution blocks must share one class, got '
                f'{names}.')

        mlp_classes = set()
        for backbone_name in ('llm_backbone', 'llm_expert'):
            backbone = getattr(self, backbone_name, None)
            try:
                layers = backbone.layers
            except AttributeError as exc:
                raise ValueError('Execution-block FSDP requires '
                                 f'{backbone_name}.layers.') from exc
            if not layers:
                raise ValueError(f'{backbone_name} has no transformer layers.')
            if any(not hasattr(layer, 'mlp') for layer in layers):
                raise ValueError(
                    f'Every {backbone_name} layer must expose an MLP.')
            mlp_classes.update(type(layer.mlp) for layer in layers)

        if len(mlp_classes) != 1:
            names = sorted(cls.__name__ for cls in mlp_classes)
            raise ValueError('Gemma execution MLPs must share one class, got '
                             f'{names}.')

        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=vision_layer_classes | mlp_classes)
