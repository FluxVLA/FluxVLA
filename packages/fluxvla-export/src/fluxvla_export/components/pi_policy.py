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
"""Shared PI policy computation; training owns initialization and extensions.

These mixins do not register modules or construct weights. Inference adapters
provide the original policy attributes and explicit noise; training adapters
also implement joint layers, random sampling, and RTC handlers.
"""

import contextlib
import math
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from .euler import plain_euler
from .tensor_utils import (create_sinusoidal_pos_embedding,
                           eager_attention_forward, make_att_2d_masks,
                           sdpa_attention_forward,
                           sdpa_math_fp32_attention_forward)


class PI0Computation:
    """PI0 tensor operations shared by training and cached inference."""

    @staticmethod
    def _disable_autocast(tensor: torch.Tensor):
        if tensor.device.type in ('cpu', 'cuda'):
            return torch.autocast(
                device_type=tensor.device.type, enabled=False)
        return contextlib.nullcontext()

    def _cast_gemma_input(self, tensor: Optional[torch.Tensor]):
        if tensor is None:
            return None
        if self.openpi_fp32_flow and self.enable_mixed_precision_training:
            if not self.preserve_fp32_residuals:
                return tensor.to(torch.bfloat16)
            # Keep the residual stream in FP32 with FP32 master parameters.
            # Autocast still controls matrix kernels. Match the cast used by
            # LeRobot when the Gemma parameters themselves are BF16.
            weight_dtype = (
                self.llm_backbone.layers[0].self_attn.q_proj.weight.dtype)
            if weight_dtype == torch.bfloat16:
                return tensor.to(torch.bfloat16)
        return tensor

    def _project_action_output(self, suffix_out: torch.Tensor):
        if not self.openpi_fp32_flow:
            return self.action_out_proj(suffix_out)
        with self._disable_autocast(suffix_out):
            return self.action_out_proj(suffix_out.float())

    def get_attention_interface(self):
        math_fp32 = [
            getattr(
                getattr(model, 'config', None), 'attention_math_fp32', False)
            for model in (self.llm_backbone, self.llm_expert)
        ]
        if any(math_fp32):
            if not all(math_fp32):
                raise ValueError(
                    'Joint attention requires attention_math_fp32 '
                    'on both the backbone and expert.')
            return sdpa_math_fp32_attention_forward
        if self.attention_implementation == 'sdpa':
            attention_interface = sdpa_attention_forward
        elif self.attention_implementation == 'eager':
            attention_interface = eager_attention_forward
        else:
            raise ValueError(
                f'Invalid attention implementation: {self.attention_implementation}. '  # noqa: E501
                "Expected one of ['sdpa', 'eager'].")
        return attention_interface

    def embed_prefix(self, images, lang_tokens, img_masks, lang_masks, *args,
                     **kwargs):
        """Embed the prefix tokens for the Pi0 head.

        Args:
            images (torch.Tensor): The image tensor of shape
                (bsize, 3, 224, 224).
            lang_tokens (torch.Tensor): The language tokens tensor
                of shape (bsize, lang_token_dim).
            img_masks (torch.Tensor): The image attention masks
                tensor of shape (bsize, num_img_embs).
            lang_masks (torch.Tensor): The language attention masks
                tensor of shape (bsize, num_lang_embs).
        """
        embs = list()
        pad_masks = list()
        attn_masks = list()
        img_masks = img_masks.permute(1, 0)
        img_emb = self.projector(self.vision_backbone(images))
        embs.append(img_emb)
        bsize, num_img_embs = img_emb.shape[:2]
        # Indexing preserves the native view order without an ONNX sequence
        # boundary for tensor iteration. The view count remains runtime-shaped.
        for view_index in range(img_masks.shape[0]):
            img_mask = img_masks[view_index]
            img_mask = img_mask[:, None].expand(bsize,
                                                num_img_embs // len(img_masks))

            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            attn_masks += [0] * (num_img_embs // len(img_masks))
        lang_emb = self.llm_backbone.embed_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        attn_masks += [0] * num_lang_embs
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        attn_masks = torch.tensor(
            attn_masks, dtype=torch.bool, device=pad_masks.device)
        attn_masks = attn_masks[None, :].expand(bsize, len(attn_masks))
        return embs, pad_masks, attn_masks

    def embed_suffix(self, states, noisy_actions, timestep):
        """Embed the suffix tokens for the Pi0 head.

        Args:
            state (torch.Tensor): The state tensor of shape (bsize, state_dim).
            noisy_actions (torch.Tensor): The noisy actions tensor of shape
                (bsize, n_action_steps, action_dim).
            timestep (torch.Tensor): The timestep tensor of shape (bsize,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
                containing the embedded suffix tokens, padding masks,
                and attention masks.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        bsize = states.shape[0]
        dtype = states.dtype
        device = states.device

        if self.state_proj is not None:
            state_emb = self.state_proj(states)
            embs.append(state_emb[:, None, :])
            pad_masks.append(
                torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        # Set attention masks so that image and language
        # inputs do not attend to state or actions

        # Embed timestep using sine-cosine positional
        # encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device)
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        if time_emb.ndim == 2:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not
        # attend to action tokens.  Use the runtime horizon rather than the
        # configured default: training/evaluation may provide a shorter
        # action window (for example, a truncated episode).
        if action_time_dim == 0:
            raise ValueError('noisy_actions must contain at least one step')
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, None

    def forward_model(self,
                      inputs_embeds: List[torch.Tensor],
                      attention_masks: List[torch.Tensor],
                      position_ids: torch.Tensor,
                      past_key_values: Optional[Union[List[torch.FloatTensor],
                                                      Cache]] = None,
                      use_cache: Optional[bool] = None,
                      fill_kv_cache: Optional[bool] = None,
                      time=None,
                      adarms_cond=None,
                      *args,
                      **kwarg):
        """
        Forward pass through the PI0 Flow Matching model.
        Args:
            images (List[torch.Tensor]): List of input image
                tensors.
            lang_tokens (torch.Tensor): Input language
                tokens.
            states (torch.Tensor): Input states tensor.
            x_t (torch.Tensor): Input noisy actions tensor.
            actions (torch.Tensor): Input actions tensor.
            img_masks (Optional[torch.Tensor]): Image attention masks.
            lang_masks (Optional[torch.Tensor]): Language attention masks.
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
            adarms_cond (Optional[torch.Tensor]): ADaRMS condition tensor for
                flow matching.

        Returns:
            torch.Tensor: Output embeddings from the VLA head.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]
        num_layers = len(self.llm_expert.layers)
        assert num_layers == len(self.llm_backbone.layers), \
            f'Number of layers in llm_model ({num_layers}) does not match the expected number ({self.llm_expert.language_model.num_hidden_layers}).'  # noqa: E501
        if inputs_embeds[1] is None:
            prefix_output = self.llm_backbone(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0]
                if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None

        elif inputs_embeds[0] is None:
            suffix_output = self.llm_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1]
                if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            if not hasattr(self, '_forward_transformer_layers'):
                raise NotImplementedError(
                    'Joint prefix/suffix execution requires the training '
                    'adapter extension _forward_transformer_layers; '
                    'inference-only policies support cached execution.')
            models = [self.llm_backbone, self.llm_expert]
            outputs_embeds = self._forward_transformer_layers(
                inputs_embeds=inputs_embeds,
                attention_masks=attention_masks,
                position_ids=position_ids,
                models=models,
                num_layers=num_layers,
                adarms_cond=adarms_cond,
            )
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None
        return suffix_output, prefix_past_key_values

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # SDPA accepts FP32 masks with either FP32 or BF16 queries. A BF16
        # mask rejects FP32 queries, making a full-precision stability run
        # impossible. Preserve the finite sentinel (including padded rows).
        mask = torch.zeros_like(att_2d_masks_4d, dtype=torch.float32)
        mask.masked_fill_(~att_2d_masks_4d, torch.finfo(torch.bfloat16).min)
        return mask

    def prefill_prefix(self,
                       images,
                       lang_tokens,
                       img_masks,
                       lang_masks,
                       past_key_values=None):
        """Return prefix padding and the canonical image/language KV cache.

        Graph adapters may provide a fresh cache with exportable empty tensor
        shapes. The ordinary policy retains the native default cache creation.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            lang_tokens=lang_tokens,
            img_masks=img_masks,
            lang_masks=lang_masks)
        prefix_embs = self._cast_gemma_input(prefix_embs)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks,
                                                prefix_att_masks)
        # PyTorch promotes boolean counts to int64 implicitly. Make that
        # promotion explicit for ONNX CumSum, which does not accept bool.
        prefix_position_ids = torch.cumsum(
            prefix_pad_masks.to(torch.int64), dim=1) - 1
        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks)
        inputs_embeds = [prefix_embs, None]
        position_ids = prefix_position_ids
        self.llm_backbone.config._attn_implementation = self.attention_implementation  # noqa: SLF001, E501
        _, past_key_values = self.forward_model(
            inputs_embeds=inputs_embeds,
            attention_masks=prefix_att_2d_masks_4d,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_key_values)
        return prefix_pad_masks, past_key_values

    def _predict_action_plain(self, x_t, denoise, bsize, dt, device):
        return plain_euler(x_t, denoise, bsize, dt, device)

    def predict_action(self,
                       images: List[torch.Tensor],
                       lang_tokens: torch.Tensor,
                       states: torch.Tensor,
                       img_masks: Optional[torch.Tensor] = None,
                       lang_masks: Optional[torch.Tensor] = None,
                       past_key_values: Optional[Union[List[torch.FloatTensor],
                                                       Cache]] = None,
                       noise=None,
                       prev_actions=None,
                       prefix_len: int = 0,
                       rtc_config: dict = None,
                       *args,
                       **kwargs):
        device = states.device
        bsize = states.shape[0]
        if noise is None:
            if not hasattr(self, 'sample_noise'):
                raise NotImplementedError(
                    'Noise sampling requires the training adapter extension '
                    'sample_noise; inference-only policies require explicit '
                    'noise.')
            actions_shape = (bsize, self.n_action_steps, self.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        if (prev_actions is not None and prefix_len > 0 and rtc_config
                and not all(
                    hasattr(self, name)
                    for name in ('_predict_action_prefix_rtc',
                                 '_predict_action_guidance_rtc'))):
            raise NotImplementedError(
                'RTC inference requires the training adapter extensions '
                '_predict_action_prefix_rtc and _predict_action_guidance_rtc; '
                'inference-only policies support plain Euler denoising.')

        prefix_pad_masks, past_key_values = self.prefill_prefix(
            images=images,
            lang_tokens=lang_tokens,
            img_masks=img_masks,
            lang_masks=lang_masks)

        dt = -1.0 / self.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        if (prev_actions is not None and self.ori_action_dim is not None
                and prev_actions.shape[-1] < self.max_action_dim):
            pad_size = self.max_action_dim - prev_actions.shape[-1]
            prev_actions = F.pad(prev_actions, (0, pad_size), value=0.0)

        rtc_method = None
        if prev_actions is not None and prefix_len > 0 and rtc_config:
            rtc_method = rtc_config.get('method', 'prefix')

        x_t = noise

        def denoise(x, t):
            return self.denoise_step(states, prefix_pad_masks, past_key_values,
                                     x, t)

        if rtc_method == 'prefix':
            x_t = self._predict_action_prefix_rtc(
                x_t=x_t,
                denoise=denoise,
                bsize=bsize,
                dt=dt,
                device=device,
                prev_actions=prev_actions,
                prefix_len=prefix_len)
        elif rtc_method == 'guidance':
            x_t = self._predict_action_guidance_rtc(
                x_t=x_t,
                denoise=denoise,
                bsize=bsize,
                dt=dt,
                device=device,
                prev_actions=prev_actions,
                prefix_len=prefix_len,
                rtc_config=rtc_config)
        else:
            x_t = self._predict_action_plain(
                x_t=x_t, denoise=denoise, bsize=bsize, dt=dt, device=device)

        return x_t

    def denoise_step(
        self,
        states,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix = self.embed_suffix(states, x_t, timestep)
        return self.denoise_suffix(prefix_pad_masks, past_key_values, x_t,
                                   suffix)

    def denoise_suffix(self, prefix_pad_masks, past_key_values, x_t, suffix):
        """Run the shared cached expert from already embedded suffix tokens."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = suffix
        suffix_embs = self._cast_gemma_input(suffix_embs)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks,
                                                suffix_att_masks)

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks)

        self.llm_expert.config._attn_implementation = self.attention_implementation  # noqa: SLF001, E501

        prefix_offsets = torch.sum(
            prefix_pad_masks.to(torch.int64), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(
            suffix_pad_masks.to(torch.int64), dim=1) - 1

        suffix_out, _ = self.forward_model(
            attention_masks=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond])
        action_time_dim = x_t.shape[1]
        suffix_out = suffix_out[:, -action_time_dim:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self._project_action_output(suffix_out)
        return v_t


class PI05Computation(PI0Computation):
    """PI0.5 action and adaptive-time embeddings, with PI0 inference."""

    def embed_suffix(self, states, noisy_actions, timestep):
        """Embed the suffix tokens for the Pi0 head.

        Args:
            state (torch.Tensor): The state tensor of shape (bsize, state_dim).
            noisy_actions (torch.Tensor): The noisy actions tensor of shape
                (bsize, n_action_steps, action_dim).
            timestep (torch.Tensor): The timestep tensor of shape (bsize,)
                for scalar time, or (bsize, n_action_steps) for per-position
                time (used in training-time RTC).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor]: A tuple containing the embedded suffix
                tokens, padding masks, attention masks, and adarms_cond.
        """
        return self.embed_suffix_with_metadata(
            noisy_actions,
            timestep,
            state_dtype=states.dtype,
            state_device=states.device)

    def embed_suffix_with_metadata(self, noisy_actions, timestep, *,
                                   state_dtype, state_device):
        """Embed PI0.5 actions with the state dtype/device, not state values.

        PI0.5 does not project state tensors. Its legacy timestep branch does
        use their dtype, and both branches allocate on their device. Keeping
        this metadata explicit permits a truthful states-free denoise graph.
        """
        embs = []
        pad_masks = []
        att_masks = []

        device = state_device

        # Set attention masks so that image and language
        # inputs do not attend to action tokens

        # Embed timestep using sine-cosine positional
        # encoding with sensitivity in the range [0, 1]
        if self.openpi_fp32_flow:
            with self._disable_autocast(noisy_actions):
                time_emb = create_sinusoidal_pos_embedding(
                    timestep.float(),
                    self.proj_width,
                    min_period=4e-3,
                    max_period=4.0,
                    device=device,
                    dtype=torch.float32)
                action_emb = self.action_in_proj(noisy_actions.float())
                time_emb = F.silu(self.time_mlp_in(time_emb))
                time_emb = F.silu(self.time_mlp_out(time_emb))
        else:
            dtype = state_dtype
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.proj_width,
                min_period=4e-3,
                max_period=4.0,
                device=device)
            time_emb = time_emb.type(dtype=dtype)

            # Fuse timestep + action information using an MLP
            action_emb = self.action_in_proj(noisy_actions)
            time_emb = F.silu(self.time_mlp_in(time_emb))
            time_emb = F.silu(self.time_mlp_out(time_emb))
        # Add to input tokens
        embs.append(action_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image and language inputs do not attend
        # to action tokens.  Derive the mask from the supplied action window
        # because evaluation and truncated episodes can use a horizon
        # different from the training configuration.
        if action_time_dim == 0:
            raise ValueError('noisy_actions must contain at least one step')
        att_masks += [1] + ([0] * (action_time_dim - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, time_emb
