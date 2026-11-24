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

# Origin: Modified from
# Upstream-Repo: roboterax/video-prediction-policy-2
# Upstream-Ref: 3365872910244f14df83003062f7c563f14c6d9e
# SPDX-License-Identifier: MIT
#
# Notes: Frozen-video-cache Euler action inference is adapted from VPP2
# (MIT, copyright 2026 The FastWAM Authors); the FluxVLA head wrapper and
# registry integration are FluxVLA-specific. MoT and the flow-matching
# scheduler reuse FluxVLA's FastWAM modules. License texts and attribution:
# ../third_party_models/vpp2/LICENSE and ATTRIBUTION.txt.
"""VPP2 frozen-video-cache Euler inference."""

import torch
from torch import nn

from fluxvla.engines import HEADS
from ..third_party_models.fastwam.modules.mot import MoT
from ..third_party_models.fastwam.modules.schedulers.scheduler_continuous import \
    WanContinuousFlowMatchScheduler  # noqa: E501
from ..third_party_models.vpp2.modules.action_dit import ActionDiT
from ..third_party_models.vpp2.modules.wan_video_dit import (
    WanVideoDiT, precompute_freqs_cis, precompute_freqs_cis_3d)


@HEADS.register_module()
class VPP2Head(nn.Module):
    """VPP2 mixture-of-transformers head for cached action inference.

    A Wan2.1 video expert and an action expert share mixed attention through
    FastWAM's :class:`MoT`. At inference the video expert runs once on noisy
    target latents at the final training timestep, conditioned on the encoded
    condition video; its key/value cache is then reused for every Euler step
    of the action flow. The video itself is never denoised. Proprioception is
    appended to the text context as one projected token.

    VAE latents, CLIP features and T5 text context are produced upstream by
    :class:`~fluxvla.models.backbones.vlms.vpp2_wan21_backbone.VPP2Wan21Backbone`.

    Args:
        video_dit_config (dict): Keyword arguments for the video expert
            (``WanVideoDiT``).
        action_dit_config (dict): Keyword arguments for the action expert
            (``ActionDiT``).
        proprio_dim (int): Width of the normalized proprioceptive state.
            Defaults to 16.
        video_scheduler (dict | None): Video flow settings; only
            ``num_train_timesteps`` (default 1000) is used, as the cached
            video timestep.
        action_scheduler (dict | None): Action flow settings:
            ``num_train_timesteps`` (default 1000) and ``infer_shift``
            (default 3.0).
        mot_checkpoint_mixed_attn (bool): Gradient-checkpoint MoT mixed
            attention in training mode. Defaults to False.
        action_visible_video_frames (int): Number of leading latent video
            frames that action tokens may attend to. Defaults to 5.
        device (str | torch.device): Device to build on. ``'meta'`` defers
            parameter allocation until checkpoint loading. Defaults to
            ``'cpu'``.
        torch_dtype (torch.dtype): Parameter dtype. Defaults to
            ``torch.float32``.
    """

    def __init__(self,
                 video_dit_config,
                 action_dit_config,
                 proprio_dim=16,
                 video_scheduler=None,
                 action_scheduler=None,
                 mot_checkpoint_mixed_attn=False,
                 action_visible_video_frames=5,
                 device='cpu',
                 torch_dtype=torch.float32):
        super().__init__()
        self.mot = MoT(
            mixtures={
                'video': WanVideoDiT(**video_dit_config),
                'action': ActionDiT(**action_dit_config)
            },
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn)
        self.proprio_encoder = nn.Linear(proprio_dim,
                                         video_dit_config['text_dim'])
        self.action_visible_video_frames = int(action_visible_video_frames)
        if self.action_visible_video_frames <= 0:
            raise ValueError('action_visible_video_frames must be positive')
        self.to(device=device, dtype=torch_dtype)
        # Scheduler scalar constants are computed on CPU, even for meta builds.
        with torch.device('cpu'):
            # RoPE tables are plain attributes in upstream modules, so they
            # must hold real values even when parameters are built on meta.
            if self.action_expert.freqs.is_meta:
                self.action_expert.freqs = precompute_freqs_cis(
                    self.action_expert.attn_head_dim, end=1024)
                self.video_expert.freqs = precompute_freqs_cis_3d(
                    self.video_expert.attn_head_dim)
            self.video_num_train_timesteps = int(
                (video_scheduler or {}).get('num_train_timesteps', 1000))
            if self.video_num_train_timesteps <= 0:
                raise ValueError('video num_train_timesteps must be positive')
            action_scheduler = dict(action_scheduler or {})
            self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=action_scheduler.get(
                    'num_train_timesteps', 1000),
                shift=action_scheduler.get('infer_shift', 3.))

    @property
    def video_expert(self):
        """The Wan2.1 video expert inside :attr:`mot`."""
        return self.mot.mixtures['video']

    @property
    def action_expert(self):
        """The action expert inside :attr:`mot`."""
        return self.mot.mixtures['action']

    def append_proprio(self, context, context_mask, states):
        """Append one projected proprioception token to the text context.

        Args:
            context (torch.Tensor): Text context ``[B, L, text_dim]``.
            context_mask (torch.Tensor): Boolean mask ``[B, L]``.
            states (torch.Tensor): ``[B, proprio_dim]``, or
                ``[B, T, proprio_dim]`` of which the first step is used.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Context ``[B, L + 1,
            text_dim]`` and mask ``[B, L + 1]``.
        """
        if states.ndim == 3:
            states = states[:, 0]
        if states.shape != (context.shape[0],
                            self.proprio_encoder.in_features):
            raise ValueError('states must be [B, proprio_dim]')
        token = self.proprio_encoder(
            states.to(context).unsqueeze(1)).to(context)
        mask = torch.ones((context.shape[0], 1),
                          device=context.device,
                          dtype=torch.bool)
        return torch.cat((context, token), 1), torch.cat((context_mask, mask),
                                                         1)

    def attention_mask(self, video_pre, action_length):
        """Build the joint video/action boolean attention mask.

        Video tokens use the video expert's video-to-video mask and never
        attend to action tokens. Action tokens attend to all action tokens
        and to the first ``action_visible_video_frames`` latent frames.

        Args:
            video_pre (dict): Output of :meth:`video_pre`.
            action_length (int): Number of action tokens.

        Returns:
            torch.Tensor: ``[N + A, N + A]`` mask, where ``N`` is the video
            token count and ``A`` is ``action_length``; True allows
            attention.
        """
        tokens = video_pre['tokens']
        n = tokens.shape[1]
        per_frame = video_pre['meta']['tokens_per_frame']
        mask = torch.zeros((n + action_length, n + action_length),
                           dtype=torch.bool,
                           device=tokens.device)
        mask[:n, :n] = self.video_expert.build_video_to_video_mask(
            n, per_frame, tokens.device)
        mask[n:, n:] = True
        mask[n:, :min(n, per_frame * self.action_visible_video_frames)] = True
        return mask

    @staticmethod
    def _context(pre):
        return {'context': pre['context'], 'mask': pre['context_mask']}

    def video_pre(self, latents, timestep, context, context_mask,
                  condition_latents, clip_fea):
        """Embed video latents and conditions for the video expert."""
        return self.video_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=False,
            condition_latents=condition_latents,
            clip_fea=clip_fea)

    def forward(self, forward_mode='predict_action', **kwargs):
        """Dispatch to :meth:`predict_action`, the only supported mode."""
        if forward_mode != 'predict_action':
            raise ValueError(f'Unknown VPP2 head mode: {forward_mode}')
        return self.predict_action(**kwargs)

    @torch.no_grad()
    def predict_action(self,
                       condition_latents,
                       clip_fea,
                       context,
                       context_mask,
                       states,
                       action_seed,
                       video_seed,
                       action_horizon=32,
                       num_video_frames=17,
                       num_inference_steps=10,
                       sigma_shift=1.):
        """Sample one action chunk with frozen-video-cache Euler steps.

        Args:
            condition_latents (torch.Tensor): VAE latents of the condition
                video, ``[1, C, T_cond, H, W]``.
            clip_fea (torch.Tensor): CLIP features of the first condition
                frame.
            context (torch.Tensor): T5 text context ``[1, L, text_dim]``.
            context_mask (torch.Tensor): Boolean text mask ``[1, L]``.
            states (torch.Tensor): Normalized proprioception, see
                :meth:`append_proprio`.
            action_seed (int): CPU generator seed for the initial action
                noise.
            video_seed (int): CPU generator seed for the target video noise.
            action_horizon (int): Actions per chunk. Defaults to 32.
            num_video_frames (int): Target video length in pixel frames;
                must be ``4k + 1``. Defaults to 17.
            num_inference_steps (int): Euler steps for the action flow.
                Defaults to 10.
            sigma_shift (float): Shift override for the action inference
                schedule. Defaults to 1.0.

        Returns:
            torch.Tensor: Normalized actions ``[1, action_horizon,
            action_dim]`` in float32.
        """
        if context.shape[0] != 1:
            raise ValueError(
                'Cached inference accepts one episode per request')
        if num_video_frames % 4 != 1:
            raise ValueError('num_video_frames must be 4k+1')
        context, context_mask = self.append_proprio(context, context_mask,
                                                    states)
        actions = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=torch.Generator('cpu').manual_seed(action_seed),
            device='cpu',
            dtype=torch.float32).to(context)
        shape = list(condition_latents.shape)
        shape[2] = (num_video_frames - 1) // 4 + 1
        if shape[2] < max(condition_latents.shape[2],
                          self.action_visible_video_frames):
            raise ValueError(
                'Target video does not cover the condition/visible frames')
        video = torch.randn(
            shape,
            generator=torch.Generator('cpu').manual_seed(video_seed),
            device='cpu',
            dtype=torch.float32).to(condition_latents)
        tv = video.new_full((1, ), self.video_num_train_timesteps)
        vp = self.video_pre(video, tv, context, context_mask,
                            condition_latents, clip_fea)
        mask = self.attention_mask(vp, action_horizon)
        n = vp['tokens'].shape[1]
        cache = self.mot.prefill_video_cache(
            video_tokens=vp['tokens'],
            video_freqs=vp['freqs'],
            video_t_mod=vp['t_mod'],
            video_context_payload=self._context(vp),
            video_attention_mask=mask[:n, :n])
        ts, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps, actions.device, actions.dtype, sigma_shift)
        for t, delta in zip(ts, deltas):
            ap = self.action_expert.pre_dit(actions, t[None], context,
                                            context_mask)
            tokens = self.mot.forward_action_with_video_cache(
                action_tokens=ap['tokens'],
                action_freqs=ap['freqs'],
                action_t_mod=ap['t_mod'],
                action_context_payload=self._context(ap),
                video_kv_cache=cache,
                attention_mask=mask,
                video_seq_len=n)
            pred = self.action_expert.post_dit(tokens, ap)
            actions = self.infer_action_scheduler.step(pred, delta, actions)
        return actions.float()
