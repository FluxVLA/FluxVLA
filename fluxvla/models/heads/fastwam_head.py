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

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import HEADS
from ..third_party_models.fastwam.modules.schedulers.scheduler_continuous import \
    WanContinuousFlowMatchScheduler  # noqa: E501

__all__ = ['FastWAMHead', 'FastWAMJointHead', 'FastWAMIDMHead']


@HEADS.register_module()
class FastWAMHead(nn.Module):
    """FastWAM MoT diffusion head (video + action experts).

    Owns the trainable components of FastWAM -- the ``video`` and ``action``
    experts wrapped by the :class:`MoT` mixed-attention module, the optional
    proprioception encoder, and the flow-matching schedulers -- together with
    the training-loss and action-inference logic.

    The video latents and text ``context`` are produced upstream by
    :class:`~fluxvla.models.backbones.vlms.wan22_backbone.Wan22Backbone`, so
    this head consumes pre-encoded tensors. The numerical computation mirrors
    ``fastwam.models.wan22.fastwam.FastWAM`` (``training_loss`` /
    ``infer_action``) verbatim; the only change is that VAE encoding lives in
    the backbone, which is deterministic and therefore preserves the random
    draw order required for exact parity.
    """

    def __init__(
        self,
        video_expert: nn.Module,
        action_expert: nn.Module,
        mot: nn.Module,
        text_dim: int,
        proprio_dim: Optional[int] = None,
        temporal_downsample_factor: int = 4,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        # Register only ``mot`` as a submodule; the experts live inside
        # ``mot.mixtures`` and are exposed via properties below. Registering
        # them again as ``self.video_expert`` / ``self.action_expert`` would
        # alias the same modules under two paths, which breaks FSDP's
        # recursive auto-wrap (a block would be wrapped twice). The
        # ``video_expert`` / ``action_expert`` args must be the very modules
        # held by ``mot`` so the property views stay consistent.
        if mot.mixtures['video'] is not video_expert \
                or mot.mixtures['action'] is not action_expert:
            raise ValueError(
                '`mot` must hold the same `video_expert` / `action_expert` '
                'instances passed to FastWAMHead.')
        self.mot = mot

        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim,
                                             self.text_dim).to(
                                                 device=device,
                                                 dtype=torch_dtype)
        else:
            self.proprio_encoder = None

        self.temporal_downsample_factor = int(temporal_downsample_factor)

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

    # ``video_expert`` / ``action_expert`` are stored once inside
    # ``mot.mixtures`` (avoids submodule aliasing that breaks FSDP wrapping);
    # expose them as read-only views for the forward / inference logic.
    @property
    def video_expert(self) -> nn.Module:
        return self.mot.mixtures['video']

    @property
    def action_expert(self) -> nn.Module:
        return self.mot.mixtures['action']

    # ------------------------------------------------------------------
    # Helpers (ported verbatim from FastWAM)
    # ------------------------------------------------------------------
    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ):
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError('`proprio` must be 2D [B, D], got shape '
                             f'{tuple(proprio.shape)}')
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f'`proprio` last dim must be {self.proprio_dim}, '
                             f'got {proprio.shape[1]}')
        proprio_token = self.proprio_encoder(
            proprio.to(device=context.device,
                       dtype=context.dtype).unsqueeze(1)).to(
                           dtype=context.dtype)  # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1),
                                  dtype=torch.bool,
                                  device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)

        mask[:video_seq_len, :video_seq_len] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(),
            reduction='none').mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError('`temporal_downsample_factor` must be positive, '
                             f'got {temporal_factor}.')
        if image_is_pad.shape[1] < 1:
            raise ValueError('`image_is_pad` must contain at least one frame.')
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                'Cannot align `image_is_pad` with video latent steps: '
                f'num_frames={image_is_pad.shape[1]}, '
                f'temporal_downsample_factor={temporal_factor}.')

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1,
                                              temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad],
                                     dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError('Video-loss mask shape mismatch: '
                             f'mask steps={video_is_pad.shape[1]}, '
                             f'loss steps={video_loss_token.shape[1]}.')

        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ):
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre['tokens'].shape[1],
            action_seq_len=action_pre['tokens'].shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_pre['tokens'].device,
        )
        tokens_out = self.mot(
            embeds_all={
                'video': video_pre['tokens'],
                'action': action_pre['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_pre['freqs'],
                'action': action_pre['freqs'],
            },
            context_all={
                'video': {
                    'context': video_pre['context'],
                    'mask': video_pre['context_mask'],
                },
                'action': {
                    'context': action_pre['context'],
                    'mask': action_pre['context_mask'],
                },
            },
            t_mod_all={
                'video': video_pre['t_mod'],
                'action': action_pre['t_mod'],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out['video'], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out['action'],
                                                  action_pre)
        return pred_video, pred_action

    def _prepare_training_inputs(
        self,
        input_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        image_is_pad: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Move/validate training inputs and build the proprio-augmented
        context (no random draws), shared by the ``uncond`` and ``idm``
        forward passes so the noise-sampling order stays identical.
        """
        device = input_latents.device

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                '`context/context_mask` must be [B,L,D]/[B,L], got '
                f'{tuple(context.shape)} and {tuple(context_mask.shape)}')
        context = context.to(
            device=device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(
            device=device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    '`proprio` is required when `proprio_dim` is enabled.')
            if proprio.ndim != 3:
                raise ValueError('`proprio` must be 3D [B, T, d], got shape '
                                 f'{tuple(proprio.shape)}')
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f'`proprio` last dim must be {self.proprio_dim}, '
                    f'got {proprio.shape[2]}')
            proprio = proprio[:, 0, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=device, dtype=self.torch_dtype, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=device, dtype=torch.bool, non_blocking=True)

        return {
            'first_frame_latents': first_frame_latents,
            'fuse_flag': fuse_flag,
            'context': context,
            'context_mask': context_mask,
            'action': action,
            'action_is_pad': action_is_pad,
            'image_is_pad': image_is_pad,
        }

    # ------------------------------------------------------------------
    # Training forward (ported from FastWAM.build_inputs + training_loss)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
        image_is_pad: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = input_latents.device
        batch_size = input_latents.shape[0]

        prep = self._prepare_training_inputs(
            input_latents=input_latents,
            context=context,
            context_mask=context_mask,
            action=action,
            action_is_pad=action_is_pad,
            image_is_pad=image_is_pad,
            proprio=proprio,
        )
        first_frame_latents = prep['first_frame_latents']
        fuse_flag = prep['fuse_flag']
        context = prep['context']
        context_mask = prep['context_mask']
        action = prep['action']
        action_is_pad = prep['action_is_pad']
        image_is_pad = prep['image_is_pad']

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=input_latents.dtype)
        latents = self.train_video_scheduler.add_noise(input_latents,
                                                       noise_video,
                                                       timestep_video)
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video)

        if first_frame_latents is not None:
            latents[:, :, 0:1] = first_frame_latents

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre['tokens']
        action_tokens = action_pre['tokens']

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                'video': video_tokens,
                'action': action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_pre['freqs'],
                'action': action_pre['freqs'],
            },
            context_all={
                'video': {
                    'context': video_pre['context'],
                    'mask': video_pre['context_mask'],
                },
                'action': {
                    'context': action_pre['context'],
                    'mask': action_pre['context_mask'],
                },
            },
            t_mod_all={
                'video': video_pre['t_mod'],
                'action': action_pre['t_mod'],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out['video'], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out['action'],
                                                  action_pre)

        include_initial_video_step = first_frame_latents is None
        if first_frame_latents is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(
            timestep_video).to(
                loss_video_per_sample.device,
                dtype=loss_video_per_sample.dtype)
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(),
            reduction='none').mean(dim=2)  # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token *
                                      valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(
            timestep_action).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video +
            self.loss_lambda_action * loss_action)
        return {
            'loss': loss_total,
            'loss_video': (self.loss_lambda_video * loss_video).detach(),
            'loss_action': (self.loss_lambda_action * loss_action).detach(),
        }

    # ------------------------------------------------------------------
    # Action inference (ported from FastWAM.infer_action denoising loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre['tokens'],
            action_freqs=action_pre['freqs'],
            action_t_mod=action_pre['t_mod'],
            action_context_payload={
                'context': action_pre['context'],
                'mask': action_pre['context_mask'],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def predict_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        if str(getattr(self.video_expert, 'video_attention_mask_mode', '')) \
                != 'first_frame_causal':
            raise ValueError(
                '`predict_action` requires '
                "`video_attention_mask_mode='first_frame_causal'`.")

        device = first_frame_latents.device
        generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)

        fuse_flag = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0], ),
            dtype=first_frame_latents.dtype,
            device=device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_pre['tokens'].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre['tokens'],
            video_freqs=video_pre['freqs'],
            video_t_mod=video_pre['t_mod'],
            video_context_payload={
                'context': video_pre['context'],
                'mask': video_pre['context_mask'],
            },
            video_attention_mask=attention_mask[:video_seq_len, :
                                                video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = \
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        schedule = zip(infer_timesteps_action, infer_deltas_action)
        for step_t_action, step_delta_action in schedule:
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action)

        return latents_action

    @torch.no_grad()
    def predict_video_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        video_latent_shape,
        action: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ):
        self.eval()
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        latents_video = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_video[:, :, 0:1] = first_frame_latents.clone()

        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if (action.ndim != 3 or action.shape[0] != 1
                    or action.shape[1] != action_horizon):
                raise ValueError(
                    '`action` must have shape [T, D] or [1, T, D] '
                    f'with action_horizon={action_horizon}, got '
                    f'{tuple(action.shape)}')
            action = action.to(device=device, dtype=self.torch_dtype)

        fuse_flag = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))
        infer_timesteps_video, infer_deltas_video = \
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        infer_timesteps_action, infer_deltas_action = \
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        for step_t_video, step_delta_video, step_t_action, step_delta_action \
                in zip(infer_timesteps_video, infer_deltas_video,
                       infer_timesteps_action, infer_deltas_action):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=device)
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=device)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return latents_video, latents_action[0].detach().to(
            device='cpu', dtype=torch.float32)


@HEADS.register_module()
class FastWAMJointHead(FastWAMHead):
    """Joint FastWAM head: action attends to *all* video tokens.

    Mirrors ``fastwam.models.wan22.fastwam_joint.FastWAMJoint``. The only
    training-time difference from :class:`FastWAMHead` is the MoT attention
    mask (action sees the full video instead of just the first frame), so
    the inherited :meth:`forward` (video + action loss) is reused verbatim.
    Inference denoises video and action jointly.
    """

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)
        mask[:video_seq_len, :video_seq_len] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        mask[video_seq_len:, video_seq_len:] = True
        # action -> full video
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ):
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre['tokens'].shape[1],
            action_seq_len=action_pre['tokens'].shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_pre['tokens'].device,
        )
        tokens_out = self.mot(
            embeds_all={
                'video': video_pre['tokens'],
                'action': action_pre['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_pre['freqs'],
                'action': action_pre['freqs'],
            },
            context_all={
                'video': {
                    'context': video_pre['context'],
                    'mask': video_pre['context_mask'],
                },
                'action': {
                    'context': action_pre['context'],
                    'mask': action_pre['context_mask'],
                },
            },
            t_mod_all={
                'video': video_pre['t_mod'],
                'action': action_pre['t_mod'],
            },
        )
        pred_video = self.video_expert.post_dit(tokens_out['video'], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out['action'],
                                                  action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def predict_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        video_latent_shape,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        latents_video = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        infer_timesteps_video, infer_deltas_video = \
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        infer_timesteps_action, infer_deltas_action = \
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        for step_t_video, step_delta_video, step_t_action, step_delta_action \
                in zip(infer_timesteps_video, infer_deltas_video,
                       infer_timesteps_action, infer_deltas_action):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=device)
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=device)
            pred_video, pred_action = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        return latents_action


@HEADS.register_module()
class FastWAMIDMHead(FastWAMJointHead):
    """IDM FastWAM head: teacher-forcing video conditioning.

    Mirrors ``fastwam.models.wan22.fastwam_idm.FastWAMIDM``. Training runs
    three branches (noisy video, noisy action, teacher-forcing cond video
    noised with probability :attr:`video_cond_noise_prob`); inference is a
    two-stage process -- denoise the video fully, then denoise the action
    against the frozen video via the MoT KV cache.
    """

    # During training the cond-video is noised with this probability.
    video_cond_noise_prob = 0.5

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                'Teacher-forcing requires identical `tokens_per_frame` for '
                'noisy and cond video branches, got '
                f'{noisy_video_tokens_per_frame} and '
                f'{cond_video_tokens_per_frame}.')

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len),
                           dtype=torch.bool,
                           device=device)

        mask[:noisy_end, :noisy_end] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=noisy_video_seq_len,
                video_tokens_per_frame=noisy_video_tokens_per_frame,
                device=device,
            )
        mask[noisy_end:cond_end, noisy_end:cond_end] = \
            self.video_expert.build_video_to_video_mask(
                video_seq_len=cond_video_seq_len,
                video_tokens_per_frame=cond_video_tokens_per_frame,
                device=device,
            )
        mask[cond_end:, cond_end:] = True
        # action -> cond_video only
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def forward(
        self,
        input_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor] = None,
        image_is_pad: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = input_latents.device
        batch_size = input_latents.shape[0]

        prep = self._prepare_training_inputs(
            input_latents=input_latents,
            context=context,
            context_mask=context_mask,
            action=action,
            action_is_pad=action_is_pad,
            image_is_pad=image_is_pad,
            proprio=proprio,
        )
        first_frame_latents = prep['first_frame_latents']
        fuse_flag = prep['fuse_flag']
        context = prep['context']
        context_mask = prep['context_mask']
        action = prep['action']
        action_is_pad = prep['action_is_pad']
        image_is_pad = prep['image_is_pad']

        # Branch A: noisy video (for video denoising target).
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=input_latents.dtype)
        latents_noisy = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video)
        if first_frame_latents is not None:
            latents_noisy[:, :, 0:1] = first_frame_latents

        # Branch B: noisy action.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action)

        # Branch C: teacher-forcing cond-video, independently noised with
        # probability ``video_cond_noise_prob`` per sample.
        cond_noise_mask = torch.rand(
            (batch_size, ), device=device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros_like(
            timestep_video, dtype=input_latents.dtype, device=device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = \
                self.train_video_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=device,
                    dtype=input_latents.dtype)
            timestep_video_cond = torch.where(cond_noise_mask,
                                              timestep_video_cond_sampled,
                                              timestep_video_cond)
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents, noise_video_cond, timestep_video_cond_sampled)
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy,
                                       input_latents)
        if first_frame_latents is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = first_frame_latents

        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        if video_pre_noisy['t_mod'].ndim != 4 \
                or video_pre_cond['t_mod'].ndim != 4:
            raise ValueError(
                'Teacher-forcing requires token-wise `t_mod`; ensure '
                '`seperated_timestep=true` and '
                '`fuse_vae_embedding_in_latents=true`.')

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        noisy_video_seq_len = int(video_pre_noisy['tokens'].shape[1])
        cond_video_seq_len = int(video_pre_cond['tokens'].shape[1])
        noisy_video_tokens_per_frame = int(
            video_pre_noisy['meta']['tokens_per_frame'])
        cond_video_tokens_per_frame = int(
            video_pre_cond['meta']['tokens_per_frame'])

        merged_video_tokens = torch.cat(
            [video_pre_noisy['tokens'], video_pre_cond['tokens']], dim=1)
        merged_video_freqs = torch.cat(
            [video_pre_noisy['freqs'], video_pre_cond['freqs']], dim=0)
        merged_video_t_mod = torch.cat(
            [video_pre_noisy['t_mod'], video_pre_cond['t_mod']], dim=1)
        merged_video_context_mask = torch.cat(
            [video_pre_noisy['context_mask'], video_pre_cond['context_mask']],
            dim=1)

        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_pre['tokens'].shape[1],
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            cond_video_tokens_per_frame=cond_video_tokens_per_frame,
            device=merged_video_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={
                'video': merged_video_tokens,
                'action': action_pre['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': merged_video_freqs,
                'action': action_pre['freqs'],
            },
            context_all={
                'video': {
                    'context': video_pre_noisy['context'],
                    'mask': merged_video_context_mask,
                },
                'action': {
                    'context': action_pre['context'],
                    'mask': action_pre['context_mask'],
                },
            },
            t_mod_all={
                'video': merged_video_t_mod,
                'action': action_pre['t_mod'],
            },
        )

        # Only the noisy-video half contributes to the video denoising loss.
        pred_video_tokens = tokens_out['video'][:, :noisy_video_seq_len]
        pred_video = self.video_expert.post_dit(pred_video_tokens,
                                                video_pre_noisy)
        pred_action = self.action_expert.post_dit(tokens_out['action'],
                                                  action_pre)

        include_initial_video_step = first_frame_latents is None
        if first_frame_latents is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(
            timestep_video).to(
                loss_video_per_sample.device,
                dtype=loss_video_per_sample.dtype)
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(),
            reduction='none').mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token *
                                      valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(
            timestep_action).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video +
            self.loss_lambda_action * loss_action)
        return {
            'loss': loss_total,
            'loss_video': (self.loss_lambda_video * loss_video).detach(),
            'loss_action': (self.loss_lambda_action * loss_action).detach(),
        }

    @torch.no_grad()
    def predict_video_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        video_latent_shape,
        action: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ):
        del action
        self.eval()
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        latents_video = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        infer_timesteps_video, infer_deltas_video = \
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        for step_t_video, step_delta_video in zip(infer_timesteps_video,
                                                  infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=device)
            pred_video = self.video_expert(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        timestep_video_cond = torch.zeros(
            (latents_video.shape[0], ),
            dtype=latents_video.dtype,
            device=device,
        )
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre['meta']['tokens_per_frame']),
            device=video_pre['tokens'].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre['tokens'],
            video_freqs=video_pre['freqs'],
            video_t_mod=video_pre['t_mod'],
            video_context_payload={
                'context': video_pre['context'],
                'mask': video_pre['context_mask'],
            },
            video_attention_mask=attention_mask[:video_seq_len, :
                                                video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = \
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        for step_t_action, step_delta_action in zip(infer_timesteps_action,
                                                    infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action)

        return latents_video, latents_action[0].detach().to(
            device='cpu', dtype=torch.float32)

    @torch.no_grad()
    def predict_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        video_latent_shape,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ) -> torch.Tensor:
        self.eval()
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        latents_video = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        # Stage 1: denoise video only.
        infer_timesteps_video, infer_deltas_video = \
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        for step_t_video, step_delta_video in zip(infer_timesteps_video,
                                                  infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=device)
            pred_video = self.video_expert(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        # Stage 2: freeze denoised video as cond, denoise action via KV cache.
        timestep_video_cond = torch.zeros((latents_video.shape[0], ),
                                          dtype=latents_video.dtype,
                                          device=device)
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre_cond['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(
                video_pre_cond['meta']['tokens_per_frame']),
            device=video_pre_cond['tokens'].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond['tokens'],
            video_freqs=video_pre_cond['freqs'],
            video_t_mod=video_pre_cond['t_mod'],
            video_context_payload={
                'context': video_pre_cond['context'],
                'mask': video_pre_cond['context_mask'],
            },
            video_attention_mask=attention_mask[:video_seq_len, :
                                                video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = \
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        for step_t_action, step_delta_action in zip(infer_timesteps_action,
                                                    infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action)

        return latents_action
