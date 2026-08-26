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

from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import HEADS
from ..third_party_models.fastwam.modules.schedulers.scheduler_continuous import \
    WanContinuousFlowMatchScheduler  # noqa: E501

__all__ = ['FastWAMHead', 'FastWAMJointHead', 'FastWAMIDMHead']

VideoLatentShape = Tuple[int, int, int, int]
FastWAMHeadForwardMode = Literal['train', 'predict_action',
                                 'predict_video_action', ]


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
        video_training_shift: float = 5.0,
        video_inference_shift: float = 5.0,
        video_num_training_timesteps: int = 1000,
        action_training_shift: float = 5.0,
        action_inference_shift: float = 5.0,
        action_num_training_timesteps: int = 1000,
        video_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
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

        self.video_training_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_training_timesteps,
            shift=video_training_shift,
        )
        self.video_inference_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_training_timesteps,
            shift=video_inference_shift,
        )
        self.action_training_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_training_timesteps,
            shift=action_training_shift,
        )
        self.action_inference_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_training_timesteps,
            shift=action_inference_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.video_loss_weight = float(video_loss_weight)
        self.action_loss_weight = float(action_loss_weight)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode proprioception and append it to the language context."""
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2:
            raise ValueError('`proprio` must be 2D [B, D], got shape '
                             f'{tuple(proprio.shape)}')
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f'proprio last dim must be {self.proprio_dim}, '
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
        video_predictions: torch.Tensor,
        video_targets: torch.Tensor,
        frame_padding_mask: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_token_losses = F.mse_loss(
            video_predictions.float(), video_targets.float(),
            reduction='none').mean(dim=(1, 3, 4))
        if frame_padding_mask is None:
            return video_token_losses.mean(dim=1)

        temporal_factor = int(self.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError('`temporal_downsample_factor` must be positive, '
                             f'got {temporal_factor}.')
        if frame_padding_mask.shape[1] < 1:
            raise ValueError(
                '`frame_padding_mask` must contain at least one frame.')
        if (frame_padding_mask.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                'Cannot align `frame_padding_mask` with video latent steps: '
                f'num_frames={frame_padding_mask.shape[1]}, '
                f'temporal_downsample_factor={temporal_factor}.')

        tail_padding_mask = frame_padding_mask[:, 1:]
        latent_tail_padding_mask = tail_padding_mask.view(
            frame_padding_mask.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_padding_mask = torch.cat(
                [frame_padding_mask[:, :1], latent_tail_padding_mask], dim=1)
        else:
            video_padding_mask = latent_tail_padding_mask

        if video_padding_mask.shape[1] != video_token_losses.shape[1]:
            raise ValueError('Video-loss mask shape mismatch: '
                             f'mask steps={video_padding_mask.shape[1]}, '
                             f'loss steps={video_token_losses.shape[1]}.')

        valid_mask = (~video_padding_mask).to(
            device=video_token_losses.device, dtype=video_token_losses.dtype)
        valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
        return (video_token_losses * valid_mask).sum(dim=1) / valid_count

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        video_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        conditioning_actions: Optional[torch.Tensor] = None,
    ):
        video_dit_inputs = self.video_expert.pre_dit(
            x=video_latents,
            timestep=video_timesteps,
            context=context,
            context_mask=context_mask,
            action=conditioning_actions,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_dit_inputs = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=action_timesteps,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_dit_inputs['tokens'].shape[1],
            action_seq_len=action_dit_inputs['tokens'].shape[1],
            video_tokens_per_frame=int(
                video_dit_inputs['meta']['tokens_per_frame']),
            device=video_dit_inputs['tokens'].device,
        )
        mot_outputs = self.mot(
            embeds_all={
                'video': video_dit_inputs['tokens'],
                'action': action_dit_inputs['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_dit_inputs['freqs'],
                'action': action_dit_inputs['freqs'],
            },
            context_all={
                'video': {
                    'context': video_dit_inputs['context'],
                    'mask': video_dit_inputs['context_mask'],
                },
                'action': {
                    'context': action_dit_inputs['context'],
                    'mask': action_dit_inputs['context_mask'],
                },
            },
            t_mod_all={
                'video': video_dit_inputs['t_mod'],
                'action': action_dit_inputs['t_mod'],
            },
        )
        video_predictions = self.video_expert.post_dit(mot_outputs['video'],
                                                       video_dit_inputs)
        pred_actions = self.action_expert.post_dit(mot_outputs['action'],
                                                   action_dit_inputs)
        return video_predictions, pred_actions

    def _prepare_training_inputs(
        self,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        actions: torch.Tensor,
        action_padding_mask: Optional[torch.Tensor],
        frame_padding_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Validate training inputs and append proprioception to context."""
        device = video_latents.device

        first_frame_latents = None
        fuse_vae_embedding_in_latents = False
        if getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False):
            first_frame_latents = video_latents[:, :, 0:1]
            fuse_vae_embedding_in_latents = True

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
                    f'proprio last dim must be {self.proprio_dim}, '
                    f'got {proprio.shape[2]}')
            proprio = proprio[:, 0, :]  # [B, T, D] -> [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=device, dtype=self.torch_dtype),
            )
        actions = actions.to(
            device=device, dtype=self.torch_dtype, non_blocking=True)
        if action_padding_mask is not None:
            action_padding_mask = action_padding_mask.to(
                device=device, dtype=torch.bool, non_blocking=True)
        if frame_padding_mask is not None:
            frame_padding_mask = frame_padding_mask.to(
                device=device, dtype=torch.bool, non_blocking=True)

        return {
            'first_frame_latents': first_frame_latents,
            'fuse_vae_embedding_in_latents': fuse_vae_embedding_in_latents,
            'context': context,
            'context_mask': context_mask,
            'actions': actions,
            'action_padding_mask': action_padding_mask,
            'frame_padding_mask': frame_padding_mask,
        }

    # ------------------------------------------------------------------
    # Training forward (ported from FastWAM.build_inputs + training_loss)
    # ------------------------------------------------------------------
    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_latents: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_padding_mask: Optional[torch.Tensor] = None,
        frame_padding_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        forward_mode: FastWAMHeadForwardMode = 'train',
        first_frame_latents: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        video_latent_shape: Optional[VideoLatentShape] = None,
        conditioning_actions: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
    ) -> Any:
        """Run training or prediction inside the head FSDP boundary.

        Args:
            context: Encoded language context, shape ``[B, L, D]``.
            context_mask: Valid language tokens, shape ``[B, L]``.
            video_latents: Training video latents.
            actions: Training action targets.
            action_padding_mask: Padding positions in ``actions``.
            frame_padding_mask: Padding positions in the input video.
            proprio: Optional proprioception input.
            forward_mode: Operation to execute through this wrapper.
            first_frame_latents: Encoded first frame for prediction.
            action_horizon: Number of actions to predict.
            video_latent_shape: Predicted video latent shape.
            conditioning_actions: Optional video-conditioning actions.
            num_inference_steps: Number of diffusion inference steps.
            sigma_shift: Optional inference scheduler shift.
            seed: Optional inference random seed.
            rand_device: Device used to initialize inference noise.

        Returns:
            Training losses, predicted actions, or video-action prediction.
        """
        if forward_mode == 'train':
            if video_latents is None or actions is None:
                raise ValueError('Training requires `video_latents` and '
                                 '`actions`.')
            return self._forward_training(
                video_latents=video_latents,
                context=context,
                context_mask=context_mask,
                actions=actions,
                action_padding_mask=action_padding_mask,
                frame_padding_mask=frame_padding_mask,
                proprio=proprio,
            )
        elif forward_mode == 'predict_action':
            if first_frame_latents is None or action_horizon is None:
                raise ValueError('Prediction requires `first_frame_latents` '
                                 'and `action_horizon`.')
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
            return self.predict_action(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                action_horizon=action_horizon,
                video_latent_shape=video_latent_shape,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        elif forward_mode == 'predict_video_action':
            if first_frame_latents is None or action_horizon is None:
                raise ValueError('Prediction requires `first_frame_latents` '
                                 'and `action_horizon`.')
            if video_latent_shape is None:
                raise ValueError('Video-action prediction requires '
                                 '`video_latent_shape`.')
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
            return self.predict_video_action(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                action_horizon=action_horizon,
                video_latent_shape=video_latent_shape,
                conditioning_actions=conditioning_actions,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        else:
            raise ValueError(
                f'Unsupported FastWAM forward mode: {forward_mode!r}')

    def _forward_training(
        self,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        actions: torch.Tensor,
        action_padding_mask: Optional[torch.Tensor] = None,
        frame_padding_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        device = video_latents.device
        batch_size = video_latents.shape[0]

        prepared_inputs = self._prepare_training_inputs(
            video_latents=video_latents,
            context=context,
            context_mask=context_mask,
            actions=actions,
            action_padding_mask=action_padding_mask,
            frame_padding_mask=frame_padding_mask,
            proprio=proprio,
        )
        first_frame_latents = prepared_inputs['first_frame_latents']
        fuse_vae_embedding_in_latents = prepared_inputs[
            'fuse_vae_embedding_in_latents']
        context = prepared_inputs['context']
        context_mask = prepared_inputs['context_mask']
        actions = prepared_inputs['actions']
        action_padding_mask = prepared_inputs['action_padding_mask']
        frame_padding_mask = prepared_inputs['frame_padding_mask']

        video_noise = torch.randn_like(video_latents)
        video_timesteps = self.video_training_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=video_latents.dtype)
        noisy_video_latents = self.video_training_scheduler.add_noise(
            video_latents, video_noise, video_timesteps)
        video_targets = self.video_training_scheduler.training_target(
            video_latents, video_noise, video_timesteps)

        if first_frame_latents is not None:
            noisy_video_latents[:, :, 0:1] = first_frame_latents

        action_noise = torch.randn_like(actions)
        action_timesteps = self.action_training_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=actions.dtype)
        noisy_actions = self.action_training_scheduler.add_noise(
            actions, action_noise, action_timesteps)
        target_actions = self.action_training_scheduler.training_target(
            actions, action_noise, action_timesteps)

        video_dit_inputs = self.video_expert.pre_dit(
            x=noisy_video_latents,
            timestep=video_timesteps,
            context=context,
            context_mask=context_mask,
            action=actions,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_dit_inputs = self.action_expert.pre_dit(
            action_tokens=noisy_actions,
            timestep=action_timesteps,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_dit_inputs['tokens']
        action_tokens = action_dit_inputs['tokens']

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(
                video_dit_inputs['meta']['tokens_per_frame']),
            device=video_tokens.device,
        )
        mot_outputs = self.mot(
            embeds_all={
                'video': video_tokens,
                'action': action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_dit_inputs['freqs'],
                'action': action_dit_inputs['freqs'],
            },
            context_all={
                'video': {
                    'context': video_dit_inputs['context'],
                    'mask': video_dit_inputs['context_mask'],
                },
                'action': {
                    'context': action_dit_inputs['context'],
                    'mask': action_dit_inputs['context_mask'],
                },
            },
            t_mod_all={
                'video': video_dit_inputs['t_mod'],
                'action': action_dit_inputs['t_mod'],
            },
        )

        video_predictions = self.video_expert.post_dit(mot_outputs['video'],
                                                       video_dit_inputs)
        pred_actions = self.action_expert.post_dit(mot_outputs['action'],
                                                   action_dit_inputs)

        include_initial_video_step = first_frame_latents is None
        if first_frame_latents is not None:
            video_predictions = video_predictions[:, :, 1:]
            video_targets = video_targets[:, :, 1:]

        video_loss_per_sample = self._compute_video_loss_per_sample(
            video_predictions=video_predictions,
            video_targets=video_targets,
            frame_padding_mask=frame_padding_mask,
            include_initial_video_step=include_initial_video_step,
        )
        video_loss_weights = self.video_training_scheduler.training_weight(
            video_timesteps).to(
                video_loss_per_sample.device,
                dtype=video_loss_per_sample.dtype)
        video_loss = (video_loss_per_sample * video_loss_weights).mean()

        action_token_losses = F.mse_loss(
            pred_actions.float(), target_actions.float(),
            reduction='none').mean(dim=2)  # [B, T]
        if action_padding_mask is not None:
            valid_mask = (~action_padding_mask).to(
                device=action_token_losses.device,
                dtype=action_token_losses.dtype)
            valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_token_losses *
                                      valid_mask).sum(dim=1) / valid_count
        else:
            action_loss_per_sample = action_token_losses.mean(dim=1)

        action_loss_weights = self.action_training_scheduler.training_weight(
            action_timesteps).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype)
        action_loss = (action_loss_per_sample * action_loss_weights).mean()

        total_loss = (
            self.video_loss_weight * video_loss +
            self.action_loss_weight * action_loss)
        return {
            'loss': total_loss,
            'loss_video': (self.video_loss_weight * video_loss).detach(),
            'loss_action': (self.action_loss_weight * action_loss).detach(),
        }

    # ------------------------------------------------------------------
    # Action inference (ported from FastWAM.infer_action denoising loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        action_latents: torch.Tensor,
        action_timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_dit_inputs = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=action_timesteps,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_dit_inputs['tokens'],
            action_freqs=action_dit_inputs['freqs'],
            action_t_mod=action_dit_inputs['t_mod'],
            action_context_payload={
                'context': action_dit_inputs['context'],
                'mask': action_dit_inputs['context_mask'],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_dit_inputs)

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
        if str(getattr(self.video_expert, 'video_attention_mask_mode', '')) \
                != 'first_frame_causal':
            raise ValueError(
                '`predict_action` requires '
                "`video_attention_mask_mode='first_frame_causal'`.")

        device = first_frame_latents.device
        generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)

        fuse_vae_embedding_in_latents = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        video_timesteps = torch.zeros(
            (first_frame_latents.shape[0], ),
            dtype=first_frame_latents.dtype,
            device=device,
        )
        video_dit_inputs = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=video_timesteps,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(video_dit_inputs['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_latents.shape[1],
            video_tokens_per_frame=int(
                video_dit_inputs['meta']['tokens_per_frame']),
            device=video_dit_inputs['tokens'].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_dit_inputs['tokens'],
            video_freqs=video_dit_inputs['freqs'],
            video_t_mod=video_dit_inputs['t_mod'],
            video_context_payload={
                'context': video_dit_inputs['context'],
                'mask': video_dit_inputs['context_mask'],
            },
            video_attention_mask=attention_mask[:video_seq_len, :
                                                video_seq_len],
        )

        action_inference_timesteps, action_inference_deltas = \
            self.action_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=action_latents.dtype,
                shift_override=sigma_shift,
            )
        schedule = zip(action_inference_timesteps, action_inference_deltas)
        for action_timestep, action_delta in schedule:
            action_timesteps = action_timestep.unsqueeze(0).to(
                dtype=action_latents.dtype, device=device)
            pred_actions = self._predict_action_noise_with_cache(
                action_latents=action_latents,
                action_timesteps=action_timesteps,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            action_latents = self.action_inference_scheduler.step(
                pred_actions, action_delta, action_latents)

        return action_latents

    @torch.no_grad()
    def predict_video_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        video_latent_shape,
        conditioning_actions: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ):
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        video_latents = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        action_latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        video_latents[:, :, 0:1] = first_frame_latents.clone()

        if conditioning_actions is not None:
            if conditioning_actions.ndim == 2:
                conditioning_actions = conditioning_actions.unsqueeze(0)
            if (conditioning_actions.ndim != 3
                    or conditioning_actions.shape[0] != 1
                    or conditioning_actions.shape[1] != action_horizon):
                raise ValueError(
                    '`conditioning_actions` must have shape [T, D] or '
                    '[1, T, D] '
                    f'with action_horizon={action_horizon}, got '
                    f'{tuple(conditioning_actions.shape)}')
            conditioning_actions = conditioning_actions.to(
                device=device, dtype=self.torch_dtype)

        fuse_vae_embedding_in_latents = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))
        video_inference_timesteps, video_inference_deltas = \
            self.video_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=video_latents.dtype,
                shift_override=sigma_shift,
            )
        action_inference_timesteps, action_inference_deltas = \
            self.action_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=action_latents.dtype,
                shift_override=sigma_shift,
            )
        for video_timestep, video_delta, action_timestep, action_delta \
                in zip(video_inference_timesteps, video_inference_deltas,
                       action_inference_timesteps, action_inference_deltas):
            video_timesteps = video_timestep.unsqueeze(0).to(
                dtype=video_latents.dtype, device=device)
            action_timesteps = action_timestep.unsqueeze(0).to(
                dtype=action_latents.dtype, device=device)
            video_predictions, pred_actions = self._predict_joint_noise(
                video_latents=video_latents,
                action_latents=action_latents,
                video_timesteps=video_timesteps,
                action_timesteps=action_timesteps,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                conditioning_actions=conditioning_actions,
            )
            video_latents = self.video_inference_scheduler.step(
                video_predictions, video_delta, video_latents)
            action_latents = self.action_inference_scheduler.step(
                pred_actions, action_delta, action_latents)
            video_latents[:, :, 0:1] = first_frame_latents.clone()

        return video_latents, action_latents[0].detach().to(
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
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        video_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        conditioning_actions: Optional[torch.Tensor] = None,
    ):
        video_dit_inputs = self.video_expert.pre_dit(
            x=video_latents,
            timestep=video_timesteps,
            context=context,
            context_mask=context_mask,
            action=conditioning_actions,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_dit_inputs = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=action_timesteps,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_dit_inputs['tokens'].shape[1],
            action_seq_len=action_dit_inputs['tokens'].shape[1],
            video_tokens_per_frame=int(
                video_dit_inputs['meta']['tokens_per_frame']),
            device=video_dit_inputs['tokens'].device,
        )
        mot_outputs = self.mot(
            embeds_all={
                'video': video_dit_inputs['tokens'],
                'action': action_dit_inputs['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': video_dit_inputs['freqs'],
                'action': action_dit_inputs['freqs'],
            },
            context_all={
                'video': {
                    'context': video_dit_inputs['context'],
                    'mask': video_dit_inputs['context_mask'],
                },
                'action': {
                    'context': action_dit_inputs['context'],
                    'mask': action_dit_inputs['context_mask'],
                },
            },
            t_mod_all={
                'video': video_dit_inputs['t_mod'],
                'action': action_dit_inputs['t_mod'],
            },
        )
        video_predictions = self.video_expert.post_dit(mot_outputs['video'],
                                                       video_dit_inputs)
        pred_actions = self.action_expert.post_dit(mot_outputs['action'],
                                                   action_dit_inputs)
        return video_predictions, pred_actions

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
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        video_latents = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        action_latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        video_latents[:, :, 0:1] = first_frame_latents.clone()
        fuse_vae_embedding_in_latents = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        video_inference_timesteps, video_inference_deltas = \
            self.video_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=video_latents.dtype,
                shift_override=sigma_shift,
            )
        action_inference_timesteps, action_inference_deltas = \
            self.action_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=action_latents.dtype,
                shift_override=sigma_shift,
            )
        for video_timestep, video_delta, action_timestep, action_delta \
                in zip(video_inference_timesteps, video_inference_deltas,
                       action_inference_timesteps, action_inference_deltas):
            video_timesteps = video_timestep.unsqueeze(0).to(
                dtype=video_latents.dtype, device=device)
            action_timesteps = action_timestep.unsqueeze(0).to(
                dtype=action_latents.dtype, device=device)
            video_predictions, pred_actions = self._predict_joint_noise(
                video_latents=video_latents,
                action_latents=action_latents,
                video_timesteps=video_timesteps,
                action_timesteps=action_timesteps,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                conditioning_actions=None,
            )
            video_latents = self.video_inference_scheduler.step(
                video_predictions, video_delta, video_latents)
            action_latents = self.action_inference_scheduler.step(
                pred_actions, action_delta, action_latents)
            video_latents[:, :, 0:1] = first_frame_latents.clone()

        return action_latents


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
        conditioning_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        conditioning_video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if noisy_video_tokens_per_frame != conditioning_video_tokens_per_frame:
            raise ValueError(
                'Teacher-forcing requires identical `tokens_per_frame` for '
                'noisy and cond video branches, got '
                f'{noisy_video_tokens_per_frame} and '
                f'{conditioning_video_tokens_per_frame}.')

        noisy_end = noisy_video_seq_len
        cond_end = noisy_video_seq_len + conditioning_video_seq_len
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
                video_seq_len=conditioning_video_seq_len,
                video_tokens_per_frame=conditioning_video_tokens_per_frame,
                device=device,
            )
        mask[cond_end:, cond_end:] = True
        # action -> cond_video only
        mask[cond_end:, noisy_end:cond_end] = True
        return mask

    def _forward_training(
        self,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        actions: torch.Tensor,
        action_padding_mask: Optional[torch.Tensor] = None,
        frame_padding_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        device = video_latents.device
        batch_size = video_latents.shape[0]

        prepared_inputs = self._prepare_training_inputs(
            video_latents=video_latents,
            context=context,
            context_mask=context_mask,
            actions=actions,
            action_padding_mask=action_padding_mask,
            frame_padding_mask=frame_padding_mask,
            proprio=proprio,
        )
        first_frame_latents = prepared_inputs['first_frame_latents']
        fuse_vae_embedding_in_latents = prepared_inputs[
            'fuse_vae_embedding_in_latents']
        context = prepared_inputs['context']
        context_mask = prepared_inputs['context_mask']
        actions = prepared_inputs['actions']
        action_padding_mask = prepared_inputs['action_padding_mask']
        frame_padding_mask = prepared_inputs['frame_padding_mask']

        # Branch A: noisy video (for video denoising target).
        video_noise = torch.randn_like(video_latents)
        video_timesteps = self.video_training_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=video_latents.dtype)
        noisy_video_latents = self.video_training_scheduler.add_noise(
            video_latents, video_noise, video_timesteps)
        video_targets = self.video_training_scheduler.training_target(
            video_latents, video_noise, video_timesteps)
        if first_frame_latents is not None:
            noisy_video_latents[:, :, 0:1] = first_frame_latents

        # Branch B: noisy action.
        action_noise = torch.randn_like(actions)
        action_timesteps = self.action_training_scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=actions.dtype)
        noisy_actions = self.action_training_scheduler.add_noise(
            actions, action_noise, action_timesteps)
        target_actions = self.action_training_scheduler.training_target(
            actions, action_noise, action_timesteps)

        # Branch C: teacher-forcing cond-video, independently noised with
        # probability ``video_cond_noise_prob`` per sample.
        conditioning_noise_mask = torch.rand(
            (batch_size, ), device=device) < float(self.video_cond_noise_prob)
        conditioning_video_timesteps = torch.zeros_like(
            video_timesteps, dtype=video_latents.dtype, device=device)
        conditioning_video_latents = video_latents
        if bool(conditioning_noise_mask.any()):
            sampled_conditioning_video_timesteps = \
                self.video_training_scheduler.sample_training_t(
                    batch_size=batch_size,
                    device=device,
                    dtype=video_latents.dtype)
            conditioning_video_timesteps = torch.where(
                conditioning_noise_mask, sampled_conditioning_video_timesteps,
                conditioning_video_timesteps)
            conditioning_video_noise = torch.randn_like(video_latents)
            noisy_conditioning_video_latents = \
                self.video_training_scheduler.add_noise(
                    video_latents, conditioning_video_noise,
                    sampled_conditioning_video_timesteps)
            conditioning_noise_selector = conditioning_noise_mask.view(
                batch_size, 1, 1, 1, 1)
            conditioning_video_latents = torch.where(
                conditioning_noise_selector,
                noisy_conditioning_video_latents,
                video_latents,
            )
        if first_frame_latents is not None:
            conditioning_video_latents = conditioning_video_latents.clone()
            conditioning_video_latents[:, :, 0:1] = first_frame_latents

        noisy_video_dit_inputs = self.video_expert.pre_dit(
            x=noisy_video_latents,
            timestep=video_timesteps,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        conditioning_video_dit_inputs = self.video_expert.pre_dit(
            x=conditioning_video_latents,
            timestep=conditioning_video_timesteps,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        if noisy_video_dit_inputs['t_mod'].ndim != 4 \
                or conditioning_video_dit_inputs['t_mod'].ndim != 4:
            raise ValueError(
                'Teacher-forcing requires token-wise `t_mod`; ensure '
                '`seperated_timestep=true` and '
                '`fuse_vae_embedding_in_latents=true`.')

        action_dit_inputs = self.action_expert.pre_dit(
            action_tokens=noisy_actions,
            timestep=action_timesteps,
            context=context,
            context_mask=context_mask,
        )

        noisy_video_seq_len = int(noisy_video_dit_inputs['tokens'].shape[1])
        conditioning_video_seq_len = int(
            conditioning_video_dit_inputs['tokens'].shape[1])
        noisy_video_tokens_per_frame = int(
            noisy_video_dit_inputs['meta']['tokens_per_frame'])
        conditioning_video_tokens_per_frame = int(
            conditioning_video_dit_inputs['meta']['tokens_per_frame'])

        merged_video_tokens = torch.cat([
            noisy_video_dit_inputs['tokens'],
            conditioning_video_dit_inputs['tokens']
        ],
                                        dim=1)
        merged_video_freqs = torch.cat([
            noisy_video_dit_inputs['freqs'],
            conditioning_video_dit_inputs['freqs']
        ],
                                       dim=0)
        merged_video_t_mod = torch.cat([
            noisy_video_dit_inputs['t_mod'],
            conditioning_video_dit_inputs['t_mod']
        ],
                                       dim=1)
        merged_video_context_mask = torch.cat([
            noisy_video_dit_inputs['context_mask'],
            conditioning_video_dit_inputs['context_mask']
        ],
                                              dim=1)

        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            conditioning_video_seq_len=conditioning_video_seq_len,
            action_seq_len=action_dit_inputs['tokens'].shape[1],
            noisy_video_tokens_per_frame=noisy_video_tokens_per_frame,
            conditioning_video_tokens_per_frame=  # noqa: E251
            conditioning_video_tokens_per_frame,
            device=merged_video_tokens.device,
        )

        mot_outputs = self.mot(
            embeds_all={
                'video': merged_video_tokens,
                'action': action_dit_inputs['tokens'],
            },
            attention_mask=attention_mask,
            freqs_all={
                'video': merged_video_freqs,
                'action': action_dit_inputs['freqs'],
            },
            context_all={
                'video': {
                    'context': noisy_video_dit_inputs['context'],
                    'mask': merged_video_context_mask,
                },
                'action': {
                    'context': action_dit_inputs['context'],
                    'mask': action_dit_inputs['context_mask'],
                },
            },
            t_mod_all={
                'video': merged_video_t_mod,
                'action': action_dit_inputs['t_mod'],
            },
        )

        # Only the noisy-video half contributes to the video denoising loss.
        video_prediction_tokens = mot_outputs['video'][:, :noisy_video_seq_len]
        video_predictions = self.video_expert.post_dit(video_prediction_tokens,
                                                       noisy_video_dit_inputs)
        pred_actions = self.action_expert.post_dit(mot_outputs['action'],
                                                   action_dit_inputs)

        include_initial_video_step = first_frame_latents is None
        if first_frame_latents is not None:
            video_predictions = video_predictions[:, :, 1:]
            video_targets = video_targets[:, :, 1:]

        video_loss_per_sample = self._compute_video_loss_per_sample(
            video_predictions=video_predictions,
            video_targets=video_targets,
            frame_padding_mask=frame_padding_mask,
            include_initial_video_step=include_initial_video_step,
        )
        video_loss_weights = self.video_training_scheduler.training_weight(
            video_timesteps).to(
                video_loss_per_sample.device,
                dtype=video_loss_per_sample.dtype)
        video_loss = (video_loss_per_sample * video_loss_weights).mean()

        action_token_losses = F.mse_loss(
            pred_actions.float(), target_actions.float(),
            reduction='none').mean(dim=2)
        if action_padding_mask is not None:
            valid_mask = (~action_padding_mask).to(
                device=action_token_losses.device,
                dtype=action_token_losses.dtype)
            valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_token_losses *
                                      valid_mask).sum(dim=1) / valid_count
        else:
            action_loss_per_sample = action_token_losses.mean(dim=1)

        action_loss_weights = self.action_training_scheduler.training_weight(
            action_timesteps).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype)
        action_loss = (action_loss_per_sample * action_loss_weights).mean()

        total_loss = (
            self.video_loss_weight * video_loss +
            self.action_loss_weight * action_loss)
        return {
            'loss': total_loss,
            'loss_video': (self.video_loss_weight * video_loss).detach(),
            'loss_action': (self.action_loss_weight * action_loss).detach(),
        }

    @torch.no_grad()
    def predict_video_action(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        video_latent_shape,
        conditioning_actions: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        **kwargs,
    ):
        del conditioning_actions
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        video_latents = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        action_latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        video_latents[:, :, 0:1] = first_frame_latents.clone()
        fuse_vae_embedding_in_latents = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        video_inference_timesteps, video_inference_deltas = \
            self.video_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=video_latents.dtype,
                shift_override=sigma_shift,
            )
        for video_timestep, video_delta in zip(video_inference_timesteps,
                                               video_inference_deltas):
            video_timesteps = video_timestep.unsqueeze(0).to(
                dtype=video_latents.dtype, device=device)
            video_predictions = self.video_expert(
                x=video_latents,
                timestep=video_timesteps,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            video_latents = self.video_inference_scheduler.step(
                video_predictions, video_delta, video_latents)
            video_latents[:, :, 0:1] = first_frame_latents.clone()

        conditioning_video_timesteps = torch.zeros(
            (video_latents.shape[0], ),
            dtype=video_latents.dtype,
            device=device,
        )
        video_dit_inputs = self.video_expert.pre_dit(
            x=video_latents,
            timestep=conditioning_video_timesteps,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(video_dit_inputs['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_latents.shape[1],
            video_tokens_per_frame=int(
                video_dit_inputs['meta']['tokens_per_frame']),
            device=video_dit_inputs['tokens'].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_dit_inputs['tokens'],
            video_freqs=video_dit_inputs['freqs'],
            video_t_mod=video_dit_inputs['t_mod'],
            video_context_payload={
                'context': video_dit_inputs['context'],
                'mask': video_dit_inputs['context_mask'],
            },
            video_attention_mask=attention_mask[:video_seq_len, :
                                                video_seq_len],
        )

        action_inference_timesteps, action_inference_deltas = \
            self.action_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=action_latents.dtype,
                shift_override=sigma_shift,
            )
        for action_timestep, action_delta in zip(action_inference_timesteps,
                                                 action_inference_deltas):
            action_timesteps = action_timestep.unsqueeze(0).to(
                dtype=action_latents.dtype, device=device)
            pred_actions = self._predict_action_noise_with_cache(
                action_latents=action_latents,
                action_timesteps=action_timesteps,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            action_latents = self.action_inference_scheduler.step(
                pred_actions, action_delta, action_latents)

        return video_latents, action_latents[0].detach().to(
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
        device = first_frame_latents.device
        z_dim, latent_t, latent_h, latent_w = video_latent_shape

        video_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        action_generator = (None if seed is None else torch.Generator(
            device=rand_device).manual_seed(seed))
        video_latents = torch.randn(
            (1, z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        action_latents = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(
            device=device, dtype=self.torch_dtype)
        video_latents[:, :, 0:1] = first_frame_latents.clone()
        fuse_vae_embedding_in_latents = bool(
            getattr(self.video_expert, 'fuse_vae_embedding_in_latents', False))

        # Stage 1: denoise video only.
        video_inference_timesteps, video_inference_deltas = \
            self.video_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=video_latents.dtype,
                shift_override=sigma_shift,
            )
        for video_timestep, video_delta in zip(video_inference_timesteps,
                                               video_inference_deltas):
            video_timesteps = video_timestep.unsqueeze(0).to(
                dtype=video_latents.dtype, device=device)
            video_predictions = self.video_expert(
                x=video_latents,
                timestep=video_timesteps,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            video_latents = self.video_inference_scheduler.step(
                video_predictions, video_delta, video_latents)
            video_latents[:, :, 0:1] = first_frame_latents.clone()

        # Stage 2: freeze denoised video as cond, denoise action via KV cache.
        conditioning_video_timesteps = torch.zeros((video_latents.shape[0], ),
                                                   dtype=video_latents.dtype,
                                                   device=device)
        conditioning_video_dit_inputs = self.video_expert.pre_dit(
            x=video_latents,
            timestep=conditioning_video_timesteps,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(conditioning_video_dit_inputs['tokens'].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_latents.shape[1],
            video_tokens_per_frame=int(
                conditioning_video_dit_inputs['meta']['tokens_per_frame']),
            device=conditioning_video_dit_inputs['tokens'].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=conditioning_video_dit_inputs['tokens'],
            video_freqs=conditioning_video_dit_inputs['freqs'],
            video_t_mod=conditioning_video_dit_inputs['t_mod'],
            video_context_payload={
                'context': conditioning_video_dit_inputs['context'],
                'mask': conditioning_video_dit_inputs['context_mask'],
            },
            video_attention_mask=attention_mask[:video_seq_len, :
                                                video_seq_len],
        )

        action_inference_timesteps, action_inference_deltas = \
            self.action_inference_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=device,
                dtype=action_latents.dtype,
                shift_override=sigma_shift,
            )
        for action_timestep, action_delta in zip(action_inference_timesteps,
                                                 action_inference_deltas):
            action_timesteps = action_timestep.unsqueeze(0).to(
                dtype=action_latents.dtype, device=device)
            pred_actions = self._predict_action_noise_with_cache(
                action_latents=action_latents,
                action_timesteps=action_timesteps,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            action_latents = self.action_inference_scheduler.step(
                pred_actions, action_delta, action_latents)

        return action_latents
