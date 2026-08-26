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

from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from fluxvla.engines import HEADS, VLAS, initialize_overwatch
from fluxvla.engines.utils.fsdp_wrapping import build_module_wrap_policy
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.video_metrics import (pil_frames_to_video_tensor,
                                                 video_psnr, video_ssim)
from ..backbones.vlms.wan22_backbone import Wan22Backbone
from ..heads.fastwam_head import FastWAMHead
from ..third_party_models.fastwam.modules.action_dit import ActionDiT
from ..third_party_models.fastwam.modules.mot import MoT
from ..third_party_models.fastwam.modules.wan_video_dit import WanVideoDiT
from ..third_party_models.fastwam.modules.wan_video_text_encoder import \
    WanTextEncoder
from ..third_party_models.fastwam.modules.wan_video_vae import WanVideoVAE38
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)

__all__ = ['FastWAMVLA']

FastWAMVLAForwardMode = Literal['train', 'training_eval']


@VLAS.register_module()
class FastWAMVLA(BaseVLA):
    """FastWAM World-Action Model (uncond / joint / idm variants).

    Implemented from the Fast-WAM paper (https://arxiv.org/abs/2603.16666).
    Wraps the upstream Wan2.2 MoT world model as a FluxVLA VLA composed of:

    * :class:`~fluxvla.models.backbones.vlms.wan22_backbone.Wan22Backbone`
      (``vlm_backbone``) -- VAE + (optional) T5 encoding frontend.
    * a FastWAM head (``vla_head``) -- video/action experts, MoT mixed
      attention, flow-matching schedulers and the training-loss /
      action-inference logic. The head variant is selected by
      ``vla_head.type`` and built through the ``HEADS`` registry
      (:class:`FastWAMHead` for ``uncond``, :class:`FastWAMJointHead` for
      ``joint``, :class:`FastWAMIDMHead` for ``idm``).

    The Wan2.2 components are loaded once via the vendored FastWAM loader and
    injected into the backbone and head, so the encoders and the diffusion
    experts own disjoint parameter sets (clean ``state_dict`` / FSDP).

    Training follows the shared FluxVLA batch contract:
    ``images`` is ``[B, 3, T, H, W]`` after ``PrepareVideo``, ``states`` holds
    proprioception, ``actions`` holds the target action window, and
    ``action_masks`` / ``frame_masks`` use ``True`` for valid entries.
    """

    def __init__(
        self,
        vlm_backbone: Optional[Dict] = None,
        vla_head: Optional[Dict] = None,
        proprio_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        frame_window_size: Optional[int] = None,
        num_views: Optional[int] = None,
        mot_checkpoint_mixed_attn: bool = False,
        pretrained_name_or_path: Optional[str] = None,
        freeze_vlm_backbone: bool = True,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            vlm_backbone=None,
            vla_head=None,
            freeze_vlm_backbone=freeze_vlm_backbone,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=None,
            strict_mapping=True,
        )
        if pretrained_name_or_path is None:
            raise ValueError(
                'FastWAMVLA requires a complete `.safetensors` checkpoint.')
        if not str(pretrained_name_or_path).endswith('.safetensors'):
            raise ValueError(
                'FastWAM base weights must be a complete `.safetensors` '
                f'checkpoint, got: {pretrained_name_or_path}')
        # ``BaseVLA.device`` is a read-only property (parameter device), so
        # keep the requested build device under a private attribute.
        self._build_device = str(device)
        if isinstance(torch_dtype, str):
            torch_dtype = str_to_dtype(torch_dtype)
        self.torch_dtype = torch_dtype
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        self.action_horizon = (None if action_horizon is None else
                               int(action_horizon))
        self.num_views = None if num_views is None else int(num_views)
        self.frame_window_size = (None if frame_window_size is None else
                                  int(frame_window_size))

        backbone, head = self._build_components(
            backbone_cfg=dict(vlm_backbone or {}),
            head_cfg=dict(vla_head or {}),
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        self.vlm_backbone = backbone
        self.vla_head = head

        self.all_module_keys = ['vlm_backbone', 'vla_head']

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_components(
        self,
        backbone_cfg: Dict,
        head_cfg: Dict,
        mot_checkpoint_mixed_attn: bool,
    ) -> Tuple[Wan22Backbone, FastWAMHead]:
        """Construct FastWAM modules for loading a complete checkpoint."""
        backbone_cfg.pop('type', None)
        # Capture the head variant before stripping ``type`` (the head is
        # built explicitly below, not via the registry).
        head_type = head_cfg.pop('type', 'FastWAMHead')

        video_dit_config = head_cfg.get('video_dit_config')
        if video_dit_config is None:
            raise ValueError(
                '`vla_head.video_dit_config` is required for FastWAMVLA.')
        if 'text_dim' not in video_dit_config:
            raise ValueError('`video_dit_config[text_dim]` is required.')
        action_dit_config = head_cfg.get('action_dit_config') or {}
        device = self._build_device

        video_expert = WanVideoDiT(**video_dit_config).to(
            device=device, dtype=self.torch_dtype)
        action_expert = ActionDiT(**action_dit_config).to(
            device=device, dtype=self.torch_dtype)
        vae = WanVideoVAE38().to(device=device, dtype=self.torch_dtype)
        text_encoder = WanTextEncoder().to(
            device=device, dtype=self.torch_dtype)

        mot = MoT(
            mixtures={
                'video': video_expert,
                'action': action_expert
            },
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        backbone = Wan22Backbone(
            vae=vae,
            text_encoder=text_encoder,
            text_embed_cache_context_len=int(
                backbone_cfg.get('text_embed_cache_context_len', 128)),
            text_embed_cache_size=int(
                backbone_cfg.get('text_embed_cache_size', 256)),
            text_embed_cache_device=backbone_cfg.get('text_embed_cache_device',
                                                     'cpu'),
            device=device,
            torch_dtype=self.torch_dtype,
            freeze=True,
        )

        video_scheduler = head_cfg.get('video_scheduler') or {}
        action_scheduler = head_cfg.get('action_scheduler') or {}
        loss = head_cfg.get('loss') or {}
        head_cls = HEADS.get(head_type)
        if head_cls is None:
            raise KeyError(
                f'vla_head.type={head_type!r} is not registered in HEADS.')
        head = head_cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            text_dim=int(video_dit_config['text_dim']),
            proprio_dim=self.proprio_dim,
            temporal_downsample_factor=int(vae.temporal_downsample_factor),
            video_training_shift=float(
                video_scheduler.get('train_shift', 5.0)),
            video_inference_shift=float(
                video_scheduler.get('infer_shift', 5.0)),
            video_num_training_timesteps=int(
                video_scheduler.get('num_train_timesteps', 1000)),
            action_training_shift=float(
                action_scheduler.get('train_shift', 5.0)),
            action_inference_shift=float(
                action_scheduler.get('infer_shift', 5.0)),
            action_num_training_timesteps=int(
                action_scheduler.get('num_train_timesteps', 1000)),
            video_loss_weight=float(loss.get('lambda_video', 1.0)),
            action_loss_weight=float(loss.get('lambda_action', 1.0)),
            device=device,
            torch_dtype=self.torch_dtype,
        )
        return backbone, head

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        task_description=None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        frame_masks: Optional[torch.Tensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        forward_mode: FastWAMVLAForwardMode = 'train',
        training_eval_batch: Optional[Dict[str, Any]] = None,
        num_inference_steps: int = 10,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run training or in-training evaluation through the VLA wrapper.

        Args:
            images: Training video tensor.
            lang_tokens: Tokenized language input.
            lang_masks: Valid language-token mask.
            task_description: Optional task metadata.
            states: Optional proprioception sequence.
            actions: Training action targets.
            action_masks: Valid action positions.
            frame_masks: Valid video-frame positions.
            img_masks: Optional image mask input.
            forward_mode: Operation selected for this FSDP forward.
            training_eval_batch: Batch used by in-training evaluation.
            num_inference_steps: Number of diffusion inference steps.
            seed: Inference random seed.
            **kwargs: Additional compatibility arguments.

        Returns:
            Training losses or in-training evaluation outputs.
        """
        if forward_mode == 'train':
            return self._forward_training(
                images=images,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                states=states,
                actions=actions,
                action_masks=action_masks,
                frame_masks=frame_masks,
            )
        elif forward_mode == 'training_eval':
            if training_eval_batch is None:
                raise ValueError('Training eval requires '
                                 '`training_eval_batch`.')
            return self.compute_training_eval(
                batch=training_eval_batch,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        else:
            raise ValueError(
                f'Unsupported FastWAM VLA forward mode: {forward_mode!r}')

    def _forward_training(
        self,
        images: Optional[torch.Tensor],
        lang_tokens: Optional[torch.Tensor],
        lang_masks: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        action_masks: Optional[torch.Tensor],
        frame_masks: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the FastWAM flow-matching training losses.

        Args:
            images: Training video tensor, shape ``[B, 3, T, H, W]``.
            lang_tokens: Tokenized language input, shape ``[B, L]``.
            lang_masks: Valid language-token mask, shape ``[B, L]``.
            states: Optional proprioception sequence.
            actions: Training action targets.
            action_masks: Valid action positions.
            frame_masks: Valid video-frame positions.

        Returns:
            Total, video, and action loss tensors.
        """
        if lang_tokens is None or lang_masks is None:
            raise ValueError(
                'FastWAMVLA.forward requires both `lang_tokens` and '
                '`lang_masks`.')
        context, context_mask = self.vlm_backbone.encode_prompt_cached(
            lang_tokens, lang_masks)

        if images is None or actions is None:
            raise ValueError(
                'FastWAMVLA.forward requires `images` and `actions`.')
        if images.ndim != 5:
            raise ValueError('`images` must be 5D [B, 3, T, H, W], got shape '
                             f'{tuple(images.shape)}')
        if images.shape[1] != 3:
            raise ValueError('`images` channel dimension must be 3, got shape '
                             f'{tuple(images.shape)}')
        _, _, num_frames, height, width = images.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError('Video spatial dims must be multiples of 16, got '
                             f'H={height}, W={width}')
        if num_frames % 4 != 1:
            raise ValueError(
                f'Video T must satisfy T % 4 == 1, got {num_frames}')
        if num_frames <= 1:
            raise ValueError(
                f'Video T must be > 1 for action-conditioned training, '
                f'got {num_frames}')

        vlm_outputs = self.vlm_backbone(
            video=images,
            tiled=False,
        )
        video_latents = vlm_outputs['input_latents']
        proprio = states
        if proprio is not None and proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        action_padding_mask = None
        if action_masks is not None:
            action_padding_mask = ~action_masks.to(dtype=torch.bool)
        frame_padding_mask = None
        if frame_masks is not None:
            frame_padding_mask = ~frame_masks.to(dtype=torch.bool)

        return self.vla_head(
            video_latents=video_latents,
            context=context,
            context_mask=context_mask,
            actions=actions,
            action_padding_mask=action_padding_mask,
            frame_padding_mask=frame_padding_mask,
            proprio=proprio,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action(
        self,
        input_image: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        task_description=None,
        states: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        tiled: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # Adapt the shared ``LiberoParquetEvalDataset`` batch
        # (images / lang_tokens / lang_masks / states) to FastWAM inputs.
        # Explicit ``input_image`` / ``proprio`` take priority; tokenization
        # itself belongs to the data transform layer.
        if input_image is None and images is not None:
            if images.ndim != 5:
                raise ValueError('`images` must be 5D [B, C, T, H, W], got '
                                 f'{tuple(images.shape)}')
            input_image = images[:, :, 0]
        if proprio is None and states is not None:
            proprio = states
        if action_horizon is None:
            action_horizon = self.action_horizon
        if action_horizon is None:
            raise ValueError(
                '`action_horizon` must be provided or configured on the '
                'model via `action_horizon=`.')
        if input_image is None:
            raise ValueError(
                'predict_action requires `input_image` or `images`.')
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        first_frame_latents = self.vlm_backbone.encode_input_image_latents(
            input_image, tiled=tiled)

        context, context_mask = self._encode_language_context(
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )

        # Joint / idm heads denoise a full imagined video, so they need the
        # video latent shape. ``frame_window_size`` must be configured for
        # those variants; the uncond head ignores ``video_latent_shape``.
        video_latent_shape = None
        if self.frame_window_size is not None:
            height, width = int(input_image.shape[-2]), int(
                input_image.shape[-1])
            vae = self.vlm_backbone.vae
            z_dim = int(vae.model.z_dim)
            temporal_factor = int(vae.temporal_downsample_factor)
            upsampling_factor = int(vae.upsampling_factor)
            latent_t = (self.frame_window_size - 1) // temporal_factor + 1
            latent_h = height // upsampling_factor
            latent_w = width // upsampling_factor
            video_latent_shape = (z_dim, latent_t, latent_h, latent_w)

        return self.vla_head(
            forward_mode='predict_action',
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            action_horizon=action_horizon,
            video_latent_shape=video_latent_shape,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            proprio=proprio,
        )

    def _build_video_latent_shape(self, input_image: torch.Tensor,
                                  num_frames: int):
        height, width = int(input_image.shape[-2]), int(input_image.shape[-1])
        vae = self.vlm_backbone.vae
        z_dim = int(vae.model.z_dim)
        temporal_factor = int(vae.temporal_downsample_factor)
        upsampling_factor = int(vae.upsampling_factor)
        latent_t = (int(num_frames) - 1) // temporal_factor + 1
        latent_h = height // upsampling_factor
        latent_w = width // upsampling_factor
        return z_dim, latent_t, latent_h, latent_w

    def _encode_language_context(
        self,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the language context shared by inference paths.

        Args:
            lang_tokens: Tokenized language input, shape ``[B, L]``.
            lang_masks: Valid language-token mask, shape ``[B, L]``.

        Returns:
            Encoded language context and its validity mask.
        """
        if lang_tokens is None or lang_masks is None:
            raise ValueError(
                'FastWAMVLA inference requires both `lang_tokens` and '
                '`lang_masks`.')
        return self.vlm_backbone.encode_prompt_cached(lang_tokens, lang_masks)

    @torch.no_grad()
    def infer(
        self,
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = 'cpu',
        tiled: bool = False,
        task_description=None,
    ):
        if action_horizon is None:
            action_horizon = self.action_horizon
        if action_horizon is None:
            raise ValueError(
                '`action_horizon` must be provided or configured on the '
                'model via `action_horizon=`.')
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (input_image.ndim != 4 or input_image.shape[0] != 1
                or input_image.shape[1] != 3):
            raise ValueError(
                '`input_image` must have shape [1,3,H,W] or [3,H,W], '
                f'got {tuple(input_image.shape)}')
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                '`input_image` spatial dims must be multiples of 16, got '
                f'H={height}, W={width}.')
        if int(num_frames) % 4 != 1:
            raise ValueError(
                f'num_frames must satisfy T % 4 == 1, got {num_frames}.')

        first_frame_latents = self.vlm_backbone.encode_input_image_latents(
            input_image, tiled=tiled)
        context, context_mask = self._encode_language_context(
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        video_latent_shape = self._build_video_latent_shape(
            input_image, num_frames)
        video_latents, pred_actions = self.vla_head(
            forward_mode='predict_video_action',
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            action_horizon=int(action_horizon),
            video_latent_shape=video_latent_shape,
            conditioning_actions=action,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            proprio=proprio,
        )
        return {
            'video':
            self.vlm_backbone.decode_latents(video_latents, tiled=tiled),
            'action': pred_actions,
        }

    @staticmethod
    def _denormalize_min_max_action(action: torch.Tensor, stats: Dict):
        action_stats = None
        if isinstance(stats, dict):
            action_stats = stats.get('action') or stats.get('actions')
        if not action_stats or 'min' not in action_stats \
                or 'max' not in action_stats:
            return None
        action_min = torch.as_tensor(
            action_stats['min'], dtype=torch.float32, device=action.device)
        action_max = torch.as_tensor(
            action_stats['max'], dtype=torch.float32, device=action.device)
        while action_min.ndim < action.ndim:
            action_min = action_min.unsqueeze(0)
            action_max = action_max.unsqueeze(0)
        return (action.float() + 1.0) * 0.5 * (action_max - action_min +
                                               1e-6) + action_min

    @staticmethod
    def _select_first_meta_value(value):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return value[0]
        return value

    @staticmethod
    def _stitch_eval_video_frames(pred_video_tensor: torch.Tensor,
                                  vae_video_tensor: torch.Tensor,
                                  gt_video_tensor: torch.Tensor):
        stitched = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        frames = []
        for t in range(stitched.shape[1]):
            frame = (stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() *
                     255.0).astype(np.uint8)
            frames.append(Image.fromarray(frame))
        return frames

    @torch.no_grad()
    def compute_training_eval(
        self,
        batch: Dict,
        num_inference_steps: int = 10,
        seed: int = 42,
        **kwargs,
    ) -> Dict:
        output = self(**batch)
        val_loss = float(output['loss'].detach().float().item())

        video0 = batch['images'][0].detach()
        action = batch.get('actions')
        action0 = action[0].detach() if action is not None else None
        proprio = batch.get('states')
        proprio0 = None
        if proprio is not None:
            proprio0 = proprio[0].detach()
            if proprio0.ndim == 2:
                proprio0 = proprio0[0]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        lang_tokens = batch.get('lang_tokens')
        lang_masks = batch.get('lang_masks')
        lang_tokens0 = (
            lang_tokens[:1].detach() if lang_tokens is not None else None)
        lang_masks0 = (
            lang_masks[:1].detach() if lang_masks is not None else None)
        task_description = self._select_first_meta_value(
            batch.get('task_description'))

        pred = self.infer(
            input_image=input_image,
            num_frames=num_frames,
            action=action0,
            action_horizon=int(action0.shape[0])
            if action0 is not None else self.action_horizon,
            proprio=proprio0,
            lang_tokens=lang_tokens0,
            lang_masks=lang_masks0,
            task_description=task_description,
            num_inference_steps=num_inference_steps,
            seed=seed,
            tiled=False,
        )

        pred_video_tensor = pil_frames_to_video_tensor(pred['video'])
        gt_video_tensor = ((video0.float().cpu().clamp(-1.0, 1.0) + 1.0) *
                           0.5).contiguous()
        if pred_video_tensor.shape != gt_video_tensor.shape:
            raise ValueError('Eval infer prediction/GT shape mismatch: '
                             f'pred={tuple(pred_video_tensor.shape)} '
                             f'gt={tuple(gt_video_tensor.shape)}')
        psnr_rg = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rg = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        gt_video_batch = video0.unsqueeze(0).to(
            device=self.device, dtype=self.torch_dtype)
        vae_latents = self.vlm_backbone.encode_video_latents(
            gt_video_batch, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(
            self.vlm_backbone.decode_latents(vae_latents, tiled=False))
        if vae_video_tensor.shape != gt_video_tensor.shape:
            raise ValueError('Eval VAE reconstruction/GT shape mismatch: '
                             f'vae={tuple(vae_video_tensor.shape)} '
                             f'gt={tuple(gt_video_tensor.shape)}')

        psnr_dg = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_dg = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)
        psnr_rd = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rd = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        metrics = {
            'val_loss': val_loss,
            'psnr_rg': psnr_rg,
            'ssim_rg': ssim_rg,
            'psnr_rd': psnr_rd,
            'ssim_rd': ssim_rd,
            'psnr_dg': psnr_dg,
            'ssim_dg': ssim_dg,
        }

        pred_actions = pred.get('action')
        stats = self._select_first_meta_value(batch.get('stats'))
        if action0 is not None and pred_actions is not None:
            pred_denorm = self._denormalize_min_max_action(
                pred_actions.detach().cpu(), stats)
            gt_denorm = self._denormalize_min_max_action(
                action0.detach().cpu(), stats)
            if pred_denorm is not None and gt_denorm is not None:
                action_diff = pred_denorm - gt_denorm
                metrics['action_l2'] = float(action_diff.pow(2).mean().item())
                metrics['action_l1'] = float(action_diff.abs().mean().item())

        video_frames = self._stitch_eval_video_frames(pred_video_tensor,
                                                      vae_video_tensor,
                                                      gt_video_tensor)
        return {'metrics': metrics, 'video_frames': video_frames}

    # ------------------------------------------------------------------
    # BaseVLA abstract method implementations
    # ------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        # Wrap the whole head (MoT + video/action experts) as a single FSDP
        # unit. FastWAM's MoT does not call ``expert.forward`` -- it invokes
        # ``expert.pre_dit`` / ``expert.post_dit`` and reads block parameters
        # directly for mixed attention -- so wrapping the experts (or their
        # inner ``DiTBlock``s) would leave their parameters sharded (flat) at
        # access time, because FSDP only all-gathers around a module's
        # ``forward``. Wrapping at ``FastWAMHead`` makes ``head.forward`` the
        # FSDP boundary, so every parameter the head touches is materialized
        # for the whole step. The frozen VAE / T5 stay in the root unit and
        # are gathered at the top-level ``FastWAMVLA.forward``.
        return build_module_wrap_policy({FastWAMHead})

    @property
    def config(self):
        from transformers import PretrainedConfig
        cfg = PretrainedConfig()
        cfg.is_encoder_decoder = False
        return cfg
