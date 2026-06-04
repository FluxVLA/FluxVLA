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

import copy
from functools import partial
from typing import Callable, Dict, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy

from fluxvla.datasets.utils.sarm_utils import pad_state_to_max_dim
from fluxvla.engines import VLAS
from fluxvla.models.backbones.llms.arm import ARMBackbone
from .base_vla import BaseVLA


class FocalLoss(nn.Module):
    """Binary focal loss for ARM task-success classification.

    Args:
        alpha (float): Focal-loss alpha balancing factor.
        gamma (float): Focal-loss gamma focusing parameter.
        reduction (str): Reduction mode, one of ``mean``, ``sum``, or
            ``none``.
    """

    def __init__(self,
                 alpha: float = 2.0,
                 gamma: float = 2.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from logits and binary targets."""
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = self.alpha * (1.0 - p_t)**self.gamma * ce_loss
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


@VLAS.register_module()
class ARMRewardModel(BaseVLA):
    """Advantage reward model for long-horizon robot manipulation.

    Official implementation of https://arxiv.org/abs/2604.03037

    ARM (Advantage Reward Modeling) estimates **relative advantage** instead of
    brittle absolute progress. A causal frame window is encoded with CLIP, and
    a shared temporal transformer predicts:

    * **Interval head**: tri-state labels over adjacent frame pairs
      (Progressive ``+1``, Stagnant ``0``, Regressive ``-1``), supervised by
      ``interval_targets`` derived from dataset ``progress``.
    * **Success head**: whether the current frame has completed the task,
      supervised by ``progress >= 1 - success_eps``.

    Training minimizes ``lambda_interval * CE + lambda_cls * focal_loss``.

    Args:
        llm_backbone (Optional[Dict]): Backbone config passed to the
            registry builder. Set ``pretrained_name_or_path`` here for the
            CLIP checkpoint used by :class:`ARMBackbone`.
        hidden_dim (int): Hidden dimension of the temporal transformer.
        num_heads (int): Number of transformer attention heads.
        num_layers (int): Number of transformer encoder layers.
        max_state_dim (int): Padded robot state feature dimension.
        dropout (float): Transformer dropout probability.
        n_history_steps (int): Number of history frames before the current
            frame in the causal window.
        frame_gap (int): Frame stride between observations in the dataset.
        freeze_clip_backbone (bool): Whether to freeze CLIP parameters.
        freeze_llm_backbone (bool): Whether to freeze the full ARM
            backbone, including temporal heads.
        lambda_interval (float): Loss weight for the interval head.
        lambda_cls (float): Loss weight for the success head.
        success_eps (float): Progress threshold for labeling success at
            the current frame.
        pretrained_name_or_path (Optional[str]): Optional compatibility
            argument passed to ``BaseVLA``.
    """

    def __init__(self,
                 llm_backbone: Optional[Dict] = None,
                 hidden_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 8,
                 max_state_dim: int = 32,
                 dropout: float = 0.1,
                 n_history_steps: int = 4,
                 frame_gap: int = 30,
                 freeze_clip_backbone: bool = True,
                 freeze_llm_backbone: bool = False,
                 lambda_interval: float = 1.0,
                 lambda_cls: float = 1.0,
                 success_eps: float = 1e-3,
                 pretrained_name_or_path: Optional[str] = None,
                 *args,
                 **kwargs) -> None:
        del args, kwargs
        self.n_history_steps = n_history_steps
        self.n_obs_steps = n_history_steps
        self.frame_gap = frame_gap
        self.max_state_dim = max_state_dim
        self.lambda_interval = lambda_interval
        self.lambda_cls = lambda_cls
        self.success_eps = success_eps
        self.freeze_clip_backbone = freeze_clip_backbone
        self.freeze_llm_backbone = freeze_llm_backbone

        llm_backbone_cfg = dict(
            type='ARMBackbone',
            pretrained_name_or_path='./checkpoints/clip-vit-base-patch32',
            hidden_dim=hidden_dim,
            max_state_dim=max_state_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_history_steps=n_history_steps,
            freeze_clip_backbone=freeze_clip_backbone,
        )
        if llm_backbone:
            llm_backbone_cfg.update(copy.deepcopy(llm_backbone))

        super().__init__(
            llm_backbone=llm_backbone_cfg,
            pretrained_name_or_path=pretrained_name_or_path,
            freeze_vision_backbone=True,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_vlm_backbone=True,
            freeze_projector=True,
            enable_mixed_precision_training=True,
            ignore_index=-100,
            norm_stats=None,
            name_mapping=None,
            strict_mapping=False,
        )
        self.focal_loss = FocalLoss(alpha=2.0, gamma=2.0, reduction='mean')

        self.all_module_keys = ['llm_backbone']
        self.trainable_module_keys = ['llm_backbone']

    @property
    def config(self):
        """Return optional Hugging Face-style model config.

        Returns:
            None: ARMRewardModel does not expose a transformer config object.
        """
        return None

    def freeze_backbones(self) -> None:
        """Apply ARM-specific freeze policy to the trainable backbone.

        ``ARMRewardModel`` only owns ``llm_backbone`` as a trainable module.
        ``freeze_llm_backbone`` freezes that whole backbone. When the backbone
        remains trainable, ``freeze_clip_backbone`` can still freeze the CLIP
        image/text encoder while leaving the temporal transformer trainable.
        ``trainable_module_keys`` is refreshed to match the resulting parameter
        state used by the runner and checkpoint utilities.
        """
        backbone = self._backbone()
        backbone.requires_grad_(not self.freeze_llm_backbone)
        if self.freeze_clip_backbone:
            backbone.clip_model.requires_grad_(False)
        self.trainable_module_keys = []
        if any(param.requires_grad for param in self.parameters()):
            self.trainable_module_keys.append('llm_backbone')

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _backbone(self) -> ARMBackbone:
        assert self.llm_backbone is not None
        return cast(ARMBackbone, self.llm_backbone)

    def _build_success_targets(self, progress: torch.Tensor) -> torch.Tensor:
        """Build binary success labels from scalar progress."""
        return (progress >= (1.0 - self.success_eps)).float()

    def forward(self,
                images: torch.Tensor,
                text_input_ids: torch.Tensor,
                text_attention_mask: torch.Tensor,
                states: torch.Tensor,
                lengths: torch.Tensor,
                interval_targets: Optional[torch.Tensor] = None,
                progress: Optional[torch.Tensor] = None,
                sparse_targets: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Compute ARM training losses and diagnostic metrics.

        Args:
            images (torch.Tensor): Image sequence, typically
                ``[B, T, N, C, H, W]`` or ``[B, T, C, H, W]``.
            text_input_ids (torch.Tensor): CLIP token ids for task text.
            text_attention_mask (torch.Tensor): Attention mask for text tokens.
            states (torch.Tensor): Robot states with shape ``[B, T, Ds]``.
            lengths (torch.Tensor): Valid frame count per sample, shape
                ``[B]``.
            interval_targets (Optional[torch.Tensor]): Tri-state interval
                labels in ``{-1, 0, +1}`` with shape ``[B, T-1]``.
            progress (Optional[torch.Tensor]): Scalar progress at the current
                frame, used to build success targets.
            sparse_targets (Optional[torch.Tensor]): Unused compatibility field
                kept for shared collators with SARM configs.
            **kwargs: Unused extra batch fields.

        Returns:
            Dict[str, torch.Tensor]: Total loss plus detached interval/success
            loss and accuracy metrics.
        """
        del kwargs, sparse_targets
        device = self._device()
        backbone = self._backbone()
        if images.dim() == 5:
            images = images.unsqueeze(2)
        images = images.to(
            device=device, dtype=next(backbone.clip_model.parameters()).dtype)
        text_input_ids = text_input_ids.to(device=device)
        text_attention_mask = text_attention_mask.to(device=device)
        states = states.to(device=device, dtype=images.dtype)
        lengths = lengths.to(device=device)

        if interval_targets is None:
            raise ValueError(
                'ARMRewardModel requires `interval_targets` in the batch.')
        if progress is None:
            raise ValueError(
                'ARMRewardModel requires `progress` in the batch.')
        interval_targets = interval_targets.to(device=device)
        progress = progress.to(device=device)

        states = pad_state_to_max_dim(states, self.max_state_dim)
        image_features = backbone.encode_images(images)
        text_features = backbone.encode_text(text_input_ids,
                                             text_attention_mask)
        # ARM currently uses one camera stream. Collapse camera dim if present.
        if image_features.shape[2] != 1:
            raise ValueError('ARMRewardModel expects one camera stream, got '
                             f'{image_features.shape[2]}.')
        image_features = image_features[:, :, 0, :]

        if progress.dim() == 2 and progress.shape[-1] == 1:
            progress = progress.squeeze(-1)
        if progress.dim() == 0:
            progress = progress.unsqueeze(0)
        progress = progress.float()
        interval_targets = interval_targets.long()
        cls_targets = self._build_success_targets(progress)

        interval_logits, cls_logits = backbone.temporal_model(
            video_features=image_features,
            state_features=states,
            text_features=text_features,
            lengths=lengths,
        )

        batch_size, num_intervals, num_classes = interval_logits.shape
        if num_classes != 3:
            raise ValueError(
                f'ARM interval head expects 3 classes, got {num_classes}.')
        if interval_targets.shape != (batch_size, num_intervals):
            raise ValueError('interval_targets shape mismatch: '
                             f'expected {(batch_size, num_intervals)}, '
                             f'got {tuple(interval_targets.shape)}')

        mapped_targets = interval_targets + 1
        interval_loss = F.cross_entropy(
            interval_logits.reshape(-1, 3),
            mapped_targets.reshape(-1),
        )
        interval_acc = (interval_logits.argmax(
            dim=-1) == mapped_targets).float().mean()

        cls_loss = self.focal_loss(cls_logits, cls_targets)
        cls_pred = (torch.sigmoid(cls_logits) >= 0.5).float()
        cls_acc = (cls_pred == cls_targets).float().mean()

        total_loss = (
            self.lambda_interval * interval_loss + self.lambda_cls * cls_loss)
        return {
            'loss': total_loss,
            'arm_interval_loss': interval_loss.detach(),
            'arm_cls_loss': cls_loss.detach(),
            'arm_interval_acc': interval_acc.detach(),
            'arm_cls_acc': cls_acc.detach(),
        }

    @torch.inference_mode()
    def predict_advantage(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        states: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_interval_probs: bool = False,
    ):
        """Predict ARM interval advantage and success probability.

        Args:
            images (torch.Tensor): Image sequence ``[B, T, N, C, H, W]`` or
                ``[B, T, C, H, W]``.
            text_input_ids (torch.Tensor): CLIP token ids, shape ``[B, L]``.
            text_attention_mask (torch.Tensor): Text attention mask, shape
                ``[B, L]``.
            states (torch.Tensor): Robot states, shape ``[B, T, D]``.
            lengths (Optional[torch.Tensor]): Valid frame counts, shape
                ``[B]``.
            return_interval_probs (bool): If ``True``, also return interval
                softmax probabilities with shape ``[B, T-1, 3]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor] or a three-tuple when
            ``return_interval_probs`` is ``True``: ``success_prob`` with shape
            ``[B]``, ``interval_pred`` in ``{-1, 0, +1}`` with shape
            ``[B, T-1]``, and optionally ``interval_probs``.
        """
        device = self._device()
        backbone = self._backbone()
        if images.dim() == 5:
            images = images.unsqueeze(2)
        images = images.to(
            device=device, dtype=next(backbone.clip_model.parameters()).dtype)
        text_input_ids = text_input_ids.to(device=device)
        text_attention_mask = text_attention_mask.to(device=device)
        states = states.to(device=device, dtype=images.dtype)
        if lengths is None:
            lengths = torch.full((images.shape[0], ),
                                 images.shape[1],
                                 dtype=torch.long,
                                 device=device)
        else:
            lengths = lengths.to(device=device)

        states = pad_state_to_max_dim(states, self.max_state_dim)
        image_features = backbone.encode_images(images)
        text_features = backbone.encode_text(text_input_ids,
                                             text_attention_mask)
        if image_features.shape[2] != 1:
            raise ValueError('ARMRewardModel expects one camera stream, got '
                             f'{image_features.shape[2]}.')
        image_features = image_features[:, :, 0, :]

        interval_logits, cls_logits = backbone.temporal_model(
            video_features=image_features,
            state_features=states,
            text_features=text_features,
            lengths=lengths,
        )
        interval_probs = F.softmax(interval_logits, dim=-1)
        interval_pred = interval_logits.argmax(dim=-1) - 1
        success_prob = torch.sigmoid(cls_logits)
        if return_interval_probs:
            return success_prob, interval_pred, interval_probs
        return success_prob, interval_pred

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return the FSDP auto-wrap policy for ARM transformer layers."""
        return partial(
            _module_wrap_policy, module_classes={nn.TransformerEncoderLayer})
