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

from typing import Tuple, cast

import torch
import torch.nn as nn
from transformers import CLIPModel

from fluxvla.engines import LLM_BACKBONES
from fluxvla.engines.utils.hf_hub import resolve_hf_local_path
from fluxvla.models.backbones.llms.clip_utils import clip_feature_tensor


class TemporalAdvantageTransformer(nn.Module):
    """Temporal transformer with interval and success heads for ARM.

    Official implementation of https://arxiv.org/abs/2604.03037

    The module fuses CLIP visual features, CLIP text features, and robot
    states over a causal frame window, then predicts:

    * **Interval head**: tri-state relative advantage between adjacent frames
      (Progressive / Stagnant / Regressive), represented as logits over
      ``{-1, 0, +1}`` (mapped to classes ``{0, 1, 2}`` during training).
    * **Success head**: whether the current frame has reached task completion.
    """

    transformer_layer_cls = nn.TransformerEncoderLayer

    def __init__(self,
                 video_dim: int = 512,
                 state_dim: int = 32,
                 text_dim: int = 512,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 8,
                 dropout: float = 0.1,
                 center_idx: int = 4) -> None:
        """Initialize the temporal advantage transformer.

        Args:
            video_dim (int): CLIP visual embedding dimension.
            state_dim (int): Padded robot state dimension.
            text_dim (int): CLIP text embedding dimension.
            d_model (int): Transformer hidden dimension.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout probability inside transformer blocks.
            center_idx (int): Index of the current frame used by the success
                head (typically ``n_history_steps`` in a causal window).
        """
        super().__init__()
        self.center_idx = center_idx
        self.video_proj = nn.Linear(video_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.interval_head = nn.Linear(2 * d_model, 3)
        self.cls_head = nn.Linear(d_model, 1)

    def forward(
        self,
        video_features: torch.Tensor,
        state_features: torch.Tensor,
        text_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the shared temporal encoder and both prediction heads.

        Args:
            video_features (torch.Tensor): Visual features with shape
                ``[B, T, Dv]``.
            state_features (torch.Tensor): State features with shape
                ``[B, T, Ds]``.
            text_features (torch.Tensor): Text features with shape
                ``[B, Dt]``.
            lengths (torch.Tensor): Valid frame count per sample, shape
                ``[B]``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Interval logits with shape
            ``[B, T-1, 3]`` and success logits with shape ``[B]``.
        """
        batch_size, seq_len, _ = video_features.shape
        device = video_features.device
        lengths = lengths.to(device)

        visual_tokens = self.video_proj(video_features)
        state_tokens = self.state_proj(state_features)
        text_tokens = self.text_proj(text_features).unsqueeze(1).expand(
            batch_size, seq_len, -1)
        fused_tokens = visual_tokens + state_tokens + text_tokens

        step_ids = torch.arange(
            seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        padding_mask = step_ids >= lengths.unsqueeze(1)
        hidden_states = self.transformer(
            fused_tokens, src_key_padding_mask=padding_mask)

        if seq_len < 2:
            raise ValueError(
                'TemporalAdvantageTransformer expects sequence length >= 2, '
                f'got {seq_len}.')
        pair_features = torch.cat(
            [hidden_states[:, :-1, :], hidden_states[:, 1:, :]], dim=-1)
        interval_logits = self.interval_head(pair_features)

        center_idx = min(self.center_idx, seq_len - 1)
        cls_logits = self.cls_head(hidden_states[:, center_idx, :]).squeeze(-1)
        return interval_logits, cls_logits


@LLM_BACKBONES.register_module()
class ARMBackbone(nn.Module):
    """CLIP encoder plus ARM temporal advantage heads.

    Official implementation of https://arxiv.org/abs/2604.03037

    The backbone loads a CLIP vision-language encoder for online feature
    extraction and attaches :class:`TemporalAdvantageTransformer` for ARM
    interval and success prediction. It is consumed by
    :class:`~fluxvla.models.vlas.arm_reward_model.ARMRewardModel`.
    """

    transformer_layer_cls = nn.TransformerEncoderLayer

    def __init__(self,
                 pretrained_name_or_path: str,
                 hidden_dim: int = 768,
                 max_state_dim: int = 32,
                 num_layers: int = 8,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 n_history_steps: int = 4,
                 freeze_clip_backbone: bool = True) -> None:
        """Initialize the ARM backbone.

        Args:
            pretrained_name_or_path (str): CLIP checkpoint path or Hugging Face
                repo id.
            hidden_dim (int): Hidden dimension for the temporal transformer.
            max_state_dim (int): Padded robot state feature dimension.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            dropout (float): Transformer dropout probability.
            n_history_steps (int): Number of history frames before the current
                frame; also used as the success-head center index.
            freeze_clip_backbone (bool): Whether CLIP parameters are frozen.
        """
        super().__init__()
        pretrained_name_or_path = resolve_hf_local_path(
            pretrained_name_or_path)
        self.pretrained_name_or_path = pretrained_name_or_path
        self.clip_model = CLIPModel.from_pretrained(pretrained_name_or_path)
        projection_dim = self.clip_model.config.projection_dim
        self.temporal_model = TemporalAdvantageTransformer(
            video_dim=projection_dim,
            state_dim=max_state_dim,
            text_dim=projection_dim,
            d_model=hidden_dim,
            n_heads=num_heads,
            n_layers=num_layers,
            dropout=dropout,
            center_idx=n_history_steps,
        )
        self.freeze_clip_backbone = freeze_clip_backbone
        if self.freeze_clip_backbone:
            self.clip_model.requires_grad_(False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image sequences with CLIP.

        Args:
            images (torch.Tensor): Image tensor with shape
                ``[B, T, N, C, H, W]``.

        Returns:
            torch.Tensor: Visual features with shape ``[B, T, N, D]``.
        """
        batch_size, seq_len, num_cameras, channels, height, width = (
            images.shape)
        flat_images = cast(
            torch.FloatTensor,
            images.reshape(batch_size * seq_len * num_cameras, channels,
                           height, width).float(),
        )
        image_features = clip_feature_tensor(
            self.clip_model.get_image_features(pixel_values=flat_images))
        return image_features.reshape(batch_size, seq_len, num_cameras, -1)

    def encode_text(self, text_input_ids: torch.Tensor,
                    text_attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode task text tokens with CLIP.

        Args:
            text_input_ids (torch.Tensor): Token ids from the CLIP tokenizer.
            text_attention_mask (torch.Tensor): Attention mask for text tokens.

        Returns:
            torch.Tensor: Text features with shape ``[B, D]``.
        """
        return clip_feature_tensor(
            self.clip_model.get_text_features(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            ))
