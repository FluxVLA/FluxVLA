# Copyright (c) 2026 The FastWAM Authors.
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Origin: Modified from
# Upstream-URL: https://github.com/roboterax/video-prediction-policy-2/blob/3365872910244f14df83003062f7c563f14c6d9e/src/vpp2/models/wan21_14b/wan_video_clip.py
# Upstream-Ref: 3365872910244f14df83003062f7c563f14c6d9e
# Additional-Upstream-URL: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/clip.py
# Notes: The ViT-H/14 tower is FluxVLA's DreamZero VisionTransformer, which
# produces identical outputs; only the visual-encoder wrapper is retained.
# License texts: ../LICENSE; attribution: ../ATTRIBUTION.txt.
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ...dreamzero.modules.wan_video_image_encoder import VisionTransformer

# Visual tower of Wan2.1's open-clip-xlm-roberta-large-vit-huge-14 CLIP.
VIT_H_14_CONFIG = dict(
    image_size=224,
    patch_size=14,
    dim=1280,
    mlp_ratio=4,
    out_dim=1024,
    num_heads=16,
    num_layers=32,
    pool_type="token",
    pre_norm=True,
    post_norm=False,
    activation="gelu",
    attn_dropout=0.0,
    proj_dropout=0.0,
    embedding_dropout=0.0,
    norm_eps=1e-5,
)


class WanImageEncoderCLIP(nn.Module):
    """Wan2.1 CLIP visual encoder applied to the first frame of each video."""

    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer(**VIT_H_14_CONFIG)
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def forward(self, videos: Sequence[torch.Tensor]):
        size = (self.visual.image_size,) * 2
        frames = torch.cat(
            [
                F.interpolate(
                    u[:, :1].transpose(0, 1), size=size, mode="bicubic", align_corners=False
                )
                for u in videos
            ],
            dim=0,
        )
        frames = self.normalize(frames.mul(0.5).add(0.5))
        return self.visual(frames, use_31_block=True)
