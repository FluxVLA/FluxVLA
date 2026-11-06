# Vendored from VPP2, revision 3365872910244f14df83003062f7c563f14c6d9e.
# See ../THIRD_PARTY_NOTICES.md for licenses and source notices.
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def pos_interpolate(pos, seq_len):
    if pos.size(1) == seq_len:
        return pos
    src_grid = int(math.sqrt(pos.size(1)))
    tar_grid = int(math.sqrt(seq_len))
    n = pos.size(1) - src_grid * src_grid
    return torch.cat(
        [
            pos[:, :n],
            F.interpolate(
                pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2),
                size=(tar_grid, tar_grid),
                mode="bicubic",
                align_corners=False,
            )
            .flatten(2)
            .transpose(1, 2),
        ],
        dim=1,
    )


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, causal=False, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, c = x.shape
        q, k, v = self.to_qkv(x).view(b, s, 3, self.num_heads, self.head_dim).unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        p = self.attn_dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=p)
        x = x.transpose(1, 2).reshape(b, s, c)
        return F.dropout(self.proj(x), self.proj_dropout, self.training)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio,
        num_heads,
        post_norm=False,
        causal=False,
        activation="gelu",
        attn_dropout=0.0,
        proj_dropout=0.0,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.post_norm = post_norm
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout, proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        act = QuickGELU() if activation == "quick_gelu" else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act,
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            return x + self.norm2(self.mlp(x))
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
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
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.pool_type = pool_type
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=not pre_norm
        )
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(embedding_dropout)
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    dim,
                    mlp_ratio,
                    num_heads,
                    post_norm,
                    False,
                    activation,
                    attn_dropout,
                    proj_dropout,
                    norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = LayerNorm(dim, eps=norm_eps)
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x, interpolation=False, use_31_block=False):
        b = x.size(0)
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        x = self.dropout(
            x
            + (
                pos_interpolate(self.pos_embedding, x.size(1))
                if interpolation
                else self.pos_embedding
            )
        )
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if use_31_block:
            return self.transformer[:-1](x)
        return self.transformer(x)


class WanImageEncoderCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer()
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
