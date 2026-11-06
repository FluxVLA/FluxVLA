# Vendored from VPP2, revision 3365872910244f14df83003062f7c563f14c6d9e.
# See ../THIRD_PARTY_NOTICES.md for licenses and source notices.
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .helpers.gradient import gradient_checkpoint_forward
from logging import getLogger as get_logger

logger = get_logger(__name__)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: Optional[torch.Tensor] = None,
):
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    if ctx_mask is not None:
        if ctx_mask.dim() == 2:
            ctx_mask = ctx_mask.unsqueeze(0).unsqueeze(0)
        elif ctx_mask.dim() == 3:
            ctx_mask = ctx_mask.unsqueeze(1)
        ctx_mask = ctx_mask.to(device=q.device, dtype=torch.bool)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor):
    assert dim % 2 == 0
    half = dim // 2
    dtype = position.dtype if position.is_floating_point() else torch.float32
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000, -torch.arange(half, dtype=torch.float64, device=position.device).div(half)
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).double()[: (dim // 2)] / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    return (
        precompute_freqs_cis(dim - 4 * (dim // 6), end, theta),
        precompute_freqs_cis(2 * (dim // 6), end, theta),
        precompute_freqs_cis(2 * (dim // 6), end, theta),
    )


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.to(device=x.device)
    if freqs.device.type == "npu":
        freqs = freqs.to(torch.complex64)
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).type_as(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: Optional[int] = None,
        num_heads: int = 16,
        eps: float = 1e-6,
        qk_norm: bool = True,
    ):
        super().__init__()
        if attn_head_dim is None:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    "`hidden_dim` must be divisible by `num_heads` when attn_head_dim is omitted."
                )
            attn_head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        return self.o(flash_attention(q, k, v, self.num_heads, ctx_mask=self_attn_mask))


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: Optional[int] = None,
        num_heads: int = 16,
        eps: float = 1e-6,
        qk_norm: bool = True,
    ):
        super().__init__()
        if attn_head_dim is None:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    "`hidden_dim` must be divisible by `num_heads` when attn_head_dim is omitted."
                )
            attn_head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = num_heads * attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q, k, v, self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class I2VCrossAttention(CrossAttention):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: Optional[int],
        num_heads: int,
        eps: float,
        qk_norm: bool = True,
    ):
        super().__init__(hidden_dim, attn_head_dim, num_heads, eps, qk_norm)
        self.k_img = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v_img = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.norm_k_img = WanRMSNorm(self.attn_hidden_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        if ctx.shape[1] < 257:
            raise ValueError(
                f"Wan2.1 i2v context must include 257 image tokens, got {ctx.shape[1]}."
            )
        ctx_img = ctx[:, :257]
        ctx_txt = ctx[:, 257:]
        text_mask = None if ctx_mask is None else ctx_mask[..., 257:]

        q = self.norm_q(self.q(x))
        k_img = self.norm_k_img(self.k_img(ctx_img))
        v_img = self.v_img(ctx_img)
        img_x = flash_attention(q, k_img, v_img, self.num_heads)

        k = self.norm_k(self.k(ctx_txt))
        v = self.v(ctx_txt)
        txt_x = flash_attention(q, k, v, self.num_heads, ctx_mask=text_mask)
        return self.o(txt_x + img_x)


class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: Optional[int],
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        cross_attn_type: str = "t2v",
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim if attn_head_dim is not None else hidden_dim // num_heads
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, self.attn_head_dim, num_heads, eps, qk_norm)
        cross_cls = I2VCrossAttention if cross_attn_type == "i2v" else CrossAttention
        self.cross_attn = cross_cls(hidden_dim, self.attn_head_dim, num_heads, eps, qk_norm)
        self.norm1 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = WanLayerNorm(hidden_dim, eps=eps, elementwise_affine=bool(cross_attn_norm))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        context_mask=None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        y = self.self_attn(
            modulate(self.norm1(x), shift_msa, scale_msa), freqs, self_attn_mask=self_attn_mask
        )
        x = self.gate(x, gate_msa, y)
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        y = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.gate(x, gate_mlp, y)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = tuple(patch_size)
        self.norm = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(self.patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t):
        if len(t.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t.dtype, device=t.device) + t.unsqueeze(2)
            ).chunk(2, dim=2)
            return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(
            2, dim=1
        )
        return self.head(self.norm(x) * (1 + scale) + shift)


class MLPProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class WanVideoDiT(nn.Module):
    def __init__(
        self,
        model_type: str = "i2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        attn_head_dim: Optional[int] = None,
        num_layers: int = 40,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        use_gradient_checkpointing: bool = False,
        video_attention_mask_mode: str = "first_frame_causal",
        condition_channels: int = 20,
        **_: Any,
    ):
        super().__init__()
        if model_type != "i2v":
            raise ValueError(
                f"Wan2.1-14B VPP2 supports only model_type='i2v', got {model_type!r}."
            )
        hidden_dim = int(
            hidden_dim if hidden_dim is not None else (dim if dim is not None else 5120)
        )
        if attn_head_dim is None:
            if hidden_dim % num_heads != 0:
                raise ValueError("`hidden_dim` must be divisible by `num_heads`.")
            attn_head_dim = hidden_dim // num_heads

        self.model_type = model_type
        self.patch_size = tuple(patch_size)
        self.text_len = int(text_len)
        self.in_dim = int(in_dim)
        self.dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = int(ffn_dim)
        self.freq_dim = int(freq_dim)
        self.text_dim = int(text_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.num_layers = int(num_layers)
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.condition_channels = int(condition_channels)
        self.fuse_vae_embedding_in_latents = False

        self.patch_embedding = nn.Conv3d(
            self.in_dim, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim,
                    self.attn_head_dim,
                    self.num_heads,
                    self.ffn_dim,
                    eps,
                    cross_attn_type="i2v",
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = Head(hidden_dim, self.out_dim, self.patch_size, eps)
        self.img_emb = MLPProj(1280, hidden_dim)
        self.freqs = precompute_freqs_cis_3d(self.attn_head_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def patchify(self, x: torch.Tensor):
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size):
        f, h, w = grid_size
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f,
            h=h,
            w=w,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def _build_condition(self, x: torch.Tensor, condition_latents: Optional[torch.Tensor]):
        if condition_latents is None:
            condition_latents = x[:, :, :1]
        if condition_latents.ndim != 5:
            raise ValueError(
                f"`condition_latents` must be 5D [B,C,T,H,W], got {tuple(condition_latents.shape)}"
            )
        b, _, lat_t, lat_h, lat_w = x.shape
        expected = (b, self.out_dim, lat_h, lat_w)
        actual = (
            condition_latents.shape[0],
            condition_latents.shape[1],
            condition_latents.shape[3],
            condition_latents.shape[4],
        )
        if actual != expected:
            raise ValueError(
                "`condition_latents` batch/channel/spatial shape mismatch: "
                f"got {tuple(condition_latents.shape)}, expected "
                f"[B={b},C={self.out_dim},T<={lat_t},H={lat_h},W={lat_w}]."
            )
        if condition_latents.shape[2] > lat_t:
            raise ValueError(
                "`condition_latents` cannot be longer than the noisy video latent "
                f"sequence, got {condition_latents.shape[2]} > {lat_t}."
            )
        mask = torch.zeros((b, 4, lat_t, lat_h, lat_w), dtype=x.dtype, device=x.device)
        mask[:, :, : condition_latents.shape[2]] = 1
        y = torch.zeros((b, self.out_dim, lat_t, lat_h, lat_w), dtype=x.dtype, device=x.device)
        y[:, :, : condition_latents.shape[2]] = condition_latents.to(device=x.device, dtype=x.dtype)
        return torch.cat([mask, y], dim=1)

    def build_video_to_video_mask(
        self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ):
        if self.video_attention_mask_mode in {"bidirectional", "three_frame"}:
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if self.video_attention_mask_mode == "first_frame_causal":
            mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            mask[:first_frame_tokens, first_frame_tokens:] = False
            return mask
        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError("`video_seq_len` must be divisible by `video_tokens_per_frame`.")
            frames = video_seq_len // video_tokens_per_frame
            mask = torch.tril(torch.ones((frames, frames), dtype=torch.bool, device=device))
            return mask.repeat_interleave(video_tokens_per_frame, 0).repeat_interleave(
                video_tokens_per_frame, 1
            )
        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        condition_latents: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[1] == self.out_dim:
            y = self._build_condition(x, condition_latents)
            x = torch.cat([x, y], dim=1)
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"Wan2.1-14B DiT expected {self.in_dim} channels after conditioning, got {x.shape[1]}."
            )
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B,L,D], got {tuple(context.shape)}")
        batch_size = x.shape[0]
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` must have shape [B] or [1], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, timestep length must match batch size.")
            timestep = timestep.expand(batch_size)
        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        if clip_fea is None:
            clip_fea = torch.zeros(
                (batch_size, 257, 1280), dtype=context.dtype, device=context.device
            )
        if clip_fea.ndim != 3 or clip_fea.shape[1:] != (257, 1280):
            raise ValueError(f"`clip_fea` must be [B,257,1280], got {tuple(clip_fea.shape)}")

        x = self.patchify(x)
        f, h, w = x.shape[2:]
        tokens_per_frame = h * w
        seq_len = f * h * w

        time_weight = self.time_embedding[0].weight
        time_input = sinusoidal_embedding_1d(self.freq_dim, timestep).to(
            device=time_weight.device,
            dtype=time_weight.dtype,
        )
        t = self.time_embedding(time_input)
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        text_context = self.text_embedding(context)
        image_context = self.img_emb(
            clip_fea.to(device=text_context.device, dtype=text_context.dtype)
        )
        context_emb = torch.cat([image_context, text_context], dim=1)
        image_mask = torch.ones(
            (batch_size, image_context.shape[1]), dtype=torch.bool, device=context_mask.device
        )
        combined_mask = torch.cat(
            [image_mask, context_mask.to(dtype=torch.bool, device=context_mask.device)], dim=1
        )
        context_attn_mask = combined_mask.unsqueeze(1).expand(-1, seq_len, -1)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(seq_len, 1, -1)
            .to(x_tokens.device)
        )

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]):
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        return self.unpatchify(x, (f, h, w))

    def forward(
        self,
        x,
        timestep,
        context,
        context_mask=None,
        clip_fea=None,
        condition_latents=None,
        **kwargs,
    ):
        pre = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            clip_fea=clip_fea,
            condition_latents=condition_latents,
            **kwargs,
        )
        tokens = pre["tokens"]
        self_attn_mask = None
        if self.video_attention_mask_mode not in {"bidirectional", "three_frame"}:
            self_attn_mask = self.build_video_to_video_mask(
                tokens.shape[1],
                int(pre["meta"]["tokens_per_frame"]),
                tokens.device,
            )
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                tokens = gradient_checkpoint_forward(
                    block,
                    True,
                    tokens,
                    pre["context"],
                    pre["t_mod"],
                    pre["freqs"],
                    context_mask=pre["context_mask"],
                    self_attn_mask=self_attn_mask,
                )
            else:
                tokens = block(
                    tokens,
                    pre["context"],
                    pre["t_mod"],
                    pre["freqs"],
                    context_mask=pre["context_mask"],
                    self_attn_mask=self_attn_mask,
                )
        return self.post_dit(tokens, pre)
