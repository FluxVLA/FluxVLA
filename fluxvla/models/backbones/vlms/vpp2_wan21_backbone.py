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
"""Frozen Wan2.1 frontend with VPP2's encoder precision contract."""

from collections import OrderedDict
from pathlib import Path

import torch

from fluxvla.engines import VLM_BACKBONES
from .wan_backbone import WanBaseBackbone


class _Wan21VAE(torch.nn.Module):
    """Reuse DreamZero's Wan2.1 network without its CUDA-only wrapper init."""

    def __init__(self, model=None):
        super().__init__()
        if model is None:
            from ...third_party_models.dreamzero.modules.wan_video_vae import \
                VideoVAE_
            model = VideoVAE_(z_dim=16)
        self.model = model
        self.temporal_downsample_factor = 4
        # Nonpersistent constants keep the released `model.*` state schema.
        self.register_buffer(
            'mean',
            torch.tensor([
                -.7571, -.7089, -.9113, .1075, -.1745, .9653, -.1517, 1.5508,
                .4134, -.0715, .5517, -.3632, -.1922, -.9497, .2503, -.2921
            ],
                         device='cpu'),
            persistent=False)
        self.register_buffer(
            'inverse_std',
            1 / torch.tensor([
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ],
                             device='cpu'),
            persistent=False)

    def encode(self, videos, tiled=False):
        """Encode ``[3, T, H, W]`` videos to normalized VAE latents."""
        if tiled:
            raise ValueError(
                'VPP2 reference inference uses untiled VAE encoding')
        return torch.stack([
            self.model.encode(v[None], [self.mean, self.inverse_std])[0]
            for v in videos
        ])


@VLM_BACKBONES.register_module()
class VPP2Wan21Backbone(WanBaseBackbone):
    """Frozen Wan2.1-I2V-14B encoders with VPP2's precision contract.

    The Wan2.1 VAE runs in fp32, the UMT5-XXL text encoder in
    ``torch_dtype`` and the CLIP ViT-H/14 visual tower in fp16, matching the
    VPP2 reference inference. These dtypes are kept fixed when the module is
    moved or cast. Encoders are built on the meta device and loaded strictly.

    Args:
        pretrained_name_or_path (str | None): Wan2.1-I2V-14B-480P directory
            containing ``Wan2.1_VAE.pth``,
            ``models_t5_umt5-xxl-enc-bf16.pth`` and
            ``models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth``. When
            None, ``vae``, ``text_encoder`` and ``clip_encoder`` are required.
        vae (torch.nn.Module | None): Prebuilt VAE wrapper.
        text_encoder (torch.nn.Module | None): Prebuilt UMT5 encoder.
        clip_encoder (torch.nn.Module | None): Prebuilt CLIP visual encoder.
        device (str | torch.device): Device for all encoders.
        torch_dtype (torch.dtype): Text-encoder and output dtype. Defaults to
            ``torch.bfloat16``.
        text_embed_cache_size (int): Number of prompt embeddings kept in an
            LRU cache; 0 disables it. Defaults to 256.
    """

    frozen_module_names = ('vae', 'text_encoder', 'clip_encoder')

    def __init__(self,
                 pretrained_name_or_path=None,
                 *,
                 vae=None,
                 text_encoder=None,
                 clip_encoder=None,
                 device='cpu',
                 torch_dtype=torch.bfloat16,
                 text_embed_cache_size=256):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.text_embed_cache_size = int(text_embed_cache_size)
        self._prompt_cache = OrderedDict()
        if self.text_embed_cache_size < 0:
            raise ValueError('text_embed_cache_size must be nonnegative')
        if pretrained_name_or_path is not None:
            from ...third_party_models.dreamzero.modules import \
                wan_video_text_encoder
            from ...third_party_models.vpp2.modules.wan_video_clip import \
                WanImageEncoderCLIP

            root = Path(pretrained_name_or_path)
            # Construct on meta to avoid a second fp32 copy of the encoders.
            with torch.device('meta'):
                vae = _Wan21VAE()
                text_encoder = wan_video_text_encoder.WanTextEncoder()
                clip_encoder = WanImageEncoderCLIP()
            weights = torch.load(
                root / 'Wan2.1_VAE.pth',
                map_location='cpu',
                weights_only=True,
                mmap=True)
            weights = weights.get('model_state', weights)
            if not any(k.startswith('model.') for k in weights):
                weights = {f'model.{k}': v for k, v in weights.items()}
            vae.load_state_dict(weights, strict=True, assign=True)
            text_encoder.load_state_dict(
                torch.load(
                    root / 'models_t5_umt5-xxl-enc-bf16.pth',
                    map_location='cpu',
                    weights_only=True,
                    mmap=True),
                strict=True,
                assign=True)
            weights = torch.load(
                root /
                'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                map_location='cpu',
                weights_only=True,
                mmap=True)
            clip_encoder.load_state_dict(
                {
                    k: v
                    for k, v in weights.items()
                    if k.startswith('visual.') or k == 'log_scale'
                },
                strict=True,
                assign=True)
        if any(m is None for m in (vae, text_encoder, clip_encoder)):
            raise ValueError(
                'Provide a Wan2.1 directory or all three encoders')
        self.vae = vae.to(device=device, dtype=torch.float32)
        self.text_encoder = text_encoder.to(device=device, dtype=torch_dtype)
        self.clip_encoder = clip_encoder.to(device=device, dtype=torch.float16)
        self.freeze_encoder_modules()

    def _apply(self, fn, recurse=True):
        # Casting then restoring fp32 loses VAE weight bits. Probe only the
        # destination device, and move the original tensor in its fixed dtype.
        def fixed_dtype(dtype):

            def apply(tensor):
                if not tensor.is_floating_point():
                    return fn(tensor)
                destination = fn(tensor.new_empty(0))
                return tensor.to(device=destination.device, dtype=dtype)

            return apply

        if recurse:
            self.vae._apply(fixed_dtype(torch.float32))
            self.clip_encoder._apply(fixed_dtype(torch.float16))
            self.text_encoder._apply(fn)
            self.torch_dtype = next(self.text_encoder.parameters()).dtype
        self._prompt_cache.clear()
        return self

    @torch.no_grad()
    def encode_video_latents(self, video):
        """Encode condition videos with the fp32 Wan2.1 VAE.

        Args:
            video (torch.Tensor): ``[B, 3, 4k + 1, H, W]`` in ``[-1, 1]``.
                RGB is rounded to ``torch_dtype`` first, as upstream does.

        Returns:
            torch.Tensor: Latents ``[B, 16, k + 1, H / 8, W / 8]`` in
            ``torch_dtype``.
        """
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] % 4 != 1:
            raise ValueError('video must be [B,3,4k+1,H,W]')
        # Upstream rounds RGB to model precision before the fp32 VAE call.
        video = video.to(device=self.device, dtype=self.torch_dtype).float()
        with torch.autocast(self.device.type, enabled=False):
            latents = self.vae.encode(list(video.unbind(0)), tiled=False)
        return latents.to(dtype=self.torch_dtype)

    @torch.no_grad()
    def encode_clip(self, video):
        """Encode the first frame of each video with the fp16 CLIP tower.

        Args:
            video (torch.Tensor): ``[B, 3, T, H, W]`` in ``[-1, 1]``.

        Returns:
            torch.Tensor: Features after 31 of the 32 transformer blocks,
            ``[B, 257, 1280]`` in ``torch_dtype``.
        """
        video = video.to(device=self.device, dtype=self.torch_dtype).half()
        with torch.autocast(self.device.type, enabled=False):
            features = self.clip_encoder(list(video[:, :, :1].unbind(0)))
        return features.to(dtype=self.torch_dtype)

    @torch.no_grad()
    def encode_prompt_cached(self, lang_tokens, lang_masks):
        """Encode prompts with UMT5, reusing cached embeddings.

        Args:
            lang_tokens (torch.Tensor): Token ids ``[B, L]`` or ``[L]``.
            lang_masks (torch.Tensor): Token mask with the same shape.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Context ``[B, L, 4096]`` in
            ``torch_dtype``, zero beyond each prompt's length, and an
            all-True mask ``[B, L]``, as in Wan2.1.
        """
        if lang_tokens.ndim == 1:
            lang_tokens, lang_masks = lang_tokens[None], lang_masks[None]
        contexts = []
        for tokens, mask in zip(lang_tokens, lang_masks):
            key = (tuple(tokens.tolist()), tuple(mask.tolist()))
            if key in self._prompt_cache:
                context = self._prompt_cache.pop(key)
            else:
                with torch.autocast(self.device.type, enabled=False):
                    context = self.encode_prompt_embeddings(tokens, mask)[0]
                context = context.cpu()
            if self.text_embed_cache_size:
                self._prompt_cache[key] = context
                while len(self._prompt_cache) > self.text_embed_cache_size:
                    self._prompt_cache.popitem(last=False)
            contexts.append(context)
        context = torch.stack(contexts).to(self.device, self.torch_dtype)
        return context, torch.ones(
            context.shape[:2], device=self.device, dtype=torch.bool)
