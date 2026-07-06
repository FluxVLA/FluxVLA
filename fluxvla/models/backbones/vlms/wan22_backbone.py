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

from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from fluxvla.engines import VLM_BACKBONES
from .wan_backbone import WanBaseBackbone

__all__ = ['Wan22Backbone']


@VLM_BACKBONES.register_module()
class Wan22Backbone(WanBaseBackbone):
    """Wan2.2 encoding frontend for FastWAM: VAE + optional T5 encoder.

    This backbone owns the frozen encoders used by FastWAM:

        ``vae`` encodes observation videos or conditioning images into latents
        and decodes predicted latents back to RGB frames. ``text_encoder`` is
        the optional umt5-xxl T5 encoder used when eval supplies tokenized
        prompts. Training usually consumes pre-computed ``context`` embeddings.

    The encoders are always frozen. Encoding helpers mirror the upstream
    ``fastwam.models.wan22.fastwam.FastWAM`` implementation verbatim so the
    split ``backbone`` + ``head`` pipeline stays numerically identical to the
    monolithic model.
    """

    frozen_module_names = ('text_encoder', 'vae')

    def __init__(
        self,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        freeze: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        if vae is None:
            raise ValueError('`Wan22Backbone` requires a `vae` module.')
        self.vae = vae
        self.text_encoder = text_encoder

        if freeze:
            self.freeze_encoder_modules()

    @property
    def temporal_downsample_factor(self) -> int:
        return int(self.vae.temporal_downsample_factor)

    # ------------------------------------------------------------------
    # Prompt token encoding (training usually uses cached ``context``)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_prompt(self, input_ids, attention_mask):
        """Encode tokenized prompts into ``(context, context_mask)``.

        Tokenization belongs to the data transform layer, matching
        :class:`Wan21Backbone`. This method only runs the frozen T5 encoder on
        ``input_ids`` / ``attention_mask`` and applies FastWAM's padded-token
        post-processing.
        """
        return self.encode_prompt_context(input_ids, attention_mask)

    @torch.no_grad()
    def encode_prompt_tokens(self, lang_tokens, lang_masks):
        """Backward-compatible alias for tokenized eval batches."""
        return self.encode_prompt(lang_tokens, lang_masks)

    # ------------------------------------------------------------------
    # Video / image latent encoding (deterministic; returns ``mu``)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_video_latents(
            self,
            video_tensor,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        return self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def encode_input_image_latents(
            self,
            input_image: torch.Tensor,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (input_image.ndim != 4 or input_image.shape[0] != 1
                or input_image.shape[1] != 3):
            raise ValueError(
                '`input_image` must have shape [1,3,H,W] or [3,H,W], got '
                f'{tuple(input_image.shape)}')
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def prepare_context(self, context, context_mask):
        if context is None or context_mask is None:
            raise ValueError(
                '`context` and `context_mask` must be provided together.')
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True)
        return context, context_mask

    def decode_latents(
            self,
            latents,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def forward(
            self,
            video: Optional[torch.Tensor] = None,
            input_image: Optional[torch.Tensor] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
            latents: Optional[torch.Tensor] = None,
            tiled: bool = False,
            tile_size=(30, 52),
            tile_stride=(15, 26),
    ):
        """Encode Wan2.2 inputs into FastWAM-ready tensors.

        Returns a dictionary keyed by the encoded products requested by the
        supplied inputs: ``input_latents`` for training videos,
        ``first_frame_latents`` for single-frame inference, ``context`` /
        ``context_mask`` for text conditioning, and ``video`` for decoded
        latents.
        """
        self.set_frozen_modules_to_eval_mode()
        outputs = {}

        if video is not None:
            outputs['input_latents'] = self.encode_video_latents(
                video,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if input_image is not None:
            outputs['first_frame_latents'] = self.encode_input_image_latents(
                input_image,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )

        has_tokens = input_ids is not None or attention_mask is not None
        has_context = context is not None or context_mask is not None
        if has_tokens and has_context:
            raise ValueError(
                '`input_ids/attention_mask` and `context/context_mask` are '
                'mutually exclusive.')
        if has_tokens:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    '`input_ids` and `attention_mask` must be provided '
                    'together.')
            outputs['context'], outputs['context_mask'] = self.encode_prompt(
                input_ids, attention_mask)
        elif has_context:
            prepared_context, prepared_mask = self.prepare_context(
                context, context_mask)
            outputs['context'] = prepared_context
            outputs['context_mask'] = prepared_mask

        if latents is not None:
            outputs['video'] = self.decode_latents(
                latents,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return outputs
