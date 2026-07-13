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

import hashlib
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

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
    DEFAULT_TEXT_PROMPT = (
        "A video recorded from a robot's point of view executing the "
        'following instruction: {task}')

    def __init__(
        self,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        text_embed_cache_dir: Optional[str] = None,
        text_embed_cache_context_len: int = 128,
        text_embed_cache_enc_id: str = 'wan22ti2v5b',
        text_embed_cache_size: int = 256,
        text_embed_cache_device: str = 'cpu',
        text_embed_prompt_template: Optional[str] = None,
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
        self.text_embed_cache_dir = (
            None if text_embed_cache_dir is None else str(
                Path(text_embed_cache_dir).expanduser()))
        self.text_embed_cache_context_len = int(text_embed_cache_context_len)
        self.text_embed_cache_enc_id = str(text_embed_cache_enc_id)
        self.text_embed_cache_size = int(text_embed_cache_size)
        self.text_embed_cache_device = str(text_embed_cache_device).lower()
        self.text_embed_prompt_template = (
            text_embed_prompt_template or self.DEFAULT_TEXT_PROMPT)
        if self.text_embed_cache_context_len <= 0:
            raise ValueError('`text_embed_cache_context_len` must be > 0.')
        if self.text_embed_cache_size < 0:
            raise ValueError('`text_embed_cache_size` must be >= 0.')
        if self.text_embed_cache_device not in {'cpu', 'model'}:
            raise ValueError(
                '`text_embed_cache_device` must be "cpu" or "model".')
        self._text_embed_cache = OrderedDict()

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

    def _text_cache_path(self, cache_key: str) -> Optional[Path]:
        if self.text_embed_cache_dir is None:
            return None
        filename = (f'{cache_key}.t5_len{self.text_embed_cache_context_len}.'
                    f'{self.text_embed_cache_enc_id}.pt')
        return Path(self.text_embed_cache_dir) / filename

    @staticmethod
    def _token_cache_key(input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> str:
        valid_ids = input_ids[attention_mask].detach().to(
            device='cpu', dtype=torch.int64).contiguous()
        return hashlib.sha256(valid_ids.numpy().tobytes()).hexdigest()

    def _resolve_text_cache_keys(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: Optional[Sequence[str]],
    ) -> list[str]:
        batch_size = int(input_ids.shape[0])
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            prompts = list(prompts)
            if len(prompts) != batch_size:
                raise ValueError(
                    '`prompts` length must match the token batch size: '
                    f'{len(prompts)} != {batch_size}.')
            return [
                hashlib.sha256(str(prompt).encode('utf-8')).hexdigest()
                for prompt in prompts
            ]
        return [
            self._token_cache_key(input_ids[index], attention_mask[index])
            for index in range(batch_size)
        ]

    def _get_memory_cached_text(self, cache_key: str):
        cached = self._text_embed_cache.get(cache_key)
        if cached is None:
            return None
        self._text_embed_cache.move_to_end(cache_key)
        context, context_mask = cached
        return (
            context.to(device=self.device, dtype=self.torch_dtype),
            context_mask.to(device=self.device, dtype=torch.bool),
        )

    def _put_memory_cached_text(self, cache_key: str, context: torch.Tensor,
                                context_mask: torch.Tensor) -> None:
        if self.text_embed_cache_size == 0:
            return
        cache_device = (
            self.device if self.text_embed_cache_device == 'model' else
            torch.device('cpu'))
        self._text_embed_cache[cache_key] = (
            context.detach().to(device=cache_device).clone(),
            context_mask.detach().to(device=cache_device).clone(),
        )
        self._text_embed_cache.move_to_end(cache_key)
        while len(self._text_embed_cache) > self.text_embed_cache_size:
            self._text_embed_cache.popitem(last=False)

    def _load_disk_cached_text(self, cache_key: str):
        cache_path = self._text_cache_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None
        payload = torch.load(cache_path, map_location='cpu', weights_only=True)
        context = payload['context']
        source_mask = payload['mask'].bool()
        expected_len = self.text_embed_cache_context_len
        if context.ndim != 2 or context.shape[0] != expected_len:
            raise ValueError(
                'Cached `context` must have shape [context_len, D], got '
                f'{tuple(context.shape)} in {cache_path}.')
        if source_mask.ndim != 1 or source_mask.shape[0] != expected_len:
            raise ValueError(
                'Cached `mask` must have shape [context_len], got '
                f'{tuple(source_mask.shape)} in {cache_path}.')
        context = context.to(device=self.device, dtype=self.torch_dtype)
        source_mask = source_mask.to(device=self.device, dtype=torch.bool)
        context = context.clone()
        context[~source_mask] = 0
        return context, torch.ones_like(source_mask)

    def _save_disk_cached_text(self, cache_key: str, context: torch.Tensor,
                               source_mask: torch.Tensor) -> None:
        cache_path = self._text_cache_path(cache_key)
        if cache_path is None or cache_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.parent / (
            f'.{cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}')
        payload = {
            'context':
            context.detach().to(device='cpu',
                                dtype=torch.bfloat16).contiguous(),
            'mask':
            source_mask.detach().to(device='cpu',
                                    dtype=torch.bool).contiguous(),
        }
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, cache_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @torch.no_grad()
    def encode_prompt_cached(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
    ):
        """Encode each unique prompt once and reuse memory/disk cache hits."""
        ids, mask = self._prepare_prompt_inputs(input_ids, attention_mask)
        if ids.shape[1] != self.text_embed_cache_context_len:
            raise ValueError(
                'Token sequence length must match '
                f'`text_embed_cache_context_len`: {ids.shape[1]} != '
                f'{self.text_embed_cache_context_len}.')
        caching_enabled = (
            self.text_embed_cache_size > 0
            or self.text_embed_cache_dir is not None)
        if not caching_enabled:
            return self.encode_prompt_context(ids, mask)

        cache_keys = self._resolve_text_cache_keys(ids, mask, prompts)
        resolved = [None] * len(cache_keys)
        missing_by_key = OrderedDict()
        newly_encoded = {}
        for index, cache_key in enumerate(cache_keys):
            cached = self._get_memory_cached_text(cache_key)
            if cached is None:
                cached = self._load_disk_cached_text(cache_key)
                if cached is not None:
                    self._put_memory_cached_text(cache_key, *cached)
            if cached is not None:
                resolved[index] = cached
            elif cache_key not in missing_by_key:
                missing_by_key[cache_key] = index

        if missing_by_key:
            missing_keys = list(missing_by_key)
            missing_indices = [missing_by_key[key] for key in missing_keys]
            index_tensor = torch.tensor(
                missing_indices, device=ids.device, dtype=torch.long)
            missing_ids = ids.index_select(0, index_tensor)
            missing_mask = mask.index_select(0, index_tensor)
            contexts, context_masks = self.encode_prompt_context(
                missing_ids, missing_mask)
            for offset, cache_key in enumerate(missing_keys):
                context = contexts[offset]
                context_mask = context_masks[offset]
                newly_encoded[cache_key] = (context, context_mask)
                self._put_memory_cached_text(cache_key, context, context_mask)
                self._save_disk_cached_text(cache_key, context,
                                            missing_mask[offset])

        for index, cache_key in enumerate(cache_keys):
            if resolved[index] is None:
                resolved[index] = newly_encoded.get(cache_key)
            if resolved[index] is None:
                resolved[index] = self._get_memory_cached_text(cache_key)
            if resolved[index] is None:
                raise RuntimeError(
                    f'Failed to resolve cached text embedding {cache_key}.')
        contexts, context_masks = zip(*resolved)
        return torch.stack(contexts), torch.stack(context_masks)

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
