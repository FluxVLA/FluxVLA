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

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from fluxvla.engines import TOKENIZERS
from fluxvla.models.third_party_models.cosmos3.model.vfm.tokenizers import \
    wan2pt2_vae_4x16x16 as wan_vae

_DEFAULT_ENCODE_CHUNK_FRAMES = {'256': 68, '480': 24, '720': 12}


@TOKENIZERS.register_module()
class Cosmos3Wan22VAE(nn.Module):
    """Frozen Wan2.2 VAE used as the Cosmos3 video latent tokenizer."""

    def __init__(
        self,
        pretrained_name_or_path: str | None = None,
        dtype: torch.dtype | str = torch.bfloat16,
        encode_chunk_frames: Mapping[str, int] | None = None,
        encode_exact_durations: list[int] | None = None,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 4,
    ) -> None:
        super().__init__()

        dtype = self._resolve_dtype(dtype)
        if encode_chunk_frames is None:
            encode_chunk_frames = dict(_DEFAULT_ENCODE_CHUNK_FRAMES)
        if any(chunk % 4 != 0 for chunk in encode_chunk_frames.values()):
            raise ValueError(
                'encode_chunk_frames values must be multiples of 4.')

        resolved_path = (None if pretrained_name_or_path is None else str(
            self.resolve_vae_path(pretrained_name_or_path)))

        wan = wan_vae.WanVAE(
            dtype=dtype,
            is_amp=False,
            vae_pth=resolved_path,
            temporal_window=encode_chunk_frames,
            encode_exact_durations=encode_exact_durations,
        )
        self.vae = wan.model.eval().requires_grad_(False)
        self.register_buffer(
            'scale_mean', wan.scale[0].clone(), persistent=False)
        self.register_buffer(
            'scale_inv_std', wan.scale[1].clone(), persistent=False)

        self.encode_chunk_frames = encode_chunk_frames
        self.encode_exact_durations = encode_exact_durations
        self.spatial_compression_factor = spatial_compression_factor
        self.temporal_compression_factor = temporal_compression_factor

    @staticmethod
    def _resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f'Unsupported dtype: {dtype}')
        return mapping[dtype]

    @staticmethod
    def resolve_vae_path(path: str | Path) -> Path:
        path = Path(path)
        if path.is_file():
            return path
        candidate = path / 'Wan2.2_VAE.pth'
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            'Could not find Wan2.2_VAE.pth. FluxVLA expects the original '
            f'Cosmos3/Wan2.2 VAE checkpoint, got {path}.')

    @property
    def dtype(self) -> torch.dtype:
        return self.scale_mean.dtype

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    def requires_grad_(self, requires_grad: bool = False):
        super().requires_grad_(requires_grad)
        self.vae.requires_grad_(requires_grad)
        return self

    @property
    def _scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scale_mean, self.scale_inv_std

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        in_dtype = state.dtype
        state = state.to(device=self.device, dtype=self.dtype)
        latents = self.vae.encode(state, self._scale)
        return latents.to(in_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        in_dtype = latent.dtype
        latent = latent.to(device=self.device, dtype=self.dtype)
        video = self.vae.decode(
            latent,
            self._scale,
            clear_decoder_cache=True,
        )
        return video.to(in_dtype)
