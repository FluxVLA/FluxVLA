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


class WanBaseBackbone(nn.Module):
    """Common frozen Wan encoder frontend utilities.

    Wan2.1 and Wan2.2 expose different model-specific image/video encoding
    APIs, but they share device tracking, frozen-module eval handling, and
    tokenized prompt encoding through a T5 text encoder.
    """

    frozen_module_names: tuple[str, ...] = ()

    def __init__(
        self,
        device: str = 'cpu',
        torch_dtype: torch.dtype = torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self.torch_dtype = torch_dtype

    @property
    def device(self) -> torch.device:
        """Infer the current module device, falling back to build device."""
        for tensor in self.parameters():
            return tensor.device
        for tensor in self.buffers():
            return tensor.device
        return self._device

    def freeze_encoder_modules(self) -> None:
        """Freeze encoder parameters and keep their stochastic layers off."""
        self.requires_grad_(False)
        self.set_frozen_modules_to_eval_mode()

    def train(self, mode: bool = True) -> 'WanBaseBackbone':
        """Set the backbone mode while keeping frozen encoders in eval mode.

        Args:
            mode: Whether to enable training mode for the backbone.

        Returns:
            This backbone with the requested mode applied.
        """
        super().train(mode)
        self.set_frozen_modules_to_eval_mode()
        return self

    def set_frozen_modules_to_eval_mode(self) -> None:
        for module_name in self.frozen_module_names:
            module = getattr(self, module_name, None)
            if module is not None:
                module.eval()

    def _prepare_prompt_inputs(self, input_ids, attention_mask):
        ids = input_ids.to(self.device)
        mask = attention_mask.to(self.device, dtype=torch.bool)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return ids, mask

    @torch.no_grad()
    def encode_prompt_embeddings(
        self,
        input_ids,
        attention_mask,
        output_dtype: Optional[torch.dtype] = None,
    ):
        """Encode tokenized prompts and zero padded embeddings."""
        text_encoder = getattr(self, 'text_encoder', None)
        if text_encoder is None:
            raise ValueError(
                'Token encoding requires a loaded text encoder. Provide '
                'precomputed context or enable the text encoder for eval.')

        ids, mask = self._prepare_prompt_inputs(input_ids, attention_mask)
        prompt_emb = text_encoder(ids, mask).clone()
        if output_dtype is not None:
            prompt_emb = prompt_emb.to(dtype=output_dtype)

        seq_lens = mask.gt(0).sum(dim=1).long()
        for index, seq_len in enumerate(seq_lens):
            prompt_emb[index, seq_len:] = 0
        return prompt_emb

    @torch.no_grad()
    def encode_prompt_context(self, input_ids, attention_mask):
        """Encode tokenized prompts into FastWAM context and all-ones mask."""
        _, mask = self._prepare_prompt_inputs(input_ids, attention_mask)
        context = self.encode_prompt_embeddings(input_ids, attention_mask)
        context_mask = torch.ones_like(mask)
        return context.to(device=self.device), context_mask


__all__ = ['WanBaseBackbone']
