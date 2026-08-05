# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Qwen-VL collators for continuous action prediction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from fluxvla.engines import COLLATORS


def _to_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (int, float, bool, np.integer, np.floating)):
        return torch.as_tensor(value)
    raise TypeError(
        f'Unsupported value type for tensor conversion: {type(value)}')


@COLLATORS.register_module()
class QwenVLSplitActionPredictionCollator:
    """Batch pre-tokenized Qwen-VL action-prediction samples.

    Previous transforms must produce ``input_ids``, ``attention_mask``,
    ``pixel_values`` and ``image_grid_thw``. This collator only pads/stacks
    tensors and does not call the Hugging Face processor.
    """

    def __init__(
        self,
        pad_token_id: Optional[int] = None,
        padding_side: str = 'left',
        pixel_values_key: str = 'pixel_values',
        fallback_pixel_values_key: str = 'images',
        mm_token_type_ids_key: str = 'mm_token_type_ids',
        meta_keys: Optional[list[str]] = None,
        drop_keys: Optional[list[str]] = None,
    ):
        if padding_side not in ('left', 'right'):
            raise ValueError(
                f"padding_side must be 'left' or 'right', got {padding_side!r}")
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pixel_values_key = pixel_values_key
        self.fallback_pixel_values_key = fallback_pixel_values_key
        self.mm_token_type_ids_key = mm_token_type_ids_key
        self.meta_keys = set(meta_keys or [])
        self.drop_keys = set(drop_keys or [
            'vlm_content',
            'text',
            'expanded_text',
        ])

    def _pad_1d(self, values: list[Any], padding_value: int) -> torch.Tensor:
        tensors = []
        for value in values:
            tensor = _to_tensor(value).to(dtype=torch.long)
            if tensor.ndim == 2 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim != 1:
                raise ValueError(
                    f'Expected 1D token tensor, got shape {tuple(tensor.shape)}')
            tensors.append(tensor)

        max_len = max(int(tensor.shape[0]) for tensor in tensors)
        padded = []
        for tensor in tensors:
            pad_len = max_len - int(tensor.shape[0])
            if self.padding_side == 'left':
                pad = (pad_len, 0)
            else:
                pad = (0, pad_len)
            padded.append(F.pad(tensor, pad, value=padding_value))
        return torch.stack(padded, dim=0)

    def _stack_values(self, values: list[Any]) -> torch.Tensor:
        return torch.stack([_to_tensor(value) for value in values], dim=0)

    def _concat_values(self, values: list[Any]) -> torch.Tensor:
        tensors = [_to_tensor(value) for value in values]
        return torch.cat(tensors, dim=0)

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        if self.pad_token_id is None:
            raise ValueError('pad_token_id is required for split token padding.')

        batch: dict[str, Any] = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        if self.pixel_values_key in keys:
            pixel_key = self.pixel_values_key
        elif self.fallback_pixel_values_key in keys:
            pixel_key = self.fallback_pixel_values_key
        else:
            pixel_key = None

        for key in keys:
            if key in self.drop_keys:
                continue
            values = [elem[key] for elem in features if key in elem]
            if key == 'input_ids':
                batch[key] = self._pad_1d(values, self.pad_token_id)
            elif key in ('attention_mask', self.mm_token_type_ids_key):
                batch[key] = self._pad_1d(values, 0)
            elif key == pixel_key:
                batch['pixel_values'] = self._concat_values(values)
            elif key == 'image_grid_thw':
                batch[key] = self._concat_values(values).to(dtype=torch.long)
            elif key in self.meta_keys:
                batch[key] = values
            elif key == self.fallback_pixel_values_key and pixel_key != key:
                continue
            else:
                batch[key] = self._stack_values(values)

        return BatchFeature(data={'inputs': batch})
