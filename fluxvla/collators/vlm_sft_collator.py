# Copyright 2026 Limx Dynamics
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Batch native Qwen VLM text, image and video processor outputs."""

import torch
from torch.nn.utils.rnn import pad_sequence

from fluxvla.engines import COLLATORS


def _sequence(value):
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError(
            'Each SFT token sequence must have shape (L,) or (1, L).')
    return value


@COLLATORS.register_module()
class VLMSFTCollator:
    """Right-pad language sequences; concatenate flattened visual patches.

    Text-only, image and video examples can coexist in one batch. Attention
    masks are based on sequence lengths, never on token IDs: pad and EOS may
    share an ID. Qwen3.5's multimodal token types must survive batching.
    """

    def __init__(self, pad_token_id, ignore_index=-100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, instances):
        if not instances:
            raise ValueError('VLMSFTCollator received an empty batch.')
        out = {}
        for key, padding in (('input_ids', self.pad_token_id),
                             ('labels', self.ignore_index),
                             ('attention_mask', 0), ('mm_token_type_ids', 0)):
            if key == 'mm_token_type_ids' and not any(key in item
                                                      for item in instances):
                continue
            values = []
            for item in instances:
                tokens = _sequence(item['input_ids'])
                value = item.get(key)
                if value is None:
                    if key == 'attention_mask':
                        value = torch.ones_like(tokens)
                    elif key == 'mm_token_type_ids':
                        if any(
                                item.get(k) is not None
                                for k in ('pixel_values',
                                          'pixel_values_videos')):
                            raise ValueError('Missing multimodal token types '
                                             'for a visual sample.')
                        value = torch.zeros_like(tokens)
                    else:
                        raise ValueError(f'Missing required SFT field: {key}')
                value = _sequence(value)
                if value.shape != tokens.shape:
                    raise ValueError(f'{key} length does not match input_ids.')
                values.append(value)
            out[key] = pad_sequence(
                values, batch_first=True, padding_value=padding)
        out['labels'] = out['labels'].masked_fill(
            ~out['attention_mask'].bool(), self.ignore_index)
        if not (out['labels'][:, 1:] != self.ignore_index).any(dim=1).all():
            raise ValueError('Every SFT sample needs a supervised next token.')

        for pixel_key, grid_key in (('pixel_values', 'image_grid_thw'),
                                    ('pixel_values_videos', 'video_grid_thw')):
            pixels, grids = [], []
            for item in instances:
                pixel, grid = item.get(pixel_key), item.get(grid_key)
                if pixel is None and grid is None:
                    continue
                if pixel is None or grid is None:
                    raise ValueError(
                        f'{pixel_key} and {grid_key} must occur together.')
                if pixel.ndim != 2 or grid.ndim != 2 or grid.shape[1] != 3:
                    raise ValueError(
                        'Expected flattened patches (P, D) and grids (N, 3).')
                patch_count = grid.prod(-1).sum().item()
                if (grid <= 0).any() or patch_count != pixel.shape[0]:
                    raise ValueError(
                        f'{pixel_key} patch count does not match {grid_key}.')
                pixels.append(pixel)
                grids.append(grid)
            if pixels:
                out[pixel_key] = torch.cat(pixels)
                out[grid_key] = torch.cat(grids)
        return out
