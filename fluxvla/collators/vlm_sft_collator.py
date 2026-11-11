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
"""Batch VLM text with padded image views or native image/video patches."""

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
    """Right-pad text and batch either image views or flattened patches.

    ``image_layout='views'`` pads ``images`` to (B, 3*N, H, W) with
    ``img_masks``. The default ``'patches'`` layout concatenates native Qwen
    image/video patches and grids. Explicit layouts distinguish encoded images
    from raw annotation paths retained as metadata by native processors.
    Layouts cannot be mixed in one batch; either can include text-only samples.
    Attention masks use sequence lengths, never token IDs: pad and EOS may
    share an ID. Native multimodal token types are preserved.
    """

    def __init__(self,
                 pad_token_id,
                 ignore_index=-100,
                 image_layout='patches'):
        if image_layout not in ('patches', 'views'):
            raise ValueError("image_layout must be 'patches' or 'views'.")
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.image_layout = image_layout

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

        if self.image_layout == 'views':
            if any(
                    key in item for item in instances
                    for key in ('pixel_values', 'pixel_values_videos',
                                'image_grid_thw', 'video_grid_thw',
                                'mm_token_type_ids')):
                raise ValueError('Cannot mix image views and native patch '
                                 'inputs in one VLM SFT batch.')
            out.update(self._collate_image_views(instances))
            return out

        if any(
                isinstance(item.get('images'), torch.Tensor)
                for item in instances):
            raise ValueError("Tensor images require image_layout='views'.")

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

    @staticmethod
    def _collate_image_views(instances):
        """Pad image views and mark only real views as valid."""
        views = [item.get('images') for item in instances]
        shapes = set()
        for view in views:
            if view is not None:
                if (not isinstance(view, torch.Tensor) or view.ndim != 3
                        or view.shape[0] % 3 or not view.is_floating_point()):
                    raise ValueError('Sample images must be float tensors '
                                     'of shape (3*N,H,W).')
                shapes.add(tuple(view.shape[-2:]))
        if len(shapes) > 1:
            raise ValueError('SFT images must share the same resolution.')
        max_views = max(
            (view.shape[0] // 3 for view in views if view is not None),
            default=0)
        if not max_views:
            return {}
        prototype = next(view for view in views if view is not None)
        pixels = prototype.new_zeros(
            len(views), max_views * 3, *prototype.shape[-2:])
        masks = torch.zeros(
            len(views), max_views, dtype=torch.bool, device=prototype.device)
        for row, view in enumerate(views):
            if view is not None:
                pixels[row, :view.shape[0]] = view
                masks[row, :view.shape[0] // 3] = True
        return dict(images=pixels, img_masks=masks)
