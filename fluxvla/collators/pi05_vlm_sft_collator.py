# Copyright 2026 Limx Dynamics
"""Batch Pi05 SFT text and variable numbers of prefix images."""

import torch

from fluxvla.engines import COLLATORS
from .vlm_sft_collator import VLMSFTCollator


@COLLATORS.register_module()
class Pi05VLMSFTCollator(VLMSFTCollator):
    """Pad text by lengths and image views with an explicit validity mask."""

    def __init__(self, pad_token_id=0, ignore_index=-100):
        super().__init__(pad_token_id, ignore_index)

    def __call__(self, instances):
        if any(
                any(
                    key in item
                    for key in ('pixel_values', 'pixel_values_videos',
                                'image_grid_thw', 'video_grid_thw',
                                'mm_token_type_ids')) for item in instances):
            raise ValueError('Pi05 SFT requires its own image transform, '
                             'not native Qwen processor inputs.')
        out = super().__call__(instances)
        views = [item.get('images') for item in instances]
        shapes = set()
        for view in views:
            if view is not None:
                if (view.ndim != 3 or view.shape[0] % 3
                        or not view.is_floating_point()):
                    raise ValueError('Pi05 sample images must be float '
                                     'tensors of shape (3*N,H,W).')
                shapes.add(tuple(view.shape[-2:]))
        if len(shapes) > 1:
            raise ValueError('Pi05 SFT images must share the same resolution.')
        max_views = max(
            (view.shape[0] // 3 for view in views if view is not None),
            default=0)
        if max_views:
            prototype = next(view for view in views if view is not None)
            pixels = prototype.new_zeros(
                len(views), max_views * 3, *prototype.shape[-2:])
            masks = torch.zeros(
                len(views),
                max_views,
                dtype=torch.bool,
                device=prototype.device)
            for row, view in enumerate(views):
                if view is not None:
                    pixels[row, :view.shape[0]] = view
                    masks[row, :view.shape[0] // 3] = True
            out.update(images=pixels, img_masks=masks)
        return out
