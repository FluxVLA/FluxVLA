# Copyright 2026 Limx Dynamics
"""Batch complete video/action blocks without truncating samples."""

import torch

from fluxvla.engines import COLLATORS
from .dict_collator import DictCollator


@COLLATORS.register_module()
class VideoActionBlockCollator(DictCollator):
    """Right-pad temporal fields and retain each sample's valid block count.

    Inputs use PrepareVideo's CTHW layout with one conditioning frame.
    Padding adds only whole future blocks for block-causal models.
    ``num_valid_blocks`` lets consumers exclude padding from temporal losses
    while preserving the per-sample weighting of unpadded training.
    """

    def __init__(self,
                 keys,
                 meta_keys=None,
                 video_frames_per_block=8,
                 actions_per_block=24,
                 states_per_block=1):
        super().__init__(keys=keys, meta_keys=meta_keys)
        self.video_frames_per_block = video_frames_per_block
        self.actions_per_block = actions_per_block
        self.states_per_block = states_per_block
        for size in (video_frames_per_block, actions_per_block,
                     states_per_block):
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    'Video/action block sizes must be positive ints.')

    @staticmethod
    def _pad(value, length, axis=0, repeat_last=False):
        value = torch.as_tensor(value)
        missing = length - value.shape[axis]
        if missing == 0:
            return value
        shape = list(value.shape)
        shape[axis] = missing
        if repeat_last:
            padding = value.narrow(axis, value.shape[axis] - 1,
                                   1).expand(shape)
        else:
            padding = value.new_zeros(shape)
        return torch.cat((value, padding), dim=axis)

    def __call__(self, batch):
        if not batch:
            raise ValueError(
                'VideoActionBlockCollator requires a non-empty batch.')
        block_counts = []
        for index, sample in enumerate(batch):
            images = sample['images']
            if images.ndim != 4:
                raise ValueError('Images must use [C, T, H, W].')
            frames = images.shape[1]
            blocks, remainder = divmod(frames - 1, self.video_frames_per_block)
            if blocks < 1 or remainder:
                raise ValueError(
                    f'Sample {index} has {frames} frames; expected 1 + N * '
                    f'{self.video_frames_per_block} frames with N >= 1.')
            lengths = {
                'states': blocks * self.states_per_block,
                'actions': blocks * self.actions_per_block,
                'action_masks': blocks * self.actions_per_block,
                'img_masks': frames,
                'frame_masks': frames,
            }
            for key, length in lengths.items():
                value = sample[key]
                if value.ndim < 1 or value.shape[0] != length:
                    raise ValueError(
                        f'Sample {index}: {key} shape {tuple(value.shape)} '
                        f'does not match {blocks} video blocks '
                        f'(expected temporal length {length}).')
            block_counts.append(blocks)

        max_blocks = max(block_counts)
        max_frames = 1 + max_blocks * self.video_frames_per_block
        lengths = {
            'states': max_blocks * self.states_per_block,
            'actions': max_blocks * self.actions_per_block,
            'action_masks': max_blocks * self.actions_per_block,
            'img_masks': max_frames,
            'frame_masks': max_frames,
        }
        padded = []
        for sample in batch:
            result = dict(sample)
            result['images'] = self._pad(
                sample['images'], max_frames, axis=1, repeat_last=True)
            for key, length in lengths.items():
                result[key] = self._pad(sample[key], length)
            padded.append(result)
        collated = super().__call__(padded)
        collated['num_valid_blocks'] = torch.tensor(
            block_counts, dtype=torch.long)
        return collated
