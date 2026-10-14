# Copyright 2026 Limx Dynamics
# SPDX-License-Identifier: Apache-2.0
"""Chunk-relative actions and temporally consistent video preprocessing."""

from collections import deque

import numpy as np
import torch
import torchvision.transforms.v2 as T

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class BufferVideoFrames:
    """Keep recent policy observations for causal video inference.

    The first request of an episode returns its latest frame for cache
    prefill. Later requests return ``num_frames`` observations, padding an
    incomplete history with its earliest frame. Place after PrepareVideo.
    """

    def __init__(self, num_frames=4, key='pixel_values'):
        if num_frames < 2:
            raise ValueError('Video history requires at least two frames')
        self.num_frames = int(num_frames)
        self.key = key
        self.frames = deque(maxlen=self.num_frames)

    def __call__(self, data):
        video = data[self.key]
        if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] < 1:
            raise ValueError('Video history requires nonempty CTHW input')
        reset = bool(data.get('is_new_episode', False)) or not self.frames
        if reset:
            self.frames.clear()
        for i in range(video.shape[1]):
            frame = video[:, i:i + 1]
            self.frames.append(frame.clone(
            ) if isinstance(frame, torch.Tensor) else frame.copy())
        frames = list(self.frames)
        if reset:
            frames = frames[-1:]
        else:
            frames = [frames[0]] * (self.num_frames - len(frames)) + frames
        data[self.key] = (
            torch.cat(frames, dim=1) if isinstance(video, torch.Tensor) else
            np.concatenate(frames, axis=1))
        data['img_masks'] = np.ones(len(frames), dtype=bool)
        data['reset_history'] = reset
        return data


@TRANSFORMS.register_module()
class ChunkRelativeActions:
    """Subtract each block's raw state, before sin/cos and normalization."""

    def __init__(self, mask, action_chunk_size=24):
        self.mask = np.asarray(mask, dtype=bool)
        self.action_chunk_size = int(action_chunk_size)
        if self.mask.ndim != 1 or self.action_chunk_size <= 0:
            raise ValueError('Expected a 1D mask and positive chunk size')

    def __call__(self, data):
        actions = np.asarray(data['actions'])
        states = np.asarray(data['states'])
        if states.ndim == 1:
            states = states[None]
        if (actions.ndim != 2 or states.ndim != 2
                or actions.shape[-1] != len(self.mask)
                or states.shape[-1] != len(self.mask)
                or len(actions) != len(states) * self.action_chunk_size):
            raise ValueError(
                'Chunk-relative actions need one raw state per action block; '
                f'got actions={actions.shape}, states={states.shape}')
        reference = np.repeat(states, self.action_chunk_size, axis=0)
        data['actions'] = actions - np.where(self.mask, reference, 0)
        return data


@TRANSFORMS.register_module()
class ConsistentVideoTransform:
    """Share crop/jitter across frames and normalize RGB video to [-1, 1].

    One torchvision-v2 invocation per video shares random crop and color
    parameters across time. Independent per-frame augmentation invents motion.
    The uint8 round-trip matches DreamZero's VideoToNumpy: its head normalizes
    that output, not the intermediate resized float image.
    """

    def __init__(self,
                 height=256,
                 width=256,
                 training=True,
                 crop_scale=0.95,
                 key='images',
                 output_key=None):
        self.height, self.width = height, width
        self.training = training
        self.crop_scale = crop_scale
        self.key = key
        self.output_key = output_key or key
        if not 0 < crop_scale <= 1:
            raise ValueError('crop_scale must be in (0, 1]')
        self.resize = T.Resize((height, width), antialias=True)
        self.jitter = T.ColorJitter(0.3, 0.4, 0.5, 0.08)

    def __call__(self, data):
        frames = torch.as_tensor(np.asarray(data[self.key]))
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4:
            raise ValueError(f'Expected TCHW/THWC video, got {frames.shape}')
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)
        if frames.shape[1] != 3 or frames.dtype != torch.uint8:
            raise ValueError(
                'ConsistentVideoTransform requires uint8 RGB frames')
        frames = frames.float() / 255.0
        crop_size = tuple(int(d * self.crop_scale) for d in frames.shape[-2:])
        crop = T.RandomCrop(crop_size) if self.training else T.CenterCrop(
            crop_size)
        frames = self.resize(crop(frames))
        if self.training:
            frames = self.jitter(frames)
        frames = (frames * 255).to(torch.uint8).float() / 127.5 - 1.0
        data[self.output_key] = frames.numpy()
        return data
