# Copyright 2026 Limx Dynamics
"""Build current-view + goal -> subtask conversations for native VLM SFT."""

import math

import torch
from PIL import Image

from fluxvla.datasets.utils.video_decode import (build_lerobot_video_path,
                                                 decode_video_frames)
from fluxvla.engines import TRANSFORMS


def build_subtask_vlm_messages(goal, images, camera_names, target=None):
    """The same prompt can be used online by omitting ``target`` entirely."""
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError('A nonempty whole-task goal is required')
    if not images or len(images) != len(camera_names):
        raise ValueError('Images must match the named camera views')
    content = []
    for name, image in zip(camera_names, images):
        if not isinstance(image, Image.Image) or image.mode != 'RGB':
            raise ValueError('Subtask VLM expects RGB PIL images')
        content.extend([
            dict(type='text', text=f'Camera: {name}'),
            dict(type='image', image=image)
        ])
    content.append(
        dict(
            type='text',
            text=f'Overall task: {goal.strip()}\n'
            'Using only the current camera views and the overall task, '
            'predict the subtask the robot should execute now. Describe '
            'the relevant arm, object and target when visible. '
            'Return only one subtask '
            'instruction, without reasoning, a list or the overall task.'))
    messages = [dict(role='user', content=content)]
    if target is not None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError('Subtask target must be a nonempty string')
        messages.append(
            dict(
                role='assistant',
                content=[dict(type='text', text=target.strip())]))
    return messages


@TRANSFORMS.register_module()
class BuildSubtaskMessages:
    """Decode exactly the current frame for each camera; never a video clip.

    Support both per-episode files and v3 shared video files using each
    camera's from_timestamp. Prompt fields are explicitly whitelisted: no
    segment number, frame number, future labels, actions or progress hints.
    ``video_keys`` defines the camera names, count and order for any dataset
    following the LeRobot video contract. Model processing is a separate
    transform; this class only produces raw image/text messages.
    """

    def __init__(self, video_keys, backend='torchcodec'):
        if (not isinstance(video_keys, (list, tuple)) or not video_keys
                or any(not isinstance(key, str) or not key
                       for key in video_keys)
                or len(set(video_keys)) != len(video_keys)):
            raise ValueError('video_keys must name distinct camera fields')
        self.video_keys = tuple(video_keys)
        self.backend = backend

    def __call__(self, sample):
        context = sample['lerobot_video']
        timestamps = context['timestamps']
        if len(timestamps) != 1:
            raise ValueError('Subtask SFT accepts only the current frame')
        images = []
        for key in self.video_keys:
            if context['info']['features'].get(key,
                                               {}).get('dtype') != 'video':
                raise ValueError(f'Missing video feature: {key}')
            offset = float(context['episode_meta'].get(
                f'videos/{key}/from_timestamp', 0.0))
            timestamp = timestamps[0] + offset
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(
                    'Video timestamp must be finite and nonnegative')
            path = build_lerobot_video_path(context['data_root_path'],
                                            context['info'],
                                            context['episode_meta'],
                                            context['episode_index'], key)
            frame = decode_video_frames(
                path, [timestamp],
                tolerance_s=0.49 / float(context['info']['fps']),
                backend=self.backend)
            if (frame.ndim != 4 or frame.shape[:2] != (1, 3)
                    or frame.dtype != torch.uint8):
                raise ValueError('Decoder must return uint8 [1,3,H,W]')
            images.append(Image.fromarray(frame[0].permute(1, 2, 0).numpy()))
        output = {
            key: value
            for key, value in sample.items() if key != 'lerobot_video'
        }
        output['messages'] = build_subtask_vlm_messages(
            sample['full_task_description'], images,
            [key.rsplit('.', 1)[-1] for key in self.video_keys],
            sample['subtask'])
        return output
