# Copyright 2026 Limx Dynamics
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Composable VLM conversation and processor transforms for SFT data."""

import copy
import math
import os

import torch

from fluxvla.datasets.utils.vlm_annotations import (build_messages_llava,
                                                    normalize_llava_item)
from fluxvla.engines import TRANSFORMS

__all__ = ['BuildVLMMessages', 'ProcessVLMSFTInputs']


def video_sampling_kwargs(num_frames=None, fps=None, max_frames=32):
    """Bound temporal sampling before decoding; FPS and count are exclusive."""
    if isinstance(max_frames,
                  bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError('video_max_frames must be a positive integer.')
    if num_frames is not None and fps is not None:
        raise ValueError(
            'video_num_frames and video_fps are mutually exclusive.')
    if num_frames is not None and (isinstance(num_frames, bool)
                                   or not isinstance(num_frames, int)
                                   or not 1 <= num_frames <= max_frames):
        raise ValueError('video_num_frames must be in [1, video_max_frames].')
    if fps is not None and (isinstance(fps, bool)
                            or not isinstance(fps, (int, float))
                            or not math.isfinite(fps) or fps <= 0):
        raise ValueError('video_fps must be finite and positive.')
    # Qwen sampling bounds apply to FPS sampling; explicit frame counts are
    # checked above because upstream does not clamp them to max_frames.
    kwargs = dict(
        do_sample_frames=True,
        max_frames=max_frames,
        min_frames=min(4, max_frames))
    kwargs.update(
        num_frames=num_frames,
        fps=None if num_frames is not None else (2 if fps is None else fps))
    return kwargs


def _decode_video(path, video_processor, sampling):
    """Retain only sampled RGB frames, not torchvision's entire decoded clip.

    PyAV may still decode sequentially up to the final selected frame. This
    bounds retained frame memory, not decoding latency or compressed file IO.
    Missing frame-count metadata requires a counting pass without storage.
    """
    import av
    import numpy as np
    from transformers.video_utils import VideoMetadata

    if '://' in path:
        raise ValueError('Download video files locally before training.')
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f'No video stream: {path}')
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        total = stream.frames
        if total <= 0:
            total = sum(1 for _ in container.decode(video=0))
            container.seek(0)
        if total <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(
                f'Video requires frame count and valid FPS: {path}')
        metadata = VideoMetadata(
            total_num_frames=total,
            fps=fps,
            duration=total / fps,
            video_backend='pyav',
            height=stream.height,
            width=stream.width)
        sampler = copy.copy(video_processor)
        sampler.max_frames = sampling['max_frames']
        sampler.min_frames = sampling['min_frames']
        indices = sampler.sample_frames(
            metadata,
            num_frames=sampling.get('num_frames'),
            fps=sampling.get('fps'))
        selected = set(int(index) for index in indices)
        frames = {}
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected:
                frames[index] = frame.to_ndarray(format='rgb24')
            if index >= indices[-1]:
                break
        if selected != frames.keys():
            raise ValueError(f'Video ended before selected frames: {path}')
        metadata.frames_indices = indices.tolist()
        # Preserve repeated indices when a short clip has fewer frames than
        # an explicitly requested count, keeping timestamps/patches aligned.
        return np.stack([frames[int(index)] for index in indices]), metadata


def get_processor_for_vlm(processor_type,
                          model_path,
                          image_max_pixels=None,
                          video_max_pixels=None,
                          **kwargs):
    if processor_type not in ('qwen3_vl', 'qwen3_5'):
        raise ValueError(f'Unsupported VLM processor: {processor_type}')
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    for kind, limit in (('image', image_max_pixels), ('video',
                                                      video_max_pixels)):
        if limit is None:
            continue
        if limit <= 0:
            raise ValueError(f'{kind}_max_pixels must be positive.')
        media_processor = getattr(processor, kind + '_processor')
        size = dict(media_processor.size)
        size['longest_edge'] = limit
        size['shortest_edge'] = min(size.get('shortest_edge', limit), limit)
        media_processor.size = size
    return processor


def _mask_labels_for_sft(input_ids, tokenizer):
    """Supervise every assistant reply, including its end-of-turn token.

    Qwen ChatML headers are matched explicitly. A missing header must not
    fall back to training on the entire user prompt (the reference branch's
    fallback did this). Do not mask EOS by token ID when EOS is also PAD.
    """
    header = tokenizer.encode(
        '<|im_start|>assistant\n', add_special_tokens=False)
    start = tokenizer.encode('<|im_start|>', add_special_tokens=False)
    end = tokenizer.encode('<|im_end|>', add_special_tokens=False)
    if not header or len(start) != 1 or len(end) != 1:
        raise ValueError('SFT requires the Qwen ChatML special tokens.')
    labels = torch.full_like(input_ids, -100)
    rows = input_ids.reshape(-1, input_ids.shape[-1])
    targets = labels.reshape_as(rows)
    for row, target in zip(rows, targets):
        tokens = row.tolist()
        cursor = 0
        while cursor < len(tokens):
            if tokens[cursor:cursor + len(header)] != header:
                cursor += 1
                continue
            left = cursor + len(header)
            right = left
            while right < len(tokens) and tokens[right] not in (start[0],
                                                                end[0]):
                right += 1
            if right < len(tokens) and tokens[right] == end[0]:
                right += 1
            target[left:right] = row[left:right]
            cursor = max(right, left)
    return labels


def _process_messages_for_sft(processor,
                              messages,
                              truncation_max_length=0,
                              video_kwargs=None):
    # Decoding replaces video paths with frame arrays. Keep the canonical
    # messages immutable so transforms/metadata never retain decoded videos.
    messages = copy.deepcopy(messages)
    has_video = any(part['type'] == 'video' for message in messages
                    for part in message['content'])
    media_kwargs = (video_kwargs if video_kwargs is not None else
                    video_sampling_kwargs()) if has_video else {}
    if has_video:
        metadata = []
        for message in messages:
            for part in message['content']:
                if part['type'] == 'video':
                    frames, info = _decode_video(part['video'],
                                                 processor.video_processor,
                                                 media_kwargs)
                    part['video'] = frames
                    metadata.append(info)
        # Pass original FPS and sampled indices, and never sample a second
        # time. Qwen uses this metadata to place timestamps in the prompt.
        media_kwargs = dict(do_sample_frames=False, video_metadata=metadata)
    # Tokenize with visuals intact. Blind tokenizer truncation can remove
    # image tokens without removing their corresponding patches/grids.
    processed = dict(
        processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors='pt',
            truncation=False,
            **media_kwargs))
    tokens = processed['input_ids']
    if truncation_max_length > 0 and tokens.shape[-1] > truncation_max_length:
        tail = tokens[..., truncation_max_length:]
        visual_ids = [
            processor.tokenizer.convert_tokens_to_ids(token)
            for token in ('<|vision_start|>', '<|vision_end|>',
                          '<|image_pad|>', '<|video_pad|>')
        ]
        has_visuals = any(
            key in processed
            for key in ('pixel_values', 'pixel_values_videos'))
        if has_visuals and any((tail == value).any()
                               for value in visual_ids if value is not None):
            raise ValueError(
                'Truncation would discard visual tokens; increase max length.')
        for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
            if key in processed:
                processed[key] = processed[key][..., :truncation_max_length]
    processed['labels'] = _mask_labels_for_sft(processed['input_ids'],
                                               processor.tokenizer)
    if 'attention_mask' in processed:
        processed['labels'].masked_fill_(~processed['attention_mask'].bool(),
                                         -100)
    if not (processed['labels'][..., 1:] != -100).any():
        raise ValueError(
            'No assistant target tokens remain in the SFT sample.')
    return processed


@TRANSFORMS.register_module()
class BuildVLMMessages:
    """Validate conversations and resolve media paths from ``data_root``.

    This transform has no processor/tokenizer dependency. Its ``messages``
    output is inspectable and can be edited by later custom transforms before
    encoding. Other source fields are retained for provenance.
    """

    def __init__(self, adapter='llava'):
        if adapter not in ('llava', 'llava-video'):
            raise ValueError(f'Unknown annotation adapter: {adapter}')
        self.adapter = adapter

    def __call__(self, data):
        item = normalize_llava_item(data, adapter=self.adapter)
        messages = build_messages_llava(item, data_root=data.get('data_root'))
        return dict(data, messages=messages)


@TRANSFORMS.register_module()
class ProcessVLMSFTInputs:
    """Encode canonical ``messages`` and construct assistant-only labels.

    Owns processor initialization, media limits/sampling and safe truncation;
    the dataset is completely independent of those choices. Processor state
    is loaded lazily per worker and excluded from spawn/pickle state. Raw
    metadata is preserved; VLMSFTCollator selects only model input tensors.
    """

    def __init__(self,
                 processor_path,
                 processor_type='qwen3_vl',
                 image_max_pixels=None,
                 video_max_pixels=None,
                 video_num_frames=None,
                 video_fps=None,
                 video_max_frames=32,
                 truncation_max_length=0,
                 processor_kwargs=None):
        if not processor_path:
            raise ValueError('ProcessVLMSFTInputs requires processor_path.')
        if processor_type not in ('qwen3_vl', 'qwen3_5'):
            raise ValueError(f'Unsupported VLM processor: {processor_type}')
        if (isinstance(truncation_max_length, bool)
                or not isinstance(truncation_max_length, int)
                or truncation_max_length < 0):
            raise ValueError('Maximum length must be a non-negative integer.')
        self.processor_path = os.fspath(processor_path)
        self.processor_type = processor_type
        self.image_max_pixels = image_max_pixels
        self.video_max_pixels = video_max_pixels
        self.video_kwargs = video_sampling_kwargs(video_num_frames, video_fps,
                                                  video_max_frames)
        self.truncation_max_length = truncation_max_length
        self.processor_kwargs = dict(processor_kwargs or {})
        self._processor = None
        self._processor_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_processor=None, _processor_pid=None)
        return state

    def save_pretrained(self, output_dir):
        """Persist the exact processor settings alongside training weights."""
        processor = get_processor_for_vlm(
            self.processor_type,
            self.processor_path,
            image_max_pixels=self.image_max_pixels,
            video_max_pixels=self.video_max_pixels,
            **self.processor_kwargs)
        processor.save_pretrained(output_dir)

    def __call__(self, data):
        messages = data.get('messages')
        if (not isinstance(messages, list) or not messages or not all(
                isinstance(message, dict)
                and isinstance(message.get('content'), list)
                for message in messages)):
            raise ValueError(
                'ProcessVLMSFTInputs requires canonical messages; '
                'configure BuildVLMMessages before it.')
        if self._processor is None or self._processor_pid != os.getpid():
            self._processor = get_processor_for_vlm(
                self.processor_type,
                self.processor_path,
                image_max_pixels=self.image_max_pixels,
                video_max_pixels=self.video_max_pixels,
                **self.processor_kwargs)
            self._processor_pid = os.getpid()
        output = _process_messages_for_sft(
            self._processor,
            messages,
            truncation_max_length=self.truncation_max_length,
            video_kwargs=self.video_kwargs)
        # Only language tensors have a batch dimension; a single media grid
        # must remain (1, 3), not (3,).
        for key in ('input_ids', 'attention_mask', 'labels',
                    'mm_token_type_ids'):
            if key in output and output[key].ndim == 2:
                output[key] = output[key].squeeze(0)
        return {**data, **output}
