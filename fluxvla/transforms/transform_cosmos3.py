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
"""Transforms for Cosmos3-Nano VLA.

Main transforms:

* ``SetCosmos3ActionMetadata`` — attach Cosmos3-specific inference metadata
  using the same names as training-time action sequence construction.
* ``ProcessCosmos3Prompt`` — append optional Cosmos3 action prompt metadata,
  then tokenise task description text with the Qwen3-VL chat template
  (text-only, no image placeholders).
* ``BuildCosmos3Sequence`` — attach action or image-to-video ``SequencePlan``
  metadata; for action samples, also attach ``raw_action_dim`` and optionally
  prepend the current state to the action sequence expected by
  ``Cosmos3FlowMatching.forward()``.
"""

from __future__ import annotations
import json
import math
import os
import random
from typing import Dict, Optional

import numpy as np

import fluxvla.models.third_party_models.cosmos3 as _c3  # noqa: F401
from fluxvla.engines import TRANSFORMS
from fluxvla.models.third_party_models.cosmos3.data.vfm import sequence_packing

SequencePlan = sequence_packing.SequencePlan
_IMAGE2VIDEO_MODE = 'image2video'
_ACTION_MODES = ('wam', 'forward_dynamics', 'inverse_dynamics')
_SEQUENCE_MODES = (_IMAGE2VIDEO_MODE, *_ACTION_MODES)
_JOINT_MODE = 'joint'


@TRANSFORMS.register_module()
class SetCosmos3ActionMetadata:
    """Attach Cosmos3-specific action-policy metadata to a sample."""

    def __init__(
        self,
        conditioning_fps: Optional[float] = None,
        prepend_state_to_action: Optional[bool] = None,
    ) -> None:
        self.conditioning_fps = conditioning_fps
        self.prepend_state_to_action = prepend_state_to_action

    def __call__(self, data: Dict) -> Dict:
        if self.conditioning_fps is not None:
            data['conditioning_fps'] = np.array(
                self.conditioning_fps, dtype=np.float32)
        if self.prepend_state_to_action is not None:
            data['prepend_state_to_action'] = bool(
                self.prepend_state_to_action)
        return data


def build_sequence_plan_from_mode(
    mode: str,
    video_length: int,
    action_length: int,
    has_text: bool = True,
    video_temporal_downsample: int = 4,
    num_history_actions: int = 0,
) -> SequencePlan:
    """Build a SequencePlan based on the training mode.

    This function determines whether action should be included and computes
    condition frame indexes for vision and action based on the mode.
    """
    valid_modes = [
        'image2video', 'forward_dynamics', 'inverse_dynamics', 'wam'
    ]
    if mode not in valid_modes:
        raise ValueError(f'Invalid mode: {mode!r}. Must be one of '
                         f'{valid_modes}')

    # Determine if action should be included based on mode
    # image2video mode: no action (pure image-to-video generation)
    # forward_dynamics, inverse_dynamics, wam: action is needed
    has_action = mode != 'image2video'

    # Determine condition frame indexes based on mode
    # image2video/forward_dynamics/wam: first frame is clean (conditioning)
    # inverse_dynamics: all frames are provided as context
    if mode in ['image2video', 'forward_dynamics', 'wam']:
        condition_frame_indexes_vision = [0]
    elif mode == 'inverse_dynamics':
        # All frames are observed for inverse dynamics
        condition_frame_indexes_vision = list(
            range(0, (video_length - 1) // video_temporal_downsample + 1))
    else:
        condition_frame_indexes_vision = []

    # For action conditioning indexes:
    # forward_dynamics: all action steps are clean (conditioning)
    # inverse_dynamics/wam: action is supervised (predicted)
    # History frames (prepended) are always conditioning.
    base_action_length = action_length - num_history_actions
    if mode == 'forward_dynamics':
        condition_frame_indexes_action = list(range(action_length))
    # This currently assumes that the action length is the same as the video
    # length - 1 and if action length is the same as the video length, then the
    # first action is the conditioning action
    elif base_action_length == video_length - 1:
        condition_frame_indexes_action = list(range(num_history_actions))
    elif base_action_length == video_length:
        condition_frame_indexes_action = list(range(num_history_actions + 1))

    if base_action_length == video_length - 1:
        action_start_frame_offset = 1 - num_history_actions
    if base_action_length == video_length:
        action_start_frame_offset = -num_history_actions

    return SequencePlan(
        has_text=has_text,
        has_vision=True,
        has_action=has_action,
        condition_frame_indexes_vision=condition_frame_indexes_vision,
        condition_frame_indexes_action=condition_frame_indexes_action,
        action_start_frame_offset=action_start_frame_offset,
    )


# ---------------------------------------------------------------------------
# ProcessCosmos3Prompt
# ---------------------------------------------------------------------------


@TRANSFORMS.register_module()
class ProcessCosmos3Prompt:
    """Prepare and tokenise a Cosmos3 text prompt.

    Action SFT configs can pass ``action_metadata`` to mirror the official
    Cosmos3 action text pipeline.  By default, viewpoint text, duration/FPS
    text, and resolution text are appended before CFG dropout and Qwen3-VL
    chat-template tokenization.  With ``format_prompt_as_json=True``, the same
    metadata is serialized as the official structured action JSON prompt.
    Observation images are not inserted into the reasoner prompt.

    Args:
        tokenizer (dict): FluxVLA tokenizer config. The built tokenizer must
            wrap or expose a HuggingFace tokenizer with
            ``apply_chat_template``.
        max_len (int): Maximum token length.  Tokens beyond this limit are
            truncated (right-side).  Defaults to 4096.
        use_system_prompt (bool): Whether to prepend a system prompt.
            Defaults to False.
        cfg_dropout_rate (float): Probability of replacing the caption with an
            empty string (classifier-free guidance training).  Defaults to 0.0.
        action_metadata (dict, optional): Metadata append settings.  When set,
            follows the official action SFT order before tokenization.
        format_prompt_as_json (bool): Serialize action metadata as the official
            Cosmos3 action JSON prompt instead of appending plain text.
    """

    DEFAULT_VIEWPOINT_TEMPLATES = {
        'ego_view':
        'This video is captured from a first-person perspective looking at '
        'the scene.',
        'third_person_view':
        'This video is captured from a third-person perspective looking '
        'towards the agent from the front.',
        'wrist_view':
        'This video is captured from a wrist-mounted camera.',
        'concat_view':
        'This video contains concatenated views from multiple camera '
        'perspectives.',
    }

    def __init__(
        self,
        tokenizer: Dict,
        max_len: int = 4096,
        use_system_prompt: bool = False,
        cfg_dropout_rate: float = 0.0,
        caption_key: str = 'task_description',
        action_metadata: Optional[Dict] = None,
        format_prompt_as_json: bool = False,
        model_path: Optional[str] = None,
        output_key: str = 'text_token_ids',
        output_attention_mask_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        self.tokenizer_cfg = dict(tokenizer)
        if model_path is not None:
            self.tokenizer_cfg['model_path'] = os.path.join(
                model_path, 'tokenizer')
        self.max_len = max_len
        self.use_system_prompt = use_system_prompt
        self.cfg_dropout_rate = cfg_dropout_rate
        self.caption_key = caption_key
        self.action_metadata = action_metadata
        self.format_prompt_as_json = format_prompt_as_json
        self.output_key = output_key
        self.output_attention_mask_key = output_attention_mask_key

        # Lazy-initialised to avoid heavy imports at registry load time.
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from fluxvla.engines import build_tokenizer_from_cfg
            tokenizer = build_tokenizer_from_cfg(self.tokenizer_cfg)
            self._tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
            if not hasattr(self._tokenizer, 'apply_chat_template'):
                raise TypeError(
                    'ProcessCosmos3Prompt requires a tokenizer with '
                    'apply_chat_template().')
        return self._tokenizer

    def _append_sentence(self, caption: str, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return caption
        separator = ' ' if caption.rstrip().endswith('.') else '. '
        return caption.rstrip() + separator + sentence

    def _scalar_float(self, value, name: str) -> float:
        if value is None:
            raise ValueError(f'ProcessCosmos3Prompt missing {name!r}.')
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError('ProcessCosmos3Prompt expected scalar '
                             f'{name!r}, got shape {array.shape}.')
        return float(array.reshape(()))

    def _action_metadata_value(self, data: Dict, key: str, default=None):
        if self.action_metadata and key in self.action_metadata:
            return self.action_metadata[key]
        return data.get(key, default)

    def _ensure_sentence(self, text: str) -> str:
        text = text.strip()
        return text if text.endswith(('.', '!', '?')) else f'{text}.'

    def _format_time_mss(self, seconds: int) -> str:
        minutes, seconds = divmod(max(seconds, 0), 60)
        return f'{minutes}:{seconds:02d}'

    def _aspect_ratio(self, width: int, height: int) -> str:
        divisor = math.gcd(width, height)
        if divisor == 0:
            raise ValueError('ProcessCosmos3Prompt cannot format aspect '
                             f'ratio for {height}x{width}.')
        return f'{width // divisor},{height // divisor}'

    def _json_viewpoint_text(self, data: Dict, viewpoint: str) -> str:
        viewpoint_text = self.DEFAULT_VIEWPOINT_TEMPLATES.get(viewpoint, '')
        if (self.action_metadata
                and 'viewpoint_description' in self.action_metadata):
            viewpoint_description = self.action_metadata[
                'viewpoint_description']
        else:
            viewpoint_description = data.pop('additional_view_description',
                                             None)
        if viewpoint_description:
            separator = ' ' if viewpoint_text.endswith('.') else '. '
            viewpoint_text = (
                viewpoint_text + separator +
                str(viewpoint_description).strip())
        return viewpoint_text

    def _format_action_json_prompt(self, caption: str, data: Dict) -> str:
        if not self.action_metadata or caption == '':
            return caption

        viewpoint = self._action_metadata_value(data, 'viewpoint',
                                                'concat_view')
        data['viewpoint'] = viewpoint
        fps = self._scalar_float(
            self._action_metadata_value(data, 'conditioning_fps'),
            'conditioning_fps',
        )
        num_frames = int(
            self._scalar_float(
                self._action_metadata_value(
                    data, 'frame_window_size',
                    self._action_metadata_value(data, 'num_frames')),
                'frame_window_size',
            ))
        if fps <= 0:
            raise ValueError('ProcessCosmos3Prompt requires positive '
                             f'conditioning_fps, got {fps}.')

        height = self._action_metadata_value(data, 'video_height')
        width = self._action_metadata_value(data, 'video_width')
        if height is None or width is None:
            image_size = data.get('image_size')
            if image_size is not None:
                height, width = image_size[:2]
        height = int(self._scalar_float(height, 'video_height'))
        width = int(self._scalar_float(width, 'video_width'))

        action_length = int(
            self._scalar_float(
                self._action_metadata_value(
                    data, 'action_horizon',
                    data.get('action_horizon', num_frames - 1)),
                'action_horizon',
            ))
        duration_seconds = num_frames / fps
        action_end_time = round(duration_seconds)
        prompt = {
            'cinematography': {
                'framing': self._json_viewpoint_text(data, viewpoint),
            },
            'actions': [{
                'time': f'0:00-{self._format_time_mss(action_end_time)}',
                'description': self._ensure_sentence(caption),
                'idle_frame': f'0 out of {action_length}.',
            }],
            'duration':
            f'{int(duration_seconds)}s',
            'fps':
            float(fps),
            'resolution': {
                'H': height,
                'W': width,
            },
            'aspect_ratio':
            self._aspect_ratio(width, height),
        }
        return json.dumps(prompt)

    def _append_action_metadata(self, caption: str, data: Dict) -> str:
        if not self.action_metadata or caption == '':
            return caption
        if self.format_prompt_as_json:
            return self._format_action_json_prompt(caption, data)

        append_viewpoint = self.action_metadata.get('append_viewpoint', True)
        append_duration_fps = self.action_metadata.get('append_duration_fps',
                                                       True)
        append_resolution = self.action_metadata.get('append_resolution', True)

        viewpoint = self._action_metadata_value(data, 'viewpoint',
                                                'concat_view')
        data['viewpoint'] = viewpoint

        if append_viewpoint:
            viewpoint_text = self.DEFAULT_VIEWPOINT_TEMPLATES.get(
                viewpoint, '')
            if (self.action_metadata
                    and 'viewpoint_description' in self.action_metadata):
                viewpoint_description = (
                    self.action_metadata['viewpoint_description'])
            else:
                viewpoint_description = data.pop('additional_view_description',
                                                 None)
            if viewpoint_description:
                separator = ' ' if viewpoint_text.endswith('.') else '. '
                viewpoint_text = (
                    viewpoint_text + separator +
                    str(viewpoint_description).strip())
            caption = self._append_sentence(caption, viewpoint_text)

        if append_duration_fps:
            fps = self._scalar_float(
                self._action_metadata_value(data, 'conditioning_fps'),
                'conditioning_fps',
            )
            num_frames = int(
                self._scalar_float(
                    self._action_metadata_value(
                        data, 'frame_window_size',
                        self._action_metadata_value(data, 'num_frames')),
                    'frame_window_size',
                ))
            if fps <= 0:
                raise ValueError('ProcessCosmos3Prompt requires positive '
                                 f'conditioning_fps, got {fps}.')
            duration = num_frames / fps
            caption = self._append_sentence(
                caption,
                'The video is '
                f'{duration:.1f} seconds long and is of {fps:.0f} FPS.',
            )

        if append_resolution:
            height = self._action_metadata_value(data, 'video_height')
            width = self._action_metadata_value(data, 'video_width')
            if height is None or width is None:
                image_size = data.get('image_size')
                if image_size is not None:
                    height, width = image_size[:2]
            height = int(self._scalar_float(height, 'video_height'))
            width = int(self._scalar_float(width, 'video_width'))
            caption = self._append_sentence(
                caption, f'This video is of {height}x{width} resolution.')

        return caption

    def __call__(self, data: Dict) -> Dict:
        tokenizer = self._get_tokenizer()

        caption = data.get(self.caption_key, '')
        if not isinstance(caption, str):
            caption = str(caption)
        caption = self._append_action_metadata(caption, data)
        data[self.caption_key] = caption

        # CFG dropout: replace with empty caption with probability
        # cfg_dropout_rate
        if self.cfg_dropout_rate > 0.0 and random.random(
        ) < self.cfg_dropout_rate:
            caption = ''
            data[self.caption_key] = caption

        # Build chat-template messages (no image/video tokens)
        conversations = []
        if self.use_system_prompt:
            conversations.append({
                'role':
                'system',
                'content':
                'You are a helpful robot assistant.',
            })
        conversations.append({'role': 'user', 'content': caption})

        try:
            token_ids = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                # add_vision_id=False is specific to processors that
                # expose it (Qwen3VLProcessor).  For a plain tokenizer
                # the kwarg is harmless or absent – use a try/except.
                add_vision_id=False,
                return_dict=False,
            )
        except TypeError:
            # Fallback for tokenizers that don't accept add_vision_id
            token_ids = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )

        # Truncate to max_len.  Eval pipelines can request a matching mask
        # under `lang_masks`, while training keeps the compact token-only key.
        token_ids = token_ids[:self.max_len]

        data[self.output_key] = np.array(token_ids, dtype=np.int64)
        if self.output_attention_mask_key is not None:
            data[self.output_attention_mask_key] = np.ones(
                len(token_ids), dtype=np.bool_)
        return data


# ---------------------------------------------------------------------------
# BuildCosmos3Sequence
# ---------------------------------------------------------------------------


@TRANSFORMS.register_module()
class BuildCosmos3Sequence:
    """Attach Cosmos3 sequence metadata after normalization.

    For action modes, ``NormalizeStatesAndActions`` owns action/state padding.
    This transform checks that ``raw_action_dim`` fits the padded action width;
    when ``prepend_state_to_action=True``, it only checks that the padded
    state/action widths match before concat. In ``image2video`` mode it emits
    a vision-only ``SequencePlan`` and clears action fields.
    """

    def __init__(
        self,
        mode: str = 'wam',
        frame_window_size: int = 5,
        video_temporal_downsample: int = 4,
        conditioning_fps: float = 15.0,
        raw_action_dim: int = 0,
        prepend_state_to_action: bool = False,
        state_key: str = 'states',
    ) -> None:
        self.mode = mode
        self.frame_window_size = frame_window_size
        self.video_temporal_downsample = video_temporal_downsample
        self.conditioning_fps = conditioning_fps
        self.raw_action_dim = int(raw_action_dim)
        self.prepend_state_to_action = bool(prepend_state_to_action)
        self.state_key = state_key

        if mode not in (*_SEQUENCE_MODES, _JOINT_MODE):
            raise ValueError(f'Unknown Cosmos3 sequence mode: {mode!r}')
        if mode == _IMAGE2VIDEO_MODE and self.prepend_state_to_action:
            raise ValueError(
                'BuildCosmos3Sequence(image2video) does not support '
                'prepend_state_to_action=True.')
        if mode != _IMAGE2VIDEO_MODE and self.raw_action_dim <= 0:
            raise ValueError(
                'BuildCosmos3Sequence requires raw_action_dim > 0.')

    def __call__(self, data: Dict) -> Dict:
        effective_mode = (
            random.choice(_ACTION_MODES)
            if self.mode == _JOINT_MODE else self.mode)

        if effective_mode == _IMAGE2VIDEO_MODE:
            for key in (
                    'actions',
                    'action_masks',
                    'raw_action_dim',
                    'embodiment_ids',
                    'action_fps',
            ):
                data.pop(key, None)
            data['sequence_plan'] = build_sequence_plan_from_mode(
                mode=effective_mode,
                video_length=self.frame_window_size,
                action_length=max(self.frame_window_size - 1, 0),
                video_temporal_downsample=self.video_temporal_downsample,
            )
            data['conditioning_fps'] = np.array(
                self.conditioning_fps, dtype=np.float32)
            return data

        action = np.asarray(data['actions'], dtype=np.float32)
        if self.raw_action_dim > action.shape[-1]:
            raise ValueError(
                f'raw_action_dim={self.raw_action_dim} exceeds padded action '
                f'dim={action.shape[-1]}. Run NormalizeStatesAndActions '
                'before BuildCosmos3Sequence.')

        if self.prepend_state_to_action:
            state = np.asarray(data[self.state_key], dtype=action.dtype)
            if state.ndim != 1:
                raise ValueError(
                    f'BuildCosmos3Sequence expects {self.state_key} to be a '
                    f'single padded state row [D], got shape {state.shape}.')
            if state.shape[-1] != action.shape[-1]:
                raise ValueError(
                    f'BuildCosmos3Sequence expects {self.state_key} to be '
                    f'padded to action dim {action.shape[-1]} before concat, '
                    f'got '
                    f'{state.shape[-1]}. Run NormalizeStatesAndActions '
                    'before BuildCosmos3Sequence.')
            action = np.concatenate([state[None, :], action], axis=0)

        history_action = data.pop('history_action', None)
        num_history_actions = 0
        if history_action is not None:
            history_action = np.asarray(history_action, dtype=action.dtype)
            if history_action.ndim != 2:
                raise ValueError(
                    'BuildCosmos3Sequence expects history_action to have '
                    f'shape [H, D], got {history_action.shape}.')
            if history_action.shape[-1] != action.shape[-1]:
                raise ValueError(
                    'BuildCosmos3Sequence expects history_action to be padded '
                    f'to action dim {action.shape[-1]}, got '
                    f'{history_action.shape[-1]}.')
            num_history_actions = int(history_action.shape[0])
            action = np.concatenate([history_action, action], axis=0)

        embodiment_id = int(np.asarray(data['embodiment_ids']).item())
        if embodiment_id < 0:
            raise ValueError('Cosmos3 embodiment_id must be non-negative, got '
                             f'{embodiment_id}.')

        seq_plan = build_sequence_plan_from_mode(
            mode=effective_mode,
            video_length=self.frame_window_size,
            action_length=int(action.shape[0]),
            video_temporal_downsample=self.video_temporal_downsample,
            num_history_actions=num_history_actions,
        )
        data['actions'] = action
        data['raw_action_dim'] = np.array(self.raw_action_dim, dtype=np.int64)
        data['sequence_plan'] = seq_plan
        data['conditioning_fps'] = np.array(
            self.conditioning_fps, dtype=np.float32)
        data['action_fps'] = np.array(self.conditioning_fps, dtype=np.float32)

        return data
