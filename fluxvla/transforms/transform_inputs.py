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

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import av
import numpy as np
import torch
from PIL import Image

from fluxvla.datasets.utils.video_decode import (build_lerobot_video_path,
                                                 decode_video_frames)
from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.eval_utils import crop_and_resize, quat2axisangle
from .transform_images import (_resize_hwc_lanczos3_numpy,
                               _resize_hwc_lanczos3_tensorflow)
from .modality_state_action import (
    ModalityStateActionCodec, resolve_groot_n17_flat_slices,
    resolve_groot_n17_metadata)
from .utils import pad_to_dim, parse_image


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@TRANSFORMS.register_module()
class ProcessLiberoInputs():
    """Process inputs for Libero dataset.
    This transform processes the inputs from the Libero
    dataset to match the expected format for the model.
    It pads the state and action dimensions to the specified
    action dimension and parses the images from the input data.
    The processed inputs are returned in a dictionary format
    that includes the state, images, image masks, and
    actions (if available). The prompt is also included
    if it exists in the input data.

    Args:
        action_dim (int): The dimension to which the state and
            actions will be padded.
        model_type (str): The type of model being used, which
            may affect how images are masked.
    """

    def __init__(self, action_dim: int, model_type: str):
        self.action_dim = action_dim
        self.model_type = model_type

    def __call__(self, data):
        state = pad_to_dim(data['state'], self.action_dim)
        # TODO: Change to opencv
        base_image = parse_image(data['image'])
        wrist_image = parse_image(data['wrist_image'])

        # Create inputs dict. Do not change the keys
        # in the dict below.
        inputs = {
            'states': state,
            'images': [base_image, wrist_image],
            'img_masks': torch.tensor(([True, True]))
        }
        if 'actions' in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = pad_to_dim(data['actions'], self.action_dim)
            inputs['actions'] = actions

        # Pass the prompt (aka language instruction)
        # to the model.
        # Keep this for your own dataset (but modify
        # the key if the instruction is not
        # stored in "prompt"; the output dict always
        # needs to have the key "prompt").
        if 'prompt' in data:
            inputs['prompt'] = data['prompt']

        return inputs


@TRANSFORMS.register_module()
class ProcessParquetInputs():
    """Process inputs for Parquet dataset.
    This transform processes the inputs from the Parquet
    dataset to match the expected format for the model.
    It pads the state and action dimensions to the specified
    action dimension and parses the images from the input data.
    The processed inputs are returned in a dictionary format
    that includes the state, images, image masks, and
    actions (if available). The prompt is also included
    if it exists in the input data.

    Args:
        parquet_keys (List[str]): List of keys to extract
            from the parquet data.
        video_keys (List[str]): List of keys corresponding
            to video data.
        data_root (str): Root directory for the video files.
        name_mappings (Dict, optional): Optional dictionary
            to map original keys to new keys.
            Defaults to None.
        video_backend (str, optional): Video decoding backend. One of
            ``'torchcodec'``, ``'pyav'`` or ``'video_reader'``. When ``None``
            (default), the ``'pyav'`` torchvision path is used. Explicitly
            selecting ``'torchcodec'`` is strict and raises instead of falling
            back. The TorchCodec path decodes by frame index
            (``round(ts * average_fps)``).
    """

    def __init__(self,
                 parquet_keys: List[str],
                 video_keys: List[str],
                 name_mappings: Dict = None,
                 embodiment_id: int = None,
                 embodiment_dim: int = None,
                 num_padding_imgs: int = 0,
                 dataset_name: str = None,
                 video_backend: str = None):
        self.parquet_keys = parquet_keys
        self.video_keys = video_keys
        self.name_mappings = name_mappings
        self.embodiment_id = embodiment_id
        self.embodiment_dim = embodiment_dim
        self.num_padding_imgs = num_padding_imgs
        self.dataset_name = dataset_name
        self.video_backend = video_backend

    def __call__(self, data):
        assert 'info' in data, "Input data must contain 'info' key"
        info = data['info']
        inputs = dict()
        # Check if the video path is provided in the info
        assert 'video_path' in info, "Input data must contain 'video_path' key"
        video_root_path = info['video_path']
        for key in self.parquet_keys:
            try:
                value = data[key]
            except KeyError as exc:
                raise KeyError(f'Missing input data key: {key}') from exc
            mapped_names = None
            if self.name_mappings is not None:
                mapped_names = self.name_mappings.get(key)
            if mapped_names is not None:
                if isinstance(mapped_names, str):
                    if isinstance(value, list) or isinstance(value, float):
                        inputs[mapped_names] = np.array(value)
                    else:
                        inputs[mapped_names] = value
                else:
                    for mapped_key in mapped_names:
                        if isinstance(value, list) or isinstance(value, float):
                            inputs[mapped_key] = np.array(value)
                        else:
                            inputs[mapped_key] = value
            else:
                if isinstance(value, list) or isinstance(value, float):
                    inputs[key] = np.array(value)
                else:
                    inputs[key] = value
        images = list()
        img_masks = list()
        timestamps = data.get('frame_timestamps', [data['timestamp']])
        for video_key in self.video_keys:
            episode_chunk = data['episode_index'] // data['info'][
                'chunks_size']  # noqa: E501
            video_path = os.path.join(
                data['data_root'],
                video_root_path.format(
                    episode_chunk=episode_chunk,
                    video_key=video_key,
                    episode_index=data['episode_index']))
            assert os.path.exists(
                video_path), f'Video file not found: {video_path}'
            # Load all requested timestamps at once (supports temporal window)
            unique_ts = sorted(set(timestamps))
            frames_tensor = decode_video_frames(
                video_path, unique_ts, 0.1, backend=self.video_backend)
            ts_to_frame = {
                ts: frames_tensor[i]
                for i, ts in enumerate(unique_ts)
            }
            for ts in timestamps:
                nearest = min(unique_ts, key=lambda x: abs(x - ts))
                images.append(ts_to_frame[nearest].numpy())
            for _ in timestamps:
                img_masks.append(True)
        # Add padding images with zero values and False masks
        if self.num_padding_imgs > 0 and len(images) > 0:
            padding_img = np.zeros_like(images[0])
            for _ in range(self.num_padding_imgs):
                images.append(padding_img)
                img_masks.append(False)
        inputs['images'] = images
        inputs['img_masks'] = np.array(img_masks)
        inputs['task_description'] = data.get('task_description', '')
        if self.dataset_name is not None:
            inputs['dataset_name'] = self.dataset_name
        if self.embodiment_id is not None:
            inputs['embodiment_ids'] = np.array(self.embodiment_id)
        if 'frame_masks' in data:
            inputs['frame_masks'] = data['frame_masks']
        if 'sample_weight' in data:
            inputs['sample_weight'] = np.asarray(
                data['sample_weight'], dtype=np.float32)

        return inputs

    def read_video_frame(self, video_path: str, frame_idx: int):
        container = av.open(video_path)
        for i, frame in enumerate(container.decode(video=0)):
            if i == frame_idx:
                return frame.to_ndarray(format='rgb24')


@TRANSFORMS.register_module()
class BuildModalityStateActionTargets:
    """Build N1.7 continuous action-head tensors from metadata layouts.

    It intentionally preserves the input sample so later transforms can build
    Qwen-VL image/text inputs from the same parquet fields.
    """

    def __init__(
        self,
        processor_path: str,
        embodiment_tag: str = 'LIBERO_PANDA',
        state_key: str = 'states',
        action_key: str = 'actions',
        action_mask_key: str = 'action_masks',
        flat_layout: str = 'auto',
        train_mode: bool = True,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        output_state_key: str = 'state',
        output_action_key: str = 'action',
        output_action_mask_key: str = 'action_mask',
        output_embodiment_id_key: str = 'embodiment_id',
    ):
        self.processor_path = processor_path
        self.state_key = state_key
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        self.flat_layout = flat_layout
        self.output_state_key = output_state_key
        self.output_action_key = output_action_key
        self.output_action_mask_key = output_action_mask_key
        self.output_embodiment_id_key = output_embodiment_id_key
        self.training = train_mode

        metadata = resolve_groot_n17_metadata(
            processor_path,
            embodiment_tag=embodiment_tag,
            **dict(processor_kwargs or {}))
        processor_kwargs = metadata['processor_kwargs']
        self.embodiment_key = metadata['embodiment_key']
        self.modality_configs = processor_kwargs['modality_configs']
        self.modality_config = metadata['modality_config']
        self.statistics = metadata['statistics']
        self.max_state_dim = processor_kwargs.get('max_state_dim', 29)
        self.max_action_dim = processor_kwargs.get('max_action_dim', 29)
        self.max_action_horizon = processor_kwargs.get(
            'max_action_horizon', 50)
        self.exclude_state = processor_kwargs.get('exclude_state', False)
        self.state_dropout_prob = processor_kwargs.get(
            'state_dropout_prob', 0.0)
        self.embodiment_id = metadata['embodiment_id']
        self.state_action_processor = ModalityStateActionCodec(
            modality_configs=self.modality_configs,
            statistics=processor_kwargs.get('statistics'),
            use_percentiles=processor_kwargs.get('use_percentiles', False),
            clip_outliers=processor_kwargs.get('clip_outliers', True),
            apply_sincos_state_encoding=processor_kwargs.get(
                'apply_sincos_state_encoding', False),
            use_relative_action=processor_kwargs.get(
                'use_relative_action', False))
        if train_mode:
            self.state_action_processor.train()
        else:
            self.state_action_processor.eval()

    def _flat_slices(self, modality: str) -> Dict[str, tuple[int, int]]:
        return resolve_groot_n17_flat_slices(
            self.modality_config,
            self.statistics,
            self.embodiment_key,
            modality,
            self.flat_layout)

    def _split_flat(self, value: Any, modality: str) -> Dict[str, np.ndarray]:
        if isinstance(value, dict):
            return {
                key: _to_numpy(item).astype(np.float32)
                for key, item in value.items()
            }

        array = _to_numpy(value).astype(np.float32)
        if array.ndim == 1:
            array = array[None, :]
        slices = self._flat_slices(modality)
        return {
            key: array[..., start:end].copy()
            for key, (start, end) in slices.items()
        }

    def _apply_external_action_mask(self, outputs: Dict[str, Any],
                                    sample: Dict[str, Any]) -> None:
        if (self.action_mask_key not in sample
                or self.output_action_mask_key not in outputs):
            return
        mask = torch.as_tensor(
            _to_numpy(sample[self.action_mask_key]),
            dtype=outputs[self.output_action_mask_key].dtype)
        if mask.ndim == 1:
            mask = mask[:, None]
        horizon = min(mask.shape[0], outputs[self.output_action_mask_key].shape[0])
        outputs[self.output_action_mask_key][:horizon] *= mask[:horizon]
        if horizon < outputs[self.output_action_mask_key].shape[0]:
            outputs[self.output_action_mask_key][horizon:] = 0

    def _build_action_targets(
            self, normalized_actions: Dict[str, np.ndarray]) -> tuple[
                Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not normalized_actions:
            assert not self.training, 'Action is required in training mode'
            return None, None

        action_keys = self.modality_config['action']['modality_keys']
        normalized_action = torch.cat(
            [torch.from_numpy(normalized_actions[key]) for key in action_keys],
            dim=-1)
        action_dim = normalized_action.shape[1]
        normalized_action = torch.cat([
            normalized_action,
            torch.zeros(normalized_action.shape[0],
                        self.max_action_dim - normalized_action.shape[1])
        ],
                                      dim=-1)
        action_horizon = normalized_action.shape[0]
        normalized_action = torch.cat([
            normalized_action,
            torch.zeros(self.max_action_horizon - normalized_action.shape[0],
                        self.max_action_dim)
        ],
                                      dim=0)
        action_mask = torch.ones_like(normalized_action)
        action_mask[action_horizon:] = 0
        action_mask[:, action_dim:] = 0
        return normalized_action, action_mask

    def _build_state(self, raw_state: Dict[str, np.ndarray],
                     normalized_state: Dict[str, np.ndarray]) -> torch.Tensor:
        state_keys = self.modality_config['state']['modality_keys']
        state_cfg = self.modality_config['state']
        exclude_state = self.exclude_state or bool(
            state_cfg.get('exclude_state', False))
        if exclude_state or (self.state_dropout_prob > 0
                             and random.random() < self.state_dropout_prob
                             and self.training):
            state = torch.cat(
                [torch.from_numpy(np.zeros_like(raw_state[key])) for key in state_keys],
                dim=-1)
        else:
            state = torch.cat(
                [torch.from_numpy(normalized_state[key]) for key in state_keys],
                dim=-1)
        state = torch.cat([
            state,
            torch.zeros(state.shape[0], self.max_state_dim - state.shape[1])
        ],
                          dim=-1)
        return state

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        state_data = self._split_flat(sample[self.state_key], 'state')
        action_data = {}
        if self.action_key in sample and sample[self.action_key] is not None:
            action_data = self._split_flat(sample[self.action_key], 'action')
        elif self.training:
            raise KeyError(f'Missing action key: {self.action_key!r}')

        normalized_state, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=self.embodiment_key)
        normalized_action, action_mask = self._build_action_targets(
            normalized_actions)
        state = self._build_state(state_data, normalized_state)

        outputs = dict(sample)
        outputs[self.output_state_key] = state.to(torch.get_default_dtype())
        if normalized_action is not None:
            outputs[self.output_action_key] = normalized_action.to(
                torch.get_default_dtype())
        if action_mask is not None:
            outputs[self.output_action_mask_key] = action_mask
        outputs[self.output_embodiment_id_key] = self.embodiment_id
        self._apply_external_action_mask(outputs, sample)
        return outputs


@TRANSFORMS.register_module()
class ProcessOBSInputs():
    """Process inputs for OBS dataset.
    This transform processes the inputs from the OBS dataset
    to match the expected format for the model.
    It pads the state and action dimensions to the specified
    action dimension and parses the images from the input data.
    The processed inputs are returned in a dictionary format
    that includes the state, images, image masks, and
    actions (if available). The prompt is also included
    if it exists in the input data.

    Args:
        action_dim (int): The dimension to which the state and
            actions will be padded.
        model_type (str): The type of model being used, which
            may affect how images are masked.
    """

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def __call__(self, inputs):
        inputs['states'] = torch.from_numpy(
            pad_to_dim(inputs['states'], self.action_dim))

        return inputs


# === Libero-specific Image Loader Transform ===
@TRANSFORMS.register_module()
class ProcessLiberoEvalInputs:
    """ Process Libero eval inputs.
    This transform loads LIBERO observation images, rotates them, converts
    them to PIL images, and leaves model-specific resizing to later image
    transforms. If enabled, center crop is applied with the OpenVLA-compatible
    crop-and-resize path.

    Args:
        img_keys (List[str]): Image keys to fetch from inputs.
            Default to ['agentview_image'].
        center_crop (bool): If True, center crop to 0.9 area and resize back
            to 224x224 before later model-specific processing.
        use_pil (bool): If True, use PIL to load the images.
            Default to True.
        resize_size (int | tuple | None): If set, lanczos-resize the rotated
            raw image before center crop.
        resize_backend (str): Resize implementation, either ``numpy`` or
            ``tensorflow``.
        jpeg_roundtrip (bool): If True, encode/decode JPEG before resizing.
            This is opt-in because the default eval path for existing
            checkpoints was trained and validated without JPEG round-trip.
    """

    def __init__(self,
                 img_keys: List[str] = ['agentview_image'],
                 center_crop: bool = False,
                 use_pil: bool = True,
                 resize_size: int = None,
                 resize_backend: str = 'numpy',
                 jpeg_roundtrip: bool = False,
                 embodiment_id: int = None) -> None:
        self.img_keys = img_keys
        self.center_crop = center_crop
        self.use_pil = use_pil
        self.resize_size = resize_size
        if resize_backend not in {'numpy', 'tensorflow'}:
            raise ValueError(
                "resize_backend must be either 'numpy' or 'tensorflow'")
        if jpeg_roundtrip and resize_backend != 'tensorflow':
            raise ValueError(
                "jpeg_roundtrip=True requires resize_backend='tensorflow'")
        self.resize_backend = resize_backend
        self.jpeg_roundtrip = jpeg_roundtrip
        self.embodiment_id = embodiment_id

    def __call__(self, inputs: Dict) -> Dict:
        # Load raw images
        imgs = list()
        replay_img = None
        for img_key in self.img_keys:
            if img_key not in inputs:
                raise KeyError(f'Missing image key: {img_key!r}')
            img = np.asarray(inputs[img_key])
            img = img[::-1, ::-1].copy()
            if self.resize_size is not None:
                if isinstance(self.resize_size, int):
                    height, width = self.resize_size, self.resize_size
                else:
                    height, width = self.resize_size
                if self.resize_backend == 'tensorflow':
                    img = _resize_hwc_lanczos3_tensorflow(
                        img, height, width, jpeg_roundtrip=self.jpeg_roundtrip)
                else:
                    img = _resize_hwc_lanczos3_numpy(img, height, width)
            if replay_img is None:
                replay_img = img.copy()
            imgs.append(img)
        images = list()
        img_masks = list()
        if self.use_pil:
            for img in imgs:
                image = Image.fromarray(img)
                image = image.convert('RGB')

                if self.center_crop:
                    image = Image.fromarray(
                        crop_and_resize(np.array(image), 0.9, 1))
                    image = image.convert('RGB')

                images.append(image)
                img_masks.append(True)
        else:
            images = imgs
            img_masks = [True] * len(imgs)
        inputs['pixel_values'] = images
        inputs['img_masks'] = img_masks
        inputs['replay_img'] = replay_img
        if self.embodiment_id is not None:
            inputs['embodiment_ids'] = np.array(
                self.embodiment_id, dtype=np.int32)
        return inputs


@TRANSFORMS.register_module()
class BuildLiberoFlatEvalObservation:
    """Build a flat modality observation from existing LIBERO env obs.

    The output observation uses explicit ``video.*`` and ``state.*`` keys so
    later model-specific transforms can consume the same adapter result without
    depending on raw LIBERO key names.
    """

    def __init__(self,
                 image_key: str = 'agentview_image',
                 wrist_image_key: str = 'robot0_eye_in_hand_image',
                 pos_key: str = 'robot0_eef_pos',
                 quat_key: str = 'robot0_eef_quat',
                 gripper_key: str = 'robot0_gripper_qpos',
                 task_key: str = 'task_description',
                 observation_key: str = 'flat_observation',
                 task_output_key: Optional[str] = None,
                 replay_image_key: Optional[str] = 'replay_img') -> None:
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.pos_key = pos_key
        self.quat_key = quat_key
        self.gripper_key = gripper_key
        self.task_key = task_key
        self.observation_key = observation_key
        self.task_output_key = task_output_key
        self.replay_image_key = replay_image_key

    def build_observation(self, inputs: Dict) -> tuple[Dict[str, Any], str]:
        for key in (self.image_key, self.wrist_image_key, self.pos_key,
                    self.quat_key, self.gripper_key):
            if key not in inputs:
                raise KeyError(f'Missing LIBERO eval input key: {key!r}')

        xyz = np.asarray(inputs[self.pos_key], dtype=np.float32)
        rpy = quat2axisangle(np.asarray(inputs[self.quat_key], dtype=np.float32))
        task = inputs.get(
            self.task_key,
            inputs.get('annotation.human.action.task_description', ''))
        observation = {
            'video.image': np.asarray(inputs[self.image_key],
                                      dtype=np.uint8)[::-1, ::-1].copy(),
            'video.wrist_image': np.asarray(inputs[self.wrist_image_key],
                                            dtype=np.uint8)[::-1, ::-1].copy(),
            'state.x': np.asarray([xyz[0]], dtype=np.float32),
            'state.y': np.asarray([xyz[1]], dtype=np.float32),
            'state.z': np.asarray([xyz[2]], dtype=np.float32),
            'state.roll': np.asarray([rpy[0]], dtype=np.float32),
            'state.pitch': np.asarray([rpy[1]], dtype=np.float32),
            'state.yaw': np.asarray([rpy[2]], dtype=np.float32),
            'state.gripper': np.asarray(inputs[self.gripper_key],
                                        dtype=np.float32).copy(),
            'annotation.human.action.task_description': task,
            'task_description': task,
        }
        return observation, task

    def __call__(self, inputs: Dict) -> Dict:
        observation, task = self.build_observation(inputs)
        outputs = dict(inputs)
        outputs[self.observation_key] = observation
        if self.task_output_key is not None:
            outputs[self.task_output_key] = task
        if self.replay_image_key is not None:
            outputs[self.replay_image_key] = observation['video.image'].copy()
        return outputs


@TRANSFORMS.register_module()
class BuildEvalInputsFromFlatObservation:
    """Build reusable eval sample fields from a flat modality observation."""

    def __init__(
        self,
        observation_key: str = 'flat_observation',
        video_keys: Optional[List[str]] = None,
        state_keys: Optional[List[str]] = None,
        video_prefix: str = 'video.',
        state_prefix: str = 'state.',
        task_key: str = 'task_description',
        output_image_key: str = 'images',
        output_state_key: str = 'states',
        output_task_key: str = 'task_description',
        add_state_step_dim: bool = True,
    ) -> None:
        self.observation_key = observation_key
        self.video_keys = video_keys
        self.state_keys = state_keys
        self.video_prefix = video_prefix
        self.state_prefix = state_prefix
        self.task_key = task_key
        self.output_image_key = output_image_key
        self.output_state_key = output_state_key
        self.output_task_key = output_task_key
        self.add_state_step_dim = add_state_step_dim

    def _resolve_keys(self, observation: Dict[str, Any], prefix: str,
                      keys: Optional[List[str]]) -> List[str]:
        if keys is not None:
            return list(keys)
        return sorted(
            key[len(prefix):]
            for key in observation
            if key.startswith(prefix))

    def _state_value(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if self.add_state_step_dim and array.ndim == 1:
            array = array[None, :]
        return array.copy()

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.observation_key not in inputs:
            raise KeyError(
                f'Missing flat observation key: {self.observation_key!r}')
        observation = inputs[self.observation_key]
        video_keys = self._resolve_keys(
            observation, self.video_prefix, self.video_keys)
        state_keys = self._resolve_keys(
            observation, self.state_prefix, self.state_keys)

        images = {}
        for key in video_keys:
            flat_key = f'{self.video_prefix}{key}'
            if flat_key not in observation:
                raise KeyError(f'Missing flat video key: {flat_key!r}')
            images[key] = [np.asarray(observation[flat_key]).copy()]

        states = {}
        for key in state_keys:
            flat_key = f'{self.state_prefix}{key}'
            if flat_key not in observation:
                raise KeyError(f'Missing flat state key: {flat_key!r}')
            states[key] = self._state_value(observation[flat_key])

        task = inputs.get(
            self.task_key,
            observation.get(
                self.task_key,
                observation.get('annotation.human.action.task_description',
                                '')))
        outputs = dict(inputs)
        outputs[self.output_image_key] = images
        outputs[self.output_state_key] = states
        outputs[self.output_task_key] = task
        return outputs


@TRANSFORMS.register_module()
class PadKeyToDim():
    """
    Pad the tensor of the specified keys in the input to an integer
        multiple of its current length, and fill the target dimension
        by copying the original tensor.

    Args:
        keys (List[str]): List of keys in the input dictionary
            to be padded.
        dim (int): The target dimension should be an integer
            multiple of the current length.
    """

    def __init__(self, keys: List[str], dim: int):
        self.keys = keys
        self.dim = dim

    def __call__(self, inputs):
        for key in self.keys:
            if key in inputs:
                tensor = inputs[key]
                orig_shape = tensor.shape
                orig_len = orig_shape[-1]
                target_len = ((orig_len + self.dim - 1) // self.dim) * self.dim
                if target_len == orig_len:
                    inputs[key] = tensor
                    continue
                # Pad by copying the entire original tensor to reach the
                # target length
                repeat_times = (target_len + orig_len - 1) // orig_len
                repeat_target = [1] * len(orig_shape)
                repeat_target[-1] = repeat_times
                tensor_padded = np.tile(tensor, repeat_target)
                inputs[key] = tensor_padded
        return inputs


@TRANSFORMS.register_module()
class DecodeLeRobotVideoSequence():
    """Decode multi-frame LeRobot episode videos into ``images``.

    Expects ``lerobot_video`` metadata emitted by :class:`SARMDataset` /
    :class:`ARMDataset` and writes ``images`` as ``[T, N, C, H, W]`` numpy.
    """

    def __init__(self,
                 video_keys: List[str],
                 tolerance_s: float = 0.1,
                 backend: str = 'pyav') -> None:
        self.video_keys = video_keys
        self.tolerance_s = tolerance_s
        self.backend = backend

    def __call__(self, inputs: Dict) -> Dict:
        ctx = inputs.pop('lerobot_video')
        data_root_path = ctx['data_root_path']
        info = ctx['info']
        episode_meta = ctx['episode_meta']
        episode_index = int(ctx['episode_index'])
        timestamps = ctx['timestamps']

        images_per_camera = []
        for video_key in self.video_keys:
            video_path = build_lerobot_video_path(
                data_root_path,
                info,
                episode_meta,
                episode_index,
                video_key,
            )
            frames = decode_video_frames(
                video_path,
                timestamps,
                tolerance_s=self.tolerance_s,
                backend=self.backend,
            )
            images_per_camera.append(frames.numpy())
        inputs['images'] = np.stack(images_per_camera, axis=1)
        return inputs
