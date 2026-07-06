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
from typing import Dict, List

import av
import numpy as np
import torch
from PIL import Image

from fluxvla.datasets.utils.video_decode import (build_lerobot_video_path,
                                                 decode_video_frames)
from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.eval_utils import crop_and_resize
from .transform_images import (_resize_hwc_lanczos3_numpy,
                               _resize_hwc_lanczos3_tensorflow)
from .utils import pad_to_dim, parse_image


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
            (default) it resolves to ``'torchcodec'`` if the package is
            importable, otherwise ``'pyav'``. The ``'torchcodec'`` path
            decodes by frame index (``round(ts * average_fps)``); on any
            decode error it transparently falls back to the ``'pyav'``
            torchvision path.
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


@TRANSFORMS.register_module()
class ConcatLeftAlign:
    """Concatenate multi-field action/state dicts and left-align pad.

    Operates on ``data['action']`` and ``data['state']`` dicts with
    ``{field_key: [T, d_field]}`` values; produces ``[T, target_dim]`` tensors
    plus ``action_dim_is_pad`` / ``state_dim_is_pad`` masks
    (``True`` where padded).

    Args:
        shape_meta (Dict): ``{'action': [{'key', 'shape'}], 'state': [...]}``.
        action_target_dim (int | None): pad action to this dim.
        state_target_dim (int | None): pad state to this dim.
    """

    def __init__(self,
                 shape_meta: Dict,
                 action_target_dim: int = None,
                 state_target_dim: int = None,
                 *args,
                 **kwargs):
        self.action_meta = shape_meta['action']
        self.state_meta = shape_meta['state']
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta) -> torch.Tensor:
        x = torch.cat([x[m['key']] for m in meta], dim=-1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _pad(x: torch.Tensor, dim):
        if dim is None:
            dim = x.shape[-1]
        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = torch.nn.functional.pad(x, (0, pad_dim))
        mask = torch.zeros_like(x[0]).bool()
        mask = torch.nn.functional.pad(mask, (0, pad_dim), value=True)
        return x_padded, mask

    def __call__(self, data: Dict) -> Dict:
        if 'action' in data:
            data['action'] = self._concat(data['action'], self.action_meta)
            data['action'], data['action_dim_is_pad'] = self._pad(
                data['action'], self.action_target_dim)
        data['state'] = self._concat(data['state'], self.state_meta)
        data['state'], data['state_dim_is_pad'] = self._pad(
            data['state'], self.state_target_dim)
        return data
