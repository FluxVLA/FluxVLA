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
"""Qwen-VL action-prediction input transforms."""

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

from fluxvla.engines import TRANSFORMS
from .modality_state_action import resolve_groot_n17_metadata


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    array = _to_numpy(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array).convert('RGB')


class LetterBoxTransform:

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        *leading_dims, c, h, w = img.shape
        if h == w:
            return img
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if leading_dims:
            batch_size = torch.tensor(leading_dims).prod().item()
            reshaped = img.reshape(batch_size, c, h, w)
            padded = transforms.functional.pad(
                reshaped,
                padding=[pad_left, pad_top, pad_right, pad_bottom],
                fill=0)
            return padded.reshape(leading_dims + [c, max_dim, max_dim])
        return transforms.functional.pad(
            img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)


def build_n17_image_transformations(image_target_size, image_crop_size,
                                    random_rotation_angle,
                                    color_jitter_params):
    train_ops = [
        transforms.ToImage(),
        LetterBoxTransform(),
        transforms.Resize(size=image_target_size),
        transforms.RandomCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_ops.append(
            transforms.RandomRotation(
                degrees=[-random_rotation_angle, random_rotation_angle]))
    if color_jitter_params is not None:
        train_ops.append(transforms.ColorJitter(**color_jitter_params))
    eval_ops = [
        transforms.ToImage(),
        LetterBoxTransform(),
        transforms.Resize(size=image_target_size),
        transforms.CenterCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_n17_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
):
    # Albumentations is optional for the rest of FluxVLA. Import its N1.7
    # helpers only when this augmentation path is explicitly enabled.
    from .groot_n17_albumentations import build_image_transformations

    return build_image_transformations(
        image_target_size=image_target_size,
        image_crop_size=image_crop_size,
        random_rotation_angle=random_rotation_angle,
        color_jitter_params=color_jitter_params,
        shortest_image_edge=shortest_image_edge,
        crop_fraction=crop_fraction,
    )


def apply_n17_albumentations(transform, images, replay=None):
    from .groot_n17_albumentations import apply_image_transformations

    return apply_image_transformations(transform, images, replay=replay)


def split_images_by_view(
        value: Any, image_keys: list[str]) -> Dict[str, list[Image.Image]]:
    if isinstance(value, dict):
        return {
            key: [_to_pil_rgb(img) for img in images]
            for key, images in value.items()
        }

    images = list(value)
    if len(image_keys) == 1:
        return {image_keys[0]: [_to_pil_rgb(img) for img in images]}

    if len(images) % len(image_keys) != 0:
        raise ValueError(
            f'Cannot split {len(images)} images across {len(image_keys)} '
            f'video keys: {image_keys}')
    per_key = len(images) // len(image_keys)
    return {
        key:
        [_to_pil_rgb(img) for img in images[i * per_key:(i + 1) * per_key]]
        for i, key in enumerate(image_keys)
    }


def stack_n17_vlm_images(image_keys: list[str], images: Dict[str, Any],
                         image_transform,
                         use_albumentations: bool) -> np.ndarray:
    temporal_stacked_images = {}
    if use_albumentations:
        replay = None
        for view in image_keys:
            transformed, replay = apply_n17_albumentations(
                image_transform, images[view], replay)
            temporal_stacked_images[view] = torch.stack(transformed)
    else:
        for view in image_keys:
            temporal_stacked_images[view] = torch.stack(
                [image_transform(img) for img in images[view]])
    return (torch.stack([temporal_stacked_images[view] for view in image_keys],
                        dim=1).flatten(0, 1).numpy())


@TRANSFORMS.register_module()
class GrootN17ImageAugmentation:
    """Apply the native GR00T N1.7 multi-view image augmentation.

    The replay-based albumentations path intentionally shares one sampled
    augmentation across all views and timesteps in a sample.  Prompt building
    and tokenization are kept out of this transform so they can be composed
    from the generic FluxVLA prompt transforms.
    """

    def __init__(
        self,
        processor_path: Optional[str] = None,
        embodiment_tag: str = 'LIBERO_PANDA',
        image_key: str = 'images',
        output_image_key: str = 'images',
        train_mode: bool = True,
        processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.processor_path = processor_path
        self.image_key = image_key
        self.output_image_key = output_image_key
        self.training = train_mode

        input_processor_kwargs = dict(processor_kwargs or {})
        metadata = resolve_groot_n17_metadata(
            processor_path,
            embodiment_tag=embodiment_tag,
            **input_processor_kwargs)
        processor_kwargs = metadata['processor_kwargs']
        self.embodiment_key = metadata['embodiment_key']
        self.modality_config = metadata['modality_config']
        self.use_albumentations = processor_kwargs.get('use_albumentations',
                                                       False)

        image_target_size = processor_kwargs.get('image_target_size')
        image_crop_size = processor_kwargs.get('image_crop_size')
        random_rotation_angle = processor_kwargs.get('random_rotation_angle')
        color_jitter_params = processor_kwargs.get('color_jitter_params')
        shortest_image_edge = processor_kwargs.get('shortest_image_edge', 256)
        crop_fraction = processor_kwargs.get('crop_fraction', 0.95)
        if self.use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_n17_image_transformations_albumentations(
                    image_target_size, image_crop_size, random_rotation_angle,
                    color_jitter_params, shortest_image_edge, crop_fraction))
        else:
            self.train_image_transform, self.eval_image_transform = (
                build_n17_image_transformations(image_target_size,
                                                image_crop_size,
                                                random_rotation_angle,
                                                color_jitter_params))

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_keys = self.modality_config['video']['modality_keys']
        images_by_view = split_images_by_view(sample[self.image_key],
                                              image_keys)
        image_transform = (
            self.train_image_transform
            if self.training else self.eval_image_transform)
        images_chw = stack_n17_vlm_images(
            image_keys,
            images_by_view,
            image_transform,
            use_albumentations=self.use_albumentations)

        outputs = dict(sample)
        outputs[self.output_image_key] = images_chw
        return outputs
