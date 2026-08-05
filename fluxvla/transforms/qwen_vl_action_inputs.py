# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Qwen-VL action-prediction input transforms."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as transforms
from transformers import Qwen3VLProcessor

from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.hf_hub import resolve_hf_local_path
from .modality_state_action import load_groot_n17_metadata


N17_EMBODIMENT_ALIASES = {
    'ROBOCASA_GR1_TABLETOP': 'robocasa_gr1_tabletop',
    'robocasa_gr1_tabletop': 'robocasa_gr1_tabletop',
    'gr1_unified': 'robocasa_gr1_tabletop',
    'LIBERO_PANDA': 'libero_sim',
    'libero_sim': 'libero_sim',
}


def resolve_qwen_vl_model_path(model_name_or_path: str) -> str:
    """Prefer local checkpoint paths for Qwen-VL processor assets.

    Official N1.7 metadata may store a Hugging Face repo id such as
    ``nvidia/Cosmos-Reason2-2B``. FluxVLA training jobs usually mount that
    asset under ``checkpoints/nvidia/Cosmos-Reason2-2B``. Resolve that local
    path before falling back to the original string.
    """
    value = str(model_name_or_path)
    resolved = resolve_hf_local_path(value)
    if resolved != value:
        return resolved

    raw_path = Path(value).expanduser()
    if raw_path.exists():
        return resolve_hf_local_path(str(raw_path))

    if raw_path.is_absolute():
        return value

    repo_root = Path(__file__).resolve().parents[2]
    for base in (Path.cwd(), repo_root):
        for candidate in (base / raw_path, base / 'checkpoints' / raw_path):
            if candidate.exists():
                return resolve_hf_local_path(str(candidate))
    return value


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


class FractionalCenterCrop(A.DualTransform):

    def __init__(self,
                 crop_fraction: float = 0.9,
                 p: float = 1.0,
                 always_apply: Optional[bool] = None):
        super().__init__(p=p, always_apply=always_apply)
        self.crop_fraction = crop_fraction

    def apply(self, img: np.ndarray, crop_coords, **params) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def get_params_dependent_on_data(self, params, data) -> dict:
        height, width = params['shape'][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2
        return {
            'crop_coords':
            (x_min, y_min, x_min + crop_width, y_min + crop_height)
        }

    def get_transform_init_args_names(self):
        return ('crop_fraction',)


class FractionalRandomCrop(FractionalCenterCrop):

    def get_params_dependent_on_data(self, params, data) -> dict:
        height, width = params['shape'][:2]
        crop_height = max(1, int(height * self.crop_fraction))
        crop_width = max(1, int(width * self.crop_fraction))
        max_y = height - crop_height
        max_x = width - crop_width
        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        return {
            'crop_coords':
            (x_min, y_min, x_min + crop_width, y_min + crop_height)
        }


class LetterBoxPad(A.DualTransform):

    def __init__(self, p: float = 1.0, always_apply: Optional[bool] = None):
        super().__init__(p=p, always_apply=always_apply)

    def apply(self,
              img: np.ndarray,
              pad_top: int = 0,
              pad_bottom: int = 0,
              pad_left: int = 0,
              pad_right: int = 0,
              **params) -> np.ndarray:
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return img
        return cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0)

    def get_params_dependent_on_data(self, params, data) -> dict:
        h, w = params['shape'][:2]
        if h == w:
            return {'pad_top': 0, 'pad_bottom': 0, 'pad_left': 0, 'pad_right': 0}
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        return {
            'pad_top': pad_h // 2,
            'pad_bottom': pad_h - pad_h // 2,
            'pad_left': pad_w // 2,
            'pad_right': pad_w - pad_w // 2,
        }

    def get_transform_init_args_names(self):
        return ()


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
    fraction = crop_fraction
    if fraction is None:
        fraction = image_crop_size[0] / image_target_size[0]
    max_size = shortest_image_edge if shortest_image_edge is not None else image_target_size[0]
    train_ops = [
        LetterBoxPad(),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=fraction),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_ops.append(A.Rotate(limit=random_rotation_angle, p=1.0))
    if color_jitter_params is not None:
        train_ops.append(A.ColorJitter(**color_jitter_params, p=1.0))
    eval_transform = A.Compose([
        LetterBoxPad(),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalCenterCrop(crop_fraction=fraction),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ])
    return A.ReplayCompose(train_ops, p=1.0), eval_transform


def apply_n17_albumentations(transform, images, replay=None):
    tensors = []
    current_replay = replay
    has_replay = hasattr(transform, 'replay')
    for img in images:
        img_array = np.array(img)
        if has_replay:
            if current_replay is None:
                augmented = transform(image=img_array)
                current_replay = augmented['replay']
            else:
                augmented = transform.replay(
                    image=img_array, saved_augmentations=current_replay)
        else:
            augmented = transform(image=img_array)
        img_array = augmented['image']
        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        tensors.append(torch.from_numpy(img_array).permute(2, 0, 1))
    return tensors, current_replay


def split_images_by_view(value: Any,
                         image_keys: list[str]) -> Dict[str, list[Image.Image]]:
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
        key: [
            _to_pil_rgb(img)
            for img in images[i * per_key:(i + 1) * per_key]
        ]
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
    return (
        torch.stack([temporal_stacked_images[view] for view in image_keys],
                    dim=1).flatten(0, 1).numpy())


def build_qwen_vl_chat_content(processor: Qwen3VLProcessor,
                               images_chw: np.ndarray,
                               language: str) -> Dict[str, Any]:
    pil_images = [
        Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images_chw
    ]
    conversation = [{
        'role':
        'user',
        'content': [
            *[{
                'type': 'image',
                'image': img
            } for img in pil_images],
            {
                'type': 'text',
                'text': language
            },
        ],
    }]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False)
    return {
        'text': text,
        'images': pil_images,
        'conversation': conversation,
    }


@TRANSFORMS.register_module()
class BuildQwenVLChatImageContent:
    """Build Qwen-VL chat text and N1.7-augmented CHW images."""

    def __init__(
        self,
        processor_path: str,
        embodiment_tag: str = 'ROBOCASA_GR1_TABLETOP',
        image_key: str = 'images',
        text_key: str = 'task_description',
        output_image_key: str = 'images',
        output_text_key: str = 'text',
        vlm_content_key: Optional[str] = 'vlm_content',
        train_mode: bool = True,
        model_name: Optional[str] = None,
        transformers_loading_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.processor_path = processor_path
        self.embodiment_key = N17_EMBODIMENT_ALIASES.get(
            embodiment_tag, N17_EMBODIMENT_ALIASES.get(
                str(embodiment_tag).lower(), str(embodiment_tag).lower()))
        self.image_key = image_key
        self.text_key = text_key
        self.output_image_key = output_image_key
        self.output_text_key = output_text_key
        self.vlm_content_key = vlm_content_key
        self.training = train_mode

        input_processor_kwargs = dict(processor_kwargs or {})
        explicit_loading_kwargs = transformers_loading_kwargs
        if explicit_loading_kwargs is None:
            explicit_loading_kwargs = input_processor_kwargs.get(
                'transformers_loading_kwargs')
        processor_kwargs = load_groot_n17_metadata(
            processor_path, **input_processor_kwargs)
        self.modality_config = processor_kwargs['modality_configs'][
            self.embodiment_key]
        self.formalize_language = processor_kwargs.get(
            'formalize_language', True)
        self.use_albumentations = processor_kwargs.get(
            'use_albumentations', False)
        self.model_name = (
            model_name or processor_kwargs.get(
                'model_name', 'nvidia/Cosmos-Reason2-2B'))
        self.model_name = resolve_qwen_vl_model_path(self.model_name)
        self.processor = Qwen3VLProcessor.from_pretrained(
            self.model_name, **dict(explicit_loading_kwargs or {}))
        self.processor.tokenizer.padding_side = 'left'

        image_target_size = processor_kwargs.get('image_target_size')
        image_crop_size = processor_kwargs.get('image_crop_size')
        random_rotation_angle = processor_kwargs.get('random_rotation_angle')
        color_jitter_params = processor_kwargs.get('color_jitter_params')
        shortest_image_edge = processor_kwargs.get('shortest_image_edge', 256)
        crop_fraction = processor_kwargs.get('crop_fraction', 0.95)
        if self.use_albumentations:
            self.train_image_transform, self.eval_image_transform = (
                build_n17_image_transformations_albumentations(
                    image_target_size,
                    image_crop_size,
                    random_rotation_angle,
                    color_jitter_params,
                    shortest_image_edge,
                    crop_fraction))
        else:
            self.train_image_transform, self.eval_image_transform = (
                build_n17_image_transformations(
                    image_target_size, image_crop_size,
                    random_rotation_angle, color_jitter_params))

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_keys = self.modality_config['video']['modality_keys']
        images_by_view = split_images_by_view(sample[self.image_key], image_keys)

        language = sample.get(self.text_key, '')
        if self.formalize_language:
            language = re.sub(r'[^\w\s]', '', str(language).lower())

        image_transform = (
            self.train_image_transform if self.training else
            self.eval_image_transform)
        images_chw = stack_n17_vlm_images(
            image_keys,
            images_by_view,
            image_transform,
            use_albumentations=self.use_albumentations)
        vlm_content = build_qwen_vl_chat_content(
            self.processor, images_chw, language)

        outputs = dict(sample)
        outputs[self.output_image_key] = images_chw
        outputs[self.output_text_key] = vlm_content['text']
        if self.vlm_content_key is not None:
            outputs[self.vlm_content_key] = vlm_content
        return outputs
