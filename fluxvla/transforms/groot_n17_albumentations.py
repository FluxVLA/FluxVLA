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
"""Optional Albumentations helpers for GR00T N1.7 image preprocessing."""

from typing import Optional

try:
    import albumentations as A
    import cv2
except ImportError as exc:
    raise ImportError(
        'GR00T N1.7 Albumentations preprocessing requires the optional '
        'dependencies `albumentations` and `opencv-python`. Install '
        'the FluxVLA requirements or set `use_albumentations=False`.') from exc

import numpy as np
import torch


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
        return ('crop_fraction', )


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
        has_no_padding = (
            pad_top == 0 and pad_bottom == 0 and pad_left == 0
            and pad_right == 0)
        if has_no_padding:
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
            return {
                'pad_top': 0,
                'pad_bottom': 0,
                'pad_left': 0,
                'pad_right': 0
            }
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


def build_image_transformations(
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
    max_size = (
        shortest_image_edge
        if shortest_image_edge is not None else image_target_size[0])
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


def apply_image_transformations(transform, images, replay=None):
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
