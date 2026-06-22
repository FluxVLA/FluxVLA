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

from typing import Tuple

import cv2
import numpy as np


def _sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=np.float64)
    nonzero = x != 0
    out[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
    return out


def _lanczos3_kernel(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x < 3.0, _sinc(x) * _sinc(x / 3.0), 0.0)


def _lanczos3_weights(in_size: int,
                      out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    scale = out_size / in_size
    inv_scale = in_size / out_size
    sample_positions = (np.arange(out_size, dtype=np.float64) +
                        0.5) * inv_scale - 0.5

    kernel_scale = scale if scale < 1.0 else 1.0
    radius = 3.0 / kernel_scale
    span = int(np.ceil(radius) * 2 + 1)
    left = np.floor(sample_positions - radius).astype(np.int64)
    indices = left[:, None] + np.arange(span, dtype=np.int64)[None, :]

    weights = _lanczos3_kernel(
        (indices - sample_positions[:, None]) * kernel_scale)
    weights = np.where((indices >= 0) & (indices < in_size), weights, 0.0)
    weight_sums = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights,
        weight_sums,
        out=np.zeros_like(weights),
        where=np.abs(weight_sums) > 1e-12)

    return np.clip(indices, 0, in_size - 1), weights


def resize_hwc_lanczos3_numpy(image: np.ndarray, height: int,
                              width: int) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f'Expected HWC image, got shape {image.shape}')

    image = image.astype(np.float64, copy=False)
    x_indices, x_weights = _lanczos3_weights(image.shape[1], width)
    resized_x = (image[:, x_indices, :] *
                 x_weights[None, :, :, None]).sum(axis=2)

    y_indices, y_weights = _lanczos3_weights(image.shape[0], height)
    resized = (resized_x[y_indices, :, :] *
               y_weights[:, :, None, None]).sum(axis=1)

    return np.clip(np.round(resized), 0, 255).astype(np.uint8)


def jpeg_roundtrip_numpy(image: np.ndarray) -> np.ndarray:
    encoded_ok, encoded = cv2.imencode('.jpg',
                                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not encoded_ok:
        raise ValueError('Failed to encode image as JPEG.')
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError('Failed to decode JPEG image.')
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def center_crop_and_resize_numpy(
        image: np.ndarray,
        crop_scale: float,
        output_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Center-crop by area scale, matching TensorFlow crop-resize semantics."""
    if image.ndim != 3:
        raise ValueError(f'Expected HWC image, got shape {image.shape}')

    orig_dtype = image.dtype
    if np.issubdtype(orig_dtype, np.integer):
        image_float = image.astype(np.float32) / np.iinfo(orig_dtype).max
    else:
        image_float = image.astype(np.float32, copy=False)

    height, width = image_float.shape[:2]
    out_height, out_width = output_size
    side_scale = float(np.clip(np.sqrt(crop_scale), 0.0, 1.0))
    top = (1.0 - side_scale) / 2.0
    left = (1.0 - side_scale) / 2.0
    bottom = top + side_scale
    right = left + side_scale

    if out_height > 1:
        ys = (
            top * (height - 1) + np.arange(out_height, dtype=np.float32) *
            (bottom - top) * (height - 1) / (out_height - 1))
    else:
        ys = np.array([0.5 * (top + bottom) * (height - 1)], dtype=np.float32)
    if out_width > 1:
        xs = (
            left * (width - 1) + np.arange(out_width, dtype=np.float32) *
            (right - left) * (width - 1) / (out_width - 1))
    else:
        xs = np.array([0.5 * (left + right) * (width - 1)], dtype=np.float32)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y_lerp = (ys - y0).astype(np.float32)
    x_lerp = (xs - x0).astype(np.float32)

    top_left = image_float[y0[:, None], x0[None, :], :]
    top_right = image_float[y0[:, None], x1[None, :], :]
    bottom_left = image_float[y1[:, None], x0[None, :], :]
    bottom_right = image_float[y1[:, None], x1[None, :], :]

    top_values = top_left + (top_right - top_left) * x_lerp[None, :, None]
    bottom_values = bottom_left + (bottom_right -
                                   bottom_left) * x_lerp[None, :, None]
    resized = top_values + (bottom_values - top_values) * y_lerp[:, None, None]

    if orig_dtype == np.uint8:
        resized = np.clip(resized, 0.0, 1.0)
        return (resized * 255.5).astype(np.uint8)
    return resized.astype(orig_dtype, copy=False)


def crop_and_resize_numpy(
        image: np.ndarray,
        crop_scale: float,
        output_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Batch-aware center crop and resize for HWC or NHWC arrays."""
    image = np.asarray(image)
    if image.ndim == 3:
        return center_crop_and_resize_numpy(image, crop_scale, output_size)
    if image.ndim != 4:
        raise ValueError(
            f'Expected HWC or NHWC image, got shape {image.shape}')
    return np.stack([
        center_crop_and_resize_numpy(single_image, crop_scale, output_size)
        for single_image in image
    ],
                    axis=0)
