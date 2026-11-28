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
"""Sensor-only change diagnostics, deliberately NOT a success verifier."""

import math

import numpy as np
from PIL import Image

from .memory import positive_integer


class VisualChangeObserver:
    """Compare small camera thumbnails after acknowledged execution.

    A moving arm/camera is not task progress, and an unchanged image is not
    proof of failure. This signal only advises scheduling. No predicates,
    rewards, segmentation labels, simulator poses or success flags are read.
    """

    def __init__(self, image_keys, threshold=0.01, thumbnail_size=32):
        if (not isinstance(image_keys, (tuple, list)) or not image_keys
                or any(not isinstance(k, str) or not k for k in image_keys)
                or len(set(image_keys)) != len(image_keys)):
            raise ValueError('image_keys must name distinct RGB cameras')
        if (isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 < threshold < 1):
            raise ValueError('visual threshold must be between zero and one')
        self.image_keys = tuple(image_keys)
        self.threshold = threshold
        self.size = positive_integer('thumbnail_size', thumbnail_size)
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_id = None
        self.images = {}

    def observe(self, observation, observation_id, *, executed):
        current, self.images = {}, {}
        for key in self.image_keys:
            value = observation.get(key)
            if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
                    or value.ndim != 3 or value.shape[-1] != 3
                    or not value.size):
                raise ValueError(f'{key} must be an RGB uint8 HWC image')
            image = Image.fromarray(value)
            thumb = image.resize((self.size, self.size))
            current[key] = np.asarray(thumb).astype(np.float32) / 255.0
            image.thumbnail((256, 256))
            self.images[key] = np.asarray(image).copy()
        differences = ({
            key: float(np.abs(current[key] - self.previous[key]).mean())
            for key in self.image_keys
        } if self.previous is not None and executed else {})
        state = ('changed' if any(v >= self.threshold
                                  for v in differences.values()) else
                 'no_observed_change') if differences else 'unknown'
        result = dict(
            state=state,
            from_observation_id=self.previous_id,
            to_observation_id=observation_id,
            observed_after_execution=executed,
            camera_mean_absolute_change=differences,
            semantic_progress='unknown',
            completion='unknown')
        self.previous, self.previous_id = current, observation_id
        return result
