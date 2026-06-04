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
"""Helpers for CLIP feature extraction across transformers versions."""

from __future__ import annotations

import torch


def clip_feature_tensor(features) -> torch.Tensor:
    """Return the projected CLIP embedding tensor from model outputs.

    ``transformers>=5.3`` returns ``BaseModelOutputWithPooling`` from
    ``get_image_features`` / ``get_text_features`` instead of a bare tensor.
    Older versions return ``torch.Tensor`` directly.
    """
    if isinstance(features, torch.Tensor):
        return features
    if (hasattr(features, 'pooler_output')
            and features.pooler_output is not None):
        return features.pooler_output
    if isinstance(features, (tuple, list)) and features:
        return clip_feature_tensor(features[0])
    raise TypeError(
        f'Unsupported CLIP feature output type: {type(features)!r}')
