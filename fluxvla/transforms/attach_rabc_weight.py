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
"""Transforms for attaching RA-BC sample weights."""

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np

import fluxvla.weighters  # noqa: F401 — register WEIGHTERS modules
from fluxvla.engines import TRANSFORMS, build_weighter_from_cfg


def _build_weighter(config: Optional[Dict[str, Any]]):
    if config is None:
        return build_weighter_from_cfg(dict(type='ConstantWeighter'))
    if callable(config):
        return config
    if not isinstance(config, dict):
        raise TypeError(
            f'weighter must be a dict or callable, got {type(config)}')
    return build_weighter_from_cfg(config)


@TRANSFORMS.register_module()
class AttachRABCWeight:
    """Attach one RA-BC sample weight to each training sample.

    Put this transform before transforms that rebuild the sample dictionary,
    such as ``ProcessParquetInputs``. Those transforms can then carry
    ``sample_weight`` through to the collator.
    """

    def __init__(self,
                 weighter: Optional[Dict[str, Any]] = None,
                 output_key: str = 'sample_weight',
                 default_weight: float = 1.0,
                 drop_index: bool = False) -> None:
        self.weighter = _build_weighter(weighter)
        self.output_key = output_key
        self.default_weight = float(default_weight)
        self.drop_index = drop_index

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        weight = self.weighter(data)
        if weight is None:
            weight = self.default_weight

        data[self.output_key] = np.asarray(weight, dtype=np.float32)
        if self.drop_index:
            data.pop('index', None)
        return data
