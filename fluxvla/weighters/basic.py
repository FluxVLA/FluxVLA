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
"""Generic BC sample weighters."""

from __future__ import annotations
from typing import Any, Mapping

import numpy as np

from fluxvla.engines import WEIGHTERS
from .utils import as_scalar


@WEIGHTERS.register_module()
class ConstantWeighter:
    """Return a fixed weight for every sample."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    def __call__(self, sample: Mapping[str, Any]) -> float:
        return self.weight


@WEIGHTERS.register_module()
class SuccessRateWeighter:
    """Map a success flag in the sample to a BC weight."""

    def __init__(self,
                 success_key: str = 'success',
                 positive_weight: float = 1.0,
                 negative_weight: float = 0.0,
                 fallback_weight: float = 1.0) -> None:
        self.success_key = success_key
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.fallback_weight = float(fallback_weight)

    def __call__(self, sample: Mapping[str, Any]) -> float:
        success = as_scalar(sample.get(self.success_key))
        if success is None:
            return self.fallback_weight
        return self.positive_weight if bool(success) else self.negative_weight


@WEIGHTERS.register_module()
class ProgressDeltaWeighter:
    """Weight samples from progress deltas already present in the sample."""

    def __init__(self,
                 progress_key: str = 'progress',
                 future_progress_key: str = 'future_progress',
                 kappa: float = 0.01,
                 fallback_weight: float = 1.0) -> None:
        self.progress_key = progress_key
        self.future_progress_key = future_progress_key
        self.kappa = float(kappa)
        self.fallback_weight = float(fallback_weight)

    def __call__(self, sample: Mapping[str, Any]) -> float:
        progress = as_scalar(sample.get(self.progress_key))
        future_progress = as_scalar(sample.get(self.future_progress_key))
        if progress is None or future_progress is None:
            return self.fallback_weight
        delta = float(future_progress) - float(progress)
        if not np.isfinite(delta):
            return self.fallback_weight
        if delta > self.kappa:
            return 1.0
        if delta < 0.0:
            return 0.0
        return delta / max(self.kappa, 1e-8)
