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

import torch
import torch.nn as nn


class LinearProjector(nn.Module):
    """Map vision features to language features with one biased linear layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Project input features of shape ``[..., in_dim]``."""
        return self.projector(input_features)
