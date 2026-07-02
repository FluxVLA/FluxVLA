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

from __future__ import annotations

import torch
import torch.nn as nn

from fluxvla.engines import PROJECTORS


@PROJECTORS.register_module()
class DomainAwareLinear(nn.Module):
    """Linear layer with domain-conditioned parameters.

    This follows the Cosmos3 VFM
    ``domain_aware_linear.DomainAwareLinear`` parameterization.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_domains: int = 50) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains

        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(
        self,
        x: torch.Tensor,
        domain_id: torch.LongTensor,
    ) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        if x.shape[0] != domain_id.shape[0]:
            raise ValueError(
                'Cosmos3 DomainAwareLinear expects one domain_id per input '
                f'row: x.shape[0]={x.shape[0]}, '
                f'domain_id={domain_id.shape[0]}.')
        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
            raise ValueError(
                f'Cosmos3 domain_id must be in [0, {self.num_domains}), '
                f'got {domain_id.tolist()}.')
        batch_size = domain_id.shape[0]
        weight = self.fc(domain_id).view(batch_size, self.input_size,
                                         self.output_size)
        bias = self.bias(domain_id).view(batch_size, self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        if x.ndim == 3:
            return torch.bmm(x, weight) + bias.unsqueeze(1)
        raise ValueError('Cosmos3 DomainAwareLinear expects rank-2 or rank-3 '
                         f'input, got shape {tuple(x.shape)}.')
