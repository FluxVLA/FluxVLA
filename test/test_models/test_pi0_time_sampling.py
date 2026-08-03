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

import pytest
import torch

from fluxvla.engines.utils.model_utils import sample_beta
from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching


def test_standard_beta_sampler_matches_openpi_torch_path():
    torch.manual_seed(7)
    expected = torch.distributions.Beta(
        torch.tensor(1.5, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32)).sample((4096, ))

    torch.manual_seed(7)
    actual = sample_beta(1.5, 1.0, 4096, 'cpu', sampler='beta')

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_sampler_remains_available_for_reproduction():
    torch.manual_seed(11)
    gamma1 = torch.rand((1024, )).pow(1 / 1.5)
    gamma2 = torch.rand((1024, )).pow(1 / 1.0)
    expected = gamma1 / (gamma1 + gamma2)

    torch.manual_seed(11)
    actual = sample_beta(1.5, 1.0, 1024, 'cpu', sampler='legacy_power_ratio')

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_pi0_sample_time_uses_configured_sampler_and_openpi_affine_map():
    model = PI0FlowMatching.__new__(PI0FlowMatching)
    torch.nn.Module.__init__(model)
    model.time_sampler = 'beta'
    model.time_beta_alpha = 1.5
    model.time_beta_beta = 1.0

    torch.manual_seed(23)
    expected = sample_beta(1.5, 1.0, 512, 'cpu', sampler='beta')
    expected = expected * 0.999 + 0.001

    torch.manual_seed(23)
    actual = model.sample_time(512, 'cpu')

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('sampler', ['power', '', 'gamma_ratio'])
def test_unknown_sampler_is_rejected(sampler):
    with pytest.raises(ValueError, match='Unsupported beta sampler'):
        sample_beta(1.5, 1.0, 1, 'cpu', sampler=sampler)


@pytest.mark.parametrize(('alpha', 'beta'), [(0.0, 1.0), (1.5, 0.0),
                                             (-1.0, 1.0)])
def test_nonpositive_beta_parameters_are_rejected(alpha, beta):
    with pytest.raises(ValueError, match='must be positive'):
        sample_beta(alpha, beta, 1, 'cpu')
