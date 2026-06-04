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
"""Registry-backed RA-BC / AW-BC sample weighters."""

from fluxvla.engines import WEIGHTERS
from .arm_rabc import ArmAWBCWeighter, ArmRABCWeighter
from .basic import ConstantWeighter, ProgressDeltaWeighter, SuccessRateWeighter
from .sarm_rabc import SarmRABCWeighter
from .utils import (as_scalar, resolve_arm_progress_path,
                    resolve_rabc_progress_path, resolve_sample_index)

# Backward-compatible config type names used in existing docs and configs.
WEIGHTERS.register_module(
    name='ARMProgressWeighter', module=ArmRABCWeighter, force=True)
WEIGHTERS.register_module(
    name='ARMProgressAWBCWeighter', module=ArmAWBCWeighter, force=True)
WEIGHTERS.register_module(
    name='SARMProgressWeighter', module=SarmRABCWeighter, force=True)

# Deprecated class-name aliases kept for external imports.
ArmRABCWeights = ArmRABCWeighter
ArmAWBCWeights = ArmAWBCWeighter
SarmRABCWeights = SarmRABCWeighter

__all__ = [
    'ArmAWBCWeighter',
    'ArmAWBCWeights',
    'ArmRABCWeighter',
    'ArmRABCWeights',
    'ConstantWeighter',
    'ProgressDeltaWeighter',
    'SarmRABCWeighter',
    'SarmRABCWeights',
    'SuccessRateWeighter',
    'as_scalar',
    'resolve_arm_progress_path',
    'resolve_rabc_progress_path',
    'resolve_sample_index',
]
