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
"""ARM RA-BC and AW-BC utilities.

Official implementation of https://arxiv.org/abs/2604.03037
"""

from fluxvla.weighters import (ArmAWBCWeighter, ArmAWBCWeights,
                               ArmRABCWeighter, ArmRABCWeights,
                               resolve_arm_progress_path)
from .progress_reconstruction import (build_cumulative_progress,
                                      extract_last_interval_delta,
                                      run_strided_episode_inference)

__all__ = [
    'ArmAWBCWeighter',
    'ArmAWBCWeights',
    'ArmRABCWeighter',
    'ArmRABCWeights',
    'build_cumulative_progress',
    'extract_last_interval_delta',
    'resolve_arm_progress_path',
    'run_strided_episode_inference',
]
