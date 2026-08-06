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

from importlib.machinery import PathFinder
from pathlib import Path
from typing import Tuple


_CUDA_EXTENSION_PACKAGES = (
    ('fluxvla.ops.cuda.gemma_rotary_embedding.gemma_rotary_embedding_ext',
     'gemma_rotary_embedding'),
    ('fluxvla.ops.cuda.rotary_pos_embedding.rotary_pos_embedding_ext',
     'rotary_pos_embedding'),
    ('fluxvla.ops.cuda.matmul_bias.matmul_bias_ext', 'matmul_bias'),
)


def missing_cuda_extensions() -> Tuple[str, ...]:
    cuda_ops_dir = Path(__file__).resolve().parents[1] / 'ops' / 'cuda'
    return tuple(
        module_name for module_name, package_name in _CUDA_EXTENSION_PACKAGES
        if PathFinder.find_spec(module_name,
                                [str(cuda_ops_dir / package_name)]) is None)


def cuda_extensions_available() -> bool:
    return not missing_cuda_extensions()