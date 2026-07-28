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

import transformers
from mmengine.utils import digit_version

transformers_version = digit_version(transformers.__version__)

transformers_supported_ranges = (
    ('4.53.0', '4.54.0'),
    ('5.3.0', '5.3.1'),
)

try:
    import robosuite
except ModuleNotFoundError:
    robosuite = None

assert any(
    transformers_version >= digit_version(minimum_version)
    and transformers_version < digit_version(maximum_version)
    for minimum_version, maximum_version in transformers_supported_ranges), \
    f'Transformers=={transformers.__version__} is used but incompatible. ' \
    f'Please install transformers==4.53.x or transformers==5.3.0.'

if robosuite is not None:
    robosuite_minimum_version = '1.5.0'
    robosuite_maximum_version = '1.5.2'
    robosuite_version = digit_version(robosuite.__version__)

    assert (robosuite_version >= digit_version(robosuite_minimum_version) and
            robosuite_version < digit_version(robosuite_maximum_version)), \
        f'Robosuite=={robosuite.__version__} is used but incompatible. ' \
        f'Please install robosuite>={robosuite_minimum_version}, ' \
        f'<{robosuite_maximum_version}.'

    assert hasattr(robosuite, 'load_controller_config'), \
        'The installed robosuite is missing load_controller_config. ' \
        'Please install the patched robosuite from ' \
        'git+https://github.com/yinchimaoliang/robosuite.git@7264a82.'

from .collators import *  # noqa: E402, F401, F403
from .datasets import *  # noqa: E402, F401, F403
from .engines import *  # noqa: E402, F401, F403
from .optimizers import *  # noqa: E402, F401, F403
from .tokenizers import *  # noqa: E402, F401, F403
from .transforms import *  # noqa: E402, F401, F403


def _is_optional_torch_distributed_error(exc: ModuleNotFoundError) -> bool:
    name = getattr(exc, 'name', '') or ''
    return name.startswith('torch.distributed') or name.startswith(
        'torch._C._distributed_c10d')


try:
    from .models import *  # noqa: E402, F401, F403
except ModuleNotFoundError as exc:
    if not _is_optional_torch_distributed_error(exc):
        raise
