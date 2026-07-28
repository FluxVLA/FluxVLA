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

from functools import partial
from typing import Callable, Iterable

try:
    from torch.distributed.fsdp.wrap import \
        _module_wrap_policy as _torch_module_wrap_policy
    from torch.distributed.fsdp.wrap import _or_policy as _torch_or_policy
    from torch.distributed.fsdp.wrap import \
        transformer_auto_wrap_policy as _torch_transformer_auto_wrap_policy
except ModuleNotFoundError as exc:
    _torch_module_wrap_policy = None
    _torch_or_policy = None
    _torch_transformer_auto_wrap_policy = None
    _FSDP_WRAP_IMPORT_ERROR = exc
else:
    _FSDP_WRAP_IMPORT_ERROR = None


def require_fsdp_wrap_support() -> None:
    """Raise a clear error if this torch build has no FSDP wrap support."""
    if _FSDP_WRAP_IMPORT_ERROR is not None:
        raise RuntimeError(
            'FSDP wrapping policies are unavailable in this torch build'
        ) from _FSDP_WRAP_IMPORT_ERROR


def module_wrap_policy(module_classes: Iterable[type]) -> Callable:
    """Build an FSDP policy that wraps instances of the given classes."""
    require_fsdp_wrap_support()
    assert _torch_module_wrap_policy is not None
    return partial(_torch_module_wrap_policy, module_classes=module_classes)


def or_policy(policies: Iterable[Callable]) -> Callable:
    """Build an FSDP policy that matches any policy in ``policies``."""
    require_fsdp_wrap_support()
    assert _torch_or_policy is not None
    return partial(_torch_or_policy, policies=policies)


def transformer_wrap_policy(transformer_layer_cls: Iterable[type]) -> Callable:
    """Build an FSDP policy for transformer layer classes."""
    require_fsdp_wrap_support()
    assert _torch_transformer_auto_wrap_policy is not None
    return partial(
        _torch_transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
