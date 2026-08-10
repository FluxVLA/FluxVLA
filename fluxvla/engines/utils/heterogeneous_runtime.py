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

from importlib import import_module
from types import ModuleType
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence


def is_heterogeneous_torch_distributed_error(exc: ImportError) -> bool:
    """Return True for torch builds missing distributed runtime support."""
    name = getattr(exc, 'name', '') or ''
    message = str(exc)
    return (name.startswith('torch.distributed')
            or name.startswith('torch._C._distributed_c10d')
            or 'torch._C._distributed_c10d' in message)


def is_heterogeneous_import_error(
        exc: ImportError,
        runtime_missing_names: Iterable[str] = (),
) -> bool:
    """Return True if ``exc`` is tolerated in a heterogeneous runtime."""
    if is_heterogeneous_torch_distributed_error(exc):
        return True

    name = getattr(exc, 'name', '') or ''
    if not name:
        return False
    return any(name == runtime_name or name.startswith(f'{runtime_name}.')
               for runtime_name in runtime_missing_names)


def _public_names(module: ModuleType) -> Sequence[str]:
    exported = getattr(module, '__all__', None)
    if exported is not None:
        return tuple(exported)
    return tuple(name for name in vars(module) if not name.startswith('_'))


def import_heterogeneous_runtime_symbols(
        package: str,
        namespace: MutableMapping[str, object],
        module_symbols: Mapping[str, Optional[Iterable[str]]],
        *,
        runtime_missing_names: Iterable[str] = (),
) -> None:
    """Import modules and expose symbols across heterogeneous runtimes.

    Args:
        package: Package name used for relative imports, usually ``__name__``.
        namespace: Caller globals where imported symbols are exported.
        module_symbols: Mapping from relative module name to exported symbols.
            Use ``None`` to export public names, matching ``from module import *``.
        runtime_missing_names: Import names that may be absent in slim or
            device-specific runtime environments. Other ImportError instances
            are re-raised.
    """
    for module_name, symbols in module_symbols.items():
        try:
            module = import_module(f'.{module_name}', package=package)
        except ImportError as exc:
            if is_heterogeneous_import_error(exc, runtime_missing_names):
                continue
            raise

        selected_symbols = _public_names(
            module) if symbols is None else symbols
        for symbol in selected_symbols:
            namespace[symbol] = getattr(module, symbol)
