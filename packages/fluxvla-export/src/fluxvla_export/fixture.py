# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Load reproducible policy inputs against their explicit tensor contract."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ._safetensors import _load_safetensors, _validate_finite, _validate_keys
from .contracts import TensorSpec, _validate_tensor
from .errors import ArtifactError, ContractError

if TYPE_CHECKING:
    import torch


def load_fixture(
    path: str | Path,
    *,
    expected_sha256: str,
    specs: Sequence[TensorSpec],
) -> dict[str, torch.Tensor]:
    """Load finite CPU inputs with exact names and contract shapes/dtypes.

    Dynamic dimensions use TensorSpec's declared min/max profile. The returned
    dictionary follows spec order, independently of safetensors storage order.
    """
    owner = f'Fixture {path}'
    specs = tuple(specs)
    if any(not isinstance(spec, TensorSpec) for spec in specs):
        raise ArtifactError(f'{owner}: specs must contain TensorSpec values.')
    names = tuple(spec.name for spec in specs)
    if any(not isinstance(name, str) for name in names):
        raise ArtifactError(f'{owner}: tensor spec names must be strings.')
    if len(names) != len(set(names)):
        raise ArtifactError(f'{owner}: tensor spec names must be unique.')
    tensors = _load_safetensors(
        path, expected_sha256=expected_sha256, label='Fixture')
    _validate_keys(tensors, names, owner=owner)
    for spec in specs:
        value = tensors[spec.name]
        try:
            _validate_tensor(value, spec, owner=owner)
        except ContractError as error:
            raise ArtifactError(str(error)) from error
        _validate_finite(value, owner=f'{owner}: tensor {spec.name!r}')
    return {name: tensors[name] for name in names}
