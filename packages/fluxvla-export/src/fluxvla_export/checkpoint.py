# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Strict, non-mutating loading of native dense model state."""

from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ._safetensors import (_load_safetensors, _validate_finite, _validate_keys,
                           validate_safetensors)
from .errors import ArtifactError

if TYPE_CHECKING:
    import torch

__all__ = ['load_checkpoint', 'validate_safetensors']


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return CPU tensors matching every native state key, shape, and dtype.

    The caller supplies its complete state_dict, including scalar and empty
    buffers. No prefix guessing, key remapping, casting, unpickling, or
    mutation of that state occurs. Metadata cannot relax these checks.
    """
    import torch

    owner = f'Checkpoint {path}'
    if not isinstance(expected_state, Mapping):
        raise ArtifactError(
            f'{owner}: expected_state must be a tensor mapping.')
    for name, value in expected_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ArtifactError(
                f'{owner}: expected_state must map string keys '
                f'to torch tensors; invalid key={name!r}.')
    tensors = _load_safetensors(
        path, expected_sha256=expected_sha256, label='Checkpoint')
    _validate_keys(tensors, expected_state, owner=owner)
    for name, expected in expected_state.items():
        value = tensors[name]
        if tuple(value.shape) != tuple(expected.shape):
            raise ArtifactError(
                f'{owner}: tensor {name!r} shape {tuple(value.shape)} '
                f'does not match expected {tuple(expected.shape)}.')
        if value.dtype != expected.dtype:
            raise ArtifactError(
                f'{owner}: tensor {name!r} dtype {value.dtype} does not '
                f'match expected {expected.dtype}.')
        _validate_finite(value, owner=f'{owner}: tensor {name!r}')
    return {name: tensors[name] for name in expected_state}
