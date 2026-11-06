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
    storage_dtype: str | None = None,
) -> dict[str, torch.Tensor]:
    """Validate native state, with optional explicit BF16-to-FP32 widening.

    With storage_dtype=None every dtype must match exactly, as before.
    Declared float32/bfloat16 storage requires all floating tensors to have
    that on-disk dtype and a float32 target. Only declared bfloat16 storage is
    widened, after all keys, shapes, dtypes, values and the hash pass. Integer
    and bool buffers always require exact dtypes. No key remapping, unpickling,
    file writes or expected-state mutation occurs.
    """
    import torch

    owner = f'Checkpoint {path}'
    if storage_dtype is not None and (type(storage_dtype) is not str
                                      or storage_dtype
                                      not in ('float32', 'bfloat16')):
        raise ArtifactError(
            f'{owner}: storage_dtype must be None, float32 or bfloat16; '
            f'got {storage_dtype!r}.')
    declared_dtype = (
        getattr(torch, storage_dtype) if storage_dtype is not None else None)
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
        required_dtype = expected.dtype
        if declared_dtype is not None and expected.is_floating_point():
            if expected.dtype != torch.float32:
                raise ArtifactError(
                    f'{owner}: tensor {name!r} target dtype {expected.dtype} '
                    f'must be float32 for declared storage {storage_dtype}.')
            required_dtype = declared_dtype
        if value.dtype != required_dtype:
            raise ArtifactError(
                f'{owner}: tensor {name!r} dtype {value.dtype} does not '
                f'match expected storage dtype {required_dtype}.')
        _validate_finite(value, owner=f'{owner}: tensor {name!r}')
    # BF16 values are represented exactly in FP32; no precision is recovered
    # or approximated. Never cast before validating the complete source state.
    return {
        name:
        (tensors[name].to(dtype=torch.float32) if storage_dtype == 'bfloat16'
         and tensors[name].is_floating_point() else tensors[name])
        for name in expected_state
    }
