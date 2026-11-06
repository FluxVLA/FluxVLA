# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Safe artifact inspection without importing optional tensor dependencies."""

from __future__ import annotations
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .errors import ArtifactError

if TYPE_CHECKING:
    import torch

# Match the safetensors format's header limit. Tensor data is never JSON.
_MAX_HEADER_SIZE = 100_000_000
_MAX_U64 = 2**64 - 1
_CHUNK_SIZE = 1024 * 1024
_DTYPE_BITS = {
    'BOOL': 8,
    'U8': 8,
    'I8': 8,
    'F8_E4M3': 8,
    'F8_E5M2': 8,
    'F8_E8M0': 8,
    'I16': 16,
    'U16': 16,
    'F16': 16,
    'BF16': 16,
    'I32': 32,
    'U32': 32,
    'F32': 32,
    'F64': 64,
    'I64': 64,
    'U64': 64,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        # Python's JSON decoder accepts escaped lone surrogates; the safe
        # format requires Unicode strings that can actually encode as UTF-8.
        key.encode('utf-8')
        if isinstance(value, str):
            value.encode('utf-8')
        if key in result:
            raise ValueError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f'non-finite JSON constant {value!r}')


def _unsigned_integer(value: str) -> int:
    # JSON -0 becomes Python int(0), but is not a valid safetensors usize.
    if value.startswith('-'):
        raise ValueError(f'negative JSON integer {value!r}')
    result = int(value)
    if result > _MAX_U64:
        raise ValueError('JSON integer exceeds uint64')
    return result


def _header_size(prefix: bytes, *, file_size: int, owner: str) -> int:
    if len(prefix) != 8:
        raise ArtifactError(f'{owner}: truncated safetensors length prefix.')
    size = int.from_bytes(prefix, 'little')
    if size == 0 or size > _MAX_HEADER_SIZE:
        raise ArtifactError(
            f'{owner}: invalid safetensors header size {size}.')
    if size > file_size - 8:
        raise ArtifactError(f'{owner}: truncated safetensors header.')
    return size


def _validate_header(header: bytes, *, data_size: int, owner: str) -> None:
    if not header.startswith(b'{'):
        raise ArtifactError(f'{owner}: safetensors header must start with {{.')
    try:
        contents = json.loads(
            header.decode('utf-8'),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_int=_unsigned_integer,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ArtifactError(
            f'{owner}: invalid safetensors JSON header: {error}') from error
    if not isinstance(contents, dict):
        raise ArtifactError(f'{owner}: safetensors header must be an object.')

    intervals = []
    for name, entry in contents.items():
        if name == '__metadata__':
            # Metadata is untrusted annotation, never a source of expected
            # names, shapes, dtypes, or digests.
            if not isinstance(entry, dict) or any(not isinstance(value, str)
                                                  for value in entry.values()):
                raise ArtifactError(f'{owner}: safetensors metadata must map '
                                    'strings to strings.')
            continue
        if not isinstance(
                entry,
                dict) or set(entry) != {'dtype', 'shape', 'data_offsets'}:
            raise ArtifactError(
                f'{owner}: tensor {name!r} has invalid safetensors fields.')
        dtype = entry['dtype']
        if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
            raise ArtifactError(
                f'{owner}: tensor {name!r} has unsupported dtype {dtype!r}.')
        shape = entry['shape']
        if not isinstance(shape, list) or any(
                type(dimension) is not int or not 0 <= dimension <= _MAX_U64
                for dimension in shape):
            raise ArtifactError(
                f'{owner}: tensor {name!r} has invalid shape {shape!r}.')
        offsets = entry['data_offsets']
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
                type(offset) is not int or not 0 <= offset <= _MAX_U64
                for offset in offsets):
            raise ArtifactError(f'{owner}: tensor {name!r} has invalid '
                                f'data_offsets {offsets!r}.')
        start, end = offsets
        if start > end or end > data_size:
            raise ArtifactError(
                f'{owner}: tensor {name!r} data_offsets exceed tensor data.')
        elements = 1
        for dimension in shape:
            elements *= dimension
            if elements > _MAX_U64:
                raise ArtifactError(
                    f'{owner}: tensor {name!r} shape element count '
                    'overflows uint64.')
        bits = elements * _DTYPE_BITS[dtype]
        if bits > _MAX_U64 or bits % 8 or end - start != bits // 8:
            raise ArtifactError(
                f'{owner}: tensor {name!r} shape/dtype byte size does not '
                'match data_offsets.')
        intervals.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise ArtifactError(f'{owner}: tensor {name!r} data_offsets '
                                'have a gap or overlap.')
        cursor = end
    if cursor != data_size:
        raise ArtifactError(f'{owner}: unindexed trailing safetensors data.')


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns)


def _read_verified(
    value: str | Path,
    *,
    expected_sha256: str,
    label: str,
    retain_bytes: bool,
) -> tuple[Path, bytes | None]:
    try:
        path = Path(value).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactError(
            f'{label} {value!r}: cannot resolve artifact path: {error}'
        ) from error
    owner = f'{label} {path}'
    if path.suffix != '.safetensors':
        raise ArtifactError(f'{owner}: must use the .safetensors format.')
    if not isinstance(expected_sha256, str) or re.fullmatch(
            r'[0-9a-fA-F]{64}', expected_sha256) is None:
        raise ArtifactError(f'{owner}: expected_sha256 must be 64 hex digits.')
    digest = hashlib.sha256()
    raw = None
    try:
        # O_NONBLOCK avoids hanging on a FIFO before fstat can reject it. A
        # pre-open stat alone would race with path replacement. One descriptor
        # pins the opened file; the loader also pins immutable decoded bytes.
        with open(
                path,
                'rb',
                opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK
                                                   ),
        ) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactError(
                    f'{owner}: artifact must be a regular file.')
            if retain_bytes:
                raw = stream.read()
                header_size = _header_size(
                    raw[:8], file_size=len(raw), owner=owner)
                header = raw[8:8 + header_size]
                data_size = len(raw) - 8 - header_size
                digest.update(raw)
            else:
                prefix = stream.read(8)
                header_size = _header_size(
                    prefix, file_size=before.st_size, owner=owner)
                header = stream.read(header_size)
                if len(header) != header_size:
                    raise ArtifactError(
                        f'{owner}: truncated safetensors header.')
                data_size = before.st_size - 8 - header_size
                digest.update(prefix)
                digest.update(header)
                count = 0
                while chunk := stream.read(_CHUNK_SIZE):
                    digest.update(chunk)
                    count += len(chunk)
                if count != data_size:
                    raise ArtifactError(
                        f'{owner}: artifact changed while reading.')
            after = os.fstat(stream.fileno())
            if _stat_identity(before) != _stat_identity(after):
                raise ArtifactError(
                    f'{owner}: artifact changed while reading.')
    except FileNotFoundError as error:
        raise ArtifactError(f'{owner}: safetensors file not found.') from error
    except OSError as error:
        raise ArtifactError(
            f'{owner}: cannot read safetensors file: {error}') from error
    actual = digest.hexdigest()
    if actual != expected_sha256.lower():
        raise ArtifactError(
            f'{owner}: SHA-256 mismatch; expected={expected_sha256.lower()}, '
            f'actual={actual}.')
    _validate_header(header, data_size=data_size, owner=owner)
    return path, raw


def validate_safetensors(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    """Stream-hash and inspect a safe tensor artifact without loading tensors.

    This preflight validates this read, not future contents of the returned
    path. Loaders independently verify their immutable byte snapshot. In-place
    writes detected by fstat fail; even undetected writes must match the
    trusted expected digest of the exact bytes inspected here.
    """
    verified, _ = _read_verified(
        path, expected_sha256=expected_sha256, label=label, retain_bytes=False)
    return verified


def _load_safetensors(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, torch.Tensor]:
    verified, raw = _read_verified(
        path, expected_sha256=expected_sha256, label=label, retain_bytes=True)
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load

    assert raw is not None
    try:
        # Force CPU even for empty tensors when the caller has changed torch's
        # default device. Decode bytes, never pickle or a newly opened path.
        with torch.device('cpu'):
            return load(raw)
    except (SafetensorError, ValueError, TypeError, RuntimeError, KeyError,
            OverflowError) as error:
        raise ArtifactError(
            f'{label} {verified}: cannot decode safetensors: {error}'
        ) from error


def _validate_keys(
    actual: Iterable[str],
    expected: Iterable[str],
    *,
    owner: str,
) -> None:
    supplied, required = set(actual), set(expected)
    if supplied != required:
        raise ArtifactError(f'{owner}: tensor keys do not match; '
                            f'missing={sorted(required - supplied)}, '
                            f'unexpected={sorted(supplied - required)}.')


def _validate_finite(value: torch.Tensor, *, owner: str) -> None:
    import torch

    if value.is_floating_point() or value.is_complex():
        try:
            finite = bool(torch.isfinite(value).all().item())
        except (RuntimeError, TypeError) as error:
            raise ArtifactError(
                f'{owner}: cannot check finite values for {value.dtype}.'
            ) from error
        if not finite:
            raise ArtifactError(f'{owner}: tensor contains non-finite values.')
