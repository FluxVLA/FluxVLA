"""Versioned multipart protocol shared by ZMQ evaluation processes."""

from __future__ import annotations
import io
from collections.abc import Mapping
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1


def encode_frame(value: Any) -> bytes:
    import msgpack
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def decode_frame(value: bytes) -> Any:
    import msgpack
    return msgpack.unpackb(
        value, object_hook=_decode, raw=False, strict_map_key=False)


def encode_header(message_type: str, **values: Any) -> bytes:
    if not isinstance(message_type, str) or not message_type:
        raise ValueError('message_type must be a non-empty string')
    return encode_frame({
        'version': PROTOCOL_VERSION,
        'type': message_type,
        **values,
    })


def decode_header(frame: bytes) -> dict[str, Any]:
    value = decode_frame(frame)
    if not isinstance(value, Mapping):
        raise TypeError('ZMQ message header must be a mapping')
    header = dict(value)
    version = header.get('version')
    if version != PROTOCOL_VERSION:
        raise ValueError(f'Unsupported ZMQ protocol version {version!r}')
    message_type = header.get('type')
    if not isinstance(message_type, str) or not message_type:
        raise ValueError('ZMQ message header.type must be a non-empty string')
    return header


def empty_payload() -> bytes:
    return encode_frame({})


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {'__ndarray__': True, 'data': buffer.getvalue()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def _decode(value: dict[str, Any]) -> Any:
    if value.get('__ndarray__') is True:
        return np.load(io.BytesIO(value['data']), allow_pickle=False)
    return value


__all__ = [
    'PROTOCOL_VERSION',
    'decode_frame',
    'decode_header',
    'empty_payload',
    'encode_frame',
    'encode_header',
]
