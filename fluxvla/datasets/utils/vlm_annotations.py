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
"""Model-independent VLM records and disk-backed JSON/JSONL random access.

This module deliberately does not import torch, transformers or FluxVLA's
registries; datasets and transforms share these model-independent helpers.
"""

import hashlib
import json
import mmap
import os
import re
import struct
import tempfile
from collections.abc import Sequence
from pathlib import Path

from filelock import FileLock

_HEADER = struct.Struct('<8s64sQ')
_ENTRY = struct.Struct('<QQ')
_MAGIC = b'FVJSONL1'
_ROLES = {
    'human': 'user',
    'user': 'user',
    'gpt': 'assistant',
    'assistant': 'assistant',
    'system': 'system',
}


def _check_record(item):
    if not isinstance(item, dict) or not any(
            key in item for key in ('conversations', 'messages')):
        raise ValueError('Expected a conversations/messages object.')
    return item


def media_paths(item, kind):
    if item.get(kind) and item.get(kind + 's'):
        raise ValueError(f'Use only one of {kind}/{kind}s.')
    value = item.get(kind)
    if value is None:
        value = item.get(kind + 's')
    if value is None:
        value = []
    paths = [value] if isinstance(value, str) else value
    if not isinstance(paths, list) or not all(
            isinstance(path, str) and path.strip() for path in paths):
        raise ValueError(f'{kind} must be a path or list of nonempty paths.')
    return list(paths)


def normalize_llava_item(item, adapter='llava'):
    """Normalize explicit LLaVA/role-content schemas; never invent answers.

    ``llava-video`` explicitly translates the legacy video-only <image>
    marker used by LLaVA-NeXT. Ambiguous image+video records are rejected.
    Structured messages accept text/image/video parts with string paths.
    """
    _check_record(item)
    if adapter not in ('llava', 'llava-video'):
        raise ValueError(f'Unknown annotation adapter: {adapter}')
    if 'conversations' in item and 'messages' in item:
        raise ValueError('Use conversations OR messages, not both.')
    turns = item.get('conversations', item.get('messages'))
    if not isinstance(turns, list) or not turns:
        raise ValueError('Conversation must be a nonempty list.')
    pools = {kind: media_paths(item, kind) for kind in ('image', 'video')}
    inline = {'image': [], 'video': []}
    conversations = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError('Each conversation turn must be an object.')
        raw_role = turn.get('from', turn.get('role'))
        role = _ROLES.get(raw_role) if isinstance(raw_role, str) else None
        if role is None:
            raise ValueError(f'Unsupported conversation role: {turn}')
        value = turn.get('value', turn.get('content'))
        if isinstance(value, list):
            parts = []
            for part in value:
                if not isinstance(part, dict):
                    raise ValueError('Message content parts must be objects.')
                kind = part.get('type')
                if not isinstance(kind, str):
                    raise ValueError('Message content type must be a string.')
                if kind == 'text' and isinstance(part.get('text'), str):
                    parts.append(part['text'])
                elif kind in inline and isinstance(part.get(kind), str):
                    if set(part) != {'type', kind}:
                        raise ValueError(
                            'Inline media supports type/path only; '
                            'sampling/cropping options are unsupported.')
                    if role != 'user' or not part[kind].strip():
                        raise ValueError(
                            'Media requires a user turn and path.')
                    inline[kind].append(part[kind])
                    parts.append(f'<{kind}>')
                else:
                    raise ValueError(f'Unsupported message content: {part}')
            value = ''.join(parts)
        if not isinstance(value, str):
            raise ValueError('Turn value/content must contain text.')
        # Otherwise text can forge a ChatML assistant header and become a
        # training target despite belonging to a user/system turn.
        if '<|im_start|>' in value or '<|im_end|>' in value:
            raise ValueError(
                'Raw conversation contains reserved ChatML tokens.')
        conversations.append(dict(from_=role, value=value))
    for kind in pools:
        if inline[kind]:
            if pools[kind]:
                raise ValueError('Do not mix inline and top-level media.')
            pools[kind] = inline[kind]
    if adapter == 'llava-video':
        if pools['image'] or not pools['video']:
            raise ValueError(
                'llava-video adapter requires video-only records.')
        for turn in conversations:
            turn['value'] = turn['value'].replace('<image>', '<video>')
    out = {
        key: value
        for key, value in item.items()
        if key not in ('messages', 'conversations', 'image', 'images', 'video',
                       'videos')
    }
    if 'system' in out and not isinstance(out['system'], str):
        raise ValueError('Top-level system must be text.')
    if any(
            token in out.get('system', '')
            for token in ('<|im_start|>', '<|im_end|>')):
        raise ValueError('System contains reserved ChatML tokens.')
    for kind, paths in pools.items():
        if paths:
            out[kind] = paths[0] if len(paths) == 1 else paths
    out['conversations'] = [{
        'from': turn['from_'],
        'value': turn['value']
    } for turn in conversations]
    return out


def build_messages_llava(item, data_root=None):
    """Validate and preserve roles, whitespace and image/video ordering."""
    item = normalize_llava_item(item)
    pools = {kind: media_paths(item, kind) for kind in ('image', 'video')}
    messages = []
    if item.get('system'):
        messages.append(
            dict(
                role='system',
                content=[dict(type='text', text=item['system'])]))
    previous = None
    for turn in item['conversations']:
        role = turn['from']
        if role == 'system' and previous not in (None, 'system'):
            raise ValueError('System turns must precede the conversation.')
        if role == 'assistant' and previous != 'user':
            raise ValueError('An assistant turn must follow a user turn.')
        content = []
        for part in re.split(r'(<image>|<video>)', turn['value']):
            if part in ('<image>', '<video>'):
                kind = part[1:-1]
                if role != 'user' or not pools[kind]:
                    raise ValueError(
                        f'Unmatched {part} placeholder in {role} turn.')
                path = os.path.expanduser(pools[kind].pop(0))
                if data_root and not os.path.isabs(path) and '://' not in path:
                    path = os.path.join(os.fspath(data_root), path)
                content.append({'type': kind, kind: path})
            elif part:
                content.append(dict(type='text', text=part))
        if not content or (role == 'assistant' and not turn['value'].strip()):
            raise ValueError(f'Empty {role} turn.')
        messages.append(dict(role=role, content=content))
        previous = role
    if any(pools.values()):
        raise ValueError('Every image/video must have a matching placeholder.')
    return messages


def _fingerprint(path):
    stat = os.stat(path)
    # An inexpensive invalidation key, NOT a content checksum. Published
    # training data must remain immutable for the lifetime of a dataset.
    identity = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return hashlib.sha256(repr(identity).encode()).hexdigest().encode()


class IndexedJSONL(Sequence):
    """One 16-byte mmap index entry per row; no resident annotation list.

    Construction scans a new source once under a cross-process file lock.
    Atomic publication prevents workers/ranks from reading partial indexes.
    File handles are reopened after fork/spawn, and reads use pread.
    """

    _magic = _MAGIC
    _index_suffix = '.idx'

    def __init__(self, path, cache_dir=None):
        self._fd = self._mapping = self._pid = None
        self.path = str(Path(path).expanduser().resolve())
        self.fingerprint = _fingerprint(self.path)
        cache = Path(cache_dir or '~/.cache/fluxvla/vlm_annotations')
        cache = cache.expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        self.index_path = str(cache /
                              (self.fingerprint.decode() + self._index_suffix))
        with FileLock(self.index_path + '.lock'):
            if not self._valid_index():
                self._build_index()
        if not self._valid_index():
            raise ValueError(f'Invalid annotation index: {self.index_path}')

    def _valid_index(self):
        try:
            with open(self.index_path, 'rb') as stream:
                magic, fingerprint, count = _HEADER.unpack(
                    stream.read(_HEADER.size))
                if (magic != self._magic or fingerprint != self.fingerprint
                        or os.fstat(stream.fileno()).st_size !=
                        _HEADER.size + count * _ENTRY.size):
                    return False
            self._length = count
            return True
        except (OSError, struct.error):
            return False

    def _iter_offsets(self):
        with open(self.path, 'rb') as source:
            offset = 0
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        _check_record(json.loads(line))
                    except ValueError as error:
                        raise ValueError(
                            f'{self.path}:{line_number}: {error}') from error
                    yield offset, len(line)
                offset += len(line)

    def _build_index(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(self.index_path), delete=False) as out:
                temporary = out.name
                out.write(_HEADER.pack(self._magic, self.fingerprint, 0))
                count = 0
                for offset, length in self._iter_offsets():
                    out.write(_ENTRY.pack(offset, length))
                    count += 1
                if _fingerprint(self.path) != self.fingerprint:
                    raise RuntimeError('Annotation changed during indexing.')
                out.seek(0)
                out.write(_HEADER.pack(self._magic, self.fingerprint, count))
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.index_path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError('Annotation index must be an integer.')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            if _fingerprint(self.path) != self.fingerprint:
                raise RuntimeError('Annotation changed; reconstruct dataset.')
            with open(self.index_path, 'rb') as stream:
                self._mapping = mmap.mmap(
                    stream.fileno(), 0, access=mmap.ACCESS_READ)
            self._fd = os.open(self.path, os.O_RDONLY)
            self._pid = os.getpid()
        offset, length = _ENTRY.unpack_from(self._mapping,
                                            _HEADER.size + index * _ENTRY.size)
        return _check_record(json.loads(os.pread(self._fd, length, offset)))

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
        if self._mapping is not None:
            self._mapping.close()
        self._fd = self._mapping = self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_fd=None, _mapping=None, _pid=None)
        return state

    def __del__(self):
        self.close()


class IndexedJSON(IndexedJSONL):
    """Index UTF-8 JSON arrays/objects directly, without writing a JSONL copy.

    Only the current record and a read buffer are held while indexing.
    Byte ranges refer to the unmodified source, including UTF-8/CRLF input.
    Locking, atomic publication and worker-safe reads are shared with JSONL.
    """

    _magic = b'FVJSONA1'
    _index_suffix = '.json.idx'

    def _iter_offsets(self):
        for _, offset, length in _iter_json_document(self.path):
            yield offset, length


def _iter_json_document(path, chunk_size=65536, max_record_chars=64 << 20):
    """Yield validated records and their original UTF-8 byte ranges.

    The buffer limit applies to ONE record, not the file's total size.
    Decoding validates nested structures, string escapes and delimiters.
    """
    if chunk_size < 1 or max_record_chars < 1:
        raise ValueError('JSON chunk and record limits must be positive.')
    decoder = json.JSONDecoder()
    # Disable universal newline translation: index offsets address raw bytes.
    with Path(path).open(encoding='utf-8', newline='') as stream:
        buffer, eof, offset = '', False, 0

        def consume(count, byte_count=None):
            nonlocal buffer, offset
            offset += (
                len(buffer[:count].encode('utf-8'))
                if byte_count is None else byte_count)
            buffer = buffer[count:]

        def fill():
            nonlocal buffer, eof
            block = stream.read(chunk_size)
            buffer += block
            eof = not block

        def whitespace():
            while True:
                count = len(buffer) - len(buffer.lstrip(' \t\r\n'))
                consume(count, count)
                if buffer or eof:
                    return
                fill()

        fill()
        if buffer.startswith('\ufeff'):
            consume(1, 3)
        whitespace()
        if not buffer:
            return
        is_array = buffer.startswith('[')
        if is_array:
            consume(1, 1)
        need_item = False
        while True:
            whitespace()
            if is_array and buffer.startswith(']'):
                if need_item:
                    raise ValueError('Trailing comma in JSON array.')
                consume(1, 1)
                break
            while True:
                try:
                    item, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if len(buffer) > max_record_chars:
                        raise ValueError(
                            'JSON record exceeds reader buffer limit.')
                    if eof:
                        raise
                    fill()
            if end > max_record_chars:
                raise ValueError('JSON record exceeds reader buffer limit.')
            item = _check_record(item)
            start = offset
            length = len(buffer[:end].encode('utf-8'))
            consume(end, length)
            yield item, start, length
            if not is_array:
                break
            whitespace()
            if buffer.startswith(','):
                consume(1, 1)
                need_item = True
            elif buffer.startswith(']'):
                consume(1, 1)
                break
            else:
                raise ValueError(
                    'Expected comma or closing JSON array bracket.')
        whitespace()
        if buffer:
            raise ValueError('Trailing data after JSON document.')


def iter_json_records(path, chunk_size=65536, max_record_chars=64 << 20):
    """Stream JSON arrays/objects or JSONL without a full-file read."""
    path = Path(path)
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        with path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield _check_record(json.loads(line))
                    except ValueError as error:
                        raise ValueError(
                            f'{path}:{number}: {error}') from error
        return
    for item, _, _ in _iter_json_document(path, chunk_size, max_record_chars):
        yield item
