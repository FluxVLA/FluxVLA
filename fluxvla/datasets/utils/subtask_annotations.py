# Copyright 2026 Limx Dynamics
"""Strict, frame-aligned episode annotations shared by VLA data and stats."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SubtaskSegment:
    name: str
    start: int
    end: int


def build_subtask_index(
        records: Iterable[Mapping],
        prefix: str = 'dense') -> dict[int, tuple[SubtaskSegment, ...]]:
    """Read ``<prefix>_subtask_*`` intervals using [start, end) semantics.

    Require a complete, ordered partition of each episode. Missing labels,
    gaps and overlaps are errors, never an implicit fallback to whole-task
    conditioning. Repeated names are legal: segment identity is positional.
    """
    result = {}
    for record in records:
        episode = int(record['episode_index'])
        context = f'episode {episode}, {prefix}_subtask'
        if episode in result:
            raise ValueError(f'Duplicate annotations for {context}')
        length = record.get('length')
        if not isinstance(length, Integral) or length <= 0:
            raise ValueError(f'Invalid episode length for {context}: {length}')
        keys = [
            f'{prefix}_subtask_{suffix}'
            for suffix in ('names', 'start_frames', 'end_frames')
        ]
        names, starts, ends = [record.get(key) for key in keys]
        if any(value is None for value in (names, starts, ends)):
            raise ValueError(f'Missing subtask annotations for {context}; '
                             f'required fields: {keys}')
        if not len(names) or not len(names) == len(starts) == len(ends):
            raise ValueError(f'Empty or mismatched annotation lists for '
                             f'{context}')
        segments = []
        previous_end = 0
        for name, start, end in zip(names, starts, ends):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f'Empty/non-string subtask name for {context}')
            if (not isinstance(start, Integral)
                    or not isinstance(end, Integral) or start != previous_end
                    or not start < end <= length):
                raise ValueError(f'Invalid subtask range [{start}, {end}) for '
                                 f'{context}; expected start {previous_end}, '
                                 f'end <= {length} (no gaps/overlaps)')
            segments.append(SubtaskSegment(name.strip(), int(start), int(end)))
            previous_end = end
        if previous_end != length:
            raise ValueError(
                f'Incomplete subtask coverage for {context}: '
                f'ends at {previous_end}, episode length {length}')
        result[episode] = tuple(segments)
    if not result:
        raise ValueError('No episode subtask annotations found')
    return result
