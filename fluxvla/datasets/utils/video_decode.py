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
"""Shared LeRobot video path resolution and torchvision decoding."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import torch
import torchvision


def build_lerobot_video_path(
    data_root_path: Union[str, Path],
    info: Mapping[str, object],
    episode_meta: Mapping[str, object],
    episode_index: int,
    video_key: str,
) -> str:
    """Resolve a LeRobot episode video file path for ``video_key``."""
    video_root_path = info['video_path']
    format_kwargs = {
        'video_key': video_key,
        'episode_index': episode_index,
    }

    def _first_available_int(keys: List[str]) -> Optional[int]:
        for key in keys:
            if key in episode_meta:
                return int(episode_meta[key])
        return None

    chunk_index = _first_available_int([
        f'videos/{video_key}/chunk_index',
        'meta/episodes/chunk_index',
        'data/chunk_index',
        'chunk_index',
    ])
    file_index = _first_available_int([
        f'videos/{video_key}/file_index',
        'meta/episodes/file_index',
        'data/file_index',
        'file_index',
    ])
    if chunk_index is not None:
        format_kwargs['chunk_index'] = chunk_index
    if file_index is not None:
        format_kwargs['file_index'] = file_index
    if 'chunks_size' in info:
        format_kwargs['episode_chunk'] = episode_index // int(
            info['chunks_size'])
    return os.path.join(
        str(data_root_path),
        str(video_root_path).format(**format_kwargs))


def decode_video_frames_torchvision(
    video_path: Union[Path, str],
    timestamps: List[float],
    tolerance_s: float = 0.1,
    backend: str = 'pyav',
) -> torch.Tensor:
    """Decode frames at ``timestamps`` from a LeRobot episode video."""
    video_path = str(video_path)
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    def _load_candidates(
            seek_ts: float) -> tuple[List[torch.Tensor], List[float]]:
        keyframes_only = backend == 'pyav'
        torchvision.set_video_backend(backend)
        reader = torchvision.io.VideoReader(video_path, 'video')
        reader.seek(seek_ts, keyframes_only=keyframes_only)

        loaded_frames: List[torch.Tensor] = []
        loaded_ts: List[float] = []
        try:
            for frame in reader:
                current_ts = float(frame['pts'])
                loaded_frames.append(frame['data'])
                loaded_ts.append(current_ts)
                if current_ts >= last_ts:
                    break
        finally:
            if backend == 'pyav':
                reader.container.close()
        return loaded_frames, loaded_ts

    def _match_candidates(loaded_frames: List[torch.Tensor],
                          loaded_ts: List[float]) -> Optional[torch.Tensor]:
        if not loaded_ts:
            return None
        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
        dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
        min_dist, argmin = dist.min(1)
        if not (min_dist <= tolerance_s).all():
            return None
        return torch.stack([loaded_frames[idx] for idx in argmin])

    loaded_frames, loaded_ts = _load_candidates(first_ts)
    matched_frames = _match_candidates(loaded_frames, loaded_ts)
    if matched_frames is None and backend == 'pyav' and first_ts > 0:
        loaded_frames, loaded_ts = _load_candidates(0.0)
        matched_frames = _match_candidates(loaded_frames, loaded_ts)
    if matched_frames is None:
        raise ValueError(
            f'Failed to find frames within tolerance for {video_path}')
    return matched_frames


def make_lerobot_video_context(
    data_root_path: Union[str, Path],
    info: Mapping[str, object],
    episode_meta: Mapping[str, object],
    episode_index: int,
    timestamps: List[float],
) -> Dict[str, object]:
    """Bundle metadata needed by :class:`DecodeLeRobotVideoSequence`."""
    return {
        'data_root_path': str(data_root_path),
        'info': dict(info),
        'episode_meta': dict(episode_meta),
        'episode_index': int(episode_index),
        'timestamps': [float(ts) for ts in timestamps],
    }
