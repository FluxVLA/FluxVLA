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
"""Shared LeRobot video path resolution and frame decoding."""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import torch
import torchvision

from fluxvla.engines import initialize_overwatch

overwatch = initialize_overwatch(__name__)


@lru_cache(maxsize=1)
def _probe_torchcodec_video_decoder():
    """Retain the result of the native TorchCodec import probe."""
    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as err:  # noqa: BLE001
        return None, err
    return VideoDecoder, None


def _get_torchcodec_video_decoder():
    """Return the decoder class or raise an actionable import error."""
    decoder_class, import_error = _probe_torchcodec_video_decoder()
    if import_error is not None:
        raise RuntimeError(
            'TorchCodec could not be imported. Run '
            '`bash scripts/update_env.sh --skip-pull`, or install a version '
            'matching PyTorch (torch 2.8: torchcodec 0.7.0; torch 2.6: '
            'torchcodec 0.2.1) and make sure FFmpeg shared libraries are '
            'available.') from import_error
    return decoder_class


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
    log_loaded_timestamps: bool = False,
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
                if log_loaded_timestamps:
                    overwatch.info('Frame loaded at timestamp=%.4f',
                                   current_ts)
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


def decode_video_frames_torchcodec(
    video_path: Union[Path, str],
    timestamps: List[float],
    tolerance_s: float = 0.1,
    device: str = 'cpu',
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Decode frames at ``timestamps`` with torchcodec."""
    VideoDecoder = _get_torchcodec_video_decoder()

    decoder = VideoDecoder(
        str(video_path), device=device, seek_mode='approximate')
    average_fps = decoder.metadata.average_fps
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    loaded_frames = []
    loaded_ts = []
    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            overwatch.info('Frame loaded at timestamp=%.4f', pts)

    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_dist, argmin = dist.min(1)

    is_within_tol = min_dist < tolerance_s
    if not is_within_tol.all():
        raise ValueError(
            'One or several query timestamps unexpectedly violate the '
            f'tolerance ({min_dist[~is_within_tol]} > {tolerance_s=}).'
            '\nqueried timestamps: '
            f'{query_ts}'
            '\nloaded timestamps: '
            f'{loaded_ts}'
            f'\nvideo: {video_path}')

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin])
    if log_loaded_timestamps:
        overwatch.info('closest_ts=%s', loaded_ts[argmin])

    if len(timestamps) != len(closest_frames):
        raise ValueError(
            'TorchCodec returned an unexpected number of frames: '
            f'{len(closest_frames)} != {len(timestamps)} for {video_path}')
    return closest_frames


def decode_video_frames(
    video_path: Union[Path, str],
    timestamps: List[float],
    tolerance_s: float = 0.1,
    backend: Optional[str] = None,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Decode frames at ``timestamps`` using the requested backend."""
    if backend is None:
        backend = 'pyav'

    if backend == 'torchcodec':
        try:
            return decode_video_frames_torchcodec(
                video_path,
                timestamps,
                tolerance_s=tolerance_s,
                log_loaded_timestamps=log_loaded_timestamps)
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(
                'The explicitly requested TorchCodec backend failed for '
                f'video={str(video_path)!r}, timestamps={timestamps!r}, '
                f'tolerance_s={tolerance_s}. No fallback was attempted. '
                'Run `bash scripts/update_env.sh --skip-pull` if the '
                'dependency is missing or incompatible.') from err

    if backend in ('pyav', 'video_reader'):
        return decode_video_frames_torchvision(
            video_path,
            timestamps,
            tolerance_s=tolerance_s,
            backend=backend,
            log_loaded_timestamps=log_loaded_timestamps)

    raise ValueError(f'Unsupported video backend: {backend}')


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
