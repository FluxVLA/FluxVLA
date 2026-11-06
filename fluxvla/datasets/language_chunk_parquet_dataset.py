# Copyright 2026 Limx Dynamics
# Copyright (c) 2025 NVIDIA Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language-centered, variable-length causal windows over LeRobot parquet.

Sampling follows ShardedLeRobotSubLangSingleActionChunkDatasetDROID in
dreamzero0/dreamzero at ab790c198fbce33503358efbbd4187ce9a89adf3.
"""

import numpy as np

from fluxvla.engines import DATASETS
from .parquet_dataset import ParquetDataset


def language_chunk_frame_indices(first_index, language, max_chunks=4):
    """Return the source's sorted 8*n+1 frames (24 actions per block)."""
    length = len(language)
    if not length:
        return np.array([], dtype=np.int64)
    first_index = min(max(int(first_index), 0), length - 1)
    target = language[first_index]
    max_frames = 8 * max_chunks + 1
    frames = []

    def add(anchor):
        if (anchor >= 0 and anchor + 23 < length
                and len(frames) + 8 <= max_frames):
            frames.extend(range(anchor, anchor + 24, 3))

    add(first_index)
    step = 1
    back_done = forward_done = False
    while len(frames) < max_frames and not (back_done and forward_done):
        if not back_done:
            anchor = first_index - 24 * step
            if anchor < 0 or language[anchor] != target:
                back_done = True
            else:
                add(anchor)
        if len(frames) >= max_frames:
            break
        if not forward_done:
            anchor = first_index + 24 * step
            if anchor >= length or language[anchor] != target:
                forward_done = True
            else:
                add(anchor)
        step += 1
    frames = sorted(set(frames))[:max_frames]
    if not frames:
        return np.array([], dtype=np.int64)
    if frames[-1] + 3 < length and len(frames) < max_frames:
        frames.append(frames[-1] + 3)
    elif len(frames) <= 8:
        return np.array([], dtype=np.int64)
    else:
        frames = frames[:-7]
    return np.asarray(frames, dtype=np.int64)


@DATASETS.register_module()
class LanguageChunkParquetDataset(ParquetDataset):
    """Reuse LeRobot decoding/stats, but sample source-compatible blocks.

    Each block uses 24 actions, video stride 3 and one raw state anchor.
    Windows have 1..max_chunks blocks. Use VideoActionBlockCollator to pad
    complete blocks and retain valid lengths when batching multiple samples.
    """

    def __init__(self,
                 *args,
                 max_chunk_size=4,
                 state_key='observation.state',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.state_key = state_key
        self.max_chunk_size = int(max_chunk_size)
        if (self.max_chunk_size <= 0 or self.use_delta
                or self.window_start_idx != 0):
            raise ValueError(
                'Chunk sampling requires positive max_chunk_size, '
                'window_start_idx=0 and use_delta=False')
        # Episode ids can repeat across roots, so also split at root edges.
        episodes = np.asarray(self.dataset['episode_index'])
        cuts = np.flatnonzero(episodes[1:] != episodes[:-1]) + 1
        self._episode_edges = np.unique(
            np.concatenate([[0], cuts, self.dataset_cumulative_sizes,
                            [len(episodes)]]))

    def __getitem__(self, index, dataset_statistics):
        index = self._resolve_index(index)
        for _ in range(64):
            dataset_idx = self._get_dataset_index(index)
            episode = np.searchsorted(
                self._episode_edges, index, side='right') - 1
            start, end = self._episode_edges[episode:episode + 2]
            rows = self.dataset[int(start):int(end)]
            # RoboCasa uses a fixed language annotation per episode. Resolve
            # actual text, not task ids, to preserve source equality semantics.
            language = [
                self.tasks[dataset_idx][task]['task']
                if 0 <= task < len(self.tasks[dataset_idx]) else 'empty'
                for task in rows['task_index']
            ]
            frame_indices = language_chunk_frame_indices(
                index - start, language, self.max_chunk_size)
            if len(frame_indices) and language[index - start] not in (
                    'empty', 'static'):
                break
            index = self._rand_another()
        else:
            raise ValueError('No valid language-centered window found in 64 '
                             'attempts; check episode lengths/language labels')

        anchors = frame_indices[:-1:8]
        action_indices = np.concatenate(
            [np.arange(a, a + 24) for a in anchors])
        data = dict(self.dataset[index])
        data[self.state_key] = np.asarray(rows[self.state_key])[anchors]
        data['actions'] = self._stack_actions(
            np.asarray(rows[self.action_key])[action_indices])
        data['action_masks'] = np.ones(len(action_indices), dtype=np.float32)
        data['frame_timestamps'] = np.asarray(
            rows['timestamp'])[frame_indices].tolist()
        data['frame_masks'] = np.ones(len(frame_indices), dtype=np.float32)
        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['task_description'] = language[index - start]
        data['data_root'] = self.data_root_path[dataset_idx]
        if self.expose_index:
            data['index'] = np.asarray(index, dtype=np.int64)
        for transform in self.transforms:
            data = transform(data)
        return data
