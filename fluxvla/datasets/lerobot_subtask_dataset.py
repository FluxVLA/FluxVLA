# Copyright 2026 Limx Dynamics
"""Frame-aligned LeRobot subtask samples, independent of robot and model."""

import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import Dataset

from fluxvla.engines import DATASETS, build_transform_from_cfg
from .utils.subtask_annotations import build_subtask_index
from .utils.video_decode import make_lerobot_video_context


@DATASETS.register_module()
class LeRobotSubtaskDataset(Dataset):
    """Sample current observations within annotated [start, end) segments.

    Requires ``meta/info.json`` and ``meta/episodes/**/*.parquet``. Each
    episode must have ``episode_index``, ``length``, one task in ``tasks``,
    and complete ``<subtask_prefix>_subtask_{names,start_frames,end_frames}``
    annotations. Videos are addressed by the LeRobot ``video_path`` template
    and per-camera episode metadata; camera names belong to transforms.

    Each segment contributes at most ``frames_per_subtask`` evenly spaced
    frames (including its start and last valid frame). Long subtasks do not
    dominate merely because they last longer. The label is the CURRENT
    segment, not the next segment. No future images or actions are inputs.

    Train/validation partition whole episodes within each task before frame
    sampling. No annotation conversion, action statistics or GPT calls are
    needed. Without transforms, samples contain the task, current subtask,
    frame/segment indices and video addressing context, not model inputs.
    Video decoding, prompt construction and tokenization belong to transforms.
    """

    def __init__(self,
                 data_root,
                 transforms=None,
                 subtask_prefix='dense',
                 frames_per_subtask=3,
                 split='train',
                 validation_fraction=0.05,
                 seed=42):
        import pyarrow.parquet as pq

        if type(frames_per_subtask) is not int or frames_per_subtask < 1:
            raise ValueError('frames_per_subtask must be a positive integer')
        if split not in ('train', 'val', 'all'):
            raise ValueError('split must be train, val or all')
        if (isinstance(validation_fraction, bool)
                or not isinstance(validation_fraction, (float, int))
                or not 0 <= validation_fraction < 1):
            raise ValueError('validation_fraction must be in [0, 1)')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        if not isinstance(subtask_prefix, str) or not subtask_prefix:
            raise ValueError('subtask_prefix must be a nonempty string')
        self.data_root = Path(data_root).expanduser().resolve()
        self.info = json.loads((self.data_root / 'meta/info.json').read_text())
        self.fps = float(self.info['fps'])
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Dataset FPS must be finite and positive')
        if not self.info.get('video_path'):
            raise ValueError('Dataset info requires a video_path template')
        self.transforms = [
            build_transform_from_cfg(cfg) for cfg in (transforms or [])
        ]
        self.split = split
        self.frames_per_subtask = frames_per_subtask
        self.subtask_prefix = subtask_prefix
        self.max_retry_times = 0  # Never silently replace an invalid sample.
        paths = sorted((self.data_root / 'meta/episodes').rglob('*.parquet'))
        if not paths:
            raise FileNotFoundError('No LeRobot episode parquet metadata')
        records = [
            row for path in paths for row in pq.read_table(path).to_pylist()
        ]
        segments = build_subtask_index(records, prefix=subtask_prefix)
        by_task = defaultdict(list)
        for record in records:
            tasks = record.get('tasks')
            if (not isinstance(tasks, list) or len(tasks) != 1
                    or not isinstance(tasks[0], str) or not tasks[0].strip()):
                raise ValueError(
                    'Each episode needs one whole-task description')
            by_task[tasks[0].strip()].append(record)

        self.episodes = {}
        self.samples = []
        self.task_episode_counts = {}
        for task, task_records in sorted(by_task.items()):
            task_records.sort(key=lambda row: row['episode_index'])
            # Stable across dataset input order, ranks and DataLoader workers.
            random.Random(f'{seed}:{task}').shuffle(task_records)
            n_val = 0
            if validation_fraction and len(task_records) > 1:
                n_val = min(
                    len(task_records) - 1,
                    max(1, round(len(task_records) * validation_fraction)))
            selected = (
                task_records[:n_val] if split == 'val' else
                task_records[n_val:] if split == 'train' else task_records)
            self.task_episode_counts[task] = len(selected)
            for record in selected:
                episode = int(record['episode_index'])
                # Keep only video addressing metadata, not the full sequence
                # of future labels/segment boundaries in the decoder context.
                metadata = {
                    key: value
                    for key, value in record.items()
                    if key.startswith(('videos/', 'data/', 'meta/episodes/'))
                }
                self.episodes[episode] = (task, metadata)
                for segment_index, segment in enumerate(segments[episode]):
                    count = min(frames_per_subtask,
                                segment.end - segment.start)
                    for position in range(count):
                        offset = (position *
                                  (segment.end - segment.start - 1) //
                                  (count - 1)) if count > 1 else 0
                        self.samples.append((episode, segment_index, segment,
                                             segment.start + offset))
        if not self.samples:
            raise ValueError(f'No samples in LeRobot {split} split')
        # Bounded validation batches should cover tasks, not just episode 0.
        random.Random(seed).shuffle(self.samples)
        self.episode_ids = tuple(sorted(self.episodes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, dataset_statistics=None):
        episode, segment_index, segment, frame = self.samples[index]
        task, metadata = self.episodes[episode]
        sample = dict(
            episode_index=episode,
            frame_index=frame,
            full_task_description=task,
            subtask=segment.name,
            subtask_index=segment_index,
            subtask_start_frame=segment.start,
            subtask_end_frame=segment.end,
            lerobot_video=make_lerobot_video_context(self.data_root, self.info,
                                                     copy.deepcopy(metadata),
                                                     episode,
                                                     [frame / self.fps]))
        for transform in self.transforms:
            sample = transform(sample)
        return sample
