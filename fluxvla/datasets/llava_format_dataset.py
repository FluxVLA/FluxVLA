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
"""LLaVA JSON/JSONL instruction data, with bounded media-error retries."""

import copy
import logging
import math
import os
import random
from bisect import bisect_right
from collections.abc import Sequence

from torch.utils.data import Dataset

from fluxvla.engines import DATASETS, build_transform_from_cfg
from .utils.vlm_annotations import IndexedJSON, IndexedJSONL

logger = logging.getLogger(__name__)


class _WeightedSource(Sequence):
    """Virtual repeats and a seeded, non-repeating fractional permutation."""

    def __init__(self, records, weight, seed):
        self.records = records
        self.full_length = len(records) * int(weight)
        self.length = self.full_length + round(len(records) * (weight % 1))
        rng = random.Random(seed)
        size = len(records)
        self.offset = rng.randrange(size) if size else 0
        self.stride = rng.randrange(1, size) if size > 1 else 1
        while size and math.gcd(self.stride, size) != 1:
            self.stride = self.stride % size + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.full_length:
            row = index % len(self.records)
        else:
            row = (self.offset + self.stride *
                   (index - self.full_length)) % len(self.records)
        return self.records[row]


def _load_json_source(source, index_cache_dir=None, seed=0):
    source = {
        'path': os.fspath(source)
    } if isinstance(source, (str, os.PathLike)) else source
    path = os.path.expanduser(source['path'])
    weight = source.get('weight', 1)
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError('Source weight must be finite and non-negative.')
    if weight == 0:
        return []
    if path.lower().endswith(('.jsonl', '.ndjson')):
        records = IndexedJSONL(path, index_cache_dir)
    else:
        records = IndexedJSON(path, index_cache_dir)
    return _WeightedSource(records, weight, seed)


@DATASETS.register_module()
class LLaVAFormatDataset(Dataset):
    """Read annotation sources and expose media roots to configured transforms.

    ``anno_json_path`` accepts a path or a list of paths / ``{path, weight}``
    dictionaries. Weights are repeat factors, not mixture probabilities.
    Fractional weights use a fixed seeded permutation, not per-epoch sampling.
    JSON arrays/objects and JSONL use disk-backed byte-offset indices.
    Raw JSON needs no file-size cap or conversion; rows are loaded on demand.
    Without transforms, samples are raw annotations plus ``data_root``.
    All conversation/media/processor handling belongs to the configured
    transforms, built with the same registry as ParquetDataset.
    """

    def __init__(self,
                 anno_json_path,
                 data_root_path=None,
                 transforms=None,
                 index_cache_dir=None,
                 seed=0,
                 max_retry_times=10,
                 statistic_name='vlm_sft'):
        sources = [anno_json_path] if isinstance(
            anno_json_path, (str, os.PathLike, dict)) else list(anno_json_path)
        if not sources:
            raise ValueError('LLaVAFormatDataset requires annotation sources.')
        if isinstance(data_root_path, (str, os.PathLike, type(None))):
            roots = [data_root_path] * len(sources)
        else:
            roots = list(data_root_path)
        if len(roots) != len(sources):
            raise ValueError(
                'data_root_path and anno_json_path must have equal lengths.')
        if (isinstance(max_retry_times, bool)
                or not isinstance(max_retry_times, int)
                or max_retry_times < 0):
            raise ValueError('Retry count must be a non-negative integer.')
        self.transforms = [
            build_transform_from_cfg(transform)
            for transform in (transforms or [])
        ]
        self._sources, self._roots, self._ends = [], [], []
        self.source_summary = []
        total = 0
        for source, root in zip(sources, roots):
            items = _load_json_source(source, index_cache_dir, seed)
            if not len(items):
                continue
            root = os.path.abspath(os.path.expanduser(root)) if root else None
            total += len(items)
            self._sources.append(items)
            self._roots.append(root)
            self._ends.append(total)
            path = source['path'] if isinstance(source, dict) else source
            self.source_summary.append(
                dict(
                    path=os.fspath(path),
                    data_root=root,
                    rows=len(items.records),
                    effective_rows=len(items)))
        if not total:
            raise ValueError('LLaVAFormatDataset has no samples.')
        self.max_retry_times = max_retry_times
        self.statistic_name = statistic_name

    def __len__(self):
        return self._ends[-1]

    def _build_sample(self, index):
        source_index = bisect_right(self._ends, index)
        start = self._ends[source_index - 1] if source_index else 0
        # Isolate in-place transforms from source records and virtual repeats.
        data = copy.deepcopy(self._sources[source_index][index - start])
        data['data_root'] = self._roots[source_index]
        for transform in self.transforms:
            data = transform(data)
        return data

    def __getitem__(self, index, dataset_statistics=None):
        current = index % len(self)
        for attempt in range(self.max_retry_times + 1):
            try:
                return self._build_sample(current)
            except (OSError, ValueError, RuntimeError) as error:
                if attempt == self.max_retry_times:
                    raise RuntimeError(
                        f'Could not process sample {index} after '
                        f'{attempt + 1} attempts.') from error
                logger.warning('Skipping sample %s: %s', current, error)
                if len(self) > 1:
                    offset = random.randrange(1, len(self))
                    current = (current + offset) % len(self)
