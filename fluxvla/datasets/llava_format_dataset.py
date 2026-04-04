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
"""LLaVA-format instruction dataset for VLM SFT."""

import json
import os
import random
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from fluxvla.engines import DATASETS, initialize_overwatch
from .utils.vlm_processors import get_processor_for_vlm, process_item_for_vlm

overwatch = initialize_overwatch(__name__)


def _load_json_source(src: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(src, str):
        path = src
        weight = 1.0
    else:
        path = src['path']
        weight = src.get('weight', 1.0)

    if not os.path.exists(path):
        overwatch.warning(f'Data source not found: {path}')
        return []

    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    if not raw:
        return []

    items: List[Dict[str, Any]] = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for _ in range(max(1, int(weight))):
                items.extend(data)
        else:
            for _ in range(max(1, int(weight))):
                items.append(data)
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for _ in range(max(1, int(weight))):
                items.append(obj)
    return items


def _resolve_media_paths_with_root(
    item: Dict[str, Any],
    data_root: Optional[str],
) -> Dict[str, Any]:
    if not data_root:
        return item

    out = dict(item)
    for key in ('image', 'images', 'video', 'videos'):
        if key not in out:
            continue
        value = out[key]
        if isinstance(value, str):
            if not os.path.isabs(value):
                out[key] = os.path.join(data_root, value)
        elif isinstance(value, list):
            out[key] = [
                os.path.join(data_root, p)
                if isinstance(p, str) and not os.path.isabs(p) else p
                for p in value
            ]
    return out


@DATASETS.register_module()
class LLaVAFormatDataset(Dataset):
    """Dataset for VLM SFT in LLaVA format.

    Args:
        anno_json_path: List of annotation JSON/JSONL files. Each entry can be
            a str or dict {"path": str, "weight": float}.
        data_root_path: Prefix root for each annotation source. When media
            paths in annotation are relative, they are resolved against the
            matched root. If str, it is used for all sources.
        processor_type: VLM processor type, e.g. 'qwen3_vl'.
        model_path: Path to VLM processor/tokenizer.
        image_max_pixels: Optional max pixels for images.
        video_max_pixels: Optional max pixels for videos.
        truncation_max_length: Max length for truncation (0 = no truncation).
        max_retry_times: Max retries for corrupted/invalid samples.
        statistic_name: Name for stats compatibility.
    """

    def __init__(
        self,
        anno_json_path: List[Union[str, Dict[str, Any]]],
        data_root_path: Optional[Union[str, List[Optional[str]]]] = None,
        processor_type: str = 'qwen3_vl',
        model_path: Optional[str] = None,
        image_max_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        truncation_max_length: int = 0,
        max_retry_times: int = 10,
        statistic_name: str = 'vlm_sft',
    ):
        super().__init__()
        self.anno_json_path = anno_json_path
        self.processor_type = processor_type
        self.model_path = model_path or ''
        self.image_max_pixels = image_max_pixels
        self.video_max_pixels = video_max_pixels
        self.truncation_max_length = truncation_max_length
        self.max_retry_times = max(0, int(max_retry_times))
        self.statistic_name = statistic_name
        self._skipped_samples = 0

        if not self.anno_json_path:
            raise ValueError('LLaVAFormatDataset requires anno_json_path.')

        if data_root_path is None:
            roots = [None] * len(self.anno_json_path)
        elif isinstance(data_root_path, str):
            roots = [data_root_path] * len(self.anno_json_path)
        else:
            roots = list(data_root_path)
            if len(roots) != len(self.anno_json_path):
                raise ValueError(
                    'data_root_path and anno_json_path must have same length.')

        self.data_root_path = [
            os.path.abspath(os.path.expanduser(p)) if p else None
            for p in roots
        ]

        self._items: List[Dict[str, Any]] = []
        self._item_data_roots: List[Optional[str]] = []
        for src, src_root in zip(self.anno_json_path, self.data_root_path):
            loaded_items = _load_json_source(src)
            self._items.extend(loaded_items)
            self._item_data_roots.extend([src_root] * len(loaded_items))

        self._processor = None
        if model_path:
            self._processor = get_processor_for_vlm(
                processor_type,
                model_path,
                image_max_pixels=image_max_pixels,
                video_max_pixels=video_max_pixels,
            )

        self.stats = [{
            'stats': {
                'length': {
                    'min': len(self._items),
                    'max': len(self._items),
                    'mean': len(self._items),
                    'std': 0.0,
                }
            }
        }]

    def __len__(self) -> int:
        return len(self._items)

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        item = self._items[idx]
        item = _resolve_media_paths_with_root(item, self._item_data_roots[idx])
        if self._processor is None:
            raise RuntimeError(
                'LLaVAFormatDataset requires model_path to build the '
                'processor.')
        out = process_item_for_vlm(
            self.processor_type,
            self._processor,
            item,
            data_root=None,
            truncation_max_length=self.truncation_max_length,
            image_max_pixels=self.image_max_pixels,
            video_max_pixels=self.video_max_pixels,
        )
        result = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                result[k] = (
                    v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v)
            else:
                result[k] = v
        return result

    def __getitem__(
        self,
        idx: int,
        dataset_statistics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if len(self._items) == 0:
            raise RuntimeError(
                'LLaVAFormatDataset has no valid items to read.')

        current_idx = idx % len(self._items)
        max_attempts = self.max_retry_times + 1
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                return self._build_sample(current_idx)
            except Exception as err:
                last_error = err
                self._skipped_samples += 1
                if attempt >= self.max_retry_times:
                    break

                next_idx = random.randint(0, len(self._items) - 1)
                if len(self._items) > 1 and next_idx == current_idx:
                    next_idx = (next_idx + 1) % len(self._items)

                overwatch.warning(
                    'Failed to load sample idx=%d (attempt %d/%d), '
                    'retry with idx=%d. skipped_samples=%d. err=%s',
                    current_idx,
                    attempt + 1,
                    max_attempts,
                    next_idx,
                    self._skipped_samples,
                    repr(err),
                )
                current_idx = next_idx

        raise RuntimeError('Failed to load sample after '
                           f'{max_attempts} attempts. idx={idx}, '
                           f'skipped_samples={self._skipped_samples}. '
                           f'last_error={repr(last_error)}') from last_error
