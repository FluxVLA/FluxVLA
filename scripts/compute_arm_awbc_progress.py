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
"""Compute ARM progress parquet files for RA-BC / AW-BC training.

Rebuilds dense per-frame ``progress`` on unlabeled policy datasets by running
strided ARM inference, accumulating interval-head deltas, anchoring completion
with the success head, and interpolating to every frame. The resulting parquet
is consumed by ``ArmRABCWeighter`` / ``ArmAWBCWeighter``.

Official implementation of https://arxiv.org/abs/2604.03037
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from mmengine import Config, DictAction
from torch.utils.data import Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from infer_arm_progress import _register_arm_runtime_modules  # noqa: E402

from tools.arm_awbc.progress_reconstruction import (  # noqa: E402
    build_cumulative_progress, run_strided_episode_inference)


def parse_args():
    """Parse command-line arguments for ARM progress computation."""
    parser = argparse.ArgumentParser(
        description='Compute ARM progress parquet files for RA/AW-BC.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument(
        '--output-path',
        type=str,
        default='./arm_progress.parquet',
        help='Output parquet path. Defaults to ./arm_progress.parquet.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--stride',
        type=int,
        default=150,
        help='Run ARM inference every N frames within each episode, then '
        'interpolate dense progress to all frames. Defaults to 150 '
        '(5 s at 30 fps), matching infer_arm_progress visualization.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config key-value pairs in xxx=yyy format')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be >= 1')
    return args


def _episode_ranges(dataset: Dataset) -> List[Tuple[int, int, int, int]]:
    ranges = getattr(dataset, 'episode_ranges', None)
    if not ranges:
        return [(0, 0, 0, len(dataset))]
    return [
        (int(dataset_idx), int(episode_idx), int(start), int(end))
        for (dataset_idx, episode_idx), (
            start, end) in sorted(ranges.items(), key=lambda item: item[1][0])
    ]


def _build_output_rows(
    dataset: Dataset,
    model: Any,
    collator: Any,
    inference_stride: int,
    device: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    episode_ranges = _episode_ranges(dataset)

    for dataset_index, episode_index, start, end in tqdm(
            episode_ranges, desc='ARM RA/AW-BC episodes'):
        episode_length = end - start
        inference_results = run_strided_episode_inference(
            model=model,
            dataset=dataset,
            collator=collator,
            ep_start=start,
            ep_end=end,
            inference_stride=inference_stride,
            device=device,
            show_progress=False,
        )
        frame_progress = build_cumulative_progress(inference_results,
                                                   episode_length)

        for local_frame_idx in range(episode_length):
            global_index = start + local_frame_idx
            rows.append({
                'index': global_index,
                'dataset_index': dataset_index,
                'episode_index': episode_index,
                'frame_index': local_frame_idx,
                'episode_length': episode_length,
                'progress': float(frame_progress[local_frame_idx]),
            })
    return rows


def _write_progress_parquet(output_path: str, rows: List[Dict[str, Any]],
                            reward_model_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No ARM progress rows were produced.')

    columns: Dict[str, List[Any]] = {
        key: [row.get(key) for row in rows]
        for key in rows[0].keys()
    }
    table = pa.Table.from_pydict(columns)
    metadata = dict(table.schema.metadata or {})
    metadata[b'reward_model_path'] = str(reward_model_path).encode()
    metadata[b'progress_source'] = b'interval_success_reconstruction'
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def main():
    """Run ARM inference and write a progress parquet for RA/AW-BC."""
    args = parse_args()
    builders = _register_arm_runtime_modules()
    build_collator_from_cfg = builders['build_collator_from_cfg']
    build_dataset_from_cfg = builders['build_dataset_from_cfg']
    build_vla_from_cfg = builders['build_vla_from_cfg']

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset_cfg = cfg.get('inference_dataset', cfg.train_dataloader.dataset)
    dataset_cfg = dataset_cfg.copy()
    dataset_cfg['training'] = False
    dataset = cast(Dataset, build_dataset_from_cfg(dataset_cfg))

    collator = build_collator_from_cfg(cfg.runner.collator.copy())
    model = cast(Any, build_vla_from_cfg(cfg.model))
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(
        checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device(args.device)).eval()

    rows = _build_output_rows(
        dataset=dataset,
        model=model,
        collator=collator,
        inference_stride=args.stride,
        device=args.device,
    )
    _write_progress_parquet(args.output_path, rows, args.ckpt_path)


if __name__ == '__main__':
    main()
