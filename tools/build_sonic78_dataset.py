# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Augment a LeRobot/Isaac-GR00T dataset with a single 78-dim SONIC action.

FluxVLA's ``ParquetDataset`` builds the temporal action window from **one**
parquet column (``action_key``). SONIC's 78-dim action space, however, is split
across three columns:

    action.motion_token        (64,)   FSQ-quantized motion tokens
    teleop.left_hand_joints    ( 7,)
    teleop.right_hand_joints   ( 7,)

This script adds a single concatenated column ``action.sonic78`` (78,) in the
fixed order ``[motion_token | left_hand | right_hand]`` (the same order used by
SonicStar's ``run_vla_inference.py``), and appends the corresponding per-episode
statistics to ``meta/episodes_stats.jsonl`` so that
``DistributedRepeatingDataset.get_dataset_statistics`` and
``NormalizeStatesAndActions`` can consume it.

It likewise adds a single concatenated proprio column
``observation.sonic_state46`` (46,) in the fixed order
``[observation.state(43) | observation.projected_gravity(3)]``. This matches the
SONIC deploy-side observation that the WBC C++ produces
(``concat[whole_q(43), projected_gravity(3)]``) and the 46-dim state used by
SonicStar-BlackOtters training, so that the FluxVLA proprio branch is aligned
with both the deploy obs and the SONIC token decoder.

By default the script is NON-DESTRUCTIVE: it writes a new dataset directory and
symlinks the (large) ``videos/`` folder back to the source. Use ``--in-place``
to instead augment the source dataset directly (additive only; existing columns
and stats are preserved).

Usage
-----
    # non-destructive (recommended): creates datasets/merged_dataset_001_sonic78
    python tools/build_sonic78_dataset.py \
        --src ~/Erwin/Datasets/merged_dataset_001 \
        --dst datasets/merged_dataset_001_sonic78

    # in-place augmentation of the source dataset
    python tools/build_sonic78_dataset.py \
        --src ~/Erwin/Datasets/merged_dataset_001 --in-place
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Concatenation order MUST match the inference-time split.
ACTION_PARTS = [
    'action.motion_token',     # 64
    'teleop.left_hand_joints',  # 7
    'teleop.right_hand_joints',  # 7
]
NEW_ACTION_KEY = 'action.sonic78'
NEW_ACTION_DIM = 78

# Proprio concatenation order MUST match the SONIC deploy observation
# (``concat[whole_q(43), projected_gravity(3)]``) and BlackOtters' 46-dim state.
STATE_PARTS = [
    'observation.state',              # 43
    'observation.projected_gravity',  # 3
]
NEW_STATE_KEY = 'observation.sonic_state46'
NEW_STATE_DIM = 46

# Each spec produces one fused column from a list of source columns.
FUSED_SPECS = [
    {'key': NEW_ACTION_KEY, 'dim': NEW_ACTION_DIM, 'parts': ACTION_PARTS},
    {'key': NEW_STATE_KEY, 'dim': NEW_STATE_DIM, 'parts': STATE_PARTS},
]


def _to_2d(col):
    """Convert a parquet list-column (pa.ChunkedArray) to (N, D) float32."""
    arr = np.asarray(col.to_pylist(), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def augment_parquet(src_path: str, dst_path: str) -> None:
    table = pq.read_table(src_path)
    changed = False
    for spec in FUSED_SPECS:
        cols = table.column_names
        for part in spec['parts']:
            if part not in cols:
                raise KeyError(
                    f"{part} not found in {src_path}; columns={cols}")
        if spec['key'] in cols:
            # Already augmented for this spec: leave as-is.
            continue
        parts = [_to_2d(table.column(p)) for p in spec['parts']]
        fused = np.concatenate(parts, axis=-1)  # (N, dim)
        assert fused.shape[-1] == spec['dim'], \
            f"Expected {spec['dim']} dims for {spec['key']}, " \
            f'got {fused.shape[-1]}'
        fused_list = pa.array(fused.tolist(), type=pa.list_(pa.float32()))
        table = table.append_column(spec['key'], fused_list)
        changed = True

    if changed or src_path != dst_path:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        pq.write_table(table, dst_path)


def augment_episode_stats(src_meta: str, dst_meta: str) -> None:
    src = os.path.join(src_meta, 'episodes_stats.jsonl')
    dst = os.path.join(dst_meta, 'episodes_stats.jsonl')
    out_lines = []
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            st = rec['stats']
            for spec in FUSED_SPECS:
                if spec['key'] in st:
                    continue
                for part in spec['parts']:
                    if part not in st:
                        raise KeyError(
                            f'{part} missing in episodes_stats for '
                            f"episode {rec.get('episode_index')}")
                fused = {}
                for field in ['min', 'max', 'mean', 'std']:
                    fused[field] = np.concatenate([
                        np.asarray(st[p][field], dtype=np.float64)
                        for p in spec['parts']
                    ]).tolist()
                # count is per-frame and identical across source columns.
                fused['count'] = list(st[spec['parts'][0]]['count'])
                st[spec['key']] = fused
            out_lines.append(json.dumps(rec))
    os.makedirs(dst_meta, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')


def augment_info(src_meta: str, dst_meta: str) -> None:
    """Copy info.json and register the new feature (optional, for validity)."""
    src = os.path.join(src_meta, 'info.json')
    with open(src, 'r', encoding='utf-8') as f:
        info = json.load(f)
    feats = info.get('features', {})
    for spec in FUSED_SPECS:
        anchor = spec['parts'][0]
        if anchor in feats and spec['key'] not in feats:
            feats[spec['key']] = {
                'dtype': 'float32',
                'shape': [spec['dim']],
                'names': None,
            }
    with open(os.path.join(dst_meta, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--src',
        default=os.path.expanduser('~/Erwin/Datasets/merged_dataset_001'),
        help='Source LeRobot dataset root (contains data/ meta/ videos/).')
    ap.add_argument(
        '--dst',
        default='datasets/merged_dataset_001_sonic78',
        help='Destination dataset root (ignored with --in-place).')
    ap.add_argument(
        '--in-place', action='store_true',
        help='Augment the source dataset directly instead of writing a copy.')
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    in_place = args.in_place
    dst = src if in_place else os.path.abspath(os.path.expanduser(args.dst))

    src_data = os.path.join(src, 'data')
    src_meta = os.path.join(src, 'meta')
    assert os.path.isdir(src_data), f'No data/ under {src}'
    assert os.path.isdir(src_meta), f'No meta/ under {src}'

    dst_data = os.path.join(dst, 'data')
    dst_meta = os.path.join(dst, 'meta')

    if not in_place:
        os.makedirs(dst, exist_ok=True)
        # Symlink the large videos directory (no copy).
        src_videos = os.path.join(src, 'videos')
        dst_videos = os.path.join(dst, 'videos')
        if os.path.isdir(src_videos) and not os.path.exists(dst_videos):
            os.symlink(src_videos, dst_videos)
        # Copy meta verbatim, then overwrite the augmented files below.
        if os.path.exists(dst_meta):
            shutil.rmtree(dst_meta)
        shutil.copytree(src_meta, dst_meta)

    # 1) parquet data: add the action.sonic78 column.
    parquets = sorted(
        glob.glob(os.path.join(src_data, '**', '*.parquet'), recursive=True))
    assert parquets, f'No parquet files under {src_data}'
    for i, p in enumerate(parquets):
        rel = os.path.relpath(p, src_data)
        out = p if in_place else os.path.join(dst_data, rel)
        augment_parquet(p, out)
        if (i + 1) % 25 == 0 or i + 1 == len(parquets):
            print(f'[data] {i + 1}/{len(parquets)} parquet augmented')

    # 2) meta: episodes_stats.jsonl + info.json.
    augment_episode_stats(src_meta, dst_meta)
    augment_info(src_meta, dst_meta)
    print(f'[meta] episodes_stats.jsonl + info.json updated under {dst_meta}')

    print(f'\nDone. Augmented dataset root: {dst}')
    print(f'New column: {NEW_ACTION_KEY} ({NEW_ACTION_DIM},) = '
          f"[{' | '.join(ACTION_PARTS)}]")
    print(f'New column: {NEW_STATE_KEY} ({NEW_STATE_DIM},) = '
          f"[{' | '.join(STATE_PARTS)}]")


if __name__ == '__main__':
    main()
