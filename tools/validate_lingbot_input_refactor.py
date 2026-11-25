#!/usr/bin/env python3
"""Check a preprocessing refactor against a pinned pre-change reader.

Uses real online video/tokenizer inputs on CPU, without loading a policy or
using model-ready replay. Historical source is loaded only in this diagnostic
process; production training never executes a Git-sourced dataset.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


class _Unregistered:

    def register_module(self):
        return lambda cls: cls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--reference-ref',
        required=True,
        help='Pinned pre-refactor Git commit')
    parser.add_argument('--data', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--stats', required=True)
    parser.add_argument(
        '--video-backend',
        choices=['torchcodec', 'pyav', 'video_reader'],
        default='torchcodec')
    parser.add_argument(
        '--resize-dtype', choices=['uint8', 'float32'], default='float32')
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 2, 4])
    parser.add_argument(
        '--worker-context',
        choices=['spawn', 'forkserver', 'fork'],
        default='spawn')
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--all-episode-boundaries', action='store_true')
    parser.add_argument('--random-samples', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    from fluxvla.datasets.lingbot_vla_v2 import LingBotParquetDataset
    from fluxvla.datasets.lingbot_vla_v2.flow_rng import sample_flow_by_index

    root = Path(__file__).resolve().parents[1]
    git = ['git', '-c', f'safe.directory={root}', '-C', str(root)]
    reference_commit = subprocess.check_output(
        git + ['rev-parse', args.reference_ref], text=True).strip()
    source = subprocess.check_output(
        git + [
            'show', f'{reference_commit}:fluxvla/datasets/'
            'lingbot_vla_v2/online_dataset.py'
        ],
        text=True)
    flow_path = 'fluxvla/datasets/lingbot_vla_v2/flow_rng.py'
    reference_flow = subprocess.check_output(
        git + ['show', f'{reference_commit}:{flow_path}'])
    if reference_flow != (root / flow_path).read_bytes():
        raise ValueError(
            'Reference flow RNG differs; use a separate pinned flow oracle '
            'before comparing inputs')
    # Avoid registering a second class with the same name. These two import
    # substitutions do not alter any data-processing operation in the oracle.
    source = source.replace('from fluxvla.engines import DATASETS',
                            'DATASETS = REFERENCE_REGISTRY')
    source = source.replace('from .flow_rng import sample_flow_by_index', '')
    scope = {
        'REFERENCE_REGISTRY': _Unregistered(),
        'sample_flow_by_index': sample_flow_by_index
    }
    exec(compile(source, f'<reference {reference_commit}>', 'exec'), scope)

    kwargs = dict(
        data_path=args.data,
        norm_stats_path=args.stats,
        tokenizer_path=args.tokenizer,
        return_teacher_images=True,
        flow_rng_seed=42)
    before = scope['LingBotOnlineDataset'](**kwargs)
    after = LingBotParquetDataset(
        **kwargs,
        video_backend=args.video_backend,
        resize_dtype=args.resize_dtype)
    if len(before) != len(after):
        raise AssertionError('Dataset length changed')
    indices = args.indices
    if indices is None:
        # A transformed HF Column can decode every image in its source row
        # on each indexed lookup. Materialize the metadata-only view once.
        episodes = [
            int(value) for value in before.dataset.hf_dataset.select_columns(
                ['episode_index'])['episode_index']
        ]
        first_boundary = next(
            (i for i in range(1, len(episodes)) if episodes[i] != episodes[0]),
            len(episodes))
        indices = sorted({
            i
            for i in (0, 1, first_boundary - 2, first_boundary - 1,
                      first_boundary, first_boundary + 1, len(after) - 2,
                      len(after) - 1) if 0 <= i < len(after)
        })
        if args.all_episode_boundaries:
            boundaries = [
                i for i in range(1, len(episodes))
                if episodes[i] != episodes[i - 1]
            ]
            indices = sorted(
                set(indices)
                | {i + offset
                   for i in boundaries for offset in (-1, 0)})
    if args.random_samples < 0:
        raise ValueError('--random-samples must be nonnegative')
    if args.random_samples:
        sampled = torch.randperm(
            len(after),
            generator=torch.Generator().manual_seed(42))[:args.random_samples]
        indices = sorted(set(indices) | set(sampled.tolist()))
    expected = [before[index] for index in indices]
    runs = []
    for workers in args.workers:
        started = time.monotonic()
        records = []
        # Decoder libraries can own native threads/caches after the reference
        # is read in the parent. Spawn avoids inheriting those decoder states.
        loader = DataLoader(
            Subset(after, indices),
            batch_size=None,
            num_workers=workers,
            multiprocessing_context=args.worker_context if workers else None)
        for index, reference, actual in zip(
                indices, expected, loader, strict=True):
            if reference.keys() != actual.keys():
                raise AssertionError(f'sample {index}: keys differ')
            hashes = {}
            for key in reference:
                a, b = reference[key], actual[key]
                if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(
                        a, b):
                    delta = float((
                        a.float() -
                        b.float()).abs().max()) if a.shape == b.shape else None
                    raise AssertionError(
                        f'workers={workers}, sample={index}, '
                        f'first-bad field={key}, '
                        f'shape={a.shape}/{b.shape}, '
                        f'dtype={a.dtype}/{b.dtype}, max_abs={delta}')
                hashes[key] = hashlib.sha256(b.contiguous().reshape(-1).view(
                    torch.uint8).numpy().tobytes()).hexdigest()
            records.append(dict(index=index, field_sha256=hashes))
        result = dict(
            workers=workers,
            exact_samples=len(records),
            fields=len(expected[0]),
            elapsed_seconds=time.monotonic() - started,
            records=records)
        runs.append(result)
        print(
            json.dumps({k: v
                        for k, v in result.items() if k != 'records'}),
            flush=True)
    files = [
        'fluxvla/datasets/lingbot_vla_v2/parquet_dataset.py',
        'fluxvla/datasets/parquet_dataset.py',
        'fluxvla/datasets/parquet_dataset_v3.py',
        'fluxvla/datasets/utils/video_decode.py',
        'fluxvla/transforms/lingbot_online.py', flow_path
    ]
    report = dict(
        reference_commit=reference_commit,
        torch_version=torch.__version__,
        package_versions={
            name: version(name)
            for name in ('lerobot', 'transformers')
        },
        worker_context=args.worker_context,
        video_backend=args.video_backend,
        resize_dtype=args.resize_dtype,
        current_file_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
        data=str(Path(args.data).resolve()),
        dataset_length=len(after),
        indices=indices,
        tokenizer=str(Path(args.tokenizer).resolve()),
        stats_sha256=hashlib.sha256(Path(args.stats).read_bytes()).hexdigest(),
        exact=True,
        runs=runs,
        scope='CPU input equivalence only; not optimizer/100-step '
        'training certification')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
