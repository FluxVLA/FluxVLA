"""Strict paired summary that rejects duplicate or missing seeds."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_episodes(folder, seeds):
    rows = [
        json.loads(line)
        for file in sorted(Path(folder).glob('worker_*.jsonl'))
        for line in file.read_text().splitlines() if line.strip()
    ]
    actual = [row['seed'] for row in rows]
    if len(actual) != len(set(actual)):
        raise ValueError(f'Duplicate completed seeds: {folder}')
    if set(actual) != set(seeds):
        raise ValueError(f'Incomplete/wrong test seeds: {folder}')
    by_seed = {row['seed']: row for row in rows}
    return np.array([by_seed[seed]['success'] for seed in seeds],
                    dtype=np.int8)


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [float(center - half), float(center + half)]


def compare(baseline, candidate):
    n = len(baseline)
    difference = candidate - baseline
    rng = np.random.default_rng(1234)
    boot = difference[rng.integers(0, n, (20000, n))].mean(axis=1) * 100
    return {
        'n':
        n,
        'baseline_successes':
        int(baseline.sum()),
        'candidate_successes':
        int(candidate.sum()),
        'baseline_rate':
        float(baseline.mean()),
        'candidate_rate':
        float(candidate.mean()),
        'delta_percentage_points':
        float(difference.mean() * 100),
        'rescue':
        int(((baseline == 0) & (candidate == 1)).sum()),
        'regression':
        int(((baseline == 1) & (candidate == 0)).sum()),
        'baseline_wilson_95ci':
        wilson(baseline.sum(), n),
        'candidate_wilson_95ci':
        wilson(candidate.sum(), n),
        'paired_delta_bootstrap_95ci_pp':
        np.quantile(boot, [0.025, 0.975]).tolist(),
        'scope':
        ('Exploratory: one task, one training seed; no test-set model '
         'selection'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument(
        '--candidate',
        action='append',
        required=True,
        metavar='NAME=EPISODE_DIR')
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seeds = json.loads(args.seeds.read_text())[args.task]['success_seeds']
    if len(seeds) != len(set(seeds)):
        raise ValueError('Expected unique test seeds')
    if args.expected_count is not None and len(seeds) != args.expected_count:
        raise ValueError(
            f'Expected {args.expected_count} seeds, found {len(seeds)}')
    baseline = load_episodes(args.baseline, seeds)
    candidates = {}
    for value in args.candidate:
        name, separator, folder = value.partition('=')
        if not separator or not name or name in candidates:
            raise ValueError(f'Invalid or duplicate candidate: {value}')
        candidates[name] = Path(folder)
    report = {
        name: compare(baseline, load_episodes(folder, seeds))
        for name, folder in candidates.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
