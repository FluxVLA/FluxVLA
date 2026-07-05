#!/usr/bin/env python3
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import glob
import json
from pathlib import Path


def _load_profiles(paths):
    profiles = []
    for pattern in paths:
        matches = [Path(match) for match in sorted(glob.glob(pattern))]
        if not matches:
            matches = [Path(pattern)]
        for path in matches:
            if path.is_file():
                with open(path, 'r') as f:
                    profile = json.load(f)
                profile['_path'] = path.as_posix()
                profiles.append(profile)
    return profiles


def _collect(profiles, event_key):
    values = []
    for profile in profiles:
        for event in profile.get('events', []):
            if event_key in event:
                values.append(float(event[event_key]))
    return values


def _collect_episode(profiles, event_key):
    values = []
    for profile in profiles:
        for event in profile.get('episode_events', []):
            if event_key in event:
                values.append(float(event[event_key]))
    return values


def _flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f'{prefix}.{key}' if prefix else key
            _flatten(next_prefix, child, output)
    elif isinstance(value, list):
        numeric = [
            float(item) for item in value
            if isinstance(item, (int, float))
        ]
        if numeric:
            output.setdefault(prefix, []).extend(numeric)
    elif isinstance(value, (int, float)):
        output.setdefault(prefix, []).append(float(value))


def _collect_model_profiles(profiles):
    output = {}
    for profile in profiles:
        for event in profile.get('events', []):
            model_profile = event.get('model_profile', {})
            _flatten('', model_profile, output)
    return output


def _stats(values):
    if not values:
        return None
    values = sorted(values)

    def percentile(pct):
        index = min(
            len(values) - 1,
            max(0, int(round((len(values) - 1) * pct / 100.0))),
        )
        return values[index]

    return {
        'count': len(values),
        'mean': sum(values) / len(values),
        'p50': percentile(50),
        'p90': percentile(90),
        'p95': percentile(95),
        'max': values[-1],
    }


def _print_stats(name, values):
    stats = _stats(values)
    if stats is None:
        print(f'{name}: no samples')
        return
    print(
        f'{name}: count={stats["count"]} '
        f'mean={stats["mean"]:.2f}ms '
        f'p50={stats["p50"]:.2f}ms '
        f'p90={stats["p90"]:.2f}ms '
        f'p95={stats["p95"]:.2f}ms '
        f'max={stats["max"]:.2f}ms')


def main():
    parser = argparse.ArgumentParser(
        description='Summarize lightweight eval profile JSON files.')
    parser.add_argument(
        'profiles',
        nargs='+',
        help='Profile JSON paths or glob patterns, e.g. work_dirs/.../*.json')
    args = parser.parse_args()

    profiles = _load_profiles(args.profiles)
    if not profiles:
        raise SystemExit('No profile files found.')

    print(f'profiles: {len(profiles)}')
    for profile in profiles:
        context = profile.get('context', {})
        print(
            f'- {profile["_path"]}: rank={context.get("rank")} '
            f'pred_events={len(profile.get("events", []))} '
            f'episodes={len(profile.get("episode_events", []))}')

    print()
    _print_stats('predict_ms', _collect(profiles, 'predict_ms'))
    _print_stats('preprocess_ms', _collect(profiles, 'preprocess_ms'))
    _print_stats('env_step_chunk_ms',
                 _collect(profiles, 'env_step_chunk_ms'))
    _print_stats('episode_ms', _collect_episode(profiles, 'episode_ms'))

    model_profiles = _collect_model_profiles(profiles)
    if model_profiles:
        print('\nmodel_profile:')
        for key in sorted(model_profiles):
            _print_stats(f'  {key}', model_profiles[key])

    max_allocated = []
    max_reserved = []
    for profile in profiles:
        memory = profile.get('summary', {}).get('cuda_memory_mb', {})
        if 'max_allocated' in memory:
            max_allocated.append(float(memory['max_allocated']))
        if 'max_reserved' in memory:
            max_reserved.append(float(memory['max_reserved']))
    if max_allocated:
        print(f'cuda_max_allocated_mb: {max(max_allocated):.1f}')
    if max_reserved:
        print(f'cuda_max_reserved_mb: {max(max_reserved):.1f}')


if __name__ == '__main__':
    main()
