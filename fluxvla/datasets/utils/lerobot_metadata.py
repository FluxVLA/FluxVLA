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
"""Local LeRobot v2.1/v3 episode readers, without HF dataset caches."""

import json
from pathlib import Path


def load_jsonl_records(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_parquet_records(paths, columns=None):
    import pyarrow.parquet as pq

    records = []
    for path in paths:
        with pq.ParquetFile(path) as parquet:
            if columns is not None:
                missing = set(columns) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f'Missing annotations in {path}: '
                                     f'{sorted(missing)}')
            for batch in parquet.iter_batches(columns=columns):
                records.extend(batch.to_pylist())
    return records


def iter_episode_records(meta_root, columns=None):
    """Yield (source path, records), retaining annotation provenance.

    JSONL takes precedence, preserving ParquetDatasetV3's legacy fallback
    behavior. Missing/corrupt annotations never fall back to another file.
    """
    meta_root = Path(meta_root)
    legacy_path = meta_root / 'episodes.jsonl'
    if legacy_path.is_file():
        records = load_jsonl_records(legacy_path)
        if columns is not None:
            selected = []
            for record in records:
                missing = set(columns) - set(record)
                if missing:
                    raise ValueError(f'Missing annotations in {legacy_path}: '
                                     f'{sorted(missing)}')
                selected.append({key: record[key] for key in columns})
            records = selected
        yield legacy_path, records
        return
    paths = sorted((meta_root / 'episodes').rglob('*.parquet'))
    if not paths:
        raise FileNotFoundError(
            f'Episodes metadata not found under {meta_root}; expected '
            'episodes.jsonl (v2.1) or episodes/**/*.parquet (v3)')
    for path in paths:
        yield path, load_parquet_records([path], columns=columns)


def load_episode_records(meta_root):
    return [
        record for _, records in iter_episode_records(meta_root)
        for record in records
    ]
