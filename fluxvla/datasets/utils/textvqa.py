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
"""Read official TextVQA validation annotations without training conversion."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEXTVQA_VAL_URL = (
    'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json')


def load_textvqa_validation(annotation_path):
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f'TextVQA annotations missing: {path}. '
                                'Download the official JSON from '
                                f'{TEXTVQA_VAL_URL} to this path.')
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected official TextVQA JSON with nonempty data.')
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get('set_name') != 'val'
                or not isinstance(row.get('question'), str)
                or not row['question'].strip()
                or type(row.get('question_id')) is not int
                or not isinstance(row.get('image_id'), str)
                or not re.fullmatch(r'[A-Za-z0-9_-]+', row['image_id'])
                or not isinstance(row.get('answers'), list)
                or len(row['answers']) != 10
                or not all(isinstance(a, str) for a in row['answers'])):
            raise ValueError('Invalid TextVQA validation record.')
        if row['question_id'] in seen:
            raise ValueError('Duplicate TextVQA question_id.')
        seen.add(row['question_id'])
    return rows


def validate_textvqa_images(rows, image_root):
    """Check unique images; missing media must never change the eval sample."""
    root = Path(image_root)
    paths = [
        root / f'{image_id}.jpg'
        for image_id in sorted({row['image_id']
                                for row in rows})
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        exists = list(pool.map(Path.is_file, paths))
    missing = [str(path) for path, found in zip(paths, exists) if not found]
    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} TextVQA images; first: {missing[0]}')
    return len(paths)
