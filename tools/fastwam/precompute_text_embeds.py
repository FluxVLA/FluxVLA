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
#
# Precompute frozen T5 (umt5-xxl) text embeddings for FastWAM training and
# cache them to disk, consumed at train time by the ``LoadCachedTextEmbedding``
# transform. This is the FluxVLA equivalent of the upstream FastWAM
# ``scripts/precompute_text_embeds.py``; the prompt template, SHA-256 cache
# filename and ``{context, mask}`` payload are kept byte-compatible so the
# produced caches match the upstream ones. Dataset dirs and the cache dir are
# passed on the CLI instead of parsed from a Hydra data config.

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List

import torch

from fluxvla.models.third_party_models.fastwam.modules.helpers.loader import (
    _load_registered_model, _resolve_configs)
from fluxvla.models.third_party_models.fastwam.modules.wan_video_text_encoder import \
    HuggingfaceTokenizer  # noqa: E501

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')
DEFAULT_MODEL_ID = 'Wan-AI/Wan2.2-TI2V-5B'
DEFAULT_TOKENIZER_MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B'
DEFAULT_BATCH_SIZE = 16


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split('/')[-1]
    enc_id = re.sub(r'[^a-z0-9]+', '', base.lower())
    return enc_id or 'textenc'


def _read_unique_prompts(dataset_dirs: List[str],
                         prompt_template: str) -> List[str]:
    prompts: List[str] = []
    seen = set()
    total = 0
    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / 'meta' / 'tasks.jsonl'
        if not tasks_path.exists():
            raise FileNotFoundError(f'Missing tasks file: {tasks_path}')
        with tasks_path.open('r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if 'task' not in record:
                    raise KeyError(
                        f'Missing `task` field at {tasks_path}:{line_idx}')
                prompt = prompt_template.format(task=str(record['task']))
                total += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
    print(f'[INFO] Loaded {total} task rows from {len(dataset_dirs)} '
          f'datasets, deduplicated to {len(prompts)} prompts.')
    return prompts


def _atomic_torch_save(payload: Dict[str, torch.Tensor],
                       output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f'.{output_path.name}.tmp.{uuid.uuid4().hex}'
    tmp_path = output_path.parent / tmp_name
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Precompute frozen T5 text-embedding caches for FastWAM.')
    parser.add_argument(
        '--dataset-dir',
        action='append',
        required=True,
        help='Dataset root containing meta/tasks.jsonl. Repeatable.')
    parser.add_argument(
        '--cache-dir', required=True, help='Output cache directory.')
    parser.add_argument('--context-len', type=int, default=128)
    parser.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument(
        '--tokenizer-model-id', default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument(
        '--redirect-common-files',
        default='false',
        help='Redirect to converted safetensors (true/false). '
        'Default false.')
    parser.add_argument(
        '--overwrite',
        default='true',
        help='Overwrite existing caches (true/false). Default true.')
    parser.add_argument(
        '--prompt-template',
        default=DEFAULT_PROMPT,
        help='Prompt template with a `{task}` field.')
    args = parser.parse_args()

    overwrite = str(
        args.overwrite).strip().lower() in {'1', 'true', 'yes', 'y'}
    redirect = str(args.redirect_common_files).strip().lower() in {
        '1', 'true', 'yes', 'y'
    }
    cache_dir = Path(args.cache_dir).expanduser()
    enc_id = _model_id_to_enc_id(args.model_id)
    context_len = int(args.context_len)

    prompts = _read_unique_prompts(args.dataset_dir, args.prompt_template)
    if not prompts:
        print('[WARN] No prompts found; nothing to do.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.bfloat16
    print(f'[INFO] Loading T5 text encoder (model_id={args.model_id}, '
          f'device={device}, context_len={context_len}).')
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=redirect,
    )
    text_config.resolve_local_path()
    tokenizer_config.resolve_local_path()
    text_encoder = _load_registered_model(
        text_config.path,
        'wan_video_text_encoder',
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path, seq_len=context_len, clean='whitespace')

    written = 0
    skipped = 0
    with torch.no_grad():
        for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
            batch_prompts = prompts[start:start + DEFAULT_BATCH_SIZE]
            ids, mask = tokenizer(
                batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context = text_encoder(ids, mask)
            for i, prompt in enumerate(batch_prompts):
                hashed = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                cache_path = cache_dir / \
                    f'{hashed}.t5_len{context_len}.{enc_id}.pt'
                if cache_path.exists() and not overwrite:
                    skipped += 1
                    continue
                payload = {
                    'context':
                    context[i].detach().to(device='cpu',
                                           dtype=torch.bfloat16).contiguous(),
                    'mask':
                    mask[i].detach().to(device='cpu',
                                        dtype=torch.bool).contiguous(),
                }
                _atomic_torch_save(payload, cache_path)
                written += 1

    print(f'[INFO] Wrote {written} caches to {cache_dir} (skipped={skipped}).')


if __name__ == '__main__':
    main()
