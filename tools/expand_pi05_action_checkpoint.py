#!/usr/bin/env python3
"""Expand PI0.5 action projections without changing the source checkpoint.

The first 32 action dimensions are copied bit-for-bit from pi05_base. Newly
effective dimensions are deterministically initialized, and the unused padded
tail is exactly zero. All unrelated tensors are copied unchanged.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ACTION_IN_WEIGHT = 'action_in_proj.weight'
ACTION_OUT_WEIGHT = 'action_out_proj.weight'
ACTION_OUT_BIAS = 'action_out_proj.bias'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform_(tensor: torch.Tensor, bound: float,
              generator: torch.Generator) -> None:
    if tensor.numel():
        tensor.uniform_(-bound, bound, generator=generator)


def expand_action_tensors(tensors: dict[str, torch.Tensor],
                          pretrained_dim: int, effective_dim: int,
                          model_dim: int, seed: int) -> None:
    if not 0 < pretrained_dim < effective_dim <= model_dim:
        raise ValueError('Expected 0 < pretrained_dim < effective_dim <= '
                         f'model_dim, got {pretrained_dim}, {effective_dim}, '
                         f'{model_dim}.')

    in_weight = tensors[ACTION_IN_WEIGHT]
    out_weight = tensors[ACTION_OUT_WEIGHT]
    out_bias = tensors[ACTION_OUT_BIAS]
    if in_weight.ndim != 2 or in_weight.shape[1] != pretrained_dim:
        raise ValueError(f'Unexpected {ACTION_IN_WEIGHT} shape: '
                         f'{tuple(in_weight.shape)}')
    if out_weight.ndim != 2 or out_weight.shape[0] != pretrained_dim:
        raise ValueError(f'Unexpected {ACTION_OUT_WEIGHT} shape: '
                         f'{tuple(out_weight.shape)}')
    if out_bias.shape != (pretrained_dim, ):
        raise ValueError(f'Unexpected {ACTION_OUT_BIAS} shape: '
                         f'{tuple(out_bias.shape)}')
    if out_weight.shape[1] != in_weight.shape[0]:
        raise ValueError('Action projection hidden dimensions disagree: '
                         f'{tuple(in_weight.shape)} vs '
                         f'{tuple(out_weight.shape)}.')

    generator = torch.Generator(device='cpu').manual_seed(seed)

    expanded_in = in_weight.new_zeros((in_weight.shape[0], model_dim))
    expanded_in[:, :pretrained_dim].copy_(in_weight)
    _uniform_(expanded_in[:, pretrained_dim:effective_dim],
              1.0 / math.sqrt(model_dim), generator)

    expanded_out = out_weight.new_zeros((model_dim, out_weight.shape[1]))
    expanded_out[:pretrained_dim].copy_(out_weight)
    _uniform_(expanded_out[pretrained_dim:effective_dim],
              1.0 / math.sqrt(out_weight.shape[1]), generator)

    expanded_bias = out_bias.new_zeros((model_dim, ))
    expanded_bias[:pretrained_dim].copy_(out_bias)
    _uniform_(expanded_bias[pretrained_dim:effective_dim],
              1.0 / math.sqrt(out_weight.shape[1]), generator)

    tensors[ACTION_IN_WEIGHT] = expanded_in.contiguous()
    tensors[ACTION_OUT_WEIGHT] = expanded_out.contiguous()
    tensors[ACTION_OUT_BIAS] = expanded_bias.contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pretrained-dim', type=int, default=32)
    parser.add_argument('--effective-dim', type=int, default=52)
    parser.add_argument('--model-dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20260901)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError('Refusing to overwrite the source checkpoint.')
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(
            f'Output already exists; refusing to overwrite: {output}')

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source, framework='pt', device='cpu') as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    original_in = tensors[ACTION_IN_WEIGHT].clone()
    original_out = tensors[ACTION_OUT_WEIGHT].clone()
    original_bias = tensors[ACTION_OUT_BIAS].clone()
    expand_action_tensors(tensors, args.pretrained_dim, args.effective_dim,
                          args.model_dim, args.seed)

    assert torch.equal(tensors[ACTION_IN_WEIGHT][:, :args.pretrained_dim],
                       original_in)
    assert torch.equal(tensors[ACTION_OUT_WEIGHT][:args.pretrained_dim],
                       original_out)
    assert torch.equal(tensors[ACTION_OUT_BIAS][:args.pretrained_dim],
                       original_bias)
    assert torch.count_nonzero(
        tensors[ACTION_IN_WEIGHT][:, args.effective_dim:]) == 0
    assert torch.count_nonzero(
        tensors[ACTION_OUT_WEIGHT][args.effective_dim:]) == 0
    assert torch.count_nonzero(
        tensors[ACTION_OUT_BIAS][args.effective_dim:]) == 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    metadata.update({
        'fluxvla_action_expansion':
        f'{args.pretrained_dim}->{args.effective_dim}->{args.model_dim}',
        'fluxvla_action_expansion_seed': str(args.seed),
        'fluxvla_source_sha256': _sha256(source),
    })
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        'source':
        str(source),
        'source_sha256':
        metadata['fluxvla_source_sha256'],
        'output':
        str(output),
        'output_sha256':
        _sha256(output),
        'pretrained_action_dim':
        args.pretrained_dim,
        'effective_action_dim':
        args.effective_dim,
        'model_action_dim':
        args.model_dim,
        'seed':
        args.seed,
        'preserved': [
            f'{ACTION_IN_WEIGHT}[:, :{args.pretrained_dim}]',
            f'{ACTION_OUT_WEIGHT}[:{args.pretrained_dim}, :]',
            f'{ACTION_OUT_BIAS}[:{args.pretrained_dim}]',
        ],
        'zero_padding': [
            f'{ACTION_IN_WEIGHT}[:, {args.effective_dim}:]',
            f'{ACTION_OUT_WEIGHT}[{args.effective_dim}:, :]',
            f'{ACTION_OUT_BIAS}[{args.effective_dim}:]',
        ],
    }
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
