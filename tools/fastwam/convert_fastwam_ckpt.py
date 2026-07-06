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
# Convert an upstream FastWAM checkpoint ({mot, proprio_encoder}) into a full
# FluxVLA ``vla.state_dict()`` safetensors so it can be evaluated with the
# standard ``scripts/eval.py`` (LiberoEvalRunner, strict load), matching the
# DreamZero-style full-checkpoint flow.

import argparse
import json
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import save_file

from fluxvla.engines import build_vla_from_cfg


def _is_fastwam_stats(stats):
    return (isinstance(stats, dict) and isinstance(stats.get('state'), dict)
            and isinstance(stats.get('action'), dict)
            and isinstance(stats['state'].get('default'), dict)
            and isinstance(stats['action'].get('default'), dict)
            and 'global_min' in stats['state']['default']
            and 'global_min' in stats['action']['default'])


def _flatten_fastwam_field(field_stats):
    default = field_stats['default']
    key_map = {
        'min': 'global_min',
        'max': 'global_max',
        'mean': 'global_mean',
        'std': 'global_std',
        'q01': 'global_q01',
        'q99': 'global_q99',
    }
    return {out_key: default[src_key] for out_key, src_key in key_map.items()}


def convert_stats_to_fluxvla_schema(stats):
    """Convert released FastWAM stats into FluxVLA eval stats schema.

    Upstream FastWAM stores state/action stats as
    ``state.default.global_min`` etc. The standard FluxVLA LIBERO eval
    transforms expect ``proprio.min`` / ``action.min`` style fields, so write
    that schema next to the converted checkpoint.
    """
    if not _is_fastwam_stats(stats):
        return stats
    converted = {
        'proprio': _flatten_fastwam_field(stats['state']),
        'action': _flatten_fastwam_field(stats['action']),
    }
    for key in ('num_episodes', 'num_transition'):
        if key in stats:
            converted[key] = stats[key]
    return converted


def parse_args():
    ap = argparse.ArgumentParser(
        description='Convert FastWAM source ckpt to a full FluxVLA '
        'state_dict safetensors for evaluation.')
    ap.add_argument('--config', required=True, help='FluxVLA config path.')
    ap.add_argument(
        '--src-ckpt',
        required=True,
        help='Upstream FastWAM checkpoint (.pt with {mot, proprio_encoder}).')
    ap.add_argument(
        '--src-stats',
        required=True,
        help='Upstream dataset stats JSON ({state, action, ...}).')
    ap.add_argument(
        '--out-dir',
        required=True,
        help='Output dir. Writes <out>/checkpoints/<name>.safetensors and '
        '<out>/dataset_statistics.json.')
    ap.add_argument(
        '--norm-stats-key',
        default='libero_10_no_noops',
        help='Key under which normalization stats are stored in '
        'dataset_statistics.json. Must match the eval config '
        "_norm_stats_key (e.g. 'libero_spatial_no_noops' for the spatial "
        'suite).')
    ap.add_argument(
        '--dtype',
        default='bf16',
        choices=['bf16', 'fp32'],
        help='Float dtype for exported tensors. Defaults to bf16 to match the '
        'source ckpt and the eval-time mixed precision.')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model_cfg = dict(cfg.inference_model if 'inference_model' in
                     cfg else cfg.model)

    # The video DiT pretrained weights (Wan2.2-TI2V-5B) are fully overwritten
    # by the source checkpoint, so skip the (potentially missing) download.
    head_cfg = dict(model_cfg['vla_head'])
    head_cfg['skip_dit_load_from_pretrain'] = True
    model_cfg['vla_head'] = head_cfg

    print(f'[convert] building model from {args.config} ...')
    vla = build_vla_from_cfg(model_cfg)
    vla.eval()

    print(f'[convert] loading source ckpt {args.src_ckpt} ...')
    vla.load_checkpoint(args.src_ckpt)

    # Strict parity check: every mot / proprio_encoder key in the source ckpt
    # must exist in the rebuilt model (verified offline, asserted here).
    src = torch.load(args.src_ckpt, map_location='cpu', mmap=True)
    model_keys = set(vla.state_dict().keys())
    miss_mot = [k for k in src['mot'] if f'vla_head.mot.{k}' not in model_keys]
    assert not miss_mot, f'mot keys missing in model: {miss_mot[:5]}'
    if 'proprio_encoder' in src and vla.vla_head.proprio_encoder is not None:
        miss_p = [
            k for k in src['proprio_encoder']
            if f'vla_head.proprio_encoder.{k}' not in model_keys
        ]
        assert not miss_p, f'proprio keys missing: {miss_p}'
    del src

    print(f'[convert] exporting full state_dict (dtype={args.dtype}) ...')
    float_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    flat = {}
    for k, v in vla.state_dict().items():
        v = v.detach().to('cpu')
        if v.is_floating_point():
            v = v.to(float_dtype)
        flat[k] = v.contiguous().clone()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.src_ckpt).stem
    sf_path = ckpt_dir / f'{name}.safetensors'
    save_file(flat, str(sf_path))
    print(f'[convert] wrote {sf_path} ({len(flat)} tensors)')

    with open(args.src_stats, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    stats = convert_stats_to_fluxvla_schema(stats)
    wrapped = {args.norm_stats_key: stats}
    stats_path = out_dir / 'dataset_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(wrapped, f)
    print(f'[convert] wrote {stats_path} (key={args.norm_stats_key})')
    print('[convert] done.')


if __name__ == '__main__':
    main()
