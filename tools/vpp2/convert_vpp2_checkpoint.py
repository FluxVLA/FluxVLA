"""Convert a released VPP2 bundle to strictly validated sharded safetensors."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from fluxvla.models.heads.vpp2_head import VPP2Head

PREFIXES = {
    'action_expert': 'vla_head.mot.mixtures.action.',
    'proprio_encoder': 'vla_head.proprio_encoder.',
    'dit': 'vla_head.mot.mixtures.video.'
}


def read_bundle(bundle):
    root = Path(bundle)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('format') != 'vpp2-robodojo-bundle-v1':
        raise ValueError('Unsupported VPP2 manifest format')
    action = torch.load(
        root / 'action.pt', map_location='cpu', mmap=True, weights_only=True)
    if action.get('format') != 'vpp2_wan21_action_v1':
        raise ValueError('Unsupported VPP2 action checkpoint format')
    video_path = Path(action['video_checkpoint']['path'])
    if not video_path.is_absolute():
        video_path = root / video_path
    video = torch.load(
        video_path, map_location='cpu', mmap=True, weights_only=True)
    if 'dit' not in video or video.get('step') != action.get(
            'step') or action.get('step') != manifest.get('step'):
        raise ValueError(
            'Bundle video/action/manifest steps or formats disagree')
    return manifest, action, video


def map_state_dict(action, video):
    return {
        PREFIXES[group] + k: v
        for group, payload in (('action_expert', action),
                               ('proprio_encoder', action), ('dit', video))
        for k, v in payload[group].items()
    }


def convert_statistics(statistics, key='robodojo_ee16'):
    converted = {}
    for source, target in [('state', 'proprio'), ('action', 'action')]:
        values = statistics[source]['default']
        stats = {}
        for name in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
            if 'global_' + name not in values:
                continue
            array = np.asarray(values['global_' + name])
            if array.shape != (16, ) or not np.isfinite(array).all():
                raise ValueError(f'{source}.global_{name} must be finite EE16')
            stats[name] = array.tolist()
        if 'mean' not in stats or 'std' not in stats or np.any(
                np.asarray(stats['std']) < 0):
            raise ValueError(f'Invalid {source} mean/std statistics')
        stats['count'] = int(statistics['num_transition'])
        converted[target] = stats
    return {key: converted}


def head_config(model):
    """Extract supported joint options; reject experimental routes."""
    for key in ('use_video_tokens', 'action_only_first_frame_fast_path',
                'joint_denoising'):
        if model.get(key, False):
            raise ValueError(f'Unsupported experimental option: {key}')
    for key in ('action_clean_first_frame_start_layer',
                'action_clean_first_frame_timestep_cutoff',
                'action_video_timestep_cutoff', 'use_fixed_video_layers'):
        if model.get(key) is not None:
            raise ValueError(f'Unsupported experimental option: {key}')
    if model.get('video_lora', {}).get(
            'enabled', False) or model.get('action_chunk_size', 1) != 1:
        raise ValueError('VPP2 supports full experts and action_chunk_size=1')
    return {
        k: model[k]
        for k in ('video_dit_config', 'action_dit_config', 'proprio_dim',
                  'video_scheduler', 'action_scheduler',
                  'mot_checkpoint_mixed_attn', 'action_visible_video_frames')
        if k in model
    }


def convert(bundle, out, max_shard_size=5_000_000_000):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a fresh output directory: {out}')
    if max_shard_size <= 0:
        raise ValueError('max_shard_size must be positive')
    manifest, action, video = read_bundle(bundle)
    config = action['resolved_config']
    with torch.device('meta'):
        model = VPP2Head(**head_config(config['model']), device='meta')
    weights = map_state_dict(action, video)
    # strict=True validates every key and shape; assign avoids allocating 37
    # GB.
    model.load_state_dict(
        {k.removeprefix('vla_head.'): v
         for k, v in weights.items()},
        strict=True,
        assign=True)
    counts = {
        name: sum(p.numel() for p in expert.parameters())
        for name, expert in model.mot.mixtures.items()
    }
    statistics = convert_statistics(
        json.loads((Path(bundle) / 'dataset_stats.json').read_text()))
    shards, shard, size = [], {}, 0
    for name, tensor in weights.items():
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > max_shard_size:
            raise ValueError(f'Tensor {name} exceeds max_shard_size')
        if shard and size + nbytes > max_shard_size:
            shards.append(shard)
            shard, size = {}, 0
        shard[name] = tensor
        size += nbytes
    if shard:
        shards.append(shard)
    out.mkdir(parents=True)
    weight_map = {}
    for index, shard in enumerate(shards, 1):
        filename = f'model-{index:05d}-of-{len(shards):05d}.safetensors'
        save_file({k: v.contiguous()
                   for k, v in shard.items()}, str(out / filename))
        weight_map.update({k: filename for k in shard})
    (out / 'model.safetensors.index.json').write_text(
        json.dumps(
            {
                'metadata': {
                    'total_size':
                    sum(t.numel() * t.element_size() for t in weights.values())
                },
                'weight_map': weight_map
            },
            indent=2) + '\n')
    (out / 'dataset_statistics.json'
     ).write_text(json.dumps(statistics, indent=2) + '\n')
    (out /
     'resolved_config.json').write_text(json.dumps(config, indent=2) + '\n')
    shutil.copyfile(Path(bundle) / 'manifest.json', out / 'manifest.json')
    (out / 'conversion.json').write_text(
        json.dumps(
            {
                'parameter_counts': counts,
                'missing_keys': [],
                'unexpected_keys': [],
                'upstream_revision': '3365872910244f14df83003062f7c563f14c6d9e'
            },
            indent=2) + '\n')
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-shard-size', type=int, default=5_000_000_000)
    args = parser.parse_args()
    print(convert(args.bundle, args.out, args.max_shard_size))
