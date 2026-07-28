#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ['ATTN_IMPLEMENTATION'] = 'flash_attention_2'
os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'flash_attention_2'

import torch
from mmengine import Config
from safetensors.torch import load_file

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def parse_args():
    p = argparse.ArgumentParser(
        description='GR00T baseline / accelerated inference benchmark')
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument(
        '--variant',
        choices=('baseline', 'accelerated'),
        default='accelerated',
        help='baseline uses cfg.model; accelerated uses cfg.inference_model.')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--predict-runs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-views', type=int, default=2)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--lang-len', type=int, default=600)
    p.add_argument('--image-token-id', type=int, default=None)
    p.add_argument('--image-tokens-per-view', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def make_dummy_lang_tokens(device, batch_size, lang_len, image_token_id,
                           num_image_tokens):
    if num_image_tokens > lang_len:
        raise ValueError(
            f'num_image_tokens={num_image_tokens} exceeds lang_len={lang_len}')
    tokens = torch.randint(
        100, 32000, (batch_size, lang_len), device=device, dtype=torch.long)
    if image_token_id is not None and num_image_tokens > 0:
        tokens[:, :num_image_tokens] = int(image_token_id)
    return tokens


def make_dummy_batch(device, batch_size, num_views, image_size, lang_len,
                     state_dim, action_dim, image_token_id,
                     image_tokens_per_view):
    num_image_tokens = num_views * image_tokens_per_view
    return {
        'images':
        torch.randn(
            batch_size,
            num_views * 3,
            image_size,
            image_size,
            device=device,
            dtype=torch.float32),
        'img_masks':
        torch.ones(batch_size, num_views, dtype=torch.bool, device=device),
        'lang_tokens':
        make_dummy_lang_tokens(device, batch_size, lang_len, image_token_id,
                               num_image_tokens),
        'lang_masks':
        torch.ones(batch_size, lang_len, dtype=torch.bool, device=device),
        'states':
        torch.randn(batch_size, state_dim, device=device, dtype=torch.float32),
        'embodiment_ids':
        torch.zeros(batch_size, device=device, dtype=torch.long),
    }


def load_checkpoint_state(ckpt_path: Path):
    if ckpt_path.suffix == '.safetensors':
        return load_file(str(ckpt_path), device='cpu')

    state = torch.load(str(ckpt_path), map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        print("Detected training checkpoint; loading checkpoint['model']")
        return state['model']
    return state


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        print(
            'ERROR: accelerated FlowMatchingInferenceHead currently uses fixed batch_size=1 buffers',
            file=sys.stderr)
        return 1

    from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere

    set_seed_everywhere(args.seed)
    device = torch.device(args.device)
    if device.type != 'cuda':
        print('ERROR: accelerated inference requires CUDA', file=sys.stderr)
        return 1

    cfg = Config.fromfile(str(Path(args.config).expanduser().resolve()))
    cfg_key = 'model' if args.variant == 'baseline' else 'inference_model'
    if cfg_key not in cfg:
        print(f'ERROR: config missing {cfg_key}', file=sys.stderr)
        return 1
    model_cfg = cfg[cfg_key].to_dict()

    print(f'Building GR00T {args.variant} {cfg_key}...')
    vla = build_vla_from_cfg(model_cfg)
    vla = vla.to(device)
    if hasattr(vla, 'to_bfloat16'):
        vla.to_bfloat16()
    else:
        vla.to(torch.bfloat16)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    print(f'Loading weights: {ckpt_path}')
    state = load_checkpoint_state(ckpt_path)
    missing, unexpected = vla.load_state_dict(state, strict=False)
    print(
        f'load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}'
    )
    if missing:
        print('missing_samples:', missing[:20])
    if unexpected:
        print('unexpected_samples:', unexpected[:20])

    vla.eval()
    print(f'CUDA: {torch.cuda.get_device_name(device)}')
    print(f'Head: {type(vla.vla_head).__name__}')
    print(f'Backbone: {type(vla.vlm_backbone).__name__}')

    state_dim = model_cfg['vla_head']['state_dim']
    action_dim = model_cfg['vla_head']['action_dim']
    image_token_id = args.image_token_id
    if image_token_id is None:
        image_token_id = getattr(vla.vlm_backbone.vlm.config,
                                 'image_token_index', None)
    batch = make_dummy_batch(device, args.batch_size, args.num_views,
                             args.image_size, args.lang_len, state_dim,
                             action_dim, image_token_id,
                             args.image_tokens_per_view)
    image_token_count = int(
        (batch['lang_tokens']
         == image_token_id).sum().item()) if image_token_id is not None else 0
    print(
        f'image_token_id={image_token_id}, image_token_count={image_token_count}'
    )

    print(
        f'Benchmark {args.variant} predict_action: warmup={args.warmup}, runs={args.predict_runs}, lang_len={args.lang_len}'
    )
    times = []
    actions = None
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for i in range(args.warmup):
                actions = vla.predict_action(**batch)
                torch.cuda.synchronize()
                print(f'warmup {i + 1}/{args.warmup} done')
            for i in range(args.predict_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                actions = vla.predict_action(**batch)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                if (i + 1) % 10 == 0:
                    print(f'run {i + 1}/{args.predict_runs} done')

    ms = [t * 1000.0 for t in times]
    print(f'predict_action output shape: {tuple(actions.shape)}')
    print(
        f'latency_ms: min={min(ms):.3f} max={max(ms):.3f} mean={statistics.mean(ms):.3f} median={statistics.median(ms):.3f} stdev={statistics.stdev(ms) if len(ms) > 1 else 0:.3f}'
    )
    print(
        f'total_wall_predict={sum(times) * 1000.0:.3f} ms ({args.predict_runs} runs, excl. warmup)'
    )
    print('OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
