#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='PI0.5 baseline / accelerated inference benchmark')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument(
        '--variant',
        choices=('baseline', 'accelerated'),
        default='accelerated',
        help='baseline uses cfg.model; accelerated uses cfg.inference_model')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--predict-runs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--lang-len', type=int, default=48)
    parser.add_argument('--state-dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_checkpoint_state(ckpt_path: Path):
    if ckpt_path.suffix == '.safetensors':
        return load_file(str(ckpt_path), device='cpu')

    state = torch.load(str(ckpt_path), map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        print("Detected training checkpoint; loading checkpoint['model']")
        return state['model']
    return state


def make_dummy_batch(device, batch_size, num_views, image_size, lang_len,
                     state_dim, vocab_size):
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
        torch.randint(
            100,
            min(32000, max(101, vocab_size - 1)),
            (batch_size, lang_len),
            device=device,
            dtype=torch.long),
        'lang_masks':
        torch.ones(batch_size, lang_len, dtype=torch.bool, device=device),
        'states':
        torch.randn(batch_size, state_dim, device=device, dtype=torch.float32),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size != 1 and args.variant == 'accelerated':
        print(
            'ERROR: accelerated PI05FlowMatchingInference currently uses fixed batch_size=1 buffers',
            file=sys.stderr)
        return 1

    from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere

    set_seed_everywhere(args.seed)
    device = torch.device(args.device)
    if device.type != 'cuda':
        print('ERROR: PI0.5 Orin benchmark requires CUDA', file=sys.stderr)
        return 1

    cfg = Config.fromfile(str(Path(args.config).expanduser().resolve()))
    cfg_key = 'model' if args.variant == 'baseline' else 'inference_model'
    if cfg_key not in cfg:
        print(f'ERROR: config missing {cfg_key}', file=sys.stderr)
        return 1
    model_cfg = cfg[cfg_key].to_dict()

    print(f'Building PI0.5 {args.variant} {cfg_key}...')
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
    vocab_size = int(getattr(vla.llm_backbone.config, 'vocab_size', 257152))
    batch = make_dummy_batch(device, args.batch_size, args.num_views,
                             args.image_size, args.lang_len, args.state_dim,
                             vocab_size)

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
                start = time.perf_counter()
                actions = vla.predict_action(**batch)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
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