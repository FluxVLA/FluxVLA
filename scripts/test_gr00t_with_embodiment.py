#!/usr/bin/env python3
"""GR00t Eagle 3B inference benchmark with embodiment_ids support."""

from __future__ import annotations
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('ATTN_IMPLEMENTATION', 'flash_attention_2')
os.environ.setdefault('TRANSFORMERS_ATTN_IMPLEMENTATION',
                      os.environ['ATTN_IMPLEMENTATION'])

import torch
from mmengine import Config

if os.environ['ATTN_IMPLEMENTATION'] == 'flash_attention_2':
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='GR00t Eagle 3B inference test')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--no-weights', action='store_true')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-views', type=int, default=2)
    p.add_argument('--lang-len', type=int, default=48)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--predict-runs', type=int, default=100)
    p.add_argument('--warmup', type=int, default=5)
    return p.parse_args()


def _make_dummy_batch(
    *,
    device: torch.device,
    batch_size: int,
    num_views: int,
    image_size: int,
    lang_len: int,
    state_dim: int,
    action_dim: int,
    vocab_size: int,
    num_embodiments: int = 1,
) -> dict[str, torch.Tensor]:
    """Build dummy tensors compatible with GR00t predict_action."""

    # Images: (B, V*3, H, W)
    images = torch.randn(
        batch_size,
        num_views * 3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    )

    # Image mask: (B, num_views)
    img_masks = torch.ones(
        batch_size,
        num_views,
        dtype=torch.bool,
        device=device,
    )

    # Language tokens
    lang_tokens = torch.randint(
        low=0,
        high=min(32000, max(1, vocab_size - 1)),
        size=(batch_size, lang_len),
        device=device,
        dtype=torch.long,
    )
    lang_masks = torch.ones(
        batch_size, lang_len, dtype=torch.bool, device=device)

    # States: (B, state_dim)
    states = torch.randn(
        batch_size, state_dim, device=device, dtype=torch.float32)

    # Noise: (B, action_dim, action_dim)
    noise = torch.randn(
        batch_size,
        action_dim,
        action_dim,
        device=device,
        dtype=torch.float32,
    )

    # embodiment_ids: (B,) - robot type ID
    embodiment_ids = torch.zeros(batch_size, device=device, dtype=torch.long)

    return {
        'images': images,
        'img_masks': img_masks,
        'lang_tokens': lang_tokens,
        'lang_masks': lang_masks,
        'states': states,
        'noise': noise,
        'embodiment_ids': embodiment_ids,
    }


def main() -> int:
    args = _parse_args()

    print(f"Attention backend: {os.environ['ATTN_IMPLEMENTATION']}")

    from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere

    set_seed_everywhere(args.seed)
    device = torch.device(args.device)

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f'ERROR: config not found: {cfg_path}', file=sys.stderr)
        return 1

    cfg = Config.fromfile(str(cfg_path))
    if 'model' not in cfg:
        print('ERROR: config missing `model` field', file=sys.stderr)
        return 1

    model_cfg = cfg.model.to_dict()

    # Clear pretrained paths when running structure-only smoke tests.
    if args.no_weights or not args.ckpt:
        model_cfg['pretrained_name_or_path'] = None
        model_cfg['name_mapping'] = None

    print('Building GR00t Eagle 3B model...')
    vla = build_vla_from_cfg(model_cfg)
    vla = vla.to(device)
    if device.type == 'cuda' and hasattr(vla, 'to_bfloat16'):
        vla.to_bfloat16()

    # Load weights.
    if args.ckpt and not args.no_weights:
        ckpt_path = Path(args.ckpt).expanduser().resolve()
        if not ckpt_path.is_file():
            print(f'ERROR: checkpoint not found: {ckpt_path}', file=sys.stderr)
            return 1
        print(f'Loading weights: {ckpt_path}')
        state = torch.load(str(ckpt_path), map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            print(
                'Detected training checkpoint; loading checkpoint[\'model\']')
            state = state['model']
        missing, unexpected = vla.load_state_dict(state, strict=False)
        print(
            f'load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}'
        )

    vla.eval()

    if device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(device)}')
    else:
        print('Device: CPU')

    # Read expected dimensions from the config.
    state_dim = model_cfg['vla_head']['state_dim']
    action_dim = model_cfg['vla_head']['action_dim']
    ori_action_dim = model_cfg['vla_head']['ori_action_dim']
    num_embodiments = model_cfg.get('num_embodiments', 1)
    vocab_size = 257152

    print(
        f'Model params: state_dim={state_dim}, action_dim={action_dim}, ori_action_dim={ori_action_dim}, num_embodiments={num_embodiments}'
    )

    # Build a dummy batch.
    batch = _make_dummy_batch(
        device=device,
        batch_size=args.batch_size,
        num_views=args.num_views,
        image_size=args.image_size,
        lang_len=args.lang_len,
        state_dim=state_dim,
        action_dim=action_dim,
        vocab_size=vocab_size,
        num_embodiments=num_embodiments,
    )

    print(
        f'Benchmark predict_action: warmup={args.warmup}, runs={args.predict_runs} (dummy tensors)...'
    )

    times = []
    actions = None

    with torch.inference_mode():
        with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == 'cuda'):
            # Warmup.
            for _ in range(args.warmup):
                try:
                    actions = vla.predict_action(**batch)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f'ERROR during warmup: {e}', file=sys.stderr)
                    return 1

            # Timed benchmark.
            for i in range(args.predict_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    actions = vla.predict_action(**batch)
                except Exception as e:
                    print(f'ERROR at run {i}: {e}', file=sys.stderr)
                    return 1
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

    ms = [t * 1000.0 for t in times]
    print(f'predict_action output shape: {tuple(actions.shape)}')
    print(
        f'latency_ms: min={min(ms):.3f} max={max(ms):.3f} '
        f'mean={statistics.mean(ms):.3f} median={statistics.median(ms):.3f}',
        end='',
    )
    if len(ms) > 1:
        print(f' stdev={statistics.stdev(ms):.3f}')
    else:
        print()

    total = sum(times)
    print(
        f'total_wall_predict={total*1000:.3f} ms ({args.predict_runs} runs, excl. warmup)'
    )

    print('OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
