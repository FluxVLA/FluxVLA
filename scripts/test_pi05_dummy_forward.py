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
"""PI05FlowMatching 干跑：随机张量输入 + 可选加载微调 safetensors。

示例::

    cd /path/to/FluxVLA
    python scripts/test_pi05_dummy_forward.py \\
        --config /home/limx/sober/checkpoints/pi05_aloha_fold_towel/config.yaml \\
        --ckpt /home/limx/sober/checkpoints/pi05_aloha_fold_towel/checkpoints/step-021062-epoch-01-loss=0.0205.safetensors

仅构建模型、不加载权重（随机初始化，用于连通性）::

    python scripts/test_pi05_dummy_forward.py \\
        --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \\
        --no-weights
"""

from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file

from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere
from fluxvla.engines.utils.model_utils import make_att_2d_masks


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='PI05 dummy-tensor forward / predict_action smoke test.')
    p.add_argument(
        '--config',
        type=str,
        required=True,
        help='mmengine 配置文件（.py 或训练目录下的 config.yaml）。',
    )
    p.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help=('微调权重：可为具体 .safetensors 文件，或训练根目录'
              '（自动在该目录或 checkpoints/ 下选一份 .safetensors）。'),
    )
    p.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='训练输出目录（含 config.yaml 与 checkpoints/*.safetensors）。',
    )
    p.add_argument(
        '--no-weights',
        action='store_true',
        help='不加载任何权重（仅随机初始化，用于快速验证前向是否可跑）。',
    )
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-views', type=int, default=3, help='Aloha 三路相机=3')
    p.add_argument('--lang-len', type=int, default=48)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    p.add_argument(
        '--strict-load',
        action='store_true',
        help=('load_state_dict(strict=True)；默认 strict=False 并打印 '
              'missing/unexpected 数量。'),
    )
    p.add_argument(
        '--predict-runs',
        type=int,
        default=100,
        help='predict_action 重复次数（用于统计延迟）。',
    )
    p.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='正式计时的预热次数（不计入统计）。',
    )
    return p.parse_args()


def _pick_safetensors_in_dir(d: Path) -> Path | None:
    """在目录 d 或 d/checkpoints 下选取一份 .safetensors（按名字排序取最后一个）。"""
    if not d.is_dir():
        return None
    cands = sorted(d.glob('*.safetensors'))
    if cands:
        return cands[-1]
    sub = d / 'checkpoints'
    if sub.is_dir():
        cands = sorted(sub.glob('*.safetensors'))
        if cands:
            return cands[-1]
    return None


def _resolve_ckpt(args: argparse.Namespace) -> Path | None:
    if args.ckpt:
        p = Path(args.ckpt).expanduser().resolve()
        if p.is_file():
            return p
        # 常见误用：把训练根目录当成 --ckpt
        picked = _pick_safetensors_in_dir(p)
        return picked
    if args.run_dir:
        return _pick_safetensors_in_dir(
            Path(args.run_dir).expanduser().resolve())
    return None


def _make_dummy_batch(
    *,
    device: torch.device,
    batch_size: int,
    num_views: int,
    image_size: int,
    lang_len: int,
    max_action_dim: int,
    n_action_steps: int,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    """构造与 embed_prefix / predict_action 兼容的 dummy 张量。"""
    # SigLIPViTBackbone: (B, V*3, H, W)，见 siglip_vit forward unflatten
    images = torch.randn(
        batch_size,
        num_views * 3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    )
    # 与 ProcessParquetInputs 一致：每路相机一个 bool，collate 后为 (B, num_views)。
    # embed_prefix 内会 img_masks.permute(1, 0) 再按视角展开到 patch 维，不能传 (B, V, n_patch)。
    img_masks = torch.ones(
        batch_size,
        num_views,
        dtype=torch.bool,
        device=device,
    )
    lang_tokens = torch.randint(
        low=0,
        high=min(32000, max(1, vocab_size - 1)),
        size=(batch_size, lang_len),
        device=device,
        dtype=torch.long,
    )
    lang_masks = torch.ones(
        batch_size, lang_len, dtype=torch.bool, device=device)
    states = torch.randn(
        batch_size, max_action_dim, device=device, dtype=torch.float32)
    noise = torch.randn(
        batch_size,
        n_action_steps,
        max_action_dim,
        device=device,
        dtype=torch.float32,
    )
    return {
        'images': images,
        'img_masks': img_masks,
        'lang_tokens': lang_tokens,
        'lang_masks': lang_masks,
        'states': states,
        'noise': noise,
    }


def _embed_prefix_forward_only(vla, batch: dict, device: torch.device) -> None:
    """可选：走 embed_prefix + embed_suffix + forward_model（与单元测试路径一致）。"""
    images = batch['images']
    with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == 'cuda'):
        prefix_embs, prefix_pad, prefix_att = vla.embed_prefix(
            images=images,
            lang_tokens=batch['lang_tokens'],
            img_masks=batch['img_masks'],
            lang_masks=batch['lang_masks'],
        )
        time = torch.full(
            (images.shape[0], ),
            0.5,
            device=device,
            dtype=torch.float32,
        )
        suffix_embs, suffix_pad, suffix_att, adarms_cond = vla.embed_suffix(
            batch['states'], batch['noise'], time)
        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = vla._prepare_attention_masks_4d(att_2d)
        suffix_out, _ = vla.forward_model(
            inputs_embeds=[prefix_embs, suffix_embs],
            attention_masks=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            fill_kv_cache=None,
            adarms_cond=[None, adarms_cond],
            time=time,
        )
        actions = vla.action_out_proj(suffix_out)
    assert actions.shape[0] == images.shape[0]


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _bench_predict_action(
    vla,
    batch: dict,
    device: torch.device,
    *,
    warmup: int,
    runs: int,
) -> tuple[list[float], torch.Tensor]:
    """返回每次 predict 的耗时（秒）列表及最后一次输出。"""
    times: list[float] = []
    actions: torch.Tensor | None = None

    def _one() -> torch.Tensor:
        return vla.predict_action(
            images=batch['images'],
            lang_tokens=batch['lang_tokens'],
            states=batch['states'],
            img_masks=batch['img_masks'],
            lang_masks=batch['lang_masks'],
            noise=batch['noise'],
        )

    with torch.inference_mode():
        with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == 'cuda'):
            for _ in range(max(0, warmup)):
                actions = _one()
                _sync_if_cuda(device)

            for _ in range(runs):
                _sync_if_cuda(device)
                t0 = time.perf_counter()
                actions = _one()
                _sync_if_cuda(device)
                times.append(time.perf_counter() - t0)

    assert actions is not None
    return times, actions


def main() -> int:
    args = _parse_args()
    set_seed_everywhere(args.seed)
    device = torch.device(args.device)

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f'ERROR: config not found: {cfg_path}', file=sys.stderr)
        return 1

    cfg = Config.fromfile(str(cfg_path))
    if 'model' not in cfg:
        print('ERROR: config 中缺少 `model` 字段。', file=sys.stderr)
        return 1

    model_cfg = cfg.model.to_dict()
    ckpt_path = None if args.no_weights else _resolve_ckpt(args)
    if ckpt_path is None and not args.no_weights:
        if args.ckpt or args.run_dir:
            print('ERROR: 未找到 safetensors，请显式指定 --ckpt。', file=sys.stderr)
            return 1
        print('WARN: 未指定权重，将使用 --no-weights 等价行为（随机初始化）。')

    if ckpt_path is not None:
        model_cfg['pretrained_name_or_path'] = None
        model_cfg['name_mapping'] = None

    print('Building PI05FlowMatching...')
    vla = build_vla_from_cfg(model_cfg)
    vla = vla.to(device)
    if device.type == 'cuda' and hasattr(vla, 'to_bfloat16'):
        vla.to_bfloat16()

    if ckpt_path is not None:
        if not ckpt_path.is_file():
            print(
                f'ERROR: checkpoint 不是有效文件: {ckpt_path}\n'
                '  请传完整路径，例如: .../checkpoints/step-xxx.safetensors\n'
                '  或传训练根目录（内含 checkpoints/*.safetensors）。',
                file=sys.stderr,
            )
            return 1
        print(f'Loading weights: {ckpt_path}')
        state = load_file(str(ckpt_path), device='cpu')
        missing, unexpected = vla.load_state_dict(
            state, strict=bool(args.strict_load))
        if not args.strict_load:
            print(f'load_state_dict(strict=False): missing={len(missing)}, '
                  f'unexpected={len(unexpected)}')
            if missing[:5]:
                print('  sample missing:', missing[:5])
            if unexpected[:5]:
                print('  sample unexpected:', unexpected[:5])
    vla.eval()

    if device.type == 'cuda':
        print('CUDA:', torch.cuda.get_device_name(device))
    else:
        print('Device: CPU')

    vocab_size = int(getattr(vla.llm_backbone.config, 'vocab_size', 257152))
    batch = _make_dummy_batch(
        device=device,
        batch_size=args.batch_size,
        num_views=args.num_views,
        image_size=args.image_size,
        lang_len=args.lang_len,
        max_action_dim=int(vla.max_action_dim),
        n_action_steps=int(vla.n_action_steps),
        vocab_size=vocab_size,
    )

    n_runs = max(1, int(args.predict_runs))
    n_warmup = max(0, int(args.warmup))
    print(
        f'Benchmark predict_action: warmup={n_warmup}, runs={n_runs} '
        f'(dummy tensors)...', )
    times_sec, actions = _bench_predict_action(
        vla,
        batch,
        device,
        warmup=n_warmup,
        runs=n_runs,
    )
    ms = [t * 1000.0 for t in times_sec]
    print('predict_action output shape:', tuple(actions.shape))
    print(
        f'latency_ms: min={min(ms):.3f} max={max(ms):.3f} '
        f'mean={statistics.mean(ms):.3f} '
        f'median={statistics.median(ms):.3f}',
        end='',
    )
    if len(ms) > 1:
        print(f' stdev={statistics.stdev(ms):.3f}')
    else:
        print()
    total = sum(times_sec)
    print(
        f'total_wall_predict={total*1000:.3f} ms '
        f'({n_runs} runs, excl. warmup)', )

    print('Running embed_prefix + forward_model slice (dummy tensors)...')
    with torch.inference_mode():
        with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == 'cuda'):
            _embed_prefix_forward_only(vla, batch, device)
    print('embed_prefix forward path OK.')

    print('OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
