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

"""Build a TensorRT engine for the FluxVLA DreamZero cached DiT path.

This follows the official DreamZero TensorRT strategy: export the Wan DiT
cached denoise forward, keep task-specific video/action shapes fixed, and make
only ``kv_cache_packed.shape[3]`` dynamic.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class DreamZeroTrtShapeSpec:
    batch_size: int
    video_channels: int
    cond_channels: int
    frames_per_forward: int
    latent_height: int
    latent_width: int
    action_horizon: int
    action_dim: int
    state_tokens: int
    state_dim: int
    context_tokens: int
    context_dim: int
    clip_tokens: int
    clip_dim: int
    num_layers: int
    kv_slots: int
    num_heads: int
    head_dim: int
    frame_seqlen: int
    min_cache_len: int
    opt_cache_len: int
    max_cache_len: int

    @property
    def min_shape_arg(self) -> str:
        return self._shape_arg(self.min_cache_len)

    @property
    def opt_shape_arg(self) -> str:
        return self._shape_arg(self.opt_cache_len)

    @property
    def max_shape_arg(self) -> str:
        return self._shape_arg(self.max_cache_len)

    def _shape_arg(self, cache_len: int) -> str:
        return (
            'kv_cache_packed:'
            f'{self.num_layers}x{self.kv_slots}x{self.batch_size}x'
            f'{cache_len}x{self.num_heads}x{self.head_dim}')


def _cfg_get(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return default


def build_shape_spec(
    head_cfg: Mapping[str, Any],
    *,
    batch_size: int = 1,
    latent_height: int = 32,
    latent_width: int = 16,
    min_cache_frames: int = 1,
    opt_cache_frames: int = 3,
    max_cache_frames: Optional[int] = None,
    context_tokens: int = 512,
    context_dim: int = 4096,
    clip_tokens: int = 257,
    clip_dim: int = 1280,
) -> DreamZeroTrtShapeSpec:
    frame_seqlen = int(_cfg_get(head_cfg, 'frame_seqlen', 880))
    inferred_frame_seqlen = latent_height * latent_width // 4
    if inferred_frame_seqlen != frame_seqlen:
        raise ValueError(
            'latent_height * latent_width / 4 must match frame_seqlen: '
            f'{latent_height} * {latent_width} / 4 = '
            f'{inferred_frame_seqlen}, frame_seqlen={frame_seqlen}')

    num_frame_per_block = int(_cfg_get(head_cfg, 'num_frame_per_block', 2))
    max_chunk_size = int(_cfg_get(head_cfg, 'max_chunk_size', -1))
    if max_cache_frames is None:
        if max_chunk_size != -1:
            max_cache_frames = max_chunk_size * num_frame_per_block + 1
        else:
            max_cache_frames = int(_cfg_get(head_cfg, 'num_frames', 1))
    max_cache_frames = max(min_cache_frames, max_cache_frames)
    opt_cache_frames = min(max(opt_cache_frames, min_cache_frames),
                           max_cache_frames)

    dit_dim = int(_cfg_get(head_cfg, 'dit_dim', 5120))
    num_heads = int(_cfg_get(head_cfg, 'dit_num_heads', 40))
    if dit_dim % num_heads != 0:
        raise ValueError(
            f'dit_dim={dit_dim} must be divisible by num_heads={num_heads}')

    video_channels = int(_cfg_get(head_cfg, 'dit_out_dim', 16))
    dit_in_dim = int(_cfg_get(head_cfg, 'dit_in_dim', 36))
    cond_channels = dit_in_dim - video_channels
    if cond_channels <= 0:
        raise ValueError(
            f'dit_in_dim={dit_in_dim} must be larger than '
            f'dit_out_dim={video_channels}')

    return DreamZeroTrtShapeSpec(
        batch_size=batch_size,
        video_channels=video_channels,
        cond_channels=cond_channels,
        frames_per_forward=num_frame_per_block,
        latent_height=latent_height,
        latent_width=latent_width,
        action_horizon=int(_cfg_get(head_cfg, 'action_horizon', 10)),
        action_dim=int(_cfg_get(head_cfg, 'max_action_dim', 32)),
        state_tokens=int(_cfg_get(head_cfg, 'num_state_per_block', 1)),
        state_dim=int(_cfg_get(head_cfg, 'max_state_dim', 64)),
        context_tokens=context_tokens,
        context_dim=context_dim,
        clip_tokens=clip_tokens,
        clip_dim=clip_dim,
        num_layers=int(_cfg_get(head_cfg, 'dit_num_layers', 40)),
        kv_slots=2,
        num_heads=num_heads,
        head_dim=dit_dim // num_heads,
        frame_seqlen=frame_seqlen,
        min_cache_len=min_cache_frames * frame_seqlen,
        opt_cache_len=opt_cache_frames * frame_seqlen,
        max_cache_len=max_cache_frames * frame_seqlen,
    )


def build_trtexec_command(
    *,
    trtexec: str,
    onnx_path: str,
    engine_path: str,
    spec: DreamZeroTrtShapeSpec,
    workspace_mib: int = 65536,
    legacy_precision_flags: bool = False,
    verbose: bool = True,
) -> list[str]:
    cmd = [
        trtexec,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        '--separateProfileRun',
        '--profilingVerbosity=detailed',
        f'--memPoolSize=workspace:{workspace_mib}',
        '--dumpProfile',
        '--dumpLayerInfo',
        '--useCudaGraph',
        f'--minShapes={spec.min_shape_arg}',
        f'--optShapes={spec.opt_shape_arg}',
        f'--maxShapes={spec.max_shape_arg}',
    ]
    if legacy_precision_flags:
        cmd.append('--fp16')
        cmd.append('--bf16')
    if verbose:
        cmd.append('--verbose')
    return cmd


def _load_checkpoint_state_dict(path: str):
    import torch
    from safetensors.torch import load_file

    if os.path.isdir(path):
        state_dict = {}
        for item in sorted(os.listdir(path)):
            if item.endswith('.safetensors'):
                state_dict.update(load_file(os.path.join(path, item),
                                            device='cpu'))
        if not state_dict:
            raise FileNotFoundError(f'No .safetensors files found in {path}')
        return state_dict
    if path.endswith('.safetensors'):
        return load_file(path, device='cpu')

    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def _extract_head_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    extracted = {}
    prefixes = (
        'module.vla_head.',
        'vla_head.',
        'module.action_head.',
        'action_head.',
    )
    for key, value in state_dict.items():
        normalized = key
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        if normalized.startswith('model.'):
            extracted[normalized] = value
    if not extracted:
        raise ValueError(
            'No DreamZero DiT weights found. Expected keys starting with '
            'vla_head.model.*, action_head.model.*, or model.*.')
    return extracted


def _make_dummy_inputs(spec: DreamZeroTrtShapeSpec, *, device: str, dtype):
    import torch

    b = spec.batch_size
    t = spec.frames_per_forward
    x = torch.randn(
        b,
        spec.video_channels,
        t,
        spec.latent_height,
        spec.latent_width,
        dtype=dtype,
        device=device,
    )
    timestep = torch.randn(b, t, dtype=dtype, device=device)
    context = torch.randn(
        b,
        spec.context_tokens,
        spec.context_dim,
        dtype=dtype,
        device=device,
    )
    kv_cache_packed = torch.zeros(
        spec.num_layers,
        spec.kv_slots,
        b,
        spec.opt_cache_len,
        spec.num_heads,
        spec.head_dim,
        dtype=dtype,
        device=device,
    )
    y = torch.randn(
        b,
        spec.cond_channels,
        t,
        spec.latent_height,
        spec.latent_width,
        dtype=dtype,
        device=device,
    )
    clip_feature = torch.randn(
        b,
        spec.clip_tokens,
        spec.clip_dim,
        dtype=dtype,
        device=device,
    )
    action = torch.randn(
        b,
        spec.action_horizon,
        spec.action_dim,
        dtype=dtype,
        device=device,
    )
    timestep_action = torch.randn(
        b, spec.action_horizon, dtype=dtype, device=device)
    state = torch.randn(
        b,
        spec.state_tokens,
        spec.state_dim,
        dtype=dtype,
        device=device,
    )
    return (
        x,
        timestep,
        context,
        kv_cache_packed,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    )


def _load_head_cfg(config_path: str, cfg_options):
    from mmengine import Config

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    return cfg.model.vla_head.copy()


def _build_head_from_config(config_path: str, cfg_options):
    os.environ.setdefault('ENABLE_TENSORRT', 'true')
    os.environ.setdefault('ATTENTION_BACKEND', 'TE')

    from fluxvla.engines import build_head_from_cfg
    import fluxvla.models.heads  # noqa: F401

    head_cfg = _load_head_cfg(config_path, cfg_options)
    return build_head_from_cfg(head_cfg).eval(), head_cfg


def _export_onnx(
    *,
    head,
    spec: DreamZeroTrtShapeSpec,
    onnx_path: str,
    device: str,
    opset: int,
) -> None:
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    wan_model = head.model.eval().to(device=device, dtype=torch.float16)
    wan_model.forward = wan_model._forward_inference_trt
    test_inputs = _make_dummy_inputs(
        spec, device=device, dtype=torch.float16)

    input_names = [
        'x',
        'timestep',
        'context',
        'kv_cache_packed',
        'y',
        'clip_feature',
        'action',
        'timestep_action',
        'state',
    ]
    output_names = ['video_noise_pred', 'action_noise_pred']
    dynamic_axes = {'kv_cache_packed': {3: 'kv_cache_len'}}

    with torch.no_grad():
        torch.onnx.export(
            wan_model,
            test_inputs,
            onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


def run(args: argparse.Namespace) -> None:
    if args.device.startswith('cuda') and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    head_cfg = _load_head_cfg(args.config, args.cfg_options)
    spec = build_shape_spec(
        head_cfg,
        batch_size=args.batch_size,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        min_cache_frames=args.min_cache_frames,
        opt_cache_frames=args.opt_cache_frames,
        max_cache_frames=args.max_cache_frames,
        context_tokens=args.context_tokens,
        context_dim=args.context_dim,
        clip_tokens=args.clip_tokens,
        clip_dim=args.clip_dim,
    )

    print('DreamZero TRT shape profile:')
    print(f'  minShapes={spec.min_shape_arg}')
    print(f'  optShapes={spec.opt_shape_arg}')
    print(f'  maxShapes={spec.max_shape_arg}')

    if args.print_shapes_only:
        return

    head, _ = _build_head_from_config(args.config, args.cfg_options)

    if args.ckpt_path is not None:
        state_dict = _load_checkpoint_state_dict(args.ckpt_path)
        head_state = _extract_head_state_dict(state_dict)
        missing, unexpected = head.load_state_dict(head_state, strict=False)
        print(f'Loaded head checkpoint: {args.ckpt_path}')
        print(f'  matched tensors: {len(head_state)}')
        print(f'  missing tensors: {len(missing)}')
        print(f'  unexpected tensors: {len(unexpected)}')
        if args.strict_load and (missing or unexpected):
            raise RuntimeError('Checkpoint did not strictly match DreamZeroHead')
        del state_dict
        del head_state
        gc.collect()

    _export_onnx(
        head=head,
        spec=spec,
        onnx_path=args.onnx_path,
        device=args.device,
        opset=args.opset,
    )
    print(f'ONNX exported: {args.onnx_path}')

    if args.export_only:
        return

    trtexec = args.trtexec or shutil.which('trtexec')
    if trtexec is None:
        raise FileNotFoundError(
            'trtexec was not found. Set PATH or pass --trtexec.')
    cmd = build_trtexec_command(
        trtexec=trtexec,
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        spec=spec,
        workspace_mib=args.workspace_mib,
        legacy_precision_flags=args.legacy_precision_flags,
        verbose=not args.no_verbose,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.engine_path)),
                exist_ok=True)
    log_path = args.build_log or args.engine_path.replace('.trt',
                                                          '_build.log')
    print('Running trtexec:')
    print('  ' + ' '.join(cmd))
    print(f'Build log: {log_path}')
    with open(log_path, 'w') as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f'trtexec failed with return code {result.returncode}; '
            f'see {log_path}')
    print(f'TensorRT engine built: {args.engine_path}')


def parse_args() -> argparse.Namespace:
    from mmengine import DictAction

    parser = argparse.ArgumentParser(
        description='Build FluxVLA DreamZero-LIBERO TensorRT engine.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--onnx-path', required=True)
    parser.add_argument('--engine-path', required=True)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cuda-device', default='0')
    parser.add_argument('--trtexec', default=None)
    parser.add_argument('--build-log', default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--latent-height', type=int, default=32)
    parser.add_argument('--latent-width', type=int, default=16)
    parser.add_argument('--min-cache-frames', type=int, default=1)
    parser.add_argument('--opt-cache-frames', type=int, default=3)
    parser.add_argument('--max-cache-frames', type=int, default=None)
    parser.add_argument('--context-tokens', type=int, default=512)
    parser.add_argument('--context-dim', type=int, default=4096)
    parser.add_argument('--clip-tokens', type=int, default=257)
    parser.add_argument('--clip-dim', type=int, default=1280)
    parser.add_argument('--opset', type=int, default=20)
    parser.add_argument('--workspace-mib', type=int, default=65536)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--strict-load', action='store_true')
    parser.add_argument(
        '--legacy-precision-flags',
        action='store_true',
        help=('Append legacy --fp16/--bf16 trtexec flags. Do not use this '
              'with TensorRT 11, where strongly typed networks are default.'))
    parser.add_argument('--no-verbose', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--print-shapes-only', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
