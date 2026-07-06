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
# Preprocess the ActionDiT backbone for FastWAM training by linearly
# interpolating the Wan2.2 video DiT weights down to the ActionDiT hidden
# size, then saving a ``.pt`` payload consumed by
# ``ActionDiT.from_pretrained``.
#
# This is the FluxVLA equivalent of the upstream FastWAM
# ``scripts/preprocess_action_dit_backbone.py``. The numerical interpolation
# (sequential 1-D linear ``align_corners=True`` resize + ``alpha=sqrt(dv/da)``
# scaling on resized last dims) is ported verbatim so the produced backbone is
# bit-for-bit identical to the upstream ``ActionDiT_linear_interp_*.pt``. The
# only difference is that the video/action DiT configs are read from a FluxVLA
# Python config (``model.vla_head.{video,action}_dit_config``) instead of a
# Hydra YAML.

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from mmengine.config import Config

from fluxvla.models.third_party_models.fastwam.modules.action_dit import \
    ActionDiT  # noqa: E501
from fluxvla.models.third_party_models.fastwam.modules.helpers.loader import \
    load_wan22_ti2v_5b_components  # noqa: E501


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == 'float32':
        return torch.float32
    if value == 'float16':
        return torch.float16
    if value == 'bfloat16':
        return torch.bfloat16
    raise ValueError(f'Unsupported dtype: {name}. Expected one of: '
                     'float32, float16, bfloat16.')


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {'1', 'true', 'yes', 'y'}:
        return True
    if value in {'0', 'false', 'no', 'n'}:
        return False
    raise ValueError(f'Cannot parse bool value: {name}')


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(
        flat, size=new_size, mode='linear', align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor,
                            target_shape: Tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError('Cannot reduce tensor rank for resize: src shape='
                             f'{tuple(src.shape)}, target={target_shape}')
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError('Resize produced wrong shape for tensor. src='
                         f'{tuple(src.shape)}, target={target_shape}, '
                         f'got={tuple(out.shape)}')
    return out.to(dtype=src.dtype)


def _load_dit_configs(config_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Read the video/action DiT configs and backbone meta from a FluxVLA
    config's ``model.vla_head`` / ``model.vlm_backbone`` sections."""
    cfg = Config.fromfile(str(config_path))
    if 'model' not in cfg:
        raise ValueError(f'`{config_path}` has no top-level `model`.')
    head = cfg.model.get('vla_head')
    backbone = cfg.model.get('vlm_backbone') or {}
    if not head or 'video_dit_config' not in head \
            or 'action_dit_config' not in head:
        raise ValueError(
            '`model.vla_head` must contain `video_dit_config` and '
            '`action_dit_config`.')
    video_cfg = dict(head['video_dit_config'])
    action_cfg = dict(head['action_dit_config'])
    return video_cfg, action_cfg, dict(backbone)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Preprocess ActionDiT backbone weights from WanVideoDiT '
        'and save as a .pt payload (FluxVLA config input).')
    parser.add_argument(
        '--config',
        required=True,
        help='FluxVLA config path, e.g. '
        'configs/fastwam/fastwam_idm_libero_10_full_finetune.py')
    parser.add_argument(
        '--output',
        required=True,
        help='Output .pt path for the preprocessed ActionDiT backbone.')
    parser.add_argument(
        '--device', default='cpu', help='Device for loading / preprocessing.')
    parser.add_argument(
        '--dtype',
        default='float32',
        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument(
        '--apply-alpha-scaling',
        default='true',
        help='Apply alpha=sqrt(dv/da) when the last dim is resized '
        '(true/false). Default: true.')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)
    torch_dtype = _parse_dtype(args.dtype)

    video_cfg, action_cfg, backbone_cfg = _load_dit_configs(Path(args.config))
    redirect_common_files = bool(
        backbone_cfg.get('redirect_common_files', False))
    model_id = str(backbone_cfg.get('model_id', 'Wan-AI/Wan2.2-TI2V-5B'))
    tokenizer_model_id = str(
        backbone_cfg.get('tokenizer_model_id', 'Wan-AI/Wan2.1-T2V-1.3B'))

    int_fields = [
        'hidden_dim', 'action_dim', 'ffn_dim', 'num_layers', 'num_heads',
        'attn_head_dim', 'text_dim', 'freq_dim'
    ]
    for key in int_fields:
        action_cfg[key] = int(action_cfg[key])
    action_cfg['eps'] = float(action_cfg['eps'])

    print(f'[INFO] Loaded DiT configs from {args.config}. Preprocessing '
          f'ActionDiT backbone with dtype={torch_dtype} on device='
          f'{args.device}, apply_alpha_scaling={apply_alpha_scaling}.')
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
        dit_config=video_cfg,
    )
    video_expert = components.dit

    action_expert = ActionDiT(**action_cfg).to(
        device=args.device, dtype=torch_dtype)
    if int(action_cfg['num_heads']) != int(video_expert.num_heads):
        raise ValueError('ActionDiT `num_heads` must match video expert.')
    if int(action_cfg['attn_head_dim']) != int(video_expert.attn_head_dim):
        raise ValueError('ActionDiT `attn_head_dim` must match video expert.')
    if int(action_cfg['num_layers']) != int(len(video_expert.blocks)):
        raise ValueError('ActionDiT `num_layers` must match video expert.')

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict: Dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f'Key `{key}` not found in video expert state.')
        src = video_state[key]
        target = action_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 \
                    and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1]))**0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(
            dtype=target.dtype, device='cpu').contiguous()

    payload: Dict[str, Any] = {
        'policy': {
            'skip_prefixes': list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            'alpha_scaling': bool(apply_alpha_scaling),
            'interpolation': 'sequential_1d_linear_align_corners_true',
        },
        'backbone_state_dict': backbone_state_dict,
        'meta': {
            'hidden_dim': int(action_cfg['hidden_dim']),
            'ffn_dim': int(action_cfg['ffn_dim']),
            'num_layers': int(action_cfg['num_layers']),
            'num_heads': int(action_cfg['num_heads']),
            'attn_head_dim': int(action_cfg['attn_head_dim']),
            'text_dim': int(action_cfg['text_dim']),
            'freq_dim': int(action_cfg['freq_dim']),
            'eps': float(action_cfg['eps']),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    print('[INFO] Saved ActionDiT backbone payload to '
          f'{output_path} (copied={copied}, interpolated={interpolated}, '
          f'skipped={skipped}).')


if __name__ == '__main__':
    main()
