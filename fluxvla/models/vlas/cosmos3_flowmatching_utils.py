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

from __future__ import annotations
import copy
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..third_party_models.cosmos3.data.vfm.sequence_packing import \
    PackedSequence

_DEFAULT_RECTIFIED_FLOW_TRAINING_CONFIG = dict(
    shift={
        '256': 3,
        '480': 5,
        '720': 10,
    },
    use_dynamic_shift=False,
    train_time_image_distribution='logitnormal',
    train_time_video_distribution='waver',
    train_time_action_distribution='logitnormal',
    train_time_weight='uniform',
    vision_loss_weight=1.0,
    independent_action_schedule=False,
    shift_action=None,
    use_high_sigma_strategy=False,
    high_sigma_ratio=0.05,
    high_sigma_timesteps_min=995,
    high_sigma_timesteps_max=1000,
    use_high_sigma_strategy_action=False,
    use_discrete_rf=False,
    normalize_loss_by_active=False,
    action_loss_weight=10.0,
)

_DEFAULT_RECTIFIED_FLOW_INFERENCE_CONFIG = dict(
    num_train_timesteps=1000,
    scheduler_type='unipc',
    num_steps=10,
    shift=1.0,
    use_dynamic_shifting=False,
    dynamic_shift_mu=None,
    use_karras_sigmas=False,
    sigma_max=80.0,
    sigma_min=0.002,
    rho=7.0,
)


def _get_vision_data_resolution(spatial_shape: tuple[int, int]) -> str:
    min_dim = min(spatial_shape)
    if min_dim <= 256:
        return '256'
    if min_dim <= 640:
        return '480'
    if min_dim <= 960:
        return '720'
    raise ValueError(f'Unsupported Cosmos3 vision resolution: {spatial_shape}')


def _move_value_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    return value


def _move_packed_sequence_to_device(packed_seq: PackedSequence,
                                    device: torch.device) -> PackedSequence:
    for field_name in (
            'text_ids',
            'text_indexes',
            'position_ids',
            'label_ids',
            'ce_loss_indexes',
            'ce_loss_weights',
    ):
        value = getattr(packed_seq, field_name)
        setattr(packed_seq, field_name, _move_value_to_device(value, device))

    for modality_name in ('vision', 'action', 'sound'):
        modality = getattr(packed_seq, modality_name)
        if modality is None:
            continue
        for field_name in (
                'sequence_indexes',
                'timesteps',
                'mse_loss_indexes',
                'tokens',
                'condition_mask',
                'noisy_frame_indexes',
                'domain_id',
                'raw_action_dim',
        ):
            setattr(
                modality, field_name,
                _move_value_to_device(getattr(modality, field_name), device))
    return packed_seq


def _expand_sampler_timestep(timestep, *, batch_size: int,
                             device: torch.device) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        values = timestep.detach().to(
            device=device, dtype=torch.float32).flatten()
        if values.numel() == 1:
            return values.expand(batch_size)
        if values.numel() == batch_size:
            return values
        raise ValueError(f'Expected 1 or {batch_size} sampler timesteps, '
                         f'got {values.numel()}.')
    return torch.full(
        (batch_size, ),
        float(timestep),
        device=device,
        dtype=torch.float32,
    )


def _as_list_of_1chw(
        latents: torch.Tensor | List[torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(latents, torch.Tensor):
        if latents.dim() == 4:
            return [latents.unsqueeze(0)]
        if latents.dim() != 5:
            raise ValueError(
                f'latents must have shape [B,C,T,H,W] or [C,T,H,W], '
                f'got {latents.shape}')
        return [latents[i:i + 1] for i in range(latents.shape[0])]

    out = []
    for latent in latents:
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)
        if latent.dim() != 5:
            raise ValueError(
                f'each latent must have shape [1,C,T,H,W] or [C,T,H,W], '
                f'got {latent.shape}')
        out.append(latent)
    return out


def _as_action_list(
        actions: torch.Tensor | List[torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(actions, torch.Tensor):
        if actions.dim() == 2:
            return [actions]
        if actions.dim() != 3:
            raise ValueError(f'actions must have shape [B,T,D] or [T,D], '
                             f'got {actions.shape}')
        return [actions[i] for i in range(actions.shape[0])]
    return actions


def _as_long_list(
    values,
    batch_size: int,
    device: torch.device,
    default_value: int = 0,
) -> List[torch.Tensor]:
    if values is None:
        value = torch.tensor([default_value], dtype=torch.long, device=device)
        return [value.clone() for _ in range(batch_size)]
    if isinstance(values, torch.Tensor):
        values = values.to(device=device, dtype=torch.long).view(-1)
        if values.numel() == 1 and batch_size > 1:
            values = values.expand(batch_size)
        if values.numel() != batch_size:
            raise ValueError(f'Expected 1 or {batch_size} values, got '
                             f'{values.numel()}.')
        return [values[i:i + 1] for i in range(batch_size)]
    if isinstance(values, (int, float)):
        value = torch.tensor([values], dtype=torch.long, device=device)
        return [value.clone() for _ in range(batch_size)]
    if len(values) not in (1, batch_size):
        raise ValueError(f'Expected 1 or {batch_size} values, got '
                         f'{len(values)}.')
    if len(values) == 1 and batch_size > 1:
        values = list(values) * batch_size
    out = []
    for value in values:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value], dtype=torch.long, device=device)
        out.append(value.to(device=device, dtype=torch.long).view(1))
    return out


def _strip_right_padding(row: List[int],
                         pad_token_id: Optional[int]) -> List[int]:
    if pad_token_id is None:
        return row
    while row and row[-1] == pad_token_id:
        row.pop()
    return row


def _as_text_ids(
    text_token_ids,
    batch_size: int,
    pad_token_id: Optional[int] = None,
) -> List[List[int]]:
    if text_token_ids is None:
        return [[] for _ in range(batch_size)]
    if isinstance(text_token_ids, torch.Tensor):
        if text_token_ids.dim() == 1:
            text_token_ids = text_token_ids.unsqueeze(0)
        if text_token_ids.shape[0] == 1 and batch_size > 1:
            text_token_ids = text_token_ids.expand(batch_size, -1)
        if text_token_ids.shape[0] != batch_size:
            raise ValueError(f'Expected 1 or {batch_size} text rows, got '
                             f'{text_token_ids.shape[0]}.')
        return [
            _strip_right_padding(
                [int(token) for token in row.detach().cpu().tolist()],
                pad_token_id,
            ) for row in text_token_ids
        ]
    if len(text_token_ids) not in (1, batch_size):
        raise ValueError(f'Expected 1 or {batch_size} text rows, got '
                         f'{len(text_token_ids)}.')
    if len(text_token_ids) == 1 and batch_size > 1:
        text_token_ids = list(text_token_ids) * batch_size
    return [
        _strip_right_padding([int(token) for token in row], pad_token_id)
        for row in text_token_ids
    ]


def _arch_invariant_randn(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Mirror cosmos_framework.utils.misc.arch_invariant_rand."""
    random_array = np.random.RandomState(
        int(seed)).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(device=device, dtype=dtype)


def _sample_arch_invariant_noise(
    shape: tuple[int, ...],
    batch_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: Optional[int | List[int]],
    label: str,
) -> torch.Tensor:
    if seed is None:
        return torch.randn((batch_size, *shape), device=device, dtype=dtype)

    seeds = seed if isinstance(seed, list) else [int(seed)]
    if len(seeds) == 1 and batch_size > 1:
        seeds = seeds * batch_size
    if len(seeds) != batch_size:
        raise ValueError(f'Expected 1 or {batch_size} {label} seeds, '
                         f'got {len(seeds)}.')
    return torch.stack(
        [
            _arch_invariant_randn(
                shape,
                dtype=dtype,
                device=device,
                seed=sample_seed,
            ) for sample_seed in seeds
        ],
        dim=0,
    )


def _apply_timestep_embeds_to_noisy_tokens(
    packed_tokens: torch.Tensor,
    packed_timestep_embeds: torch.Tensor,
    noisy_frame_indexes: List[torch.Tensor],
    token_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    if packed_timestep_embeds.numel() == 0:
        return packed_tokens

    start_noisy_index = 0
    flattened_noisy_frame_indexes = []
    for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes,
                                              token_shapes):
        spatial_numel_i = math.prod(token_shape_i[1:])
        spatial_indexes_i = torch.arange(
            spatial_numel_i, device=packed_tokens.device)
        noisy_indexes_i = noisy_indexes_i.to(
            device=packed_tokens.device, dtype=torch.long)
        noisy_indexes_i = (noisy_indexes_i *
                           spatial_numel_i).unsqueeze(-1).expand(
                               -1, spatial_numel_i)
        noisy_indexes_i = noisy_indexes_i.clone(
        ) + spatial_indexes_i + start_noisy_index
        flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())
        start_noisy_index += math.prod(token_shape_i)

    if not flattened_noisy_frame_indexes:
        return packed_tokens

    flattened_noisy_frame_indexes = torch.cat(
        flattened_noisy_frame_indexes, dim=0)
    if flattened_noisy_frame_indexes.numel() == 0:
        return packed_tokens

    flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(
        -1).expand(-1, packed_tokens.shape[1])
    return packed_tokens.scatter_add(
        dim=0,
        index=flattened_noisy_frame_indexes,
        src=packed_timestep_embeds,
    )


def merge_config(defaults: Dict, config: Optional[Dict]) -> Dict:
    merged = copy.deepcopy(defaults)
    if config is not None:
        merged.update(copy.deepcopy(config))
    return merged


def resolve_torch_dtype(
        dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f'Unsupported torch_dtype: {dtype}')
    return mapping[dtype]


def resolve_training_config(config: Optional[Dict]) -> Dict:
    merged = merge_config(_DEFAULT_RECTIFIED_FLOW_TRAINING_CONFIG, config)
    if config is not None and 'train_time_vision_distribution' in config:
        if 'train_time_video_distribution' not in config:
            merged['train_time_video_distribution'] = config[
                'train_time_vision_distribution']
    merged['train_time_vision_distribution'] = merged[
        'train_time_video_distribution']
    supported_distributions = {'uniform', 'logitnormal', 'waver'}
    for key in (
            'train_time_image_distribution',
            'train_time_video_distribution',
            'train_time_action_distribution',
    ):
        if merged[key] not in supported_distributions:
            raise ValueError(
                f'Unsupported {key}={merged[key]!r}; expected one of '
                f'{sorted(supported_distributions)}.')
    if merged['train_time_weight'] == 'reweighting':
        merged['train_time_weight'] = 'uniform'
    if merged['train_time_weight'] != 'uniform':
        raise ValueError(
            'Cosmos3FlowMatching follows the original Cosmos3 RF loss path, '
            'which supports train_time_weight="uniform" ("reweighting" is '
            'treated as uniform for checkpoint compatibility).')
    if merged['use_discrete_rf']:
        raise ValueError('Cosmos3FlowMatching does not support discrete RF.')
    return merged


def resolve_inference_config(config: Optional[Dict]) -> Dict:
    merged = merge_config(_DEFAULT_RECTIFIED_FLOW_INFERENCE_CONFIG, config)
    merged['scheduler_type'] = str(merged['scheduler_type']).lower()
    if merged['scheduler_type'] != 'unipc':
        raise ValueError(
            'Cosmos3FlowMatching supports scheduler_type="unipc" only, '
            f'got {merged["scheduler_type"]!r}.')
    return merged


def read_json_if_exists(path: str | Path) -> Optional[Dict]:
    path = Path(path)
    if not path.exists():
        return None
    import json
    with open(path, encoding='utf-8') as reader:
        return json.load(reader)
