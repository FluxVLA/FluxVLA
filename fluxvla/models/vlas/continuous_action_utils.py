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
"""Shared helpers for VLM plus continuous-action-head policies."""

from typing import Dict, Mapping, Optional

import torch


def normalize_action_head_output(output) -> Dict:
    """Normalize tensor and mapping action-head outputs to a loss dict."""
    if torch.is_tensor(output):
        return {'loss': output, 'action_loss': output}
    if not isinstance(output, Mapping):
        raise TypeError('Action head must return a Tensor or mapping, got '
                        f'{type(output)!r}.')
    normalized = dict(output)
    if normalized.get('loss') is None:
        action_loss = normalized.get('action_loss')
        if action_loss is None:
            raise KeyError('Action head output must contain `loss` or '
                           '`action_loss`.')
        normalized['loss'] = action_loss
    return normalized


def add_auxiliary_losses(
    output: Mapping,
    auxiliary_losses: Mapping[str, torch.Tensor],
    loss_weights: Optional[Mapping[str, float]] = None,
) -> Dict:
    """Add backbone-owned objectives without hard-coding model names."""
    merged = dict(output)
    total_loss = merged.get('loss')
    if total_loss is None:
        raise KeyError('VLA output must contain `loss` before adding '
                       'auxiliary losses.')
    weights = loss_weights or {}
    for name, loss in auxiliary_losses.items():
        if loss is None:
            continue
        if not torch.is_tensor(loss):
            raise TypeError(f'Auxiliary loss {name!r} must be a Tensor, got '
                            f'{type(loss)!r}.')
        merged[name] = loss
        total_loss = total_loss + float(weights.get(name, 1.0)) * loss
    merged['loss'] = total_loss
    return merged


def repeat_batch_tensor(value, repeats: int, batch_size: int):
    """Repeat a batch in the same order as ``Tensor.repeat`` used before."""
    if value is None or repeats <= 1 or not torch.is_tensor(value):
        return value
    if value.ndim == 0 or value.shape[0] != batch_size:
        return value
    return value.repeat(repeats, *([1] * (value.ndim - 1)))


__all__ = [
    'add_auxiliary_losses', 'normalize_action_head_output',
    'repeat_batch_tensor'
]
