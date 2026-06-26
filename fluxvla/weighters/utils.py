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
"""Shared helpers for RA-BC / AW-BC sample weighters."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def as_scalar(value: Any) -> Any:
    """Return a Python scalar from common tensor and array containers."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.flatten()[0].item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return as_scalar(value[0])
    return value


def resolve_sample_index(
    sample: Mapping[str, Any],
    index_key: str,
) -> int | None:
    """Resolve a global frame index from a dataset sample dict."""
    index = as_scalar(sample.get(index_key))
    if index is None:
        index = as_scalar(sample.get('current_index'))
    if index is None:
        return None
    return int(index)


def resolve_rabc_progress_path(path: str | Path) -> Path:
    """Resolve a local or ``hf://datasets/...`` RA-BC progress path."""
    path_str = str(path)
    if not path_str.startswith('hf://datasets/'):
        return Path(path).expanduser()

    parts = path_str.replace('hf://datasets/', '').split('/')
    if len(parts) < 3:
        raise ValueError(
            'Expected hf://datasets/<namespace>/<repo>/<filename>, got '
            f'{path_str!r}')
    repo_id = '/'.join(parts[:2])
    filename = '/'.join(parts[2:])
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type='dataset',
        ))


resolve_arm_progress_path = resolve_rabc_progress_path
