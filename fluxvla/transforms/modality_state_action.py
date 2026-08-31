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
"""Metadata helpers used by GR00T N1.7 transforms."""

from __future__ import annotations
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

GROOT_N17_EMBODIMENT_ALIASES = {
    'LIBERO_PANDA': 'libero_sim',
    'libero_sim': 'libero_sim',
}

GROOT_N17_VALIDATED_DEFAULT_EMBODIMENT_IDS = {'libero_sim': 2}


def resolve_groot_n17_embodiment_key(embodiment_tag: Optional[str] = None,
                                     env_name: Optional[str] = None) -> str:
    """Resolve a public or environment embodiment name to a metadata key."""
    value = env_name.split('/', 1)[0] if env_name else embodiment_tag
    if value is None:
        raise ValueError('An N1.7 embodiment tag or environment is required.')
    return GROOT_N17_EMBODIMENT_ALIASES.get(
        value,
        GROOT_N17_EMBODIMENT_ALIASES.get(
            str(value).lower(),
            str(value).lower()))


def select_groot_n17_metadata(
        processor_kwargs: Dict[str, Any],
        statistics: Dict[str, Any],
        embodiment_id_mapping: Optional[Dict[str, int]],
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
        require_statistics: bool = True) -> Dict[str, Any]:
    """Select one embodiment from checkpoint-owned N1.7 metadata."""
    embodiment_key = resolve_groot_n17_embodiment_key(embodiment_tag, env_name)
    modality_configs = processor_kwargs.get('modality_configs', {})
    if embodiment_key not in modality_configs:
        raise KeyError(f'No checkpoint modality config for {embodiment_key!r}')

    selected_statistics = statistics.get(embodiment_key)
    if require_statistics and selected_statistics is None:
        raise KeyError(f'No checkpoint statistics for {embodiment_key!r}')

    ids = dict(embodiment_id_mapping or {})
    if embodiment_key in ids:
        embodiment_id = int(ids[embodiment_key])
        embodiment_id_source = 'checkpoint'
    elif embodiment_key in GROOT_N17_VALIDATED_DEFAULT_EMBODIMENT_IDS:
        embodiment_id = GROOT_N17_VALIDATED_DEFAULT_EMBODIMENT_IDS[
            embodiment_key]
        embodiment_id_source = 'validated_default'
    else:
        raise KeyError(f'No checkpoint embodiment id for {embodiment_key!r}')

    return {
        'embodiment_key': embodiment_key,
        'embodiment_id': embodiment_id,
        'embodiment_id_source': embodiment_id_source,
        'modality_config': modality_configs[embodiment_key],
        'modality_source': 'checkpoint',
        'statistics': selected_statistics,
    }


def resolve_groot_n17_metadata(
        pretrained_model_name_or_path: Optional[str | Path] = None,
        embodiment_tag: Optional[str] = None,
        env_name: Optional[str] = None,
        require_statistics: bool = True,
        **kwargs) -> Dict[str, Any]:
    """Load checkpoint metadata and select one embodiment."""
    processor_kwargs = load_groot_n17_metadata(pretrained_model_name_or_path,
                                               **kwargs)
    selected = select_groot_n17_metadata(
        processor_kwargs,
        processor_kwargs.get('statistics', {}),
        processor_kwargs.get('embodiment_id_mapping'),
        embodiment_tag=embodiment_tag,
        env_name=env_name,
        require_statistics=require_statistics)
    selected['processor_kwargs'] = processor_kwargs
    return selected


def load_groot_n17_metadata(
        pretrained_model_name_or_path: Optional[str | Path] = None,
        **kwargs) -> Dict[str, Any]:
    """Load official N1.7 processor metadata without building HF processors."""
    kwargs = dict(kwargs)
    inline_statistics = kwargs.pop('statistics', None)
    inline_embodiment_ids = kwargs.pop('embodiment_id_mapping', None)
    modality_configs = kwargs.pop('modality_configs', {})
    if pretrained_model_name_or_path is None:
        # With no metadata directory, the config is the metadata source. Keep
        # every processor option instead of filtering it through the small
        # override allowlist used for checkpoint-backed metadata below.
        processor_kwargs = deepcopy(kwargs)
        kwargs = {}
        statistics = inline_statistics or {}
        embodiment_id_mapping = inline_embodiment_ids
    else:
        root = Path(pretrained_model_name_or_path)
        with open(root / 'processor_config.json', 'r') as f:
            config = json.load(f)
        with open(root / 'statistics.json', 'r') as f:
            statistics = json.load(f)
        embodiment_file = root / 'embodiment_id.json'
        embodiment_id_mapping = None
        if os.path.exists(embodiment_file):
            with open(embodiment_file, 'r') as f:
                embodiment_id_mapping = json.load(f)
        processor_kwargs = deepcopy(config['processor_kwargs'])
        if inline_statistics is not None:
            statistics = inline_statistics
        if inline_embodiment_ids is not None:
            embodiment_id_mapping = inline_embodiment_ids

    processor_kwargs['statistics'] = statistics
    processor_kwargs['embodiment_id_mapping'] = embodiment_id_mapping
    processor_kwargs.setdefault('model_type', 'qwen')
    processor_kwargs.setdefault('clip_outliers', True)

    processor_kwargs.setdefault('modality_configs', {})
    for embodiment_tag, modality_config in modality_configs.items():
        processor_kwargs['modality_configs'][embodiment_tag] = modality_config
    for key in (
            'random_rotation_angle',
            'color_jitter_params',
            'use_relative_action',
            'exclude_state',
            'state_dropout_prob',
            'model_name',
            'model_type',
            'max_action_horizon',
            'max_state_dim',
            'max_action_dim',
    ):
        if key in kwargs and kwargs[key] is not None:
            processor_kwargs[key] = kwargs[key]
    return processor_kwargs
