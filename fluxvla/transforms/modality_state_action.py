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
"""Metadata-driven state/action codec used by GR00T-style transforms."""

from __future__ import annotations
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

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


def resolve_groot_n17_flat_slices(
        modality_config: Dict[str, Any],
        statistics: Dict[str, Any],
        embodiment_key: str,
        modality: str,
        flat_layout: str = 'auto') -> Dict[str, tuple[int, int]]:
    """Resolve flat slices from checkpoint statistics for validated layouts."""
    layout = str(flat_layout).lower()
    if layout != 'auto':
        raise ValueError(
            f'Unsupported N1.7 flat layout: {flat_layout!r}. Expected '
            "'auto'.")
    if embodiment_key != 'libero_sim':
        raise ValueError('Automatic flat N1.7 layout is validated only for '
                         f"'libero_sim', got {embodiment_key!r}.")

    start = 0
    slices = {}
    for key in modality_config[modality]['modality_keys']:
        dim = _normalization_dim(statistics[modality][key])
        slices[key] = (start, start + dim)
        start += dim
    return slices


def normalize_tag_value(tag: Any) -> str:
    if hasattr(tag, 'value'):
        return str(tag.value)
    return str(tag)


def _normalization_dim(stats: Dict[str, Any]) -> int:
    for field in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
        if field in stats:
            return int(np.asarray(stats[field]).shape[-1])
    raise KeyError(f'No supported statistics fields in {sorted(stats)}')


def _apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    return np.concatenate([np.sin(values), np.cos(values)], axis=-1)


def _normalize_minmax(values: np.ndarray,
                      params: Dict[str, np.ndarray]) -> np.ndarray:
    min_vals = params['min']
    max_vals = params['max']
    normalized = np.zeros_like(values)
    mask = ~np.isclose(max_vals, min_vals)
    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask])
    normalized[..., mask] = 2 * normalized[..., mask] - 1
    return normalized


def _unnormalize_minmax(values: np.ndarray,
                        params: Dict[str, np.ndarray]) -> np.ndarray:
    min_vals = params['min']
    max_vals = params['max']
    return (np.clip(values, -1.0, 1.0) + 1.0) / 2.0 * (max_vals -
                                                       min_vals) + min_vals


def _normalize_meanstd(values: np.ndarray,
                       params: Dict[str, np.ndarray]) -> np.ndarray:
    return (values - params['mean']) / params['std']


def _unnormalize_meanstd(values: np.ndarray,
                         params: Dict[str, np.ndarray]) -> np.ndarray:
    return values * params['std'] + params['mean']


def _action_config_value(config: Dict[str, Any], key: str,
                         default: str) -> str:
    return str(config.get(key, default)).upper()


class ModalityStateActionCodec:
    """Per-modality state/action codec for metadata-driven VLA transforms.

    This owns normalization, relative action conversion, sin/cos state
    encoding, and eval-time action decode. It is intentionally independent
    from the legacy N1.7 processor facade.
    """

    def __init__(self,
                 modality_configs: Dict[str, Any],
                 statistics: Optional[Dict[str, Any]] = None,
                 use_percentiles: bool = False,
                 clip_outliers: bool = True,
                 apply_sincos_state_encoding: bool = False,
                 use_relative_action: bool = False):
        self.modality_configs = deepcopy(modality_configs)
        self.statistics: Dict[str, Any] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.norm_params: Dict[str, Any] = {}
        self.training = True
        if statistics is not None:
            self.set_statistics(statistics)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(self,
                       statistics: Dict[str, Any],
                       override: bool = False) -> None:
        for key, value in statistics.items():
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(value)
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        self.norm_params = {}
        for embodiment_tag, emb_stats in self.statistics.items():
            self.norm_params[embodiment_tag] = {}
            for modality in ('state', 'action'):
                if modality not in emb_stats:
                    continue
                self.norm_params[embodiment_tag][modality] = {}
                for key, stats in emb_stats[modality].items():
                    low_field, high_field = (('q01', 'q99')
                                             if self.use_percentiles else
                                             ('min', 'max'))
                    params = {
                        'min': np.asarray(stats[low_field]),
                        'max': np.asarray(stats[high_field]),
                        'mean': np.asarray(stats['mean']),
                        'std': np.asarray(stats['std']),
                        'dim': np.array(_normalization_dim(stats)),
                    }
                    self.norm_params[embodiment_tag][modality][key] = params
            action_cfg = self.modality_configs.get(embodiment_tag,
                                                   {}).get('action', {})
            for key, cfg in zip(
                    action_cfg.get('modality_keys') or [],
                    action_cfg.get('action_configs') or []):
                if (self.use_relative_action and _action_config_value(
                        cfg, 'rep', '') == 'RELATIVE'):
                    action_dim = self.norm_params[embodiment_tag]['action'][
                        key]['dim']
                    rel_stats = emb_stats['relative_action'][key]
                    self.norm_params[embodiment_tag]['action'][key] = {
                        'min': np.asarray(rel_stats['min']),
                        'max': np.asarray(rel_stats['max']),
                        'mean': np.asarray(rel_stats['mean']),
                        'std': np.asarray(rel_stats['std']),
                        'dim': action_dim,
                    }

    def _use_mean_std(self, embodiment_tag: str, modality: str,
                      key: str) -> bool:
        cfg = self.modality_configs[embodiment_tag][modality]
        keys = cfg.get('mean_std_embedding_keys')
        return bool(keys and key in keys)

    def _normalize(self, values: np.ndarray, embodiment_tag: str,
                   modality: str, key: str) -> np.ndarray:
        params = self.norm_params[embodiment_tag][modality][key]
        if self._use_mean_std(embodiment_tag, modality, key):
            normalized = _normalize_meanstd(values, params)
        else:
            normalized = _normalize_minmax(values, params)
        if self.clip_outliers:
            normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

    def _unnormalize(self, values: np.ndarray, embodiment_tag: str,
                     modality: str, key: str) -> np.ndarray:
        params = self.norm_params[embodiment_tag][modality][key]
        if self._use_mean_std(embodiment_tag, modality, key):
            return _unnormalize_meanstd(values, params)
        return _unnormalize_minmax(values, params)

    def apply_state(self, state: Dict[str, np.ndarray],
                    embodiment_tag: str) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['state']
        sincos_keys = set(cfg.get('sin_cos_embedding_keys') or [])
        result = {}
        for key in cfg['modality_keys']:
            if self.apply_sincos_state_encoding and key in sincos_keys:
                result[key] = _apply_sin_cos_encoding(state[key])
            else:
                result[key] = self._normalize(state[key], embodiment_tag,
                                              'state', key)
        return result

    def _relative(self, action: np.ndarray, reference_state: np.ndarray,
                  action_config: Dict[str,
                                      Any], to_absolute: bool) -> np.ndarray:
        action_type = _action_config_value(action_config, 'type', 'NON_EEF')
        action_format = _action_config_value(action_config, 'format',
                                             'DEFAULT')
        if action_type != 'NON_EEF' or action_format != 'DEFAULT':
            raise NotImplementedError(
                'Native N1.7 processor currently supports NON_EEF/DEFAULT '
                f'relative actions only, got {action_type}/{action_format}.')
        action = action.astype(np.float64)
        reference_state = reference_state.astype(np.float64)
        if to_absolute:
            return action + reference_state
        return action - reference_state

    def _maybe_relative(self, values: np.ndarray, state: Dict[str, np.ndarray],
                        key: str, action_config: Dict[str, Any],
                        to_absolute: bool) -> np.ndarray:
        if (not self.use_relative_action or
                _action_config_value(action_config, 'rep', '') != 'RELATIVE'):
            return values
        state_key = action_config.get('state_key') or key
        reference = np.asarray(state[state_key])[-1]
        return self._relative(values, reference, action_config, to_absolute)

    def apply_action(self, action: Dict[str, np.ndarray], embodiment_tag: str,
                     state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['action']
        action_configs = cfg.get('action_configs') or [
            {} for _ in cfg['modality_keys']
        ]
        result = {}
        for key, action_config in zip(cfg['modality_keys'], action_configs):
            values = deepcopy(action[key])
            values = self._maybe_relative(
                values, state, key, action_config, to_absolute=False)
            result[key] = self._normalize(values, embodiment_tag, 'action',
                                          key)
        return result

    def unapply_action(
        self,
        action: Dict[str, np.ndarray],
        embodiment_tag: str,
        state: Optional[Dict[str,
                             np.ndarray]] = None) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['action']
        action_configs = cfg.get('action_configs') or [
            {} for _ in cfg['modality_keys']
        ]
        result = {}
        for key, action_config in zip(cfg['modality_keys'], action_configs):
            values = self._unnormalize(action[key], embodiment_tag, 'action',
                                       key)
            if (self.use_relative_action and _action_config_value(
                    action_config, 'rep', '') == 'RELATIVE'):
                if state is None:
                    raise ValueError(f'State is required to decode {key!r}.')
                values = self._maybe_relative(
                    values, state, key, action_config, to_absolute=True)
            result[key] = values
        return result

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: Any,
        state: Optional[Dict[str,
                             np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Decode a padded action using the configured modality layout."""
        embodiment_key = normalize_tag_value(embodiment_tag)
        action_cfg = self.modality_configs[embodiment_key]['action']
        horizon = len(action_cfg['delta_indices'])
        decoded = {}
        start_idx = 0
        for key in action_cfg['modality_keys']:
            dim = int(self.norm_params[embodiment_key]['action'][key]['dim'])
            decoded[key] = action[..., :horizon, start_idx:start_idx + dim]
            start_idx += dim
        return self.unapply_action(decoded, embodiment_key, state=state)

    def apply(self, state: Dict[str, np.ndarray],
              action: Dict[str, np.ndarray], embodiment_tag: str):
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state)
        else:
            assert not self.training, 'Action is required in training mode'
            processed_action = {}
        return processed_state, processed_action

    def get_action_dim(self, embodiment_tag: str) -> int:
        total = 0
        for key in self.modality_configs[embodiment_tag]['action'][
                'modality_keys']:
            total += int(
                self.norm_params[embodiment_tag]['action'][key]['dim'])
        return total


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


# Backward-compatible aliases while the old processor facade remains as a
# golden-reference path.
