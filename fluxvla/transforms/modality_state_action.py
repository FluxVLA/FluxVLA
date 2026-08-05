# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Metadata-driven state/action codec used by GR00T-style transforms."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    'oxe_droid_relative_eef_relative_joint': 24,
    'xdof_relative_eef_relative_joint': 27,
    'xdof_relative_eef_relative_joint_subtask': 27,
    'real_g1_relative_eef_relative_joints': 25,
    'real_r1_pro_sharpa_relative_eef': 26,
    'real_r1_pro_sharpa_relative_eef_human': 26,
    'real_r1_pro_sharpa_relative_eef_maxinsights': 26,
    'real_r1_pro_sharpa_relative_eef_mecka': 26,
    'unitree_g1_full_body_with_waist_height_nav_cmd': 25,
    'unitree_g1_sonic': 11,
    'simpler_env_google': 0,
    'simpler_env_widowx': 1,
    'libero_sim': 2,
    'new_embodiment': 10,
    'robocasa_panda_omron': 10,
    'robocasa_gr1_tabletop': 10,
}


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
    return (np.clip(values, -1.0, 1.0) + 1.0) / 2.0 * (
        max_vals - min_vals) + min_vals


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
                    low_field, high_field = (
                        ('q01', 'q99') if self.use_percentiles else
                        ('min', 'max'))
                    params = {
                        'min': np.asarray(stats[low_field]),
                        'max': np.asarray(stats[high_field]),
                        'mean': np.asarray(stats['mean']),
                        'std': np.asarray(stats['std']),
                        'dim': np.array(_normalization_dim(stats)),
                    }
                    self.norm_params[embodiment_tag][modality][key] = params
            action_cfg = self.modality_configs.get(embodiment_tag, {}).get(
                'action', {})
            for key, cfg in zip(action_cfg.get('modality_keys') or [],
                                action_cfg.get('action_configs') or []):
                if (self.use_relative_action
                        and _action_config_value(cfg, 'rep', '') == 'RELATIVE'):
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
                result[key] = self._normalize(
                    state[key], embodiment_tag, 'state', key)
        return result

    def _relative(self, action: np.ndarray, reference_state: np.ndarray,
                  action_config: Dict[str, Any], to_absolute: bool) -> np.ndarray:
        action_type = _action_config_value(action_config, 'type', 'NON_EEF')
        action_format = _action_config_value(action_config, 'format', 'DEFAULT')
        if action_type != 'NON_EEF' or action_format != 'DEFAULT':
            raise NotImplementedError(
                'Native N1.7 processor currently supports NON_EEF/DEFAULT '
                f'relative actions only, got {action_type}/{action_format}.')
        action = action.astype(np.float64)
        reference_state = reference_state.astype(np.float64)
        return action + reference_state if to_absolute else action - reference_state

    def _maybe_relative(self, values: np.ndarray, state: Dict[str, np.ndarray],
                        key: str, action_config: Dict[str, Any],
                        to_absolute: bool) -> np.ndarray:
        if (not self.use_relative_action
                or _action_config_value(action_config, 'rep', '') != 'RELATIVE'):
            return values
        state_key = action_config.get('state_key') or key
        reference = np.asarray(state[state_key])[-1]
        return self._relative(values, reference, action_config, to_absolute)

    def apply_action(self, action: Dict[str, np.ndarray], embodiment_tag: str,
                     state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['action']
        action_configs = cfg.get('action_configs') or [{} for _ in cfg[
            'modality_keys']]
        result = {}
        for key, action_config in zip(cfg['modality_keys'], action_configs):
            values = deepcopy(action[key])
            values = self._maybe_relative(
                values, state, key, action_config, to_absolute=False)
            result[key] = self._normalize(
                values, embodiment_tag, 'action', key)
        return result

    def unapply_action(self, action: Dict[str, np.ndarray], embodiment_tag: str,
                       state: Optional[Dict[str, np.ndarray]] = None
                       ) -> Dict[str, np.ndarray]:
        cfg = self.modality_configs[embodiment_tag]['action']
        action_configs = cfg.get('action_configs') or [{} for _ in cfg[
            'modality_keys']]
        result = {}
        for key, action_config in zip(cfg['modality_keys'], action_configs):
            values = self._unnormalize(
                action[key], embodiment_tag, 'action', key)
            if (self.use_relative_action
                    and _action_config_value(action_config, 'rep', '') ==
                    'RELATIVE'):
                if state is None:
                    raise ValueError(f'State is required to decode {key!r}.')
                values = self._maybe_relative(
                    values, state, key, action_config, to_absolute=True)
            result[key] = values
        return result

    def decode_action(self,
                      action: np.ndarray,
                      embodiment_tag: Any,
                      state: Optional[Dict[str, np.ndarray]] = None
                      ) -> Dict[str, np.ndarray]:
        """Decode a padded action tensor using the configured modality layout."""
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

    def apply(self, state: Dict[str, np.ndarray], action: Dict[str, np.ndarray],
              embodiment_tag: str):
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
            total += int(self.norm_params[embodiment_tag]['action'][key]['dim'])
        return total


def load_groot_n17_metadata(pretrained_model_name_or_path: str | Path,
                            **kwargs) -> Dict[str, Any]:
    """Load official N1.7 processor metadata without building HF processors."""
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
    processor_kwargs['statistics'] = statistics
    processor_kwargs['embodiment_id_mapping'] = embodiment_id_mapping
    processor_kwargs.setdefault('model_name', 'nvidia/Cosmos-Reason2-2B')
    processor_kwargs.setdefault('model_type', 'qwen')
    processor_kwargs.setdefault('clip_outliers', True)

    kwargs = dict(kwargs)
    modality_configs = kwargs.pop('modality_configs', {})
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
