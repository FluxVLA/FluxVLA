"""Transport-neutral FluxVLA serving runner and configuration helpers."""

from __future__ import annotations
import copy
import json
import os
import random
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...utils.root import RUNNERS

_MISSING = object()


@RUNNERS.register_module()
class FluxVLAServingRunner:
    """Own model preprocessing, inference, and action denormalization."""

    def __init__(self,
                 vla: Any,
                 dataset: Any,
                 denormalize_action: Any = None,
                 device: str = 'cuda:0',
                 mixed_precision_dtype: torch.dtype = torch.bfloat16,
                 enable_mixed_precision: bool = True,
                 keep_params_fp32: bool = False,
                 model_outputs_environment_actions: bool = False,
                 denormalize_context: Mapping[str, Any] | None = None,
                 denormalize_per_action: bool = False,
                 expected_unnorm_key: str = '') -> None:
        if not callable(dataset):
            raise TypeError('FluxVLA server dataset must be callable')
        if (not model_outputs_environment_actions
                and not callable(denormalize_action)):
            raise RuntimeError(
                'A denormalize_action transform is required unless '
                'model_outputs_environment_actions=True is explicitly set.')
        if not isinstance(mixed_precision_dtype, torch.dtype):
            raise TypeError('mixed_precision_dtype must be a torch dtype')
        if mixed_precision_dtype not in {
                torch.float32, torch.float16, torch.bfloat16
        }:
            raise ValueError(
                'mixed_precision_dtype must be fp32, fp16, or bf16')
        if not isinstance(enable_mixed_precision, bool):
            raise TypeError('enable_mixed_precision must be a bool')
        if not isinstance(keep_params_fp32, bool):
            raise TypeError('keep_params_fp32 must be a bool')
        if not isinstance(model_outputs_environment_actions, bool):
            raise TypeError('model_outputs_environment_actions must be a bool')
        if not isinstance(denormalize_per_action, bool):
            raise TypeError('denormalize_per_action must be a bool')
        if denormalize_context is not None and not isinstance(
                denormalize_context, Mapping):
            raise TypeError('denormalize_context must be a mapping')
        if not isinstance(expected_unnorm_key, str):
            raise TypeError('expected_unnorm_key must be a string')

        self.vla = vla
        self.dataset = dataset
        self.denormalize_action = denormalize_action
        self.device = torch.device(device)
        self.mixed_precision_dtype = mixed_precision_dtype
        self.enable_mixed_precision = enable_mixed_precision
        self.keep_params_fp32 = keep_params_fp32
        self.model_outputs_environment_actions = \
            model_outputs_environment_actions
        self.denormalize_context = dict(denormalize_context or {})
        self.denormalize_per_action = denormalize_per_action
        self.expected_unnorm_key = expected_unnorm_key
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.vla.eval()
        if self.enable_mixed_precision and not self.keep_params_fp32:
            self.vla.to(device=self.device, dtype=self.mixed_precision_dtype)
        else:
            self.vla.to(self.device)

    def predict_action(self, observation: Mapping[str, Any], unnorm_key: str,
                       policy_seed: int) -> tuple[np.ndarray, float]:
        """Process one self-contained request and return ``[T, A]`` actions."""
        if not isinstance(observation, Mapping):
            raise TypeError('observation must be a mapping')
        if (unnorm_key and self.expected_unnorm_key
                and unnorm_key != self.expected_unnorm_key):
            raise ValueError(
                f'Request unnorm_key {unnorm_key!r} does not match the '
                f'configured key {self.expected_unnorm_key!r}')
        unnorm_key = unnorm_key or self.expected_unnorm_key
        self._seed_policy_rngs(policy_seed)
        result = self.dataset(dict(observation))
        batch = result[0] if isinstance(result, tuple) else result
        if not isinstance(batch, Mapping):
            raise TypeError(
                'FluxVLA dataset pipeline must return a mapping or a '
                'tuple whose first item is a mapping')
        batch = dict(batch)
        batch['reset_history'] = True
        if unnorm_key:
            batch['unnorm_key'] = unnorm_key
        batch['seed'] = int(policy_seed)
        batch = self._move_to_device(batch)

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad(), self._autocast_context():
            raw_actions = self.vla.predict_action(**batch)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        inference_time_s = time.perf_counter() - started
        request_context = {}
        raw_state = getattr(self.dataset, 'last_raw_state', None)
        if raw_state is not None:
            request_context['state'] = np.asarray(
                raw_state, dtype=np.float32).copy()
        elif observation.get('states') is not None:
            request_context['state'] = observation['states']
        actions = self._to_environment_actions(
            raw_actions,
            unnorm_key=unnorm_key,
            request_context=request_context)
        return actions, inference_time_s

    @staticmethod
    def _seed_policy_rngs(policy_seed: int) -> None:
        random.seed(int(policy_seed))
        np.random.seed(int(policy_seed) % (2**32))
        torch.manual_seed(int(policy_seed))

    def _autocast_context(self):
        enabled = (
            self.enable_mixed_precision and self.device.type == 'cuda'
            and self.mixed_precision_dtype in {torch.bfloat16, torch.float16})
        if not enabled:
            return nullcontext()
        return torch.autocast(
            device_type='cuda', dtype=self.mixed_precision_dtype)

    def _to_environment_actions(
            self,
            raw_actions: Any,
            unnorm_key: str = '',
            request_context: Mapping[str, Any] | None = None) -> np.ndarray:
        raw_array = self._as_numpy(raw_actions)
        if self.model_outputs_environment_actions:
            return self._canonicalize_actions(raw_array)

        context = dict(self.denormalize_context)
        context.update(request_context or {})
        if unnorm_key:
            context.setdefault('unnorm_key', unnorm_key)
            context.setdefault('norm_stats_key', unnorm_key)
            context.setdefault('task_suite_name', unnorm_key)

        normalized = self._canonicalize_actions(raw_array)
        stats_horizon_getter = getattr(self.denormalize_action,
                                       'get_action_stats_horizon', None)
        stats_horizon = (
            stats_horizon_getter() if callable(stats_horizon_getter) else None)
        if stats_horizon is not None:
            normalized = normalized[:stats_horizon]

        if self.denormalize_per_action or stats_horizon is not None:
            denormalized = []
            for index, action in enumerate(normalized):
                action_context = context
                if stats_horizon is not None:
                    action_context = dict(context, action_horizon_index=index)
                value = self._call_denormalizer(action, action_context)
                value = np.asarray(value)
                if value.ndim == 2 and value.shape[0] == 1:
                    value = value[0]
                if value.ndim != 1:
                    raise ValueError(
                        'Per-action denormalizer must return shape [A], got '
                        f'{value.shape}')
                denormalized.append(value)
            return self._canonicalize_actions(np.stack(denormalized))

        value = self._call_denormalizer(normalized, context)
        return self._canonicalize_actions(value)

    def _call_denormalizer(self, actions: np.ndarray,
                           context: Mapping[str, Any]) -> np.ndarray:
        payload = dict(context)
        payload['action'] = actions
        value = self.denormalize_action(payload)
        if isinstance(value, Mapping):
            if 'action' not in value:
                raise KeyError(
                    'denormalize_action returned a mapping without `action`')
            value = value['action']
        return self._as_numpy(value)

    def _move_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, Mapping):
            return {
                key: self._move_to_device(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        return value

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _canonicalize_actions(value: Any) -> np.ndarray:
        actions = np.asarray(value)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(
                    'FluxVLA inference only supports batch size 1, got '
                    f'{actions.shape}')
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]
        elif actions.ndim != 2:
            raise ValueError('FluxVLA actions must have shape [A], [T, A], or '
                             f'[1, T, A], got {actions.shape}')
        try:
            actions = np.asarray(actions, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError('FluxVLA actions must be numeric') from exc
        if actions.shape[0] == 0 or actions.shape[1] == 0:
            raise ValueError('FluxVLA returned an empty action chunk')
        if not np.isfinite(actions).all():
            raise ValueError('FluxVLA returned NaN or infinite actions')
        return actions


def build_serving_runner_from_config(
        cfg: Any,
        ckpt_path: str | None = None,
        device: str | None = None) -> FluxVLAServingRunner:
    """Build one complete serving runner from an authoritative config."""
    themis_cfg = require_mapping(
        config_get(cfg, 'themis', _MISSING), 'config.themis')
    transport = dict(
        require_mapping(
            themis_cfg.get('transport', _MISSING), 'themis.transport'))
    server_cfg = get_server_config(themis_cfg)

    section_name = server_cfg.get('dataset_section', 'eval')
    if section_name not in {'eval', 'inference'}:
        raise ValueError('themis.server.dataset_section must be `eval` or '
                         '`inference`')
    section_cfg = require_mapping(
        config_get(cfg, section_name, _MISSING), f'config.{section_name}')
    dataset_cfg = copy.deepcopy(
        require_mapping(
            section_cfg.get('dataset', _MISSING),
            f'config.{section_name}.dataset'))
    resolved_ckpt = resolve_checkpoint_path(
        ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    stats_path = resolve_statistics_path(
        server_cfg.get('norm_stats_path'), resolved_ckpt)
    model_outputs_environment_actions = server_cfg.get(
        'model_outputs_environment_actions', False)
    if not model_outputs_environment_actions and stats_path is None:
        raise FileNotFoundError(
            'dataset_statistics.json is required to prove the action unit '
            'contract')

    from fluxvla.engines import (build_dataset_from_cfg,
                                 build_transform_from_cfg, build_vla_from_cfg)
    from fluxvla.engines.utils import str_to_dtype

    model_cfg = config_get(cfg, 'inference_model', None)
    if model_cfg is None:
        model_cfg = config_get(cfg, 'model', _MISSING)
    if model_cfg is _MISSING:
        raise KeyError('FluxVLA config must define `model` or '
                       '`inference_model`')
    vla = build_vla_from_cfg(model_cfg)
    checkpoint_model_cfg = config_get(cfg, 'model', model_cfg)
    load_checkpoint(
        vla,
        resolved_ckpt,
        name_mapping=config_get(checkpoint_model_cfg, 'name_mapping'),
    )
    if stats_path is not None:
        with stats_path.open('r', encoding='utf-8') as stream:
            vla.norm_stats = json.load(stream)

    prepare_dataset_config(
        dataset_cfg=dataset_cfg,
        section_cfg=section_cfg,
        transport=transport,
        stats_path=stats_path,
        model_root=resolved_ckpt.parent.parent,
    )
    dataset = build_dataset_from_cfg(dataset_cfg)

    denormalize_action = None
    denormalize_context = dict(
        require_mapping(
            server_cfg.get('denormalize_context', {}),
            'themis.server.denormalize_context'))
    if not model_outputs_environment_actions:
        denorm_cfg = copy.deepcopy(
            require_mapping(
                section_cfg.get('denormalize_action', _MISSING),
                f'config.{section_name}.denormalize_action'))
        denorm_cfg['norm_stats'] = str(stats_path)
        denormalize_action = build_transform_from_cfg(denorm_cfg)
        prepare_denormalize_context(denormalize_context, denorm_cfg,
                                    section_cfg, transport)
        validate_denormalization_stats(denormalize_action, denormalize_context)

    resolved_device = device or server_cfg.get('device', 'cuda:0')
    dtype_name = server_cfg.get('mixed_precision_dtype', 'bf16')
    from fluxvla.engines import build_runner_from_cfg

    runner_cfg = copy.deepcopy(
        require_mapping(
            server_cfg.get('runner', _MISSING), 'themis.server.runner'))
    return build_runner_from_cfg(
        runner_cfg,
        default_args=dict(
            vla=vla,
            dataset=dataset,
            denormalize_action=denormalize_action,
            device=resolved_device,
            mixed_precision_dtype=str_to_dtype(str(dtype_name)),
            enable_mixed_precision=server_cfg.get('enable_mixed_precision',
                                                  True),
            keep_params_fp32=server_cfg.get('keep_params_fp32', False),
            model_outputs_environment_actions=(
                model_outputs_environment_actions),
            denormalize_context=denormalize_context,
            denormalize_per_action=server_cfg.get('denormalize_per_action',
                                                  False),
            expected_unnorm_key=str(transport.get('unnorm_key', '')),
        ))


def get_server_config(themis_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the required transport-neutral server configuration."""
    server_cfg = dict(
        require_mapping(themis_cfg.get('server', _MISSING), 'themis.server'))
    if server_cfg.get('mode', 'stateless') != 'stateless':
        raise ValueError('themis.server.mode must be `stateless`')
    return server_cfg


def resolve_report_config_path(
        cfg: Any, explicit_path: str | os.PathLike | None) -> Path:
    value = explicit_path
    if value is None:
        value = getattr(cfg, 'filename', None)
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(
            'Evaluation reporting requires the authoritative FluxVLA config '
            'path')
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'FluxVLA config not found: {path}')
    return path


def resolve_report_result_root(value: Any, checkpoint_path: Path) -> Path:
    checkpoint_root = checkpoint_path.parent.parent.resolve()
    if value is None:
        return checkpoint_root
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise TypeError(
            'themis.server.evaluation_reporting.result_output_dir must be a '
            'non-empty path')
    requested = Path(value).expanduser()
    if requested.is_absolute():
        return requested.resolve()
    return (checkpoint_root / requested).resolve()


def resolve_checkpoint_path(value: Any) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ValueError(
            'Checkpoint path is required via --ckpt-path, '
            'themis.server.ckpt_path, or the selected section.ckpt_path')
    path = Path(value).expanduser().resolve()
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def resolve_statistics_path(value: Any, checkpoint_path: Path) -> Path | None:
    explicit = value is not None
    path = (
        Path(value).expanduser().resolve() if explicit else
        checkpoint_path.parent.parent / 'dataset_statistics.json')
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(
                f'Configured norm_stats_path not found: {path}')
        return None
    return path


def load_checkpoint(vla: Any,
                    checkpoint_path: Path,
                    name_mapping: Mapping[str, str] | None = None) -> None:
    if checkpoint_path.is_dir():
        from safetensors.torch import load_file
        shards = sorted(checkpoint_path.glob('model-*.safetensors'))
        if not shards:
            raise FileNotFoundError(
                f'No model-*.safetensors files found in {checkpoint_path}')
        state_dict = {}
        for shard in shards:
            state_dict.update(load_file(str(shard), device='cpu'))
    elif checkpoint_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path), device='cpu')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = (
            checkpoint['model'] if isinstance(checkpoint, Mapping)
            and 'model' in checkpoint else checkpoint)

    if name_mapping:
        if not isinstance(name_mapping, Mapping):
            raise TypeError('model.name_mapping must be a mapping')

        def has_prefix(key: str, prefix: str) -> bool:
            return key == prefix or key.startswith(f'{prefix}.')

        native_prefixes = tuple(name_mapping)
        source_prefixes = tuple(name_mapping.values())
        has_native_keys = any(
            any(has_prefix(key, prefix) for key in state_dict)
            for prefix in native_prefixes)
        needs_mapping = (
            any(
                any(has_prefix(key, prefix) for key in state_dict)
                for prefix in source_prefixes) and not has_native_keys)
        if needs_mapping:
            mapped_state_dict = {}
            for native_prefix, source_prefix in name_mapping.items():
                if not isinstance(native_prefix, str) or not isinstance(
                        source_prefix, str):
                    raise TypeError(
                        'model.name_mapping keys and values must be strings')
                for key, value in state_dict.items():
                    if has_prefix(key, source_prefix):
                        mapped_key = native_prefix + key[len(source_prefix):]
                        mapped_state_dict[mapped_key] = value
            state_dict = mapped_state_dict

    from fluxvla.engines.utils.checkpoint_utils import handle_shared_tensors
    state_dict = handle_shared_tensors(state_dict, vla.state_dict())
    vla.load_state_dict(state_dict, strict=True)


def prepare_dataset_config(dataset_cfg: dict, section_cfg: Mapping[str, Any],
                           transport: Mapping[str, Any],
                           stats_path: Path | None, model_root: Path) -> None:
    if stats_path is not None:
        dataset_cfg['norm_stats'] = str(stats_path)
    dataset_type = dataset_cfg.get('type', '')
    dataset_type_name = (
        dataset_type if isinstance(dataset_type, str) else getattr(
            dataset_type, '__name__', str(dataset_type)))
    task_suite_name = section_cfg.get('task_suite_name')

    if 'Libero' in dataset_type_name:
        if not task_suite_name:
            raise KeyError('Libero serving requires section.task_suite_name')
        dataset_cfg.setdefault('task_suite_name', task_suite_name)
        dataset_cfg.setdefault(
            'norm_stats_key',
            section_cfg.get('norm_stats_key') or f'{task_suite_name}_no_noops')
    if 'PrivateInferenceDataset' in dataset_type_name:
        dataset_cfg.setdefault('model_path', str(model_root))
    if 'Robocasa' in dataset_type_name:
        unnorm_key = transport.get('unnorm_key')
        if unnorm_key:
            dataset_cfg.setdefault('unnorm_key', unnorm_key)


def prepare_denormalize_context(context: dict[str,
                                              Any], denorm_cfg: Mapping[str,
                                                                        Any],
                                section_cfg: Mapping[str, Any],
                                transport: Mapping[str, Any]) -> None:
    transform_type = denorm_cfg.get('type', '')
    transform_name = (
        transform_type if isinstance(transform_type, str) else getattr(
            transform_type, '__name__', str(transform_type)))
    if 'Robocasa' in transform_name:
        section_key = section_cfg.get('unnorm_key')
        transport_key = transport.get('unnorm_key')
        if section_key and transport_key and section_key != transport_key:
            raise ValueError('RoboCasa section.unnorm_key and '
                             'themis.transport.unnorm_key must match')
        stats_key = transport_key or section_key
        if not stats_key:
            raise KeyError(
                'RoboCasa serving requires an unnorm_key in the eval section '
                'or themis.transport')
        configured_key = context.get('task_suite_name')
        if configured_key and configured_key != stats_key:
            raise ValueError(
                'RoboCasa denormalize_context.task_suite_name must match '
                'the configured unnorm_key')
        context['task_suite_name'] = stats_key
        return
    task_suite_name = section_cfg.get('task_suite_name')
    if task_suite_name:
        context.setdefault('task_suite_name', task_suite_name)
    if 'Libero' in transform_name:
        context.setdefault(
            'norm_stats_key',
            section_cfg.get('norm_stats_key')
            or (f'{task_suite_name}_no_noops'
                if task_suite_name else transport.get('unnorm_key', '')))


def validate_denormalization_stats(transform: Any,
                                   context: Mapping[str, Any]) -> None:
    norm_stats = getattr(transform, 'norm_stats', None)
    transform_name = type(transform).__name__
    statistic_name = getattr(transform, 'statistic_name', None)
    if statistic_name:
        stats_key = statistic_name
    elif 'Libero' in transform_name:
        stats_key = context.get('norm_stats_key')
    else:
        stats_key = (
            context.get('task_suite_name') or context.get('norm_stats_key'))
    if not isinstance(norm_stats, Mapping) or not stats_key:
        return
    if stats_key not in norm_stats:
        raise KeyError(
            f'Normalization statistics key {stats_key!r} is missing. '
            f'available keys: {list(norm_stats)}')
    stats = norm_stats[stats_key]
    if not isinstance(stats, Mapping) or 'action' not in stats:
        raise KeyError(
            f'Normalization statistics {stats_key!r} must contain action')


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is _MISSING:
        raise KeyError(f'{name} is required')
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    return value


__all__ = [
    'FluxVLAServingRunner',
    'build_serving_runner_from_config',
    'config_get',
    'get_server_config',
    'require_mapping',
    'resolve_checkpoint_path',
    'resolve_report_config_path',
    'resolve_report_result_root',
    'resolve_statistics_path',
]
