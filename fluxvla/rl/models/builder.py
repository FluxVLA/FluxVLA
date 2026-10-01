"""Build Flux models from Python configs and strictly load initial weights."""

import copy
import json
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file

from fluxvla.engines import build_vla_from_cfg
from fluxvla.rl.benchmarks.registry import build_observation_adapter
from fluxvla.rl.models.registry import (checkpoint_resolver_for,
                                        get_policy_spec, resolve_symbol)


def read_checkpoint(path):
    """Read native/indexed safetensors, validating index paths."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = [
            path / name for name in ('model.safetensors.index.json',
                                     'model.safetensors', 'model.pt')
            if (path / name).is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f'No supported checkpoint in {path}')
        path = candidates[0]
    if path.name.endswith('.safetensors.index.json'):
        index = json.loads(path.read_text())
        weight_map = index.get('weight_map', {})
        if not weight_map:
            raise ValueError('Empty safetensors weight_map')
        checkpoint = {}
        for shard_name in sorted(set(weight_map.values())):
            shard = (path.parent / shard_name).resolve()
            if not shard.is_relative_to(path.parent) or not shard.is_file():
                raise ValueError(
                    f'Missing/unsafe checkpoint shard: {shard_name}')
            tensors = load_file(str(shard), device='cpu')
            expected = {
                key
                for key, value in weight_map.items() if value == shard_name
            }
            if set(tensors) != expected:
                raise ValueError(f'Shard/index key mismatch: {shard_name}')
            checkpoint.update(tensors)
        return checkpoint
    if path.suffix == '.safetensors':
        return load_file(str(path), device='cpu')
    if path.suffix in ('.pt', '.pth'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get(
                'model', checkpoint.get('state_dict', checkpoint))
        return checkpoint
    raise ValueError(
        'Expected a PT, safetensors, index, or checkpoint directory')


def load_sft_weights(model, path):
    """Load before adding the critic; exact Flux keys precede name_mapping."""
    path = Path(path).expanduser().resolve()
    checkpoint = read_checkpoint(path)
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must contain a model state dict')
    expected = model.state_dict()
    resolver = checkpoint_resolver_for(model)
    loaded, errors = {}, []
    for name, target in expected.items():
        if name in checkpoint:
            candidates = [name]
        elif model.name_mapping:
            candidates = [
                candidate
                for _, candidate in model._mapped_name_candidates(name)
                if candidate in checkpoint
            ]
        else:
            candidates = []
        candidates = list(dict.fromkeys(candidates))
        if resolver is not None and model.name_mapping:
            try:
                candidates = resolver(name, candidates, checkpoint)
            except ValueError as error:
                errors.append(f'{name}: {error}')
                continue
        if len(candidates) != 1:
            errors.append(
                f'{name}: expected one checkpoint match, found {candidates}')
            continue
        source = checkpoint[candidates[0]]
        if not isinstance(source,
                          torch.Tensor) or source.shape != target.shape:
            errors.append(
                f'{name}: shape mismatch, expected {tuple(target.shape)}')
        else:
            loaded[name] = source
    if errors:
        raise ValueError('Incomplete/incompatible SFT checkpoint:\n' +
                         '\n'.join(errors[:20]))
    model.load_state_dict(loaded, strict=True)
    model.pretrained_name_or_path = str(path)


def build_flow_policy(cfg, torch_dtype=None):
    """Shared strict-SFT builder for declared flow-matching policies."""
    spec = get_policy_spec(cfg.model_type)
    if torch_dtype not in (None, torch.float32):
        raise ValueError('FluxVLA RL requires FP32 master weights; '
                         'use compute_dtype for autocast')
    path = Path(cfg.model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Initial SFT checkpoint not found: {path}')
    options = cfg.fluxvla
    spec.validate_adapter(options.get('observation_adapter', 'libero'))
    for key in ('joint_logprob', 'is_lora'):
        if cfg.get(key, False):
            raise ValueError(f'FluxVLA PPO v1 does not support {key}=True')
    if not cfg.get('add_value_head', True):
        raise ValueError('FluxVLA PPO requires add_value_head=True')
    if options.get('train_expert_only', True) is not True:
        raise ValueError('FluxVLA PPO v1 requires train_expert_only=True')
    if options.get('noise_method', 'flow_sde') != 'flow_sde':
        raise ValueError('FluxVLA PPO v1 requires noise_method=flow_sde')
    flux_cfg = Config.fromfile(
        str(Path(options.config_path).expanduser().resolve()))
    model_cfg = copy.deepcopy(flux_cfg.model)
    if model_cfg.type != spec.native_model_type:
        raise ValueError(f'Expected Flux model type {spec.native_model_type}')
    model_cfg.type = resolve_symbol(spec.policy_class)
    model_cfg.pretrained_name_or_path = None
    model_cfg.num_steps = int(cfg.num_steps)
    if options.get('action_horizon') is not None:
        model_cfg[spec.horizon_config_key] = int(options.action_horizon)
    # Gemma creates some norm parameters in BF16 even for an FP32 config.
    # Promote BEFORE copying checkpoint tensors or their FP32 values would be
    # irreversibly rounded, despite the final model reporting FP32 parameters.
    model = build_vla_from_cfg(model_cfg).float()
    load_sft_weights(model, path)
    stats = None
    if options.get('norm_stats_path'):
        with Path(options.norm_stats_path).expanduser().open(
                encoding='utf-8') as stream:
            stats = json.load(stream)
    observation = build_observation_adapter(flux_cfg, options, stats)
    dtype_name = options.get('compute_dtype', 'bf16')
    if dtype_name not in ('bf16', 'fp32'):
        raise ValueError('compute_dtype must be bf16 or fp32')
    model.configure_rl(
        observation,
        action_chunk=int(cfg.num_action_chunks),
        action_dim=int(cfg.action_dim),
        noise_level=float(options.get('noise_level', 0.5)),
        rollout_micro_batch_size=options.get('rollout_micro_batch_size'),
        compute_dtype=torch.bfloat16
        if dtype_name == 'bf16' else torch.float32)
    return model.to(dtype=torch_dtype or torch.float32)


def build_pi05_policy(cfg, torch_dtype=None):
    """Backward-compatible entrypoint; new models use build_flow_policy."""
    if cfg.model_type != 'fluxvla_pi05':
        raise ValueError('Expected model_type=fluxvla_pi05')
    return build_flow_policy(cfg, torch_dtype)


def build_smolvla_policy(cfg, torch_dtype=None):
    """Backward-compatible entrypoint; new models use build_flow_policy."""
    if cfg.model_type != 'fluxvla_smolvla':
        raise ValueError('Expected model_type=fluxvla_smolvla')
    return build_flow_policy(cfg, torch_dtype)
