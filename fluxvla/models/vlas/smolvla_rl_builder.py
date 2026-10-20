"""Build the registered SmolVLA RL policy from a native SFT checkpoint."""

import copy
import json
from pathlib import Path

import torch
from mmengine import Config

import fluxvla.datasets.adapters  # noqa: F401
from fluxvla.engines import build_processor_from_cfg, build_vla_from_cfg
from .pi05_rl_builder import _deep_update, load_sft_weights
from .smolvla_rl_policy import FluxSmolVLARLPolicy


def build_smolvla_policy(cfg, torch_dtype=None):
    """RLinf ModelBuilder for the Flux-native SmolVLA policy."""
    if cfg.model_type != 'fluxvla_smolvla':
        raise ValueError('Expected model_type=fluxvla_smolvla')
    if torch_dtype not in (None, torch.float32):
        raise ValueError(
            'FluxVLA RL requires FP32 master weights; use compute_dtype for '
            'autocast')
    path = Path(cfg.model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Initial SFT checkpoint not found: {path}')
    options = cfg.fluxvla
    for key in ('joint_logprob', 'is_lora'):
        if cfg.get(key, False):
            raise ValueError(f'FluxVLA PPO does not support {key}=True')
    if not cfg.get('add_value_head', True):
        raise ValueError('FluxVLA PPO requires add_value_head=True')
    if options.get('train_expert_only', True) is not True:
        raise ValueError('FluxVLA PPO requires train_expert_only=True')
    if options.get('noise_method', 'flow_sde') != 'flow_sde':
        raise ValueError('FluxVLA PPO requires noise_method=flow_sde')

    flux_cfg = Config.fromfile(
        str(Path(options.config_path).expanduser().resolve()))
    model_cfg = copy.deepcopy(flux_cfg.model)
    if model_cfg.type != 'SmolVLAFlowMatching':
        raise ValueError('Expected Flux model type SmolVLAFlowMatching')
    model_cfg.type = FluxSmolVLARLPolicy
    model_cfg.pretrained_name_or_path = None
    model_cfg.num_steps = int(cfg.num_steps)
    _deep_update(model_cfg, options.get('model_overrides', {}))
    if options.get('action_horizon') is not None:
        model_cfg.chunk_size = int(options.action_horizon)
    model = build_vla_from_cfg(model_cfg).float()
    load_sft_weights(model, path)

    adapter_cfg = copy.deepcopy(dict(options.observation_adapter))
    if adapter_cfg.get('type') != 'LiberoRLAdapter':
        raise ValueError('SmolVLA RL currently supports LiberoRLAdapter only')
    stats = None
    if options.get('norm_stats_path'):
        with Path(options.norm_stats_path).expanduser().open(
                encoding='utf-8') as stream:
            stats = json.load(stream)
    adapter_cfg.setdefault('flux_cfg', flux_cfg)
    adapter_cfg.setdefault('norm_stats', stats)
    adapter_cfg.setdefault('tokenizer_path', options.get('tokenizer_path'))
    observation = build_processor_from_cfg(adapter_cfg)

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
