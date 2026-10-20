"""Build the registered PI0.5 RL policy and strictly load initial weights."""

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file

import fluxvla.datasets.adapters  # noqa: F401
from fluxvla.engines import build_processor_from_cfg, build_vla_from_cfg
from .pi05_rl_policy import FluxPI05RLPolicy


def _deep_update(target, updates):
    """Apply simulator overrides without another model config."""
    for key, value in dict(updates or {}).items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def read_checkpoint(path):
    """Read native or indexed safetensors with safe index paths."""
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
        # OpenPI serializes both sides of its tied vocabulary matrix. Accept
        # only this documented alias and only when the tensors agree exactly.
        aliases = {
            ('paligemma_with_expert.paligemma.model.language_model.'
             'embed_tokens.weight'),
            'paligemma_with_expert.paligemma.lm_head.weight',
        }
        if name == 'llm_backbone.embed_tokens.weight' and set(
                candidates) == aliases:
            if not torch.equal(checkpoint[candidates[0]],
                               checkpoint[candidates[1]]):
                errors.append(f'{name}: shared embedding aliases disagree')
                continue
            candidates = candidates[:1]
        # Flux allocates an unused expert vocabulary embedding; OpenPI drops
        # it but retains the matching LM head. Never fill effective trunk
        # parameters with random data to tolerate an incomplete checkpoint.
        if (not candidates and name == 'llm_expert.embed_tokens.weight'
                and model.name_mapping):
            alias = 'paligemma_with_expert.gemma_expert.lm_head.weight'
            if alias in checkpoint:
                candidates = [alias]
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


def build_pi05_policy(cfg, torch_dtype=None):
    """RLinf ModelBuilder(cfg, torch_dtype), with all model code in FluxVLA."""
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
    if model_cfg.type != 'PI05FlowMatching':
        raise ValueError('The first RL bridge supports PI05FlowMatching only')
    model_cfg.type = FluxPI05RLPolicy
    model_cfg.pretrained_name_or_path = None
    model_cfg.num_steps = int(cfg.num_steps)
    _deep_update(model_cfg, options.get('model_overrides', {}))
    if options.get('action_horizon') is not None:
        model_cfg.n_action_steps = int(options.action_horizon)
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
    adapter_cfg = options.get('observation_adapter')
    if not isinstance(adapter_cfg, (Mapping, Config)):
        # Programmatic compatibility only; public configs use registry dicts so
        # adding a simulator does not require another builder branch.
        adapter_cfg = {
            'type':
            'RoboTwinRLAdapter'
            if adapter_cfg == 'robotwin' else 'LiberoRLAdapter'
        }
    adapter_cfg = copy.deepcopy(dict(adapter_cfg))
    adapter_type = adapter_cfg.get('type')
    if adapter_type == 'RoboTwinRLAdapter':
        adapter_cfg.setdefault('norm_stats', stats)
        adapter_cfg.setdefault('tokenizer_path', options.get('tokenizer_path'))
        adapter_cfg.setdefault('image_size', options.get('image_size', 224))
        adapter_cfg.setdefault('max_token_len',
                               options.get('max_token_len', 200))
    elif adapter_type == 'LiberoRLAdapter':
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
