# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Emit deployment artifacts from a live canonical PI0.5 policy.

This is a training-side handoff, not a config template or model converter.
Imports remain lightweight; the producer lazily uses the already installed
canonical component implementations. No training package is imported here.
"""

from __future__ import annotations
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..._safetensors import _stat_identity, _validate_finite
from ..._validation import validate_token_vocabulary
from ...checkpoint import load_checkpoint
from ...config import (GemmaConfig, PreprocessingConfig, VisionConfig,
                       config_from_document, config_to_document)
from ...errors import ArtifactError, ConfigError

if TYPE_CHECKING:
    import torch

_INPUT_NAMES = ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'states',
                'noise')
_LINEAR_NAMES = ('projector', 'action_in_proj', 'action_out_proj',
                 'time_mlp_in', 'time_mlp_out')
_COMPONENT_NAMES = ('vision_backbone', 'llm_backbone', 'llm_expert',
                    *_LINEAR_NAMES)
_ABSENT_NAMES = ('vlm_backbone', 'vla_head', 'state_proj',
                 'action_time_mlp_in', 'action_time_mlp_out')


@dataclass(frozen=True)
class PI05Artifacts:
    """Explicit paths for prepare_reference/prepare_tensorrt and Hermes.

    ``checkpoint`` is the original, validated file, never a copied checkpoint.
    Config hashes identify the validated snapshot; later consumers reverify it.
    """

    config: Path
    inputs: Path
    checkpoint: Path


def _error(path: str, message: str) -> None:
    raise ConfigError(f'PI0.5 artifact producer {path}: {message}')


def _attribute(value: Any, name: str, path: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise ConfigError(
            f'PI0.5 artifact producer {path}.{name}: required live attribute '
            'is missing') from error


def _path(value: str | Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ArtifactError(f'{label}: expected a nonempty filesystem path.')
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactError(f'{label} {value!r}: {error}') from error


def _checkpoint_digest(path: Path) -> str:
    """Discover a digest for subsequent verification by the strict reader."""
    if path.suffix != '.safetensors':
        raise ArtifactError(f'Checkpoint {path}: expected .safetensors.')
    digest = hashlib.sha256()
    try:
        with open(
                path,
                'rb',
                opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK)
        ) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactError(
                    f'Checkpoint {path}: expected a regular file.')
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            if _stat_identity(before) != _stat_identity(
                    os.fstat(stream.fileno())):
                raise ArtifactError(
                    f'Checkpoint {path}: changed while deriving SHA256.')
    except OSError as error:
        raise ArtifactError(
            f'Checkpoint {path}: cannot read: {error}') from error
    return digest.hexdigest()


def _same_method(actual: Any, expected: Any, name: str, path: str) -> None:
    left = _attribute(actual, name, path)
    right = _attribute(expected, name, path)
    if getattr(left, '__func__', left) is not getattr(right, '__func__',
                                                      right):
        _error(f'{path}.{name}', 'overridden computation is not supported')


def _validate_root(model: Any) -> None:
    import torch

    from ...components.pi_policy import PI0Computation, PI05Computation
    from .model import PI05Model

    if not (isinstance(model, torch.nn.Module)
            and isinstance(model, PI05Computation)):
        _error('model', 'requires an unwrapped canonical PI0.5 nn.Module')
    cls = type(model)
    training_type = ('fluxvla.models.vlas.pi05_flowmatching',
                     'PI05FlowMatching')
    if cls is not PI05Model and (cls.__module__,
                                 cls.__qualname__) != training_type:
        _error(
            'model', 'only the canonical PI05Model or training '
            'PI05FlowMatching instance is supported; unwrap externally')
    if 'forward' in model.__dict__:
        _error('model.forward', 'instance-overridden forward is unsupported')
    _same_method(model, torch.nn.Module, 'state_dict', 'model')
    _same_method(model, torch.nn.Module, '_save_to_state_dict', 'model')
    for base in (PI0Computation, PI05Computation):
        for name, value in vars(base).items():
            if callable(value) or isinstance(value, staticmethod):
                _same_method(model, PI05Computation, name, 'model')
    if model._parameters or model._buffers:
        _error('model', 'unexpected root parameters or buffers')
    if set(model._modules) != set(_COMPONENT_NAMES):
        _error(
            'model', 'unsupported or missing root modules: '
            f'{sorted(set(model._modules) ^ set(_COMPONENT_NAMES))}')
    for name in _ABSENT_NAMES:
        if _attribute(model, name, 'model') is not None:
            _error(f'model.{name}', 'must be None for PI0.5 schema 1')
    # RTC training metadata is not representable by this plain-only handoff.
    for name in ('rtc_training_config', 'rtc_config'):
        if getattr(model, name, None) is not None:
            _error(f'model.{name}', 'RTC is not supported by this producer')
    for name, module in model.named_modules():
        path = f'model.{name}' if name else 'model'
        if module.training:
            _error(path, 'all modules must already be in eval mode')
        for hook in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks',
                     '_state_dict_hooks', '_state_dict_pre_hooks',
                     '_load_state_dict_pre_hooks',
                     '_load_state_dict_post_hooks'):
            if getattr(module, hook, None):
                _error(path, f'custom {hook} are not supported')
        for key, value in module._parameters.items():
            if value is not None and (value.dtype != torch.float32
                                      or value.device.type == 'meta'):
                _error(f'{path}.{key}',
                       'requires materialized FP32 parameters')


def _gemma_document(module: Any, path: str) -> dict[str, Any]:
    from ...components.condition_gemma import ConditionGemmaModel

    if not isinstance(module, ConditionGemmaModel):
        _error(path, 'requires the canonical ConditionGemmaModel')
    config = module.config
    for name, child in module.named_modules():
        if hasattr(child, 'config') and child.config is not config:
            _error(f'{path}.{name}.config',
                   'must share the canonical branch configuration')
    rope = _attribute(config, 'rope_parameters', f'{path}.config')
    if (type(rope) is not dict or set(rope) != {'rope_type', 'rope_theta'}
            or rope['rope_type'] != 'default'):
        _error(f'{path}.config.rope_parameters',
               'schema 1 requires explicit default rope_type and rope_theta')
    result = {'type': 'condition_gemma'}
    for field in fields(GemmaConfig):
        if field.name == 'type':
            continue
        result[field.name] = (
            rope['rope_theta'] if field.name == 'rope_theta' else _attribute(
                config, field.name, f'{path}.config'))
    if _attribute(config, 'hidden_act',
                  f'{path}.config') != result['hidden_activation']:
        _error(f'{path}.config.hidden_act', 'conflicts with hidden_activation')
    return result


def _derive_document(model: Any, inputs: Mapping[str, Any],
                     preprocessing: PreprocessingConfig,
                     checkpoint_sha256: str, inputs_sha256: str) -> dict:
    import torch
    from transformers import SiglipVisionModel

    from ...components.projector import LinearProjector
    from ...components.siglip import SigLIPViTBackbone

    vision = model.vision_backbone
    if not isinstance(vision, SigLIPViTBackbone):
        _error('model.vision_backbone', 'requires canonical SigLIPViTBackbone')
    tower = _attribute(vision, 'vision', 'model.vision_backbone')
    if type(tower) is not SiglipVisionModel:
        _error('model.vision_backbone.vision',
               'requires an unwrapped canonical SiglipVisionModel')
    for name, child in tower.named_modules():
        if hasattr(child,
                   'config') and child.config is not vision.vision.config:
            _error(f'model.vision_backbone.vision.{name}.config',
                   'must share the canonical vision configuration')
    vision_document = {'type': 'siglip'}
    for field in fields(VisionConfig):
        if field.name == 'type':
            continue
        name = field.name
        if name == 'hidden_activation':
            name = 'hidden_act'
        vision_document[field.name] = _attribute(
            vision.vision.config, name, 'model.vision_backbone.vision.config')
    architecture = {
        'type': 'pi05',
        'vision': vision_document,
        'backbone': _gemma_document(model.llm_backbone, 'model.llm_backbone'),
        'expert': _gemma_document(model.llm_expert, 'model.llm_expert'),
        'proj_width': _attribute(model, 'proj_width', 'model'),
        'state_dim': inputs['states'].shape[-1],
        'action_dim': _attribute(model, 'max_action_dim', 'model'),
        'action_horizon': _attribute(model, 'n_action_steps', 'model'),
        'state_projection': model.state_proj is not None,
    }
    for name in _LINEAR_NAMES:
        projection = getattr(model, name)
        if not isinstance(projection, LinearProjector):
            _error(f'model.{name}', 'requires the canonical LinearProjector')
        layer = _attribute(projection, 'projector', f'model.{name}')
        if type(layer) is not torch.nn.Linear:
            _error(f'model.{name}.projector',
                   'requires an unwrapped nn.Linear')
        architecture[name] = {
            'type': 'linear',
            'in_dim': layer.in_features,
            'out_dim': layer.out_features,
            'bias': layer.bias is not None,
        }
    backbone, expert = model.llm_backbone.config, model.llm_expert.config
    for branch, config in (('backbone', backbone), ('expert', expert)):
        layers = _attribute(config, 'fp32_layers',
                            f'model.llm_{branch}.config')
        if type(layers) not in (list, tuple):
            _error(f'model.llm_{branch}.config.fp32_layers',
                   'requires an explicit list or tuple of layer indices')
    common = {}
    for name in ('adarms_fp32', 'attention_math_fp32'):
        left = _attribute(backbone, name, 'model.llm_backbone.config')
        right = _attribute(expert, name, 'model.llm_expert.config')
        if type(left) is not bool or type(right) is not bool or left != right:
            _error(f'precision.{name}', 'both Gemma branches must agree')
        common[name] = left
    amp = _attribute(model, 'enable_mixed_precision_training', 'model')
    if type(amp) is not bool:
        _error('model.enable_mixed_precision_training', 'expected a boolean')
    provenance = {
        field.name: getattr(preprocessing, field.name)
        for field in fields(PreprocessingConfig)
    }
    provenance['view_order'] = list(preprocessing.view_order)
    return {
        'schema_version': 1,
        'recipe': 'pi05-v1',
        'checkpoint': {
            'format': 'fluxvla-native',
            'kind': 'dense',
            'sha256': checkpoint_sha256
        },
        'inputs': {
            'format': 'safetensors',
            'sha256': inputs_sha256,
            'tensors': {
                name: {
                    'dtype': str(value.dtype).removeprefix('torch.'),
                    'shape': list(value.shape)
                }
                for name, value in inputs.items()
            }
        },
        'model': architecture,
        'inference': {
            'mode': 'plain',
            'solver': 'euler',
            'num_steps': _attribute(model, 'num_steps', 'model'),
            'rtc': False
        },
        'precision': {
            'parameter_dtype':
            'float32',
            'autocast_dtype':
            'bfloat16' if amp else None,
            'attention_backend':
            _attribute(model, 'attention_implementation', 'model'),
            'flow_fp32':
            _attribute(model, 'openpi_fp32_flow', 'model'),
            'residual_fp32':
            _attribute(model, 'preserve_fp32_residuals', 'model'),
            'vision_stem_fp32':
            vision.openpi_stem_fp32,
            'vision_residual_fp32':
            vision.preserve_fp32_residuals,
            **common,
            'backbone_fp32_layers':
            list(
                _attribute(backbone, 'fp32_layers',
                           'model.llm_backbone.config')),
            'expert_fp32_layers':
            list(_attribute(expert, 'fp32_layers', 'model.llm_expert.config')),
        },
        'preprocessing': provenance,
    }


def _compare_component(actual: Any, expected: Any, path: str) -> None:
    """Validate topology against the canonical constructor's meta-only model.

    No forward or numerical baseline uses meta tensors. This avoids allocating
    another full model merely to reject added/removed state or changed modules.
    """
    import torch

    adapters = {
        ('fluxvla.models.backbones.llms.condition_gemma',
         'ConditionGemmaModel'),
        ('fluxvla.models.backbones.visions.siglip_vit', 'SigLIPViTBackbone'),
    }
    cls = type(actual)
    allowed_adapter = (
        isinstance(actual, type(expected))
        and (cls.__module__, cls.__qualname__) in adapters)
    if type(actual) is not type(expected) and not allowed_adapter:
        _error(
            path, f'unsupported module {cls.__module__}.{cls.__name__}; '
            f'expected {type(expected).__name__}')
    for name in ('forward', '_forward_openpi_stem', '_norm',
                 'compute_default_rope_parameters'):
        if hasattr(expected, name):
            _same_method(actual, expected, name, path)
    for member in ('_modules', '_parameters', '_buffers',
                   '_non_persistent_buffers_set'):
        left, right = getattr(actual, member), getattr(expected, member)
        if set(left) != set(right):
            _error(
                path, f'{member} layout differs from canonical schema: '
                f'{sorted(set(left) ^ set(right))}')
    for member in ('_parameters', '_buffers'):
        for name, right in getattr(expected, member).items():
            left = getattr(actual, member)[name]
            if (left is None) != (right is None):
                _error(f'{path}.{name}', 'missing or unexpected tensor')
            if left is not None and left.device.type == 'meta':
                _error(f'{path}.{name}', 'requires a materialized tensor')
            if left is not None and (left.shape != right.shape
                                     or left.dtype != right.dtype):
                _error(f'{path}.{name}', 'shape/dtype differs from schema')
    # Structural constructor fields and runtime precision flags must agree;
    # accepting only config dictionaries would miss post-construction edits.
    for name in ('in_features', 'out_features', 'kernel_size', 'stride',
                 'padding', 'dilation', 'groups', 'padding_mode',
                 'num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                 'norm_type', 'scale_grad_by_freq', 'sparse',
                 'normalized_shape', 'eps', 'elementwise_affine', 'dim',
                 'cond_dim', 'adarms_fp32', 'head_dim', 'num_key_value_groups',
                 'scaling', 'attention_dropout', 'is_causal', 'compute_fp32',
                 'layer_idx', 'hidden_size', 'intermediate_size', 'num_heads',
                 'scale', 'dropout', 'embed_dim', 'image_size', 'patch_size',
                 'num_patches', 'num_positions', 'use_head', 'rope_type',
                 'attention_scaling'):
        if hasattr(expected, name):
            left, right = getattr(actual, name, None), getattr(expected, name)
            if isinstance(right,
                          (str, int, float, bool, tuple)) or right is None:
                if type(left) is not type(right) or left != right:
                    _error(
                        f'{path}.{name}', 'live computation differs from '
                        f'derived schema: {left!r} != {right!r}')
    for name, module in expected._modules.items():
        if module is not None:
            _compare_component(actual._modules[name], module, f'{path}.{name}')
    # Nonpersistent buffers are absent from the authoritative checkpoint but
    # must still agree with canonical construction, not caller modifications.
    if 'position_ids' in expected._buffers:
        value = actual._buffers['position_ids']
        canonical = torch.arange(value.shape[-1], device=value.device)[None]
        if not torch.equal(value, canonical):
            _error(f'{path}.position_ids', 'modified nonpersistent positions')
    if 'inv_freq' in expected._buffers:
        canonical, _ = type(expected).compute_default_rope_parameters(
            actual.config, device=actual.inv_freq.device)
        if not torch.equal(actual.inv_freq, canonical) or not torch.equal(
                actual.original_inv_freq, canonical):
            _error(f'{path}.inv_freq', 'modified nonpersistent RoPE buffer')


def _validate_components(model: Any, config: Any) -> None:
    import torch

    from .model import PI05Model

    # Use the sole canonical constructor, including its exact master-parameter
    # normalization, but allocate no real model weights and run no inference.
    with torch.random.fork_rng(devices=[]):
        canonical = PI05Model(config, _initialization_device='meta')
    if (model.vision_backbone.vision.config._attn_implementation
            != canonical.vision_backbone.vision.config._attn_implementation):
        _error('model.vision_backbone.vision.config._attn_implementation',
               'schema 1 cannot represent an overridden vision backend')
    for name in _COMPONENT_NAMES:
        _compare_component(
            getattr(model, name), getattr(canonical, name), f'model.{name}')
    if model.attention_interface is not model.get_attention_interface():
        _error('model.attention_interface', 'does not match declared backend')


def _snapshot_inputs(inputs: Mapping[str, torch.Tensor]) -> dict:
    import torch

    if not isinstance(inputs, Mapping) or set(inputs) != set(_INPUT_NAMES):
        _error('inputs', f'requires exactly {_INPUT_NAMES}, including noise')
    result = {}
    for name in _INPUT_NAMES:
        value = inputs[name]
        if type(value) is not torch.Tensor or value.layout != torch.strided:
            _error(f'inputs.{name}', 'requires a dense torch.Tensor')
        if value.device.type == 'meta':
            _error(f'inputs.{name}', 'requires actual materialized values')
        rank = 4 if name == 'images' else 3 if name == 'noise' else 2
        if value.ndim != rank:
            _error(f'inputs.{name}', f'expected rank {rank}')
        _validate_finite(value, owner=f'Producer inputs.{name}')
        # Preserve exact dtype/value. CPU contiguous copies decouple the file
        # writer from caller storage, gradients and noncontiguous tensor views.
        result[name] = value.detach().to(device='cpu').contiguous().clone()
    return result


def produce_pi05_artifacts(
    *,
    model: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    checkpoint: str | Path,
    preprocessing: PreprocessingConfig,
    output_dir: str | Path,
) -> PI05Artifacts:
    """Derive and write strict, hash-bound PI0.5 deployment artifacts.

    ``model`` must be an unwrapped canonical training PI05FlowMatching or
    standalone PI05Model, in eval mode with FP32 master parameters. Compute
    precision comes from its mixed-precision/protection attributes, never
    ambient autocast. The existing native dense checkpoint is authoritative:
    its keys, shapes, dtypes and finite values are checked. Its values need not
    equal the live policy, allowing an explicitly selected EMA checkpoint.

    All six preprocessed tensors, including fixed FP32 noise, and a complete
    PreprocessingConfig are mandatory. The training caller is responsible for
    the provenance's semantic truth. The producer does not preprocess inputs,
    run model forward, convert/copy checkpoints, or mutate caller data.

    The output directory must be new, with an existing parent. Validation
    precedes directory creation. Files use exclusive creation. Failed writes
    can leave a partial directory; this is reported and never reused.
    """
    from safetensors.torch import save

    destination = _path(output_dir, 'Output directory')
    requested = Path(output_dir).expanduser()
    if destination.exists() or requested.is_symlink():
        raise ArtifactError(f'Output directory {destination}: must be fresh; '
                            'refusing overwrite.')
    if type(preprocessing) is not PreprocessingConfig:
        _error('preprocessing',
               'requires an explicit complete PreprocessingConfig')
    if type(preprocessing.view_order) is not tuple:
        _error('preprocessing.view_order',
               'requires an explicit tuple of views')
    _validate_root(model)
    snapshot = _snapshot_inputs(inputs)
    try:
        fixture_bytes = save(snapshot)
    except (ValueError, TypeError, RuntimeError) as error:
        raise ArtifactError(
            f'Cannot serialize exact input tensors: {error}') from error
    checkpoint_path = _path(checkpoint, 'Checkpoint')
    checkpoint_sha256 = _checkpoint_digest(checkpoint_path)
    document = _derive_document(model, snapshot, preprocessing,
                                checkpoint_sha256,
                                hashlib.sha256(fixture_bytes).hexdigest())
    config = config_from_document(document, recipe='pi05-v1')
    validate_token_vocabulary(
        snapshot['lang_tokens'],
        vocab_size=config.model.vocab_size,
        label='Producer inputs')
    _validate_components(model, config)
    # No load_state_dict is called on the live model. The reader rejects any
    # missing/extra key, non-FP32 tensor, mismatched shape, or nonfinite value.
    validated_weights = load_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha256,
        expected_state=model.state_dict())
    del validated_weights
    config_bytes = (
        json.dumps(config_to_document(config), indent=2, allow_nan=False) +
        '\n').encode('utf-8')
    config_path = destination / 'export_config.json'
    inputs_path = destination / 'export_inputs.safetensors'
    try:
        destination.mkdir(exist_ok=False)
        for path, data in ((inputs_path, fixture_bytes), (config_path,
                                                          config_bytes)):
            with path.open('xb') as stream:
                stream.write(data)
    except OSError as error:
        raise ArtifactError(
            f'Output directory {destination}: cannot create artifacts: '
            f'{error}; a partial new directory may remain, and is never reused'
        ) from error
    return PI05Artifacts(
        config=config_path, inputs=inputs_path, checkpoint=checkpoint_path)
