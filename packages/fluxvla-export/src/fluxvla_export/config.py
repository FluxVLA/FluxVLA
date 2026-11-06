# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Strict, stdlib-only deployment configuration; no model is constructed.

Schema 1 describes the proposed pi05-v1 boundary, not an installed recipe.
It accepts native dense FP32 or explicitly declared BF16 checkpoint storage,
FP32 target parameters, optional BF16 autocast, plain Euler inference, and
fixed-shape six-tensor fixtures.
Training Python configs, dynamic profiles, RTC, LoRA, quantization, arbitrary
HF config extensions, and per-parameter storage dtype maps are not supported.
JSON fields are required except checkpoint.storage_dtype (default float32);
every object rejects additional fields.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from .contracts import TensorSpec
from .errors import ConfigError

_INPUT_NAMES = ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'states',
                'noise')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')


@dataclass(frozen=True)
class VisionConfig:
    type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    hidden_activation: str
    layer_norm_eps: float
    attention_dropout: float
    vision_use_head: bool


@dataclass(frozen=True)
class GemmaConfig:
    type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    hidden_activation: str
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    attention_dropout: float
    use_adarms: bool
    adarms_cond_dim: int | None
    use_cache: bool
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | None


@dataclass(frozen=True)
class LinearConfig:
    type: str
    in_dim: int
    out_dim: int
    bias: bool


@dataclass(frozen=True)
class ModelConfig:
    type: str
    vision: VisionConfig
    backbone: GemmaConfig
    expert: GemmaConfig
    projector: LinearConfig
    action_in_proj: LinearConfig
    action_out_proj: LinearConfig
    time_mlp_in: LinearConfig
    time_mlp_out: LinearConfig
    proj_width: int
    state_dim: int
    action_dim: int
    action_horizon: int
    state_projection: bool

    @property
    def vocab_size(self) -> int:
        """Vocabulary bound for all stored tokens, including padding."""
        return self.backbone.vocab_size


@dataclass(frozen=True)
class InferenceConfig:
    mode: str
    solver: str
    num_steps: int
    rtc: bool


@dataclass(frozen=True)
class PrecisionConfig:
    parameter_dtype: str
    autocast_dtype: str | None
    attention_backend: str
    flow_fp32: bool
    vision_stem_fp32: bool
    residual_fp32: bool
    vision_residual_fp32: bool
    adarms_fp32: bool
    attention_math_fp32: bool
    backbone_fp32_layers: tuple[int, ...]
    expert_fp32_layers: tuple[int, ...]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Training-side provenance; this package does not preprocess inputs."""

    id: str
    tokenizer_id: str
    tokenizer_sha256: str
    prompt_id: str
    normalization_id: str
    normalization_sha256: str
    state_semantics: str
    action_semantics: str
    image_transform_id: str
    view_order: tuple[str, ...]
    state_in_prompt: bool


@dataclass(frozen=True)
class DeploymentConfig:
    """Validated deployment description returned by :func:`load_config`.

    Digests bind the supplied files, not inferred architecture or preprocessing
    equivalence. The artifact loaders must verify the two artifact digests.
    """

    schema_version: int
    recipe: str
    checkpoint_sha256: str
    inputs_sha256: str
    input_specs: tuple[TensorSpec, ...]
    model: ModelConfig
    inference: InferenceConfig
    precision: PrecisionConfig
    preprocessing: PreprocessingConfig
    checkpoint_storage_dtype: str = 'float32'


def _fail(path: str, message: str) -> None:
    raise ConfigError(f'{path}: {message}')


def _object(value: Any, path: str, keys: tuple[str, ...]) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, 'expected a JSON object')
    if any(type(key) is not str for key in value):
        _fail(path, 'object keys must be strings')
    missing = sorted(set(keys) - value.keys())
    unknown = sorted(value.keys() - set(keys))
    if missing or unknown:
        _fail(path, f'missing fields={missing}; unknown fields={unknown}')
    return dict(value)


def _record(value: Any, path: str, cls: type) -> dict[str, Any]:
    return _object(value, path, tuple(field.name for field in fields(cls)))


def _int(value: Any, path: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f'expected an integer >= {minimum}, got {value!r}')
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        _fail(path, f'expected a finite number, got {value!r}')
    try:
        result = float(value)
    except OverflowError:
        _fail(path, 'expected a finite number')
    if not math.isfinite(result) or (positive and result <= 0):
        _fail(
            path, 'expected a finite positive number'
            if positive else 'expected a finite number')
    return result


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, f'expected a boolean, got {value!r}')
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(path, 'expected a nonempty string')
    return value


def _choice(value: Any, path: str, choices: tuple[Any, ...]) -> Any:
    # Python considers True == 1, which is not a valid JSON type equivalence.
    if not any(
            type(value) is type(item) and value == item for item in choices):
        _fail(path,
              f'unsupported value {value!r}; expected one of {choices!r}')
    return value


def _digest(value: Any, path: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(path, 'expected a lowercase 64-character SHA256 hex digest')
    return value


def _array(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        _fail(path, 'expected a JSON array')
    return value


def _vision(value: Any, path: str) -> VisionConfig:
    data = _record(value, path, VisionConfig)
    _choice(data['type'], f'{path}.type', ('siglip', ))
    for key in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_channels', 'image_size',
                'patch_size'):
        _int(data[key], f'{path}.{key}')
    _choice(data['num_channels'], f'{path}.num_channels', (3, ))
    _choice(data['hidden_activation'], f'{path}.hidden_activation',
            ('gelu_pytorch_tanh', ))
    data['layer_norm_eps'] = _number(
        data['layer_norm_eps'], f'{path}.layer_norm_eps', positive=True)
    data['attention_dropout'] = _number(data['attention_dropout'],
                                        f'{path}.attention_dropout')
    _choice(data['attention_dropout'], f'{path}.attention_dropout', (0.0, ))
    _choice(data['vision_use_head'], f'{path}.vision_use_head', (False, ))
    if data['hidden_size'] % data['num_attention_heads']:
        _fail(f'{path}.hidden_size',
              'must be divisible by num_attention_heads')
    if data['image_size'] % data['patch_size']:
        _fail(f'{path}.image_size', 'must be divisible by patch_size')
    return VisionConfig(**data)


def _gemma(value: Any, path: str) -> GemmaConfig:
    data = _record(value, path, GemmaConfig)
    _choice(data['type'], f'{path}.type', ('condition_gemma', ))
    for key in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'head_dim',
                'vocab_size', 'max_position_embeddings'):
        _int(data[key], f'{path}.{key}')
    if data['num_attention_heads'] % data['num_key_value_heads']:
        _fail(f'{path}.num_attention_heads',
              'must be divisible by num_key_value_heads')
    if data['head_dim'] % 2:
        _fail(f'{path}.head_dim',
              'must be even for rotary position embeddings')
    # Gemma permits hidden_size != num_attention_heads * head_dim. The real
    # PI0.5 expert uses hidden_size=1024 with 8 heads of dimension 256.
    _choice(data['hidden_activation'], f'{path}.hidden_activation',
            ('gelu_pytorch_tanh', ))
    for key in ('rms_norm_eps', 'rope_theta'):
        data[key] = _number(data[key], f'{path}.{key}', positive=True)
    _choice(data['attention_bias'], f'{path}.attention_bias', (False, ))
    data['attention_dropout'] = _number(data['attention_dropout'],
                                        f'{path}.attention_dropout')
    _choice(data['attention_dropout'], f'{path}.attention_dropout', (0.0, ))
    _bool(data['use_adarms'], f'{path}.use_adarms')
    _choice(data['use_cache'], f'{path}.use_cache', (True, ))
    if data['adarms_cond_dim'] is not None:
        _int(data['adarms_cond_dim'], f'{path}.adarms_cond_dim')
    for key in ('pad_token_id', 'bos_token_id', 'eos_token_id'):
        if data[key] is not None:
            _int(data[key], f'{path}.{key}', minimum=0)
            if data[key] >= data['vocab_size']:
                _fail(f'{path}.{key}', 'must be smaller than vocab_size')
    return GemmaConfig(**data)


def _linear(value: Any, path: str) -> LinearConfig:
    data = _record(value, path, LinearConfig)
    _choice(data['type'], f'{path}.type', ('linear', ))
    _int(data['in_dim'], f'{path}.in_dim')
    _int(data['out_dim'], f'{path}.out_dim')
    _choice(data['bias'], f'{path}.bias', (True, ))
    return LinearConfig(**data)


def _model(value: Any) -> ModelConfig:
    path = 'model'
    data = _record(value, path, ModelConfig)
    _choice(data['type'], 'model.type', ('pi05', ))
    _choice(data['state_projection'], 'model.state_projection', (False, ))
    for key in ('proj_width', 'state_dim', 'action_dim', 'action_horizon'):
        _int(data[key], f'model.{key}')
    if data['proj_width'] % 2:
        _fail('model.proj_width', 'must be even for timestep embeddings')
    data['vision'] = _vision(data['vision'], 'model.vision')
    data['backbone'] = _gemma(data['backbone'], 'model.backbone')
    data['expert'] = _gemma(data['expert'], 'model.expert')
    for key in ('projector', 'action_in_proj', 'action_out_proj',
                'time_mlp_in', 'time_mlp_out'):
        data[key] = _linear(data[key], f'model.{key}')
    model = ModelConfig(**data)
    if model.backbone.adarms_cond_dim is not None:
        _fail('model.backbone.adarms_cond_dim', 'must be null for the prefix')
    if not model.expert.use_adarms:
        _fail('model.expert.use_adarms', 'must be true for PI0.5')
    if model.expert.adarms_cond_dim != model.proj_width:
        _fail('model.expert.adarms_cond_dim', 'must equal model.proj_width')
    if model.expert.hidden_size != model.proj_width:
        _fail('model.expert.hidden_size', 'must equal model.proj_width')
    for key in ('num_hidden_layers', 'num_key_value_heads', 'head_dim',
                'rope_theta', 'vocab_size'):
        if getattr(model.backbone, key) != getattr(model.expert, key):
            _fail(f'model.expert.{key}', f'must match model.backbone.{key}')
    expected = {
        'projector': (model.vision.hidden_size, model.backbone.hidden_size),
        'action_in_proj': (model.action_dim, model.proj_width),
        'action_out_proj': (model.proj_width, model.action_dim),
        'time_mlp_in': (model.proj_width, model.proj_width),
        'time_mlp_out': (model.proj_width, model.proj_width),
    }
    for key, dimensions in expected.items():
        projection = getattr(model, key)
        if (projection.in_dim, projection.out_dim) != dimensions:
            _fail(f'model.{key}',
                  f'in_dim/out_dim must equal {dimensions} for PI0.5')
    return model


def _inference(value: Any) -> InferenceConfig:
    data = _record(value, 'inference', InferenceConfig)
    _choice(data['mode'], 'inference.mode', ('plain', ))
    _choice(data['solver'], 'inference.solver', ('euler', ))
    _int(data['num_steps'], 'inference.num_steps')
    _choice(data['rtc'], 'inference.rtc', (False, ))
    return InferenceConfig(**data)


def _precision(value: Any, model: ModelConfig) -> PrecisionConfig:
    data = _record(value, 'precision', PrecisionConfig)
    _choice(data['parameter_dtype'], 'precision.parameter_dtype',
            ('float32', ))
    _choice(data['autocast_dtype'], 'precision.autocast_dtype',
            (None, 'bfloat16'))
    _choice(data['attention_backend'], 'precision.attention_backend',
            ('sdpa', 'eager'))
    for key in ('flow_fp32', 'vision_stem_fp32', 'residual_fp32',
                'vision_residual_fp32', 'adarms_fp32', 'attention_math_fp32'):
        _bool(data[key], f'precision.{key}')
    if data['attention_math_fp32'] and data['attention_backend'] != 'sdpa':
        _fail('precision.attention_backend',
              'attention_math_fp32 requires sdpa')
    for branch in ('backbone', 'expert'):
        key = f'{branch}_fp32_layers'
        path = f'precision.{key}'
        layers = _array(data[key], path)
        for index, layer in enumerate(layers):
            _int(layer, f'{path}[{index}]', minimum=0)
            if layer >= getattr(model, branch).num_hidden_layers:
                _fail(f'{path}[{index}]', 'layer index is outside the model')
        if len(set(layers)) != len(layers):
            _fail(path, 'layer indices must be unique')
        data[key] = tuple(layers)
    return PrecisionConfig(**data)


def _preprocessing(value: Any) -> PreprocessingConfig:
    data = _record(value, 'preprocessing', PreprocessingConfig)
    for key in ('id', 'tokenizer_id', 'prompt_id', 'normalization_id',
                'state_semantics', 'action_semantics', 'image_transform_id'):
        _string(data[key], f'preprocessing.{key}')
    for key in ('tokenizer_sha256', 'normalization_sha256'):
        _digest(data[key], f'preprocessing.{key}')
    _bool(data['state_in_prompt'], 'preprocessing.state_in_prompt')
    views = _array(data['view_order'], 'preprocessing.view_order')
    if not views:
        _fail('preprocessing.view_order', 'at least one view is required')
    for index, view in enumerate(views):
        _string(view, f'preprocessing.view_order[{index}]')
    if len(set(views)) != len(views):
        _fail('preprocessing.view_order', 'view identifiers must be unique')
    data['view_order'] = tuple(views)
    return PreprocessingConfig(**data)


def _inputs(value: Any, model: ModelConfig, precision: PrecisionConfig,
            preprocessing: PreprocessingConfig) -> tuple[TensorSpec, ...]:
    data = _object(value, 'inputs.tensors', _INPUT_NAMES)
    specs = []
    for name in _INPUT_NAMES:
        path = f'inputs.tensors.{name}'
        tensor = _object(data[name], path, ('dtype', 'shape'))
        allowed = {
            'images': ('float32', 'bfloat16'),
            'img_masks': ('bool', ),
            'lang_tokens': ('int64', ),
            'lang_masks': ('bool', ),
            'states': ('float32', 'bfloat16'),
            'noise': ('float32', ),
        }
        _choice(tensor['dtype'], f'{path}.dtype', allowed[name])
        if (tensor['dtype'] == 'bfloat16'
                and precision.autocast_dtype != 'bfloat16'):
            _fail(f'{path}.dtype', 'bfloat16 inputs require bfloat16 autocast')
        shape = _array(tensor['shape'], f'{path}.shape')
        rank = {'images': 4, 'noise': 3}.get(name, 2)
        if len(shape) != rank:
            _fail(f'{path}.shape', f'expected rank {rank}, got {len(shape)}')
        for index, dimension in enumerate(shape):
            _int(dimension, f'{path}.shape[{index}]')
        specs.append(TensorSpec(name, tensor['dtype'], tuple(shape)))
    shapes = {spec.name: spec.shape for spec in specs}
    batch = shapes['images'][0]
    views = len(preprocessing.view_order)
    language_length = shapes['lang_tokens'][1]
    expected = {
        'images':
        (batch, 3 * views, model.vision.image_size, model.vision.image_size),
        'img_masks': (batch, views),
        'lang_tokens': (batch, language_length),
        'lang_masks': (batch, language_length),
        'states': (batch, model.state_dim),
        'noise': (batch, model.action_horizon, model.action_dim),
    }
    for name, shape in expected.items():
        if shapes[name] != shape:
            _fail(
                f'inputs.tensors.{name}.shape',
                f'expected {shape} from model, batch and view_order; '
                f'got {shapes[name]}')
    prefix_length = (
        views * (model.vision.image_size // model.vision.patch_size)**2 +
        language_length)
    if prefix_length > model.backbone.max_position_embeddings:
        _fail('model.backbone.max_position_embeddings',
              f'must cover all {prefix_length} unmasked prefix positions')
    if (prefix_length + model.action_horizon
            > model.expert.max_position_embeddings):
        _fail('model.expert.max_position_embeddings',
              'must cover the full prefix plus action_horizon')
    return tuple(specs)


def _deployment(value: Any, recipe: str) -> DeploymentConfig:
    data = _object(value, 'config',
                   ('schema_version', 'recipe', 'checkpoint', 'inputs',
                    'model', 'inference', 'precision', 'preprocessing'))
    _choice(data['schema_version'], 'schema_version', (1, ))
    _choice(data['recipe'], 'recipe', ('pi05-v1', ))
    if data['recipe'] != recipe:
        _fail(
            'recipe', f'configuration {data["recipe"]!r} does not match '
            f'requested recipe {recipe!r}')
    checkpoint_data = data['checkpoint']
    if type(checkpoint_data) is dict:
        checkpoint_data = {'storage_dtype': 'float32', **checkpoint_data}
    checkpoint = _object(checkpoint_data, 'checkpoint',
                         ('format', 'kind', 'sha256', 'storage_dtype'))
    storage_dtype = _choice(checkpoint['storage_dtype'],
                            'checkpoint.storage_dtype',
                            ('float32', 'bfloat16'))
    _choice(checkpoint['format'], 'checkpoint.format', ('fluxvla-native', ))
    _choice(checkpoint['kind'], 'checkpoint.kind', ('dense', ))
    checkpoint_sha256 = _digest(checkpoint['sha256'], 'checkpoint.sha256')
    inputs = _object(data['inputs'], 'inputs', ('format', 'sha256', 'tensors'))
    _choice(inputs['format'], 'inputs.format', ('safetensors', ))
    inputs_sha256 = _digest(inputs['sha256'], 'inputs.sha256')
    model = _model(data['model'])
    inference = _inference(data['inference'])
    precision = _precision(data['precision'], model)
    preprocessing = _preprocessing(data['preprocessing'])
    input_specs = _inputs(inputs['tensors'], model, precision, preprocessing)
    return DeploymentConfig(
        schema_version=data['schema_version'],
        recipe=recipe,
        checkpoint_sha256=checkpoint_sha256,
        inputs_sha256=inputs_sha256,
        input_specs=input_specs,
        model=model,
        inference=inference,
        precision=precision,
        preprocessing=preprocessing,
        checkpoint_storage_dtype=storage_dtype)


def config_from_document(document: Any, *, recipe: str) -> DeploymentConfig:
    """Validate an in-memory JSON document with the same schema as file input."""
    _string(recipe, 'recipe argument')
    return _deployment(document, recipe)


def _config_document(config: DeploymentConfig) -> dict[str, Any]:
    if type(config) is not DeploymentConfig:
        _fail('config', 'expected DeploymentConfig')
    _choice(config.checkpoint_storage_dtype, 'checkpoint.storage_dtype',
            ('float32', 'bfloat16'))
    if type(config.input_specs) is not tuple:
        _fail('input_specs', 'expected an ordered tuple of TensorSpec')
    if any(type(spec) is not TensorSpec for spec in config.input_specs):
        _fail('input_specs', 'expected TensorSpec entries')
    if tuple(spec.name for spec in config.input_specs) != _INPUT_NAMES:
        _fail('input_specs', f'expected exact input order {_INPUT_NAMES}')
    for spec in config.input_specs:
        if any(
                value is not None
                for value in (spec.min_shape, spec.opt_shape, spec.max_shape)):
            _fail(f'input_specs.{spec.name}',
                  'schema 1 requires static specs without profiles')

    def document_value(value):
        # Convert dataclass containers without copying tensor/foreign objects.
        # The schema below rejects such objects at their actual field path.
        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: document_value(getattr(value, item.name))
                for item in fields(value)
            }
        if type(value) in (tuple, list):
            return [document_value(item) for item in value]
        if type(value) is dict:
            return {key: document_value(item) for key, item in value.items()}
        return value

    return {
        'schema_version': config.schema_version,
        'recipe': config.recipe,
        'checkpoint': {
            'format': 'fluxvla-native',
            'kind': 'dense',
            'sha256': config.checkpoint_sha256,
            **({'storage_dtype': config.checkpoint_storage_dtype}
               if config.checkpoint_storage_dtype != 'float32' else {}),
        },
        'inputs': {
            'format': 'safetensors',
            'sha256': config.inputs_sha256,
            'tensors': {
                spec.name: {
                    'dtype': spec.dtype,
                    'shape': list(spec.shape)
                }
                for spec in config.input_specs
            }
        },
        'model': document_value(config.model),
        'inference': document_value(config.inference),
        'precision': document_value(config.precision),
        'preprocessing': document_value(config.preprocessing),
    }


def config_to_document(config: DeploymentConfig) -> dict[str, Any]:
    """Serialize validated schema fields without discarding unsupported data."""
    document = _config_document(config)
    config_from_document(document, recipe=config.recipe)
    return document


def validate_config(config: DeploymentConfig) -> DeploymentConfig:
    """Revalidate caller-constructed/replaced dataclasses and return canonical data."""
    return config_from_document(_config_document(config), recipe=config.recipe)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail('JSON', f'duplicate object key {key!r}')
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail('JSON', f'non-finite number {value!r} is not allowed')


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _fail('JSON', f'non-finite number {value!r} is not allowed')
    return result


def load_config(path: str | Path, *, recipe: str) -> DeploymentConfig:
    """Read a strict UTF-8 JSON deployment description, without artifact I/O.

    Only this explicit path is read. No neighboring files, training config
    execution, weight-based architecture inference or recipe registration
    occurs. Artifact digests are checked here for syntax; loaders verify them.
    Errors include the source path and the invalid JSON field when available.
    """
    _string(recipe, 'recipe argument')
    if not isinstance(path, (str, Path)) or not str(path).strip():
        _fail('config path', 'expected a nonempty filesystem path')
    try:
        source = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(
            f'{path}: cannot resolve config path: {error}') from error
    try:
        # On supported POSIX deployment hosts, a FIFO must not block before
        # fstat can reject it. open owns/closes its descriptor on every path.
        with open(
                source,
                encoding='utf-8',
                opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK)
        ) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                _fail(str(source), 'config must be a regular file')
            text = stream.read()
    except ConfigError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError(
            f'{source}: cannot read UTF-8 config: {error}') from error
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float)
        return _deployment(document, recipe)
    except ConfigError as error:
        raise ConfigError(f'{source}: {error}') from error
    except (ValueError, RecursionError) as error:
        raise ConfigError(f'{source}: invalid JSON: {error}') from error
