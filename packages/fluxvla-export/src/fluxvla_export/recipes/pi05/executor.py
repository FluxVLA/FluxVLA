# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""PI0.5 engine orchestration reconstructed from JSON-safe runtime metadata.

Importing this module does not load torch, model weights, Transformers or the
training stack. Engines own graph execution; this module owns the six-input
contract and delegates the exact plain Euler loop to the shared torch helper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..._validation import validate_token_vocabulary
from ...contracts import InferenceEngine, TensorSpec, _validate_tensor
from ...errors import ContractError

if TYPE_CHECKING:
    import torch

_RUNTIME_ENTRYPOINT = 'fluxvla_export.recipes.pi05.executor:build_executor'
_INPUT_NAMES = ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'states',
                'noise')
_GRAPH_NAMES = ('prefill', 'denoise')


def tensor_spec_document(spec: TensorSpec) -> dict[str, Any]:
    """Serialize one fixed graph boundary, rejecting dynamic profiles."""
    if not isinstance(spec, TensorSpec) or any(
            value is not None
            for value in (spec.min_shape, spec.opt_shape, spec.max_shape)):
        raise ContractError('PI0.5 runtime requires fixed TensorSpec values.')
    if any(size <= 0 for size in spec.shape):
        raise ContractError(
            'PI0.5 runtime requires positive fixed dimensions.')
    return {'name': spec.name, 'dtype': spec.dtype, 'shape': list(spec.shape)}


def _object(value: Any, keys: Sequence[str], owner: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContractError(f'{owner}: expected an object.')
    if any(type(name) is not str for name in value):
        raise ContractError(f'{owner}: object keys must be strings.')
    missing = sorted(set(keys) - set(value))
    unknown = sorted(set(value) - set(keys))
    if missing or unknown:
        raise ContractError(
            f'{owner}: missing fields={missing}; unknown fields={unknown}.')
    return value


def _positive_int(value: Any, owner: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(f'{owner}: expected a positive integer.')
    return value


def _spec(value: Any, owner: str) -> TensorSpec:
    data = _object(value, ('name', 'dtype', 'shape'), owner)
    if type(data['name']) is not str or not data['name']:
        raise ContractError(f'{owner}.name: expected a nonempty string.')
    if type(data['dtype']) is not str or data['dtype'] not in (
            'float32', 'bfloat16', 'int64', 'bool'):
        raise ContractError(f'{owner}.dtype: unsupported tensor dtype.')
    if type(data['shape']) is not list or not data['shape']:
        raise ContractError(f'{owner}.shape: expected a nonempty JSON array.')
    shape = tuple(
        _positive_int(size, f'{owner}.shape[{index}]')
        for index, size in enumerate(data['shape']))
    return TensorSpec(data['name'], data['dtype'], shape)


def _specs(value: Any, owner: str) -> tuple[TensorSpec, ...]:
    if type(value) is not list or not value:
        raise ContractError(f'{owner}: expected a nonempty JSON array.')
    specs = tuple(
        _spec(item, f'{owner}[{index}]') for index, item in enumerate(value))
    if len({spec.name for spec in specs}) != len(specs):
        raise ContractError(f'{owner}: tensor names must be unique.')
    return specs


def _require_spec(spec: TensorSpec,
                  *,
                  name: str,
                  dtypes: tuple[str, ...],
                  shape: tuple[int, ...] | None = None,
                  rank: int | None = None) -> None:
    if spec.name != name or spec.dtype not in dtypes:
        raise ContractError(
            f'PI0.5 runtime {name}: expected name={name!r}, dtype in '
            f'{dtypes}; got name={spec.name!r}, dtype={spec.dtype!r}.')
    if shape is not None and spec.shape != shape:
        raise ContractError(
            f'PI0.5 runtime {name}: shape {spec.shape} must equal {shape}.')
    if rank is not None and len(spec.shape) != rank:
        raise ContractError(f'PI0.5 runtime {name}: expected rank {rank}.')


@dataclass(frozen=True)
class _RuntimeContract:
    num_steps: int
    vocab_size: int
    input_specs: tuple[TensorSpec, ...]
    prefill_output_specs: tuple[TensorSpec, ...]
    velocity_spec: TensorSpec
    output_spec: TensorSpec


def _runtime_contract(metadata: Mapping[str, Any]) -> _RuntimeContract:
    data = _object(metadata, (
        'entrypoint',
        'schema_version',
        'recipe',
        'num_steps',
        'vocab_size',
        'input_specs',
        'prefill_output_specs',
        'velocity_spec',
        'output_spec',
    ), 'PI0.5 runtime metadata')
    if data['entrypoint'] != _RUNTIME_ENTRYPOINT:
        raise ContractError('PI0.5 runtime metadata: unsupported entrypoint.')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ContractError(
            'PI0.5 runtime metadata: unsupported schema_version.')
    if data['recipe'] != 'pi05-v1':
        raise ContractError('PI0.5 runtime metadata: expected recipe pi05-v1.')
    num_steps = _positive_int(data['num_steps'], 'PI0.5 runtime num_steps')
    vocab_size = _positive_int(data['vocab_size'], 'PI0.5 runtime vocab_size')
    inputs = _specs(data['input_specs'], 'PI0.5 runtime input_specs')
    if tuple(spec.name for spec in inputs) != _INPUT_NAMES:
        raise ContractError(
            'PI0.5 runtime input_specs: expected six input names '
            f'in order {_INPUT_NAMES}.')
    by_name = {spec.name: spec for spec in inputs}
    for name in _INPUT_NAMES:
        dtypes = {
            'images': ('float32', 'bfloat16'),
            'img_masks': ('bool', ),
            'lang_tokens': ('int64', ),
            'lang_masks': ('bool', ),
            'states': ('float32', 'bfloat16'),
            'noise': ('float32', ),
        }[name]
        rank = {'images': 4, 'noise': 3}.get(name, 2)
        _require_spec(by_name[name], name=name, dtypes=dtypes, rank=rank)
    batch, channels, height, width = by_name['images'].shape
    views = by_name['img_masks'].shape[1]
    language_length = by_name['lang_tokens'].shape[1]
    if channels != 3 * views or height != width:
        raise ContractError(
            'PI0.5 runtime images: expected packed square RGB views.')
    for spec in inputs:
        if spec.shape[0] != batch:
            raise ContractError(f'PI0.5 runtime {spec.name}: batch mismatch.')
    if by_name['lang_masks'].shape != by_name['lang_tokens'].shape:
        raise ContractError(
            'PI0.5 runtime lang_masks: must match lang_tokens.')

    prefill = _specs(data['prefill_output_specs'],
                     'PI0.5 runtime prefill_output_specs')
    if len(prefill) < 3 or len(prefill) % 2 != 1:
        raise ContractError(
            'PI0.5 runtime prefill requires a mask and complete '
            'interleaved key/value pairs.')
    _require_spec(
        prefill[0], name='prefix_pad_masks', dtypes=('bool', ), rank=2)
    prefix_length = prefill[0].shape[1]
    if prefill[0].shape[0] != batch or prefix_length <= language_length or (
            prefix_length - language_length) % views:
        raise ContractError('PI0.5 runtime prefix_pad_masks: inconsistent '
                            'batch, views or language length.')
    cache_shape = None
    for layer in range((len(prefill) - 1) // 2):
        key, value = prefill[1 + 2 * layer:3 + 2 * layer]
        for kind, spec in (('key', key), ('value', value)):
            _require_spec(
                spec,
                name=f'{kind}_{layer}',
                dtypes=('float32', 'bfloat16'),
                rank=4)
            if spec.shape[0] != batch or spec.shape[2] != prefix_length:
                raise ContractError(f'PI0.5 runtime {spec.name}: cache shape '
                                    'does not match prefix batch/length.')
            if cache_shape is None:
                cache_shape = spec.shape
            elif spec.shape != cache_shape:
                raise ContractError('PI0.5 runtime cache shapes must agree.')
        if key.dtype != value.dtype:
            raise ContractError(
                f'PI0.5 runtime cache layer {layer}: key/value '
                'dtypes must agree.')
    velocity = _spec(data['velocity_spec'], 'PI0.5 runtime velocity_spec')
    actions = _spec(data['output_spec'], 'PI0.5 runtime output_spec')
    _require_spec(
        velocity,
        name='velocity',
        dtypes=('float32', 'bfloat16'),
        shape=by_name['noise'].shape)
    _require_spec(
        actions,
        name='actions',
        dtypes=('float32', ),
        shape=by_name['noise'].shape)
    return _RuntimeContract(num_steps, vocab_size, inputs, prefill, velocity,
                            actions)


def make_runtime_metadata(
    *,
    num_steps: int,
    vocab_size: int,
    input_specs: Sequence[TensorSpec],
    prefill_output_specs: Sequence[TensorSpec],
    velocity_spec: TensorSpec,
    output_spec: TensorSpec,
) -> dict[str, Any]:
    """Build and validate the self-contained JSON-safe runtime block."""
    result = {
        'entrypoint':
        _RUNTIME_ENTRYPOINT,
        'schema_version':
        1,
        'recipe':
        'pi05-v1',
        'num_steps':
        num_steps,
        'vocab_size':
        vocab_size,
        'input_specs': [tensor_spec_document(spec) for spec in input_specs],
        'prefill_output_specs':
        [tensor_spec_document(spec) for spec in prefill_output_specs],
        'velocity_spec':
        tensor_spec_document(velocity_spec),
        'output_spec':
        tensor_spec_document(output_spec),
    }
    _runtime_contract(result)
    return result


def _checked_tensors(
        values: Mapping[str, torch.Tensor],
        specs: Sequence[TensorSpec],
        *,
        owner: str,
        device: torch.device | None = None) -> dict[str, torch.Tensor]:
    import torch

    ordered = _object(values, tuple(spec.name for spec in specs), owner)
    result = {}
    for spec in specs:
        tensor = ordered[spec.name]
        if not isinstance(tensor, torch.Tensor):
            raise ContractError(
                f'{owner}.{spec.name}: expected a torch Tensor.')
        _validate_tensor(tensor, spec, owner=owner)
        if tensor.device.type not in ('cpu', 'cuda'):
            raise ContractError(f'{owner}.{spec.name}: unsupported device '
                                f'{tensor.device}.')
        if device is not None and tensor.device != device:
            raise ContractError(f'{owner}.{spec.name}: device {tensor.device} '
                                f'does not match {device}.')
        if tensor.is_floating_point(
        ) and not torch.isfinite(tensor).all().item():
            raise ContractError(f'{owner}.{spec.name}: tensor must be finite.')
        result[spec.name] = tensor
    return result


class PI05Executor:
    """Execute real prefill/denoise engines over validated policy tensors.

    Engines must obey the InferenceEngine contract without modifying input
    storage. This executor clones only the evolving noise and never updates
    the prefix cache or caller fixture. Model weights are not held here.
    """

    def __init__(self, engines: Mapping[str, InferenceEngine], *,
                 metadata: Mapping[str, Any]) -> None:
        self._contract = _runtime_contract(metadata)
        ordered = _object(engines, _GRAPH_NAMES, 'PI0.5 engines')
        for name in _GRAPH_NAMES:
            if not callable(getattr(ordered[name], 'infer', None)):
                raise ContractError(
                    f'PI0.5 engine {name}: infer must be callable.')
        self._engines = MappingProxyType(
            {name: ordered[name]
             for name in _GRAPH_NAMES})

    def infer(self,
              inputs: Mapping[str, torch.Tensor],
              *,
              synchronize: bool = False) -> dict[str, torch.Tensor]:
        import torch

        from ...components.euler import plain_euler

        if type(synchronize) is not bool:
            raise ContractError('PI0.5 synchronize must be a boolean.')
        contract = self._contract
        tensors = _checked_tensors(
            inputs, contract.input_specs, owner='PI0.5 inputs')
        device = tensors['noise'].device
        if any(value.device != device for value in tensors.values()):
            raise ContractError('PI0.5 inputs must all use the same device.')
        validate_token_vocabulary(
            tensors['lang_tokens'],
            vocab_size=contract.vocab_size,
            label='PI0.5 inputs')
        # The graphs own configured autocast. The host timestep and Euler
        # arithmetic use the same FP32 scalars as the canonical policy.
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            prefix = _checked_tensors(
                self._engines['prefill'].infer(
                    {name: tensors[name]
                     for name in _INPUT_NAMES[:4]},
                    synchronize=synchronize),
                contract.prefill_output_specs,
                owner='PI0.5 prefill',
                device=device)
            x_t = tensors['noise'].clone()
            dt = torch.tensor(
                -1.0 / contract.num_steps, dtype=torch.float32, device=device)

            def denoise(actions, timestep):
                outputs = self._engines['denoise'].infer(
                    {
                        'noisy_actions': actions,
                        'timestep': timestep,
                        **prefix,
                    },
                    synchronize=synchronize)
                checked = _checked_tensors(
                    outputs, (contract.velocity_spec, ),
                    owner='PI0.5 denoise',
                    device=device)
                return checked['velocity']

            actions = plain_euler(x_t, denoise, x_t.shape[0], dt, device)
        return _checked_tensors({'actions': actions}, (contract.output_spec, ),
                                owner='PI0.5 outputs',
                                device=device)


def build_executor(engines: Mapping[str, InferenceEngine], *,
                   metadata: Mapping[str, Any]) -> PI05Executor:
    """Reconstruct the executor from the bundle's ``metadata['runtime']``."""
    return PI05Executor(engines, metadata=metadata)
