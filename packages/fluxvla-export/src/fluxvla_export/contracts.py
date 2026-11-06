# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence,
                    Tuple)

from .errors import ContractError

if TYPE_CHECKING:
    from .config import DeploymentConfig

Shape = Tuple[int, ...]


class TensorLike(Protocol):
    """Tensor surface used without importing PyTorch at package import time."""

    @property
    def shape(self) -> Sequence[int]:
        ...

    @property
    def dtype(self) -> Any:
        ...

    def detach(self) -> 'TensorLike':
        ...


class ExportModule(Protocol):
    """PyTorch module surface required by the ONNX handoff."""

    def __call__(self, *inputs: TensorLike) -> Any:
        ...

    def forward(self, *inputs: TensorLike) -> Any:
        ...

    def eval(self) -> 'ExportModule':
        ...

    def state_dict(self) -> Mapping[str, TensorLike]:
        ...


def _normalize_shape(
    value: tuple[int, ...],
    *,
    profile: bool = False,
) -> Shape:
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool):
            raise ContractError(
                f'Tensor dimensions must be integers, got {dimension!r}.')
        try:
            dimensions.append(operator.index(dimension))
        except TypeError as error:
            raise ContractError(
                f'Tensor dimensions must be integers, got {dimension!r}.'
            ) from error
    shape = tuple(dimensions)
    if not shape:
        raise ContractError(
            'Tensor shapes must contain at least one dimension.')
    if profile:
        if any(dimension <= 0 for dimension in shape):
            raise ContractError(
                'Profile shapes must contain only positive dimensions: '
                f'{shape}')
    elif any(dimension == 0 or dimension < -1 for dimension in shape):
        raise ContractError(f'Invalid tensor shape: {shape}')
    return shape


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix('torch.')


def _validate_module(module: ExportModule, *, graph: str) -> None:
    required = ('forward', 'eval', 'state_dict')
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if not callable(module) or missing:
        raise ContractError(
            f'{graph}: module must implement callable forward(), eval(), and '
            f'state_dict(); missing={missing}.')


def _validate_tensor(
    value: TensorLike,
    spec: 'TensorSpec',
    *,
    owner: str,
) -> None:
    if not callable(getattr(value, 'detach', None)):
        raise ContractError(
            f'{owner}.{spec.name}: value must be a tensor with detach().')
    try:
        actual_shape = _normalize_shape(tuple(value.shape), profile=True)
    except AttributeError as error:
        raise ContractError(
            f'{owner}.{spec.name}: tensor has no shape.') from error
    if len(actual_shape) != len(spec.shape):
        raise ContractError(
            f'{owner}.{spec.name}: shape {actual_shape} does not match '
            f'contract {spec.shape}.')
    for index, (actual, declared) in enumerate(zip(actual_shape, spec.shape)):
        if declared != -1 and actual != declared:
            raise ContractError(
                f'{owner}.{spec.name}: dimension {index}={actual} does not '
                f'match {declared}.')
        if declared == -1:
            assert spec.min_shape is not None
            assert spec.max_shape is not None
            lower = spec.min_shape[index]
            upper = spec.max_shape[index]
            if not lower <= actual <= upper:
                raise ContractError(
                    f'{owner}.{spec.name}: dimension {index}={actual} is '
                    f'outside [{lower}, {upper}].')
    actual_dtype = _dtype_name(getattr(value, 'dtype', None))
    if actual_dtype != spec.dtype:
        raise ContractError(
            f'{owner}.{spec.name}: dtype {actual_dtype!r} does not match '
            f'{spec.dtype!r}.')


@dataclass(frozen=True)
class TensorSpec:
    """One stable tensor boundary shared with FluxHermes."""

    name: str
    dtype: str
    shape: Shape
    min_shape: Shape | None = None
    opt_shape: Shape | None = None
    max_shape: Shape | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError('TensorSpec.name cannot be empty.')
        if not self.dtype:
            raise ContractError(f'{self.name}: dtype cannot be empty.')
        object.__setattr__(self, 'shape', _normalize_shape(self.shape))
        profiles = (self.min_shape, self.opt_shape, self.max_shape)
        if any(value is not None
               for value in profiles) and not all(value is not None
                                                  for value in profiles):
            raise ContractError(
                f'{self.name}: min/opt/max shapes must be provided together.')
        if all(value is not None for value in profiles):
            normalized = tuple(
                _normalize_shape(value, profile=True) for value in profiles
                if value is not None)
            minimum, optimum, maximum = normalized
            object.__setattr__(self, 'min_shape', minimum)
            object.__setattr__(self, 'opt_shape', optimum)
            object.__setattr__(self, 'max_shape', maximum)
            if any(len(value) != len(self.shape) for value in normalized):
                raise ContractError(
                    f'{self.name}: profile rank must match declared shape.')
            for index, values in enumerate(zip(minimum, optimum, maximum)):
                lower, typical, upper = values
                if not lower <= typical <= upper:
                    raise ContractError(
                        f'{self.name}: profile dimension {index} must satisfy '
                        f'min <= opt <= max, got {values}.')
                declared = self.shape[index]
                if declared != -1 and values != (declared, ) * 3:
                    raise ContractError(
                        f'{self.name}: static dimension {index}={declared} '
                        f'does not match profile {values}.')
        if -1 in self.shape and self.min_shape is None:
            raise ContractError(
                f'{self.name}: dynamic dimensions require min/opt/max shapes.')


class InferenceEngine(Protocol):
    """Small engine surface required by a model-specific executor."""

    def infer(
        self,
        inputs: Mapping[str, TensorLike],
        *,
        synchronize: bool = False,
    ) -> Mapping[str, TensorLike]:
        ...


class DeploymentExecutor(Protocol):
    """Run the complete policy over one or more named engines."""

    def infer(
        self,
        inputs: Mapping[str, TensorLike],
        *,
        synchronize: bool = False,
    ) -> Mapping[str, TensorLike]:
        ...


@dataclass
class PreparedGraph:
    """One named module boundary; validity does not establish exportability."""

    name: str
    module: ExportModule
    inputs: Mapping[str, TensorLike]
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    export_options: Mapping[str, Any] = field(default_factory=dict)
    build_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError('PreparedGraph.name cannot be empty.')
        _validate_module(self.module, graph=self.name)
        self.inputs = dict(self.inputs)
        self.input_specs = tuple(self.input_specs)
        self.output_specs = tuple(self.output_specs)
        self.export_options = dict(self.export_options)
        self.build_options = dict(self.build_options)
        if not self.inputs:
            raise ContractError(f'{self.name}: graph inputs cannot be empty.')
        input_names = tuple(self.inputs)
        spec_names = tuple(spec.name for spec in self.input_specs)
        if input_names != spec_names:
            raise ContractError(
                f'{self.name}: input order {input_names} does not match '
                f'input_specs {spec_names}.')
        for spec in self.input_specs:
            _validate_tensor(
                self.inputs[spec.name], spec, owner=f'graph {self.name}')
        output_names = tuple(spec.name for spec in self.output_specs)
        if not output_names:
            raise ContractError(f'{self.name}: output_specs cannot be empty.')
        if len(output_names) != len(set(output_names)):
            raise ContractError(f'{self.name}: output names must be unique.')


ReferenceRunner = Callable[[], Mapping[str, TensorLike]]
ExecutorFactory = Callable[[Mapping[str, InferenceEngine]], DeploymentExecutor]


@dataclass
class PreparedDeployment:
    """Complete build-time handoff from FluxVLA to FluxHermes."""

    recipe: str
    graphs: Mapping[str, PreparedGraph]
    policy_inputs: Mapping[str, TensorLike]
    policy_input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    reference_runner: ReferenceRunner
    executor_factory: ExecutorFactory
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.recipe:
            raise ContractError('PreparedDeployment.recipe cannot be empty.')
        self.graphs = dict(self.graphs)
        self.policy_inputs = dict(self.policy_inputs)
        self.policy_input_specs = tuple(self.policy_input_specs)
        self.output_specs = tuple(self.output_specs)
        self.metadata = dict(self.metadata)
        if not self.graphs:
            raise ContractError('PreparedDeployment.graphs cannot be empty.')
        for name, graph in self.graphs.items():
            if name != graph.name:
                raise ContractError(
                    f'Graph mapping key {name!r} does not match graph name '
                    f'{graph.name!r}.')
        input_names = tuple(self.policy_inputs)
        spec_names = tuple(spec.name for spec in self.policy_input_specs)
        if input_names != spec_names:
            raise ContractError(
                f'Policy input order {input_names} does not match '
                f'policy_input_specs {spec_names}.')
        for spec in self.policy_input_specs:
            _validate_tensor(
                self.policy_inputs[spec.name], spec, owner='policy input')
        output_names = tuple(spec.name for spec in self.output_specs)
        if not output_names:
            raise ContractError(
                'PreparedDeployment.output_specs cannot be empty.')
        if len(output_names) != len(set(output_names)):
            raise ContractError('Policy output names must be unique.')
        if not callable(self.reference_runner):
            raise ContractError('reference_runner must be callable.')
        if not callable(self.executor_factory):
            raise ContractError('executor_factory must be callable.')

    def run_reference(self) -> dict[str, TensorLike]:
        outputs = dict(self.reference_runner())
        expected = tuple(spec.name for spec in self.output_specs)
        actual = tuple(outputs)
        if actual != expected:
            raise ContractError(
                f'Reference outputs {actual} do not match contract '
                f'{expected}.')
        for spec in self.output_specs:
            _validate_tensor(outputs[spec.name], spec, owner='reference')
        return outputs

    def create_executor(
        self,
        engines: Mapping[str, InferenceEngine],
    ) -> DeploymentExecutor:
        supplied = set(engines)
        expected = set(self.graphs)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unknown = sorted(supplied - expected)
            raise ContractError(
                f'Engine names do not match graphs; missing={missing}, '
                f'unknown={unknown}.')
        ordered = {name: engines[name] for name in self.graphs}
        invalid = [
            name for name, engine in ordered.items()
            if not callable(getattr(engine, 'infer', None))
        ]
        if invalid:
            raise ContractError(
                f'Engines must implement callable infer(); invalid={invalid}.')
        executor = self.executor_factory(ordered)
        if not callable(getattr(executor, 'infer', None)):
            raise ContractError(
                'executor_factory must return an object with callable '
                'infer().')
        return executor


class ExportRecipe(Protocol):
    """Interface implemented once per supported model recipe."""

    name: str

    def prepare(
        self,
        *,
        config: DeploymentConfig,
        checkpoint: Path,
        inputs: Path,
        device: str,
    ) -> PreparedDeployment:
        ...
