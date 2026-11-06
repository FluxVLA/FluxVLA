# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Strict family-specific preparation and real PyTorch reference inference.

This is not a registered production recipe or an export/ONNX entry point.
Model libraries are loaded only when preparation or inference is requested.
"""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..._safetensors import validate_safetensors
from ..._validation import validate_device_syntax, validate_token_vocabulary
from ...checkpoint import load_checkpoint
from ...config import DeploymentConfig, load_config, validate_config
from ...contracts import TensorSpec, _validate_tensor
from ...errors import ContractError
from ...fixture import load_fixture

if TYPE_CHECKING:
    import torch

    from .model import PI05Model


@dataclass(frozen=True)
class PI05Reference:
    """A strictly loaded eval model and its validated six-tensor fixture.

    The input mapping is read-only. Tensor and model objects remain available
    for inspection; callers must not edit them between reference calls.
    """

    model: PI05Model
    inputs: Mapping[str, torch.Tensor]
    config: DeploymentConfig
    device: torch.device

    def __post_init__(self) -> None:
        object.__setattr__(self, 'inputs', MappingProxyType(dict(self.inputs)))

    def run_reference(self) -> dict[str, torch.Tensor]:
        """Run deterministic plain Euler inference without modifying inputs.

        The model retains FP32 master parameters; BF16, when configured, is
        compute autocast only. Explicitly disabling autocast for an FP32
        reference also prevents an ambient caller context from changing it.
        """
        import torch

        if any(module.training for module in self.model.modules()):
            raise ContractError(
                'PI0.5 reference requires all modules in eval mode.')
        with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.config.precision.autocast_dtype == 'bfloat16'):
            actions = self.model(**self.inputs)
        if not isinstance(actions, torch.Tensor):
            raise ContractError(
                'PI0.5 reference.actions must be a torch Tensor.')
        _validate_tensor(
            actions,
            TensorSpec('actions', 'float32', (
                self.config.input_specs[0].shape[0],
                self.config.model.action_horizon,
                self.config.model.action_dim,
            )),
            owner='PI0.5 reference',
        )
        if not torch.isfinite(actions).all().item():
            raise ContractError(
                'PI0.5 reference.actions must contain finite values.')
        return {'actions': actions}


def _resolve_device(device: str, *, bf16: bool) -> torch.device:
    import torch

    try:
        target = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ContractError(
            f'device {device!r}: cannot represent this device in torch.'
        ) from error
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise ContractError(f'device {device!r}: CUDA is not available.')
        index = (
            torch.cuda.current_device()
            if target.index is None else target.index)
        if index >= torch.cuda.device_count():
            raise ContractError(
                f'device {device!r}: CUDA index must be smaller than '
                f'{torch.cuda.device_count()}.')
        target = torch.device('cuda', index)
        if bf16:
            with torch.cuda.device(target):
                if not torch.cuda.is_bf16_supported():
                    raise ContractError(
                        f'device {device!r}: BF16 autocast is not supported.')
    return target


def prepare_reference(
    *,
    config: str | Path,
    checkpoint: str | Path,
    inputs: str | Path,
    device: str,
) -> PI05Reference:
    """Validate bound JSON/safetensors and strictly load the native
    PI0.5 graph.

    CPU construction is wrapped in ``fork_rng`` to restore the caller's RNG
    even on failure. The raw builder itself preserves the original normal
    seeded initialization semantics. Every checkpoint key, including the
    unused expert embedding, is required with its exact native shape/dtype.
    """
    validate_device_syntax(device)
    resolved = load_config(config, recipe='pi05-v1')
    return prepare_reference_from_config(
        config=resolved, checkpoint=checkpoint, inputs=inputs, device=device)


def prepare_reference_from_config(
    *,
    config: DeploymentConfig,
    checkpoint: str | Path,
    inputs: str | Path,
    device: str,
) -> PI05Reference:
    """Prepare from a revalidated typed config without reopening its JSON file.

    Recipe callers already loaded strict JSON. Revalidate the full dataclass
    and all artifact bindings rather than trusting manually replaced fields;
    all checkpoint/fixture reads and CPU RNG protection use this same path.
    """
    validate_device_syntax(device)
    resolved = validate_config(config)
    target = _resolve_device(
        device, bf16=resolved.precision.autocast_dtype == 'bfloat16')
    # Stream header/hash preflight before allocating a potentially large model.
    # This does not authorize later bytes: load_checkpoint revalidates below.
    checkpoint_path = validate_safetensors(
        checkpoint,
        expected_sha256=resolved.checkpoint_sha256,
        label='Checkpoint',
    )
    tensors = load_fixture(
        inputs,
        expected_sha256=resolved.inputs_sha256,
        specs=resolved.input_specs,
    )
    validate_token_vocabulary(
        tensors['lang_tokens'],
        vocab_size=resolved.model.vocab_size,
        label=f'Export inputs {inputs}')

    import torch

    from .model import build_pi05_model

    with torch.random.fork_rng(devices=[]):
        model = build_pi05_model(resolved)
    weights = load_checkpoint(
        checkpoint_path,
        expected_sha256=resolved.checkpoint_sha256,
        expected_state=model.state_dict(),
    )
    model.load_state_dict(weights, strict=True)
    model.to(device=target)
    model.eval()
    prepared_inputs = {
        name: value.to(device=target)
        for name, value in tensors.items()
    }
    return PI05Reference(
        model=model, inputs=prepared_inputs, config=resolved, device=target)
