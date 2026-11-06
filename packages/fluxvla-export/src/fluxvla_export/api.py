# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations
from pathlib import Path

from . import recipes as _recipes  # noqa: F401
from ._safetensors import validate_safetensors
from ._validation import validate_device_syntax, validate_token_vocabulary
from .config import load_config
from .contracts import PreparedDeployment
from .errors import ContractError
from .fixture import load_fixture
from .registry import get_recipe


def prepare_tensorrt(
    *,
    recipe: str,
    config: str | Path,
    checkpoint: str | Path,
    inputs: str | Path,
    device: str,
) -> PreparedDeployment:
    """Validate deployment artifacts, then dispatch a model-family recipe.

    Preflight verifies configuration, hashes, file structure and fixture
    tensors. Only a real recipe can construct the model and validate its
    complete checkpoint state. No PI0.5 model recipe is registered yet.
    """
    validate_device_syntax(device)
    resolved = load_config(config, recipe=recipe)
    # Fail before reading potentially large weights when no model exists.
    implementation = get_recipe(recipe)
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
        label=f'Export inputs {inputs}',
    )
    # Recipes must revalidate hashes when loading paths. Preflight is not a
    # persistent authorization to trust files that may change afterwards.
    prepared = implementation.prepare(
        config=resolved,
        checkpoint=checkpoint_path,
        inputs=Path(inputs).expanduser().resolve(),
        device=device,
    )
    if not isinstance(prepared, PreparedDeployment):
        raise ContractError(
            f'Recipe {recipe!r} must return PreparedDeployment, got '
            f'{type(prepared).__name__}.')
    if prepared.recipe != recipe:
        raise ContractError(f'Recipe {recipe!r} returned deployment for '
                            f'{prepared.recipe!r}.')
    return prepared
