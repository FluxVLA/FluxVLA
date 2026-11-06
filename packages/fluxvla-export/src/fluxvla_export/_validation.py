# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""Shared preflight checks; importing this module does not import torch."""

from __future__ import annotations

import re
from typing import Any

from .errors import ArtifactError, ContractError


def validate_device_syntax(device: str) -> None:
    """Check the public device spelling, without probing or initializing CUDA."""
    if not isinstance(device, str) or not re.fullmatch(
            r'cpu|cuda(?::(?:0|[1-9][0-9]*))?', device):
        raise ContractError('device must be cpu, cuda, or cuda:<index>.')


def validate_token_vocabulary(
    tokens: Any, *, vocab_size: int, label: str,
) -> None:
    """Check actual loaded token tensors, including masked/padding positions.

    Shape and dtype validation belongs to the fixture reader or caller's input
    contract. Keep this model-specific range check out of the generic reader.
    """
    if (tokens < 0).any().item() or (tokens >= vocab_size).any().item():
        raise ArtifactError(
            f'{label}: lang_tokens must be in [0, {vocab_size}).')
