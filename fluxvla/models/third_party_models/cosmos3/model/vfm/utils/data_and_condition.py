# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Condition data retained for FluxVLA Cosmos3 sequence packing."""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GenerationDataClean:
    """Tokenized states and conditioning metadata for a packed MoT batch."""

    batch_size: int
    is_image_batch: bool

    # Vision
    raw_state_vision: list[torch.Tensor] | None = None
    x0_tokens_vision: list[torch.Tensor] | None = None
    fps_vision: torch.Tensor | None = None
    num_vision_items_per_sample: list[int] | None = None

    # Sound
    raw_state_sound: torch.Tensor | None = None
    x0_tokens_sound: list[torch.Tensor] | None = None
    fps_sound: torch.Tensor | None = None

    # Action
    raw_state_action: list[torch.Tensor] | None = None
    x0_tokens_action: list[torch.Tensor] | None = None
    fps_action: torch.Tensor | None = None
    action_domain_id: list[torch.Tensor] | None = None
    raw_action_dim: list[torch.Tensor] | None = None
