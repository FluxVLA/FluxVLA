# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch


class TrainTimeWeight:
    def __init__(
        self,
        noise_scheduler,
        weight: str = 'uniform',
    ):
        # Map reweighting -> uniform to support inference for existing checkpoints.
        if weight == 'reweighting':
            weight = 'uniform'

        self.weight = weight
        self.noise_scheduler = noise_scheduler

        assert self.weight == 'uniform', 'Only uniform loss weight is supported in RF'

    def __call__(self, t, tensor_kwargs) -> torch.Tensor:  # t: [B], returns [B]
        if self.weight == 'uniform':
            wts = torch.ones_like(t)  # [B]
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")

        return wts
