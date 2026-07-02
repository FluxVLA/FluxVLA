# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Timestep embedding retained for FluxVLA Cosmos3 flow matching."""

import math

import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        torch.nn.init.trunc_normal_(
            self.mlp[0].weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.mlp[0].bias)

        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(
            self.mlp[2].weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32) / half).to(
                    device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # FluxVLA vendor patch: avoid bf16/float32 mixed Linear inputs.
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)
