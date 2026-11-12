"""Tests for MCore data-parallel sampling and RNG semantics."""

import os
from unittest.mock import patch

import torch

from fluxvla.engines.runners.base_train_runner import BaseTrainRunner


class _Runner:
    _build_dataloader_generator = BaseTrainRunner._build_dataloader_generator

    def __init__(self, data_parallel_rank):
        self.data_parallel_rank = data_parallel_rank

    def _data_parallel_rank(self):
        return self.data_parallel_rank


def test_dataloader_seed_uses_data_parallel_rank():
    with patch.dict(os.environ, {"EXPERIMENT_GLOBAL_SEED": "123"}):
        tp_rank_zero = _Runner(data_parallel_rank=0)
        tp_rank_one = _Runner(data_parallel_rank=0)
        next_dp_replica = _Runner(data_parallel_rank=1)

        tp_zero_sample = torch.rand(4, generator=tp_rank_zero._build_dataloader_generator())
        tp_one_sample = torch.rand(4, generator=tp_rank_one._build_dataloader_generator())
        dp_one_sample = torch.rand(4, generator=next_dp_replica._build_dataloader_generator())

    torch.testing.assert_close(tp_zero_sample, tp_one_sample, rtol=0, atol=0)
    assert not torch.equal(tp_zero_sample, dp_one_sample)
