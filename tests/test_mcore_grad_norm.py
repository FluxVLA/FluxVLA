"""Tests for logical-model gradient clipping under tensor parallelism."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from fluxvla.engines.runners.megatron_train_runner import MegatronNativeTrainRunner
from fluxvla.mcore.diagnostics import (
    _fsdp_gradient_divide_factors,
    _ordered_grad_reduce_entries,
    _tp_aware_fsdp_total_norm,
)


def test_fsdp_gradient_divide_factors_match_dp_collective_contract():
    assert _fsdp_gradient_divide_factors(1) == (1.0, 1.0)
    assert _fsdp_gradient_divide_factors(2) == (2.0, 1.0)
    assert _fsdp_gradient_divide_factors(4) == (2.0, 2.0)
    assert _fsdp_gradient_divide_factors(8) == (4.0, 2.0)


def test_grad_reduce_entries_include_ordered_zero_segments():
    before = torch.nn.Parameter(torch.zeros(2, 3))
    after = torch.nn.Parameter(torch.zeros(5))

    entries = _ordered_grad_reduce_entries(
        [("after", after), ("before", before)],
        grad_order=lambda name: {"before": (0,), "after": (2,)}[name],
        zero_segments=[((1,), 7)],
    )

    assert [(name, parameter, numel) for name, parameter, numel in entries] == [
        ("before", before, 6),
        ("__zero_reduce_segment_0", None, 7),
        ("after", after, 5),
    ]
    assert sum(numel for _, _, numel in entries) == 18


def test_grad_reduce_entries_skip_non_allreduce_parameters():
    included = torch.nn.Parameter(torch.zeros(2))
    excluded = torch.nn.Parameter(torch.zeros(3))
    excluded.allreduce = False

    entries = _ordered_grad_reduce_entries(
        [("excluded", excluded), ("included", included)],
        grad_order=lambda name: (name,),
        zero_segments=[],
    )

    assert entries == [("included", included, 2)]


def test_tp_aware_fsdp_norm_gathers_shards_and_counts_replicas_once():
    replicated = torch.nn.Parameter(torch.zeros(1))
    replicated.main_grad = torch.tensor([3.0])
    sharded = torch.nn.Parameter(torch.zeros(1))
    sharded.main_grad = torch.tensor([4.0])
    sharded.tensor_model_parallel = True
    sharded.partition_dim = 0

    def all_gather(outputs, input_tensor, *, group):
        assert group is tp_group
        torch.testing.assert_close(input_tensor, torch.tensor([4.0]))
        outputs[0].copy_(torch.tensor([4.0]))
        outputs[1].copy_(torch.tensor([12.0]))

    tp_group = object()
    with (
        patch(
            "megatron.core.parallel_state.get_tensor_model_parallel_group",
            return_value=tp_group,
        ),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather", side_effect=all_gather),
    ):
        total_norm = _tp_aware_fsdp_total_norm(
            [("replicated", replicated), ("zero", None), ("sharded", sharded)]
        )

    torch.testing.assert_close(total_norm, torch.tensor(13.0))


def test_tp_grad_norm_counts_replicas_once_and_all_shards():
    replicated = torch.nn.Parameter(torch.tensor([1.0]))
    replicated.grad = torch.tensor([3.0])
    sharded = torch.nn.Parameter(torch.tensor([1.0]))
    sharded.grad = torch.tensor([4.0])
    sharded._tp_sharded = True

    model = Mock()
    model.parameters.return_value = iter([replicated, sharded])
    runtime = Mock()
    runtime.tensor_model_parallel_group.return_value = object()
    runner = SimpleNamespace(
        max_grad_norm=2.5,
        device_id=torch.device("cpu"),
        vla=model,
        megatron_runtime=runtime,
    )

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.all_reduce"),
    ):
        total_norm = MegatronNativeTrainRunner.clip_grad_norm(runner)

    torch.testing.assert_close(total_norm, torch.tensor(5.0))
    torch.testing.assert_close(replicated.grad, torch.tensor([1.5]))
    torch.testing.assert_close(sharded.grad, torch.tensor([2.0]))
