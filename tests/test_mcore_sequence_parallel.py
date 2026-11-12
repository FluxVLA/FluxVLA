"""Distributed correctness tests for the shared MCore SP boundary."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


def _run_replicated_mlp_sp_parity(rank: int, world_size: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=world_size,
    )
    try:
        from fluxvla.mcore.tensor_parallel.activation import SequenceParallelMLP

        torch.manual_seed(17)
        reference = nn.Sequential(
            nn.Linear(8, 13, bias=True, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(13, 8, bias=True, dtype=torch.float64),
        ).cuda()
        candidate = copy.deepcopy(reference)
        sequence_parallel = SequenceParallelMLP(candidate)

        reference_inputs = []
        candidate_inputs = []
        for microbatch in range(2):
            generator = torch.Generator().manual_seed(100 + microbatch)
            value = torch.randn(
                2,
                6,
                8,
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            ).cuda()
            output_gradient = torch.randn(
                2,
                6,
                8,
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            ).cuda()
            reference_input = value.clone().requires_grad_(True)
            candidate_input = value.clone().requires_grad_(True)
            reference_output = reference(reference_input)
            candidate_output = sequence_parallel(candidate_input)

            torch.testing.assert_close(candidate_output, reference_output, rtol=1e-12, atol=1e-12)
            reference_output.backward(output_gradient)
            candidate_output.backward(output_gradient)
            reference_inputs.append(reference_input)
            candidate_inputs.append(candidate_input)

        for candidate_input, reference_input in zip(
            candidate_inputs,
            reference_inputs,
            strict=True,
        ):
            torch.testing.assert_close(
                candidate_input.grad,
                reference_input.grad,
                rtol=1e-12,
                atol=1e-12,
            )
        for candidate_parameter, reference_parameter in zip(
            candidate.parameters(),
            reference.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                candidate_parameter.grad,
                reference_parameter.grad,
                rtol=1e-12,
                atol=1e-12,
            )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("FLUXVLA_RUN_DISTRIBUTED_TESTS") != "1"
    or torch.cuda.device_count() < 2,
    reason="requires explicit opt-in and two CUDA devices",
)
def test_replicated_mlp_sequence_parallel_matches_two_microbatch_baseline() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous = str(Path(temporary_directory) / "distributed_init")
        mp.start_processes(
            _run_replicated_mlp_sp_parity,
            args=(2, rendezvous),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
