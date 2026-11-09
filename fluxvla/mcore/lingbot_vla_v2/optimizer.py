"""LingBot-specific optimizer grouping installed at the MCore boundary."""

from __future__ import annotations

import json
import math
import os
import re
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch


_ROUTED_EXPERT = re.compile(r"(?:^|\.)layers\.\d+\.mlp\.(?:module\.)?experts\.")


@dataclass(frozen=True)
class _FSDP2ReductionUnit:
    name: str
    parameters: tuple[torch.nn.Parameter, ...]


_FSDP2_REDUCTION_UNITS: list[_FSDP2ReductionUnit] = []
_LAST_FSDP2_CLIP_COEFFICIENT: torch.Tensor | None = None
_FSDP2_PARTIAL_CLIP_DENOMINATOR: torch.Tensor | None = None
_FSDP2_MICROBATCH_REDUCTION_COMPLETE = False


def _strict_phase_log(message: str) -> None:
    """Emit opt-in timing markers without synchronizing CUDA work."""
    if os.environ.get("FLUXVLA_STRICT_PHASE_LOG", "0") != "1":
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    print(
        f"[fluxvla-strict-phase] t={time.monotonic():.6f} rank={rank} {message}",
        flush=True,
    )


def _write_fsdp2_reduction_layout(adapter: torch.nn.Module) -> None:
    """Write the logical reduction layout without materializing parameters."""
    root_value = os.environ.get("FLUXVLA_FSDP2_LAYOUT_TRACE_DIR")
    if not root_value:
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    names_by_id = {
        id(parameter): name.replace(".linear.", ".")
        for name, parameter in adapter.named_parameters()
    }
    from megatron.core import parallel_state

    tensor_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
    units = []
    for unit_index, unit in enumerate(_FSDP2_REDUCTION_UNITS):
        parameters = []
        for parameter_index, parameter in enumerate(unit.parameters):
            local_shape = list(parameter.shape)
            logical_shape = list(local_shape)
            is_tensor_parallel = bool(
                getattr(parameter, "tensor_model_parallel", False)
            )
            partition_dim = int(getattr(parameter, "partition_dim", 0))
            if is_tensor_parallel:
                logical_shape[partition_dim] *= tensor_parallel_size
            parameters.append(
                {
                    "index": parameter_index,
                    "name": names_by_id.get(id(parameter), "<unnamed>"),
                    "shape": logical_shape,
                    "local_shape": local_shape,
                    "numel": math.prod(logical_shape),
                    "dtype": str(parameter.dtype),
                    "tensor_parallel": is_tensor_parallel,
                    "partition_dim": partition_dim if is_tensor_parallel else None,
                }
            )
        units.append(
            {
                "index": unit_index,
                "name": unit.name,
                "parameter_count": len(parameters),
                "numel": sum(item["numel"] for item in parameters),
                "parameters": parameters,
            }
        )
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    (root / "mcore_fsdp2_layout.json").write_text(
        json.dumps(
            {"backend": "mcore", "unit_count": len(units), "units": units},
            indent=2,
        ),
        encoding="utf-8",
    )


def register_lingbot_fsdp2_reduction_units(adapter: torch.nn.Module) -> None:
    """Recreate Native FSDP2's child-first parameter ownership.

    ``fully_shard`` claims parameters when it is applied. Native LingBot first
    applies it to every FP32 basic module, then each VLM decoder layer, and
    finally the adapter root. A parent unit must therefore exclude parameters
    already claimed by a child unit.
    """
    claimed: set[int] = set()
    units: list[_FSDP2ReductionUnit] = []

    def child_first_parameters(module: torch.nn.Module):
        # FSDP2 enumerates descendants before parameters owned directly by the
        # current module. This differs from ``module.parameters()``, which
        # yields direct parameters first, and changes reduce-scatter packing
        # for modules such as FixAdaRMSNorm and the adapter root.
        for child in module.children():
            yield from child_first_parameters(child)
        yield from module.parameters(recurse=False)

    def claim(name: str, module: torch.nn.Module) -> None:
        parameters = []
        local_seen = set()
        for parameter in child_first_parameters(module):
            parameter_id = id(parameter)
            if (
                parameter.requires_grad
                and parameter_id not in claimed
                and parameter_id not in local_seen
            ):
                local_seen.add(parameter_id)
                parameters.append(parameter)
        parameters = tuple(parameters)
        claimed.update(id(parameter) for parameter in parameters)
        if parameters:
            for parameter in parameters:
                parameter._fluxvla_autograd_grad_seen = False

                def mark_grad_seen(gradient, *, target=parameter):
                    target._fluxvla_autograd_grad_seen = True
                    return gradient

                parameter.register_hook(mark_grad_seen)
            units.append(_FSDP2ReductionUnit(name, parameters))

    for index, module in enumerate(adapter.get_fsdp2_full_precision_modules()):
        claim(f"fp32_basic_{index}:{module.__class__.__name__}", module)
    for index, module in enumerate(adapter.get_fsdp2_wrapping_modules()):
        claim(f"vlm_layer_{index}:{module.__class__.__name__}", module)
    claim("root", adapter)

    missing = [
        parameter
        for parameter in adapter.parameters()
        if parameter.requires_grad and id(parameter) not in claimed
    ]
    if missing:
        raise RuntimeError(
            f"LingBot FSDP2 reduction layout left {len(missing)} trainable parameters unclaimed"
        )
    _FSDP2_REDUCTION_UNITS.clear()
    _FSDP2_REDUCTION_UNITS.extend(units)
    _write_fsdp2_reduction_layout(adapter)


@torch.no_grad()
def _strict_fsdp2_reduce_unit(
    unit: _FSDP2ReductionUnit,
    process_group,
    *,
    tensor_parallel_group=None,
    tensor_parallel_rank: int = 0,
) -> None:
    """Reduce one logical FSDP2 unit with PyTorch's exact packing layout."""
    world_size = torch.distributed.get_world_size(process_group)
    tensor_parallel_size = (
        1
        if tensor_parallel_group is None
        else torch.distributed.get_world_size(tensor_parallel_group)
    )
    active_parameters = [
        parameter
        for parameter in unit.parameters
        if getattr(parameter, "_fluxvla_autograd_grad_seen", False)
    ]
    if not active_parameters:
        return
    local_gradients = [parameter.main_grad for parameter in active_parameters]
    if any(gradient is None for gradient in local_gradients):
        raise RuntimeError(f"LingBot strict DP reduction found a missing gradient in {unit.name}")
    gradients = []
    for parameter, local_gradient in zip(
        active_parameters, local_gradients, strict=True
    ):
        if (
            tensor_parallel_size > 1
            and getattr(parameter, "tensor_model_parallel", False)
        ):
            shards = [
                torch.empty_like(local_gradient)
                for _ in range(tensor_parallel_size)
            ]
            torch.distributed.all_gather(
                shards,
                local_gradient.contiguous(),
                group=tensor_parallel_group,
            )
            gradients.append(
                torch.cat(
                    shards,
                    dim=int(getattr(parameter, "partition_dim", 0)),
                )
            )
        else:
            gradients.append(local_gradient)
    dtypes = {gradient.dtype for gradient in gradients}
    devices = {gradient.device for gradient in gradients}
    if len(dtypes) != 1 or len(devices) != 1:
        raise RuntimeError(
            f"LingBot strict DP reduction requires one dtype/device per unit; "
            f"{unit.name} has dtypes={dtypes}, devices={devices}"
        )

    # This is the same padded dim-0 layout used by PyTorch 2.7 FSDP2's
    # foreach_reduce_scatter_copy_in. ``chunk_cat`` writes a [DP, shard_numel]
    # matrix where each row contains that rank's shard of every unit parameter.
    padded_sizes = [
        torch.Size(
            [math.ceil(gradient.shape[0] / world_size) * world_size]
            + list(gradient.shape[1:])
        )
        for gradient in gradients
    ]
    total_numel = sum(size.numel() for size in padded_sizes)
    reduce_input = torch.empty(
        total_numel,
        dtype=gradients[0].dtype,
        device=gradients[0].device,
    )
    torch.ops.fsdp.chunk_cat(
        gradients,
        dim=0,
        num_chunks=world_size,
        out=reduce_input.view(world_size, -1),
    )
    reduced_shard = torch.empty(
        total_numel // world_size,
        dtype=reduce_input.dtype,
        device=reduce_input.device,
    )
    torch.distributed.reduce_scatter_tensor(
        reduced_shard,
        reduce_input,
        op=torch.distributed.ReduceOp.AVG,
        group=process_group,
    )

    # FSDP2 keeps each sharded gradient norm as ``_NormPartial``. Its later
    # ``total_norm + 1e-6`` therefore adds epsilon to every rank-local norm
    # before the p-norm redistribution. Preserve those exact local norms; a
    # coefficient reconstructed only from the materialized global norm loses
    # this information and eventually differs by one FP32 ULP.
    rank = torch.distributed.get_rank(process_group)
    local_shards = []
    offset = 0
    for gradient, padded_size in zip(gradients, padded_sizes, strict=True):
        shard_numel = padded_size.numel() // world_size
        shard_dim0 = padded_size[0] // world_size
        start = rank * shard_dim0
        valid_dim0 = max(0, min(shard_dim0, gradient.shape[0] - start))
        shard = reduced_shard[offset : offset + shard_numel].view(
            shard_dim0, *padded_size[1:]
        )
        local_shards.append(shard[:valid_dim0])
        offset += shard_numel
    local_norms = torch._foreach_norm(local_shards, 2.0)
    for parameter, local_norm in zip(
        active_parameters, local_norms, strict=True
    ):
        parameter._fluxvla_fsdp2_local_grad_norm = (
            local_norm if tensor_parallel_rank == 0 else torch.zeros_like(local_norm)
        )

    # DistOpt owns different flat-buffer ranges than FSDP2. Gather the already
    # reduced shards without further arithmetic, reconstruct each full tensor,
    # and copy it into the existing MCore main-grad views.
    # The reduce-scatter input is dead after the collective. Reuse that large
    # allocation for all-gather output so strict parity does not hold two full
    # unit-sized buffers at once (the root unit can contain billions of values).
    torch.distributed.all_gather_into_tensor(
        reduce_input,
        reduced_shard,
        group=process_group,
    )
    reduced_rows = reduce_input.view(world_size, -1)
    offset = 0
    for parameter, local_gradient, gradient, padded_size in zip(
        active_parameters,
        local_gradients,
        gradients,
        padded_sizes,
        strict=True,
    ):
        shard_numel = padded_size.numel() // world_size
        shard_dim0 = padded_size[0] // world_size
        for rank in range(world_size):
            start = rank * shard_dim0
            valid_dim0 = min(shard_dim0, gradient.shape[0] - start)
            if valid_dim0 <= 0:
                break
            shard = reduced_rows[rank, offset : offset + shard_numel].view(
                padded_size[0] // world_size, *padded_size[1:]
            )
            gradient.narrow(0, start, valid_dim0).copy_(shard[:valid_dim0])
        if (
            tensor_parallel_size > 1
            and getattr(parameter, "tensor_model_parallel", False)
        ):
            partition_dim = int(getattr(parameter, "partition_dim", 0))
            local_extent = local_gradient.shape[partition_dim]
            local_gradient.copy_(
                gradient.narrow(
                    partition_dim,
                    tensor_parallel_rank * local_extent,
                    local_extent,
                )
            )
        elif gradient.data_ptr() != local_gradient.data_ptr():
            local_gradient.copy_(gradient)
        offset += shard_numel


@torch.no_grad()
def _strict_fsdp2_finish_grad_sync(ddp) -> None:
    """Run strict Native-FSDP2-compatible DP reduction for MCore DistOpt."""
    from megatron.core import parallel_state

    global _FSDP2_MICROBATCH_REDUCTION_COMPLETE

    if not _FSDP2_REDUCTION_UNITS:
        raise RuntimeError("LingBot strict DP reduction units were not registered")
    expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
    config = ddp.ddp_config
    if (
        not config.use_distributed_optimizer
        or config.overlap_grad_reduce
        or config.num_distributed_optimizer_instances != 1
    ):
        raise RuntimeError(
            "LingBot strict FSDP2 DP reduction requires synchronous single-instance DistOpt"
        )
    if _FSDP2_MICROBATCH_REDUCTION_COMPLETE:
        _FSDP2_MICROBATCH_REDUCTION_COMPLETE = False
        for bucket_group in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
            bucket_group.param_gather_dispatched = False
            bucket_group.grad_reduce_handle = None
        return
    process_group = ddp.dp_cp_group
    tensor_parallel_group = parallel_state.get_tensor_model_parallel_group()
    tensor_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
    for unit in _FSDP2_REDUCTION_UNITS:
        for parameter in unit.parameters:
            parameter._fluxvla_fsdp2_local_grad_norm = None
    for unit in _FSDP2_REDUCTION_UNITS:
        if expert_parallel_size > 1:
            _strict_fsdp2_average_ep_dense_parameters(unit, process_group)
        else:
            _strict_fsdp2_reduce_unit(
                unit,
                process_group,
                tensor_parallel_group=tensor_parallel_group,
                tensor_parallel_rank=tensor_parallel_rank,
            )
        for parameter in unit.parameters:
            parameter._fluxvla_autograd_grad_seen = False
    for bucket_group in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
        bucket_group.param_gather_dispatched = False
        bucket_group.grad_reduce_handle = None


@torch.no_grad()
def _strict_fsdp2_average_ep_dense_parameters(
    unit: _FSDP2ReductionUnit,
    process_group,
) -> None:
    """Average EP-replicated dense gradients in Native FSDP2 ring order."""
    parameters = [
        parameter
        for parameter in unit.parameters
        if getattr(parameter, "_fluxvla_autograd_grad_seen", False)
        and not getattr(parameter, "_fluxvla_grad_already_reduced", False)
    ]
    if not parameters:
        return
    if any(parameter.main_grad is None for parameter in parameters):
        raise RuntimeError(
            f"LingBot strict EP reduction found a missing gradient in {unit.name}"
        )

    world_size = torch.distributed.get_world_size(process_group)
    max_chunk_numel = int(
        os.environ.get("FLUXVLA_STRICT_EP_GATHER_CHUNK_NUMEL", str(16 * 1024 * 1024))
    )
    from torch.distributed.algorithms._comm_hooks.default_hooks import DefaultState

    predivide = DefaultState._get_gradient_predivide_factor(world_size)
    postdivide = world_size / predivide

    def reduce_chunk(owner: int, entries: list[tuple]) -> None:
        local = torch.cat([entry[1] for entry in entries])
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, local, group=process_group)
        order = [
            (owner + offset) % world_size
            for offset in range(1, world_size + 1)
        ]
        offset = 0
        for parameter, local_shard, valid_dim0 in entries:
            numel = local_shard.numel()
            reduced = gathered[order[0]][offset : offset + numel].clone()
            if predivide > 1.0:
                reduced.div_(predivide)
            for source in order[1:]:
                value = gathered[source][offset : offset + numel]
                reduced.add_(value / predivide if predivide > 1.0 else value)
            if postdivide > 1.0:
                reduced.div_(postdivide)
            gradient = parameter.main_grad
            shard_dim0 = local_shard.numel() // math.prod(gradient.shape[1:])
            start = owner * shard_dim0
            gradient.narrow(0, start, valid_dim0).copy_(
                reduced.view(shard_dim0, *gradient.shape[1:])[:valid_dim0]
            )
            offset += numel

    for owner in range(world_size):
        entries = []
        chunk_numel = 0
        for parameter in parameters:
            gradient = parameter.main_grad
            shard_dim0 = math.ceil(gradient.shape[0] / world_size)
            start = owner * shard_dim0
            valid_dim0 = max(0, min(shard_dim0, gradient.shape[0] - start))
            local_shard = torch.zeros(
                shard_dim0,
                *gradient.shape[1:],
                dtype=gradient.dtype,
                device=gradient.device,
            )
            if valid_dim0:
                local_shard[:valid_dim0].copy_(
                    gradient.narrow(0, start, valid_dim0)
                )
            flat = local_shard.view(-1)
            if entries and chunk_numel + flat.numel() > max_chunk_numel:
                reduce_chunk(owner, entries)
                entries = []
                chunk_numel = 0
            entries.append((parameter, flat, valid_dim0))
            chunk_numel += flat.numel()
        if entries:
            reduce_chunk(owner, entries)


@torch.no_grad()
def _refresh_strict_fsdp2_local_grad_norms() -> None:
    """Rebuild Native FSDP2 rank-local norms after microbatch accumulation."""
    from megatron.core import parallel_state

    process_group = parallel_state.get_data_parallel_group()
    tensor_parallel_group = parallel_state.get_tensor_model_parallel_group()
    data_parallel_size = torch.distributed.get_world_size(process_group)
    data_parallel_rank = torch.distributed.get_rank(process_group)
    tensor_parallel_size = torch.distributed.get_world_size(tensor_parallel_group)
    tensor_parallel_rank = torch.distributed.get_rank(tensor_parallel_group)

    active_parameters = []
    local_shards = []
    for unit in _FSDP2_REDUCTION_UNITS:
        for parameter in unit.parameters:
            if not getattr(parameter, "_fluxvla_autograd_grad_seen", False):
                parameter._fluxvla_fsdp2_local_grad_norm = None
                continue
            gradient = parameter.main_grad
            if gradient is None:
                raise RuntimeError(
                    f"LingBot strict DP norm refresh found a missing gradient in {unit.name}"
                )
            logical_gradient = gradient
            if (
                tensor_parallel_size > 1
                and getattr(parameter, "tensor_model_parallel", False)
            ):
                shards = [
                    torch.empty_like(gradient)
                    for _ in range(tensor_parallel_size)
                ]
                torch.distributed.all_gather(
                    shards,
                    gradient.contiguous(),
                    group=tensor_parallel_group,
                )
                logical_gradient = torch.cat(
                    shards,
                    dim=int(getattr(parameter, "partition_dim", 0)),
                )
            shard_dim0 = math.ceil(
                logical_gradient.shape[0] / data_parallel_size
            )
            start = data_parallel_rank * shard_dim0
            valid_dim0 = max(
                0,
                min(shard_dim0, logical_gradient.shape[0] - start),
            )
            local_shard = logical_gradient.narrow(0, start, valid_dim0)
            active_parameters.append(parameter)
            local_shards.append(local_shard)

    # PyTorch FSDP2's clip_grad_norm_ groups the DTensor local shards by
    # device/dtype and computes all per-parameter norms in one foreach call.
    # Calling vector_norm separately changes hundreds of results by one FP32
    # ULP on CUDA and eventually changes clipped AdamW updates.
    local_norms = torch._foreach_norm(local_shards, 2.0)
    for parameter, local_norm in zip(
        active_parameters, local_norms, strict=True
    ):
        parameter._fluxvla_fsdp2_local_grad_norm = (
            local_norm
            if tensor_parallel_rank == 0
            else torch.zeros_like(local_norm)
        )


def _install_strict_fsdp2_dp_reduction() -> None:
    """Patch MCore's final DP-sync boundary, leaving its optimizer unchanged."""
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.training import get_args

    original = DistributedDataParallel.finish_grad_sync
    if getattr(original, "_fluxvla_lingbot_strict_fsdp2", False):
        return

    def finish_grad_sync(ddp):
        if not getattr(get_args(), "lingbot_fsdp2_strict_dp_reduction", False):
            return original(ddp)
        return _strict_fsdp2_finish_grad_sync(ddp)

    finish_grad_sync._fluxvla_lingbot_strict_fsdp2 = True
    DistributedDataParallel.finish_grad_sync = finish_grad_sync


@torch.no_grad()
def _accumulate_strict_microbatch_gradients(
    active_parameters: list[torch.nn.Parameter],
    all_parameters: list[torch.nn.Parameter],
    accumulator: dict[int, torch.Tensor],
    *,
    final_microbatch: bool,
    accumulator_device: torch.device | str | None = None,
    cuda_accumulator_limit_bytes: int | None = None,
    transfer_state: dict | None = None,
) -> list[torch.nn.Parameter]:
    """Stage reduced gradients while preserving microbatch addition order."""
    active_ids = {id(parameter) for parameter in active_parameters}
    if not final_microbatch:
        cuda_accumulator_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in accumulator.values()
            if tensor.device.type == "cuda"
        )
        cuda_entries = []
        cpu_entries = []
        new_cuda_keys = set()
        if (
            accumulator_device is not None
            and torch.device(accumulator_device).type == "cuda"
            and cuda_accumulator_limit_bytes is not None
        ):
            available_bytes = cuda_accumulator_limit_bytes - cuda_accumulator_bytes
            new_parameters = [
                parameter
                for parameter in active_parameters
                if id(parameter) not in accumulator
            ]
            for parameter in sorted(
                new_parameters,
                key=lambda item: item.main_grad.numel()
                * item.main_grad.element_size(),
                reverse=True,
            ):
                current_bytes = (
                    parameter.main_grad.numel()
                    * parameter.main_grad.element_size()
                )
                if current_bytes <= available_bytes:
                    new_cuda_keys.add(id(parameter))
                    available_bytes -= current_bytes
        for parameter in active_parameters:
            key = id(parameter)
            current = parameter.main_grad.detach()
            if key in accumulator:
                target_device = accumulator[key].device
            else:
                target_device = accumulator_device or current.device
                current_bytes = current.numel() * current.element_size()
                if (
                    torch.device(target_device).type == "cuda"
                    and cuda_accumulator_limit_bytes is not None
                    and key not in new_cuda_keys
                ):
                    target_device = "cpu"
                if torch.device(target_device).type == "cuda":
                    cuda_accumulator_bytes += current_bytes
            entry = (parameter, key, current)
            if torch.device(target_device).type == "cuda":
                cuda_entries.append(entry)
            else:
                cpu_entries.append(entry)

        for _, key, current in cuda_entries:
            if key in accumulator:
                accumulator[key].add_(current)
            else:
                accumulator[key] = current.clone()

        # Packing CPU-spilled tensors turns thousands of synchronous D2H
        # copies into one transfer per microbatch. Tensor views are unpacked
        # before the temporary flat buffer is released, so accumulator
        # ownership remains per parameter and dynamic activity still works.
        if cpu_entries:
            cpu_keys = [entry[1] for entry in cpu_entries]
            total_cpu_numel = sum(entry[2].numel() for entry in cpu_entries)
            source_device = cpu_entries[0][2].device
            use_pinned_transfer = source_device.type == "cuda" and transfer_state is not None
            if use_pinned_transfer:
                host_packed = transfer_state.get("host_transfer")
                if (
                    host_packed is None
                    or host_packed.numel() != total_cpu_numel
                    or host_packed.dtype != cpu_entries[0][2].dtype
                ):
                    host_packed = torch.empty(
                        total_cpu_numel,
                        dtype=cpu_entries[0][2].dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    transfer_state["host_transfer"] = host_packed
                max_chunk_numel = int(
                    os.environ.get(
                        "FLUXVLA_STRICT_TRANSFER_CHUNK_NUMEL",
                        str(256 * 1024 * 1024),
                    )
                )
                offset = 0
                chunk = []
                chunk_numel = 0
                for entry in cpu_entries:
                    current_numel = entry[2].numel()
                    if chunk and chunk_numel + current_numel > max_chunk_numel:
                        packed = torch.cat([item[2].reshape(-1) for item in chunk])
                        host_packed[offset : offset + chunk_numel].copy_(
                            packed, non_blocking=True
                        )
                        torch.cuda.current_stream(source_device).synchronize()
                        offset += chunk_numel
                        chunk = []
                        chunk_numel = 0
                    chunk.append(entry)
                    chunk_numel += current_numel
                if chunk:
                    packed = torch.cat([item[2].reshape(-1) for item in chunk])
                    host_packed[offset : offset + chunk_numel].copy_(
                        packed, non_blocking=True
                    )
                    torch.cuda.current_stream(source_device).synchronize()
            else:
                packed = torch.cat([entry[2].reshape(-1) for entry in cpu_entries])
                host_packed = packed.cpu()

            if transfer_state is not None and transfer_state.get("cpu_keys") == cpu_keys:
                transfer_state["cpu_accumulator"].add_(host_packed)
                for parameter in active_parameters:
                    parameter.main_grad.zero_()
                    parameter._fluxvla_autograd_grad_seen = False
                    parameter._fluxvla_fsdp2_local_grad_norm = None
                return []

            first_cpu_layout = transfer_state is not None and "cpu_keys" not in transfer_state
            if first_cpu_layout:
                cpu_accumulator = host_packed.clone()
                transfer_state["cpu_keys"] = cpu_keys
                transfer_state["cpu_accumulator"] = cpu_accumulator
            else:
                cpu_accumulator = None
            offset = 0
            for _, key, current in cpu_entries:
                numel = current.numel()
                if first_cpu_layout:
                    value = cpu_accumulator[offset : offset + numel].view_as(current)
                    accumulator[key] = value
                else:
                    value = host_packed[offset : offset + numel].view_as(current)
                    if key in accumulator:
                        accumulator[key].add_(value)
                    else:
                        accumulator[key] = value.clone()
                offset += numel

        for parameter in active_parameters:
            parameter.main_grad.zero_()
            parameter._fluxvla_autograd_grad_seen = False
            parameter._fluxvla_fsdp2_local_grad_norm = None
        return []

    finalized = []
    cpu_entries = []
    for parameter in all_parameters:
        key = id(parameter)
        staged = accumulator.pop(key, None)
        if staged is None and key not in active_ids:
            continue
        if staged is not None:
            if staged.device.type == "cpu" and parameter.main_grad.device.type != "cpu":
                cpu_entries.append((parameter, staged, key in active_ids))
            else:
                if key in active_ids:
                    parameter.main_grad.add_(staged)
                else:
                    parameter.main_grad.copy_(staged)
        parameter._fluxvla_fsdp2_local_grad_norm = None
        parameter._fluxvla_autograd_grad_seen = False
        finalized.append(parameter)

    if cpu_entries:
        cpu_keys = [id(entry[0]) for entry in cpu_entries]
        if transfer_state is not None and transfer_state.get("cpu_keys") == cpu_keys:
            host_packed = transfer_state["cpu_accumulator"]
        else:
            host_packed = torch.cat([entry[1].reshape(-1) for entry in cpu_entries])
        max_chunk_numel = int(
            os.environ.get(
                "FLUXVLA_STRICT_TRANSFER_CHUNK_NUMEL",
                str(256 * 1024 * 1024),
            )
        )
        offset = 0
        chunk = []
        chunk_numel = 0

        def restore_chunk(entries, start, numel):
            device_packed = host_packed[start : start + numel].to(
                device=entries[0][0].main_grad.device,
                dtype=entries[0][0].main_grad.dtype,
                non_blocking=True,
            )
            inner_offset = 0
            for parameter, staged, is_active in entries:
                value = device_packed[
                    inner_offset : inner_offset + staged.numel()
                ].view_as(parameter.main_grad)
                if is_active:
                    parameter.main_grad.add_(value)
                else:
                    parameter.main_grad.copy_(value)
                inner_offset += staged.numel()

        for entry in cpu_entries:
            current_numel = entry[1].numel()
            if chunk and chunk_numel + current_numel > max_chunk_numel:
                restore_chunk(chunk, offset, chunk_numel)
                offset += chunk_numel
                chunk = []
                chunk_numel = 0
            chunk.append(entry)
            chunk_numel += current_numel
        if chunk:
            restore_chunk(chunk, offset, chunk_numel)
    if transfer_state is not None:
        transfer_state.clear()
    if accumulator:
        raise RuntimeError(
            "LingBot strict FSDP2 accumulator contains parameters outside "
            f"the registered reduction units ({len(accumulator)} tensors)"
        )
    return finalized


def _install_strict_fsdp2_microbatch_reduction() -> None:
    """Match FSDP2's reduce-each-microbatch accumulation order for any GA.

    MCore normally accumulates local microbatch gradients before one DP
    collective. Composable FSDP2 reduces each microbatch and then accumulates
    the reduced gradients. The operations are algebraically equivalent but
    differ in FP32 rounding. This correctness-only path streams every reduced
    non-final microbatch into a separate accumulator so subsequent backward
    calls can reuse MCore's fixed main-grad buffers. The default CUDA staging
    trades memory for throughput; CPU staging remains available as an explicit
    low-memory fallback through ``FLUXVLA_STRICT_ACCUMULATOR_DEVICE=cpu``.
    """
    from megatron.core import parallel_state
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.pipeline_parallel import schedules
    from megatron.training import get_args

    original = schedules.backward_step
    if getattr(original, "_fluxvla_lingbot_strict_microbatch", False):
        return
    state = {
        "backward_count": 0,
        "accumulator": {},
        "transfer_state": {},
    }

    def accumulator_device() -> str:
        value = os.environ.get("FLUXVLA_STRICT_ACCUMULATOR_DEVICE", "cuda")
        if value not in {"cuda", "cpu"}:
            raise ValueError(
                "FLUXVLA_STRICT_ACCUMULATOR_DEVICE must be 'cuda' or 'cpu'; "
                f"got {value!r}"
            )
        return value

    def cuda_accumulator_limit_bytes() -> int | None:
        if accumulator_device() != "cuda":
            return None
        limit_gib = float(
            os.environ.get("FLUXVLA_STRICT_CUDA_ACCUMULATOR_GIB", "13")
        )
        if limit_gib <= 0:
            raise ValueError(
                "FLUXVLA_STRICT_CUDA_ACCUMULATOR_GIB must be positive; "
                f"got {limit_gib}"
            )
        return int(limit_gib * 1024**3)

    def reduce_current_microbatch() -> None:
        process_group = parallel_state.get_data_parallel_group()
        tensor_parallel_group = parallel_state.get_tensor_model_parallel_group()
        tensor_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
        detailed_phase_log = os.environ.get("FLUXVLA_STRICT_PHASE_LOG") == "2"
        for unit_index, unit in enumerate(_FSDP2_REDUCTION_UNITS):
            if detailed_phase_log:
                _strict_phase_log(
                    f"reduce-unit-start index={unit_index} name={unit.name}"
                )
            _strict_fsdp2_reduce_unit(
                unit,
                process_group,
                tensor_parallel_group=tensor_parallel_group,
                tensor_parallel_rank=tensor_parallel_rank,
            )
            if detailed_phase_log:
                _strict_phase_log(
                    f"reduce-unit-end index={unit_index} name={unit.name}"
                )

    def strict_microbatch_backward_step(*args, **kwargs):
        _strict_phase_log("backward-start")
        result = original(*args, **kwargs)
        _strict_phase_log("backward-end")
        if not getattr(get_args(), "lingbot_fsdp2_strict_dp_reduction", False):
            return result
        num_microbatches = get_num_microbatches()
        if num_microbatches == 1:
            return result
        microbatch_index = state["backward_count"] % num_microbatches
        state["backward_count"] += 1
        if microbatch_index == 0 and state["accumulator"]:
            raise RuntimeError(
                "LingBot strict FSDP2 started a new optimizer step with a "
                "non-empty microbatch accumulator"
            )
        _strict_phase_log(f"microbatch-reduce-start index={microbatch_index}")
        reduce_current_microbatch()
        _strict_phase_log(f"microbatch-reduce-end index={microbatch_index}")
        active_parameters = [
            parameter
            for unit in _FSDP2_REDUCTION_UNITS
            for parameter in unit.parameters
            if getattr(parameter, "_fluxvla_autograd_grad_seen", False)
        ]
        final_microbatch = microbatch_index == num_microbatches - 1
        all_parameters = [
            parameter
            for unit in _FSDP2_REDUCTION_UNITS
            for parameter in unit.parameters
        ]
        if not final_microbatch:
            device = accumulator_device()
            _strict_phase_log(
                f"accumulator-stage-start index={microbatch_index} "
                f"device={device} "
                f"parameters={len(active_parameters)}"
            )
            _accumulate_strict_microbatch_gradients(
                active_parameters,
                all_parameters,
                state["accumulator"],
                final_microbatch=False,
                accumulator_device=device,
                cuda_accumulator_limit_bytes=cuda_accumulator_limit_bytes(),
                transfer_state=state["transfer_state"],
            )
            _strict_phase_log("accumulator-stage-end")
            return result

        _strict_phase_log(
            f"accumulator-finalize-start parameters={len(active_parameters)}"
        )
        finalized_parameters = _accumulate_strict_microbatch_gradients(
            active_parameters,
            all_parameters,
            state["accumulator"],
            final_microbatch=True,
            transfer_state=state["transfer_state"],
        )
        _strict_phase_log("accumulator-finalize-end")
        state["accumulator"] = {}
        # The per-microbatch norms no longer describe the accumulated gradient.
        # Recreate the exact Native FSDP2 rank-local shard norms before the
        # optimizer computes its global norm and clipping coefficient.
        for parameter in finalized_parameters:
            parameter._fluxvla_autograd_grad_seen = True
        _strict_phase_log("norm-refresh-start")
        _refresh_strict_fsdp2_local_grad_norms()
        _strict_phase_log("norm-refresh-end")
        for parameter in finalized_parameters:
            parameter._fluxvla_autograd_grad_seen = False
        global _FSDP2_MICROBATCH_REDUCTION_COMPLETE
        _FSDP2_MICROBATCH_REDUCTION_COMPLETE = True
        return result

    strict_microbatch_backward_step._fluxvla_lingbot_strict_microbatch = True
    schedules.backward_step = strict_microbatch_backward_step


def _chunk_parameter_groups(groups: list[dict], chunk_size: int) -> list[dict]:
    """Split large AdamW groups while preserving their scheduling metadata."""
    if chunk_size <= 0:
        raise ValueError("LingBot AdamW parameter-group chunk size must be positive")
    chunked = []
    for group in groups:
        parameters = group["params"]
        for start in range(0, len(parameters), chunk_size):
            item = dict(group)
            item["params"] = parameters[start : start + chunk_size]
            chunked.append(item)
    return chunked


def _restore_fp32_master_parameters(optimizer) -> int:
    """Restore DistOpt owned shards from pre-BF16 checkpoint parameters."""
    restored = 0
    children = getattr(optimizer, "chained_optimizers", (optimizer,))
    model_parameters = set()
    for child in children:
        model_map = getattr(child, "model_param_group_index_map", {})
        for model_parameter in model_map:
            model_parameters.add(model_parameter)
            initial = getattr(model_parameter, "_fluxvla_fp32_init", None)
            main_parameter = getattr(model_parameter, "main_param", None)
            if initial is None or main_parameter is None:
                continue
            param_range = child._get_model_param_range_map(model_parameter)["param"]
            source = initial.view(-1)[param_range.start : param_range.end]
            if source.numel() != main_parameter.numel():
                raise RuntimeError(
                    "LingBot FP32 master shard does not match MCore DistOpt range"
                )
            main_parameter.copy_(source.to(device=main_parameter.device))
            restored += 1
        for model_chunk in getattr(child, "model_chunks", ()):
            model_parameters.update(model_chunk.parameters())
    for model_parameter in model_parameters:
        if hasattr(model_parameter, "_fluxvla_fp32_init"):
            del model_parameter._fluxvla_fp32_init
    return restored


def is_routed_expert_parameter(name: str, _parameter) -> bool:
    return bool(_ROUTED_EXPERT.search(name))


@torch.no_grad()
def _fsdp2_virtual_dp_norms(
    parameters: list[torch.nn.Parameter],
    local_optimizer_ranges: dict[int, tuple[torch.Tensor, tuple[int, int] | None]],
    *,
    data_parallel_size: int,
    trace_path: Path | None = None,
) -> list[torch.Tensor]:
    """Rebuild FSDP2 rank-local norms from TP-sharded model gradients.

    The parity-reference forward retains MCore TP parameter ownership, while
    the Native oracle shards each logical parameter over DP ranks. Recreating
    those virtual DP shards is required for DTensor ``_NormPartial`` clipping;
    a single materialized global norm loses the rank-local epsilon placement.
    """
    from megatron.core import parallel_state

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dp_group = parallel_state.get_data_parallel_group()
    norms_by_dp_rank: list[list[torch.Tensor]] = [
        [] for _ in range(data_parallel_size)
    ]
    for index, parameter in enumerate(parameters):
        owned = local_optimizer_ranges.get(index)
        local_gradient = torch.zeros(
            parameter.numel(),
            dtype=torch.float32,
            device=parameter.device,
        )
        if owned is not None:
            gradient, model_range = owned
            start, end = (0, parameter.numel()) if model_range is None else model_range
            local_gradient[start:end].copy_(gradient.detach().view(-1))
        torch.distributed.all_reduce(
            local_gradient,
            op=torch.distributed.ReduceOp.SUM,
            group=dp_group,
        )
        logical_gradient = local_gradient.view(parameter.shape)
        if tp_size > 1 and getattr(parameter, "tensor_model_parallel", False):
            shards = [torch.empty_like(logical_gradient) for _ in range(tp_size)]
            torch.distributed.all_gather(shards, logical_gradient, group=tp_group)
            logical_gradient = torch.cat(
                shards,
                dim=int(getattr(parameter, "partition_dim", 0)),
            )
        flat = logical_gradient.contiguous().view(-1)
        # DTensor/FSDP uses uneven shards when a parameter is not divisible by
        # the DP size (for example, a 55-element action bias becomes 28/27).
        for rank, shard in enumerate(torch.tensor_split(flat, data_parallel_size)):
            norms_by_dp_rank[rank].append(
                torch.linalg.vector_norm(shard, 2.0, dtype=torch.float32)
            )
    stacked_norms = [torch.stack(norms) for norms in norms_by_dp_rank]
    local_norms = [
        torch.linalg.vector_norm(norms, 2.0) for norms in stacked_norms
    ]
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "per_parameter_norms": [
                    norms.detach().cpu() for norms in stacked_norms
                ],
                "local_total_norms": [norm.detach().cpu() for norm in local_norms],
            },
            trace_path,
        )
    return local_norms


def _install_fsdp2_parameterwise_grad_norm(model, optimizer) -> None:
    """Match FSDP2's per-parameter FP32 norm reduction for DP/TP/EP parity.

    MCore's Apex norm kernel first flattens all optimizer-owned shards and
    performs the final square root on the host. FSDP2 instead computes a norm
    for every DTensor parameter and combines those norms on the GPU. The two
    paths can differ by one FP32 ULP, which changes clipped AdamW updates.

    DistributedOptimizer partitions each local model parameter over its DP
    group. Its grad-stat group spans the corresponding TP/EP model-parallel
    ranks, so summing owned shard squares reconstructs the global norm. A
    replicated parameter contributes only on TP rank zero; genuine TP shards
    and disjoint EP local experts contribute on every owning rank.
    """
    from megatron.training import get_args

    args = get_args()
    if not getattr(args, "lingbot_fsdp2_parameterwise_grad_norm", True):
        return
    if args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1:
        return

    from ..diagnostics import _optimizer_tensors
    from megatron.core.tensor_parallel.layers import (
        param_is_not_tensor_parallel_duplicate,
    )

    model_chunks = model if isinstance(model, list) else [model]
    parameters = []
    parameter_names = {}
    seen = set()
    for chunk in model_chunks:
        for name, parameter in chunk.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
                parameter_names[id(parameter)] = name

    owned_entries = []
    local_optimizer_ranges = {}
    trace_device = None
    for index, parameter in enumerate(parameters):
        tensors = _optimizer_tensors(optimizer, parameter)
        if tensors is None:
            continue
        gradient = tensors.get("model_grad")
        if gradient is not None and trace_device is None:
            trace_device = gradient.device
        if gradient is not None:
            local_optimizer_ranges[index] = (
                gradient,
                tensors.get("model_range"),
            )
        if (
            gradient is not None
            and param_is_not_tensor_parallel_duplicate(parameter)
        ):
            owned_entries.append((index, parameter, gradient))
    if trace_device is None:
        return
    indices = torch.tensor(
        [index for index, _, _ in owned_entries],
        dtype=torch.long,
        device=trace_device,
    )
    owned_gradients = [gradient for _, _, gradient in owned_entries]
    # Keep the EP ownership classification beside the frozen parameter order
    # for an opt-in diagnostic norm experiment.  It is deliberately not used
    # by the default parity path until parameter-level ownership is proven.
    expert_entry_mask = [
        not bool(getattr(parameter, "allreduce", True))
        for _, parameter, _ in owned_entries
    ]
    trace_state = {"step": 0}

    @torch.no_grad()
    def get_grad_norm_fsdp2(chained_optimizer):
        global _FSDP2_PARTIAL_CLIP_DENOMINATOR
        trace_state["step"] += 1
        _FSDP2_PARTIAL_CLIP_DENOMINATOR = None
        partial_norms = [
            getattr(parameter, "_fluxvla_fsdp2_local_grad_norm", None)
            for parameter in parameters
        ]
        trace_root = os.environ.get("FLUXVLA_FSDP2_NORM_TRACE_DIR")
        trace_target = int(
            os.environ.get("FLUXVLA_FSDP2_NORM_TRACE_STEP", "0")
        )
        trace_path = None
        if trace_root and (trace_target == 0 or trace_target == trace_state["step"]):
            rank = int(os.environ.get("RANK", "0"))
            trace_path = (
                Path(trace_root)
                / f"mcore_norm_step_{trace_state['step']:06d}_rank_{rank:02d}.pt"
            )
        if any(value is not None for value in partial_norms):
            from megatron.core import parallel_state

            present_indices = [
                index for index, value in enumerate(partial_norms) if value is not None
            ]
            present_norms = [partial_norms[index] for index in present_indices]
            local_total_norm = torch.linalg.vector_norm(
                torch.stack(present_norms), 2.0
            )
            group = chained_optimizer.get_grad_stats_parallel_group()
            contributes = parallel_state.get_tensor_model_parallel_rank() == 0
            global_squared = (
                local_total_norm.square()
                if contributes
                else torch.zeros_like(local_total_norm)
            )
            torch.distributed.all_reduce(
                global_squared,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
            clip_squared = (
                (local_total_norm + 1.0e-6).square()
                if contributes
                else torch.zeros_like(local_total_norm)
            )
            torch.distributed.all_reduce(
                clip_squared,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
            _FSDP2_PARTIAL_CLIP_DENOMINATOR = clip_squared.sqrt()
            if trace_path is not None:
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "mode": "partial",
                        "step": trace_state["step"],
                        "present_indices": present_indices,
                        "parameter_names": [
                            parameter_names[id(parameters[index])]
                            for index in present_indices
                        ],
                        "per_parameter_norms": torch.stack(present_norms)
                        .detach()
                        .cpu(),
                        "local_total_norm": local_total_norm.detach().cpu(),
                        "contributes": contributes,
                        "global_total_norm": global_squared.sqrt().detach().cpu(),
                        "clip_denominator": _FSDP2_PARTIAL_CLIP_DENOMINATOR
                        .detach()
                        .cpu(),
                    },
                    trace_path,
                )
            for parameter in parameters:
                parameter._fluxvla_fsdp2_local_grad_norm = None
            return float(global_squared.sqrt().item())
        if (
            getattr(args, "lingbot_tp_parity_reference", False)
            and args.tensor_model_parallel_size > 1
        ):
            virtual_norms = _fsdp2_virtual_dp_norms(
                parameters,
                local_optimizer_ranges,
                data_parallel_size=args.data_parallel_size,
                trace_path=trace_path,
            )
            total_norm = torch.linalg.vector_norm(
                torch.stack(virtual_norms), 2.0
            )
            _FSDP2_PARTIAL_CLIP_DENOMINATOR = torch.linalg.vector_norm(
                torch.stack(
                    [norm + 1.0e-6 for norm in virtual_norms]
                ),
                2.0,
            )
            return float(total_norm.item())

        local_squared = torch.zeros(
            len(parameters), dtype=torch.float32, device=trace_device
        )
        if owned_gradients:
            local_norms = torch._foreach_norm(owned_gradients, 2.0)
            local_squared.index_copy_(
                0,
                indices,
                torch.stack(local_norms).float().square(),
            )
        trace_root = os.environ.get("FLUXVLA_GRAD_NORM_INPUT_TRACE_DIR")
        trace_step = int(os.environ.get("FLUXVLA_GRAD_NORM_INPUT_TRACE_STEP", "0"))
        if trace_root and (trace_step == 0 or trace_step == trace_state["step"]):
            rank = int(os.environ.get("RANK", "0"))
            path = Path(trace_root) / f"mcore_owned_rank{rank}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            owned_by_index = {
                index: (parameter, gradient)
                for index, parameter, gradient in owned_entries
            }
            group = chained_optimizer.get_grad_stats_parallel_group()
            with path.open("a", encoding="utf-8") as stream:
                for index, parameter in enumerate(parameters):
                    owned = owned_by_index.get(index)
                    gradient = None if owned is None else owned[1]
                    row = {
                        "backend": "mcore",
                        "step": trace_state["step"],
                        "rank": rank,
                        "index": index,
                        "name": parameter_names.get(id(parameter), "<unnamed>"),
                        "parameter_shape": list(parameter.shape),
                        "gradient_shape": None if gradient is None else list(gradient.shape),
                        "gradient_norm": (
                            None
                            if gradient is None
                            else float(
                                torch.linalg.vector_norm(
                                    gradient.detach(), 2.0, dtype=torch.float32
                                ).item()
                            )
                        ),
                        "tensor_model_parallel": bool(
                            getattr(parameter, "tensor_model_parallel", False)
                        ),
                        "partition_dim": getattr(parameter, "partition_dim", None),
                        "allreduce": bool(getattr(parameter, "allreduce", True)),
                        "grad_stats_group_size": torch.distributed.get_world_size(group),
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
        stats_group = chained_optimizer.get_grad_stats_parallel_group()
        from megatron.core import parallel_state

        expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        # EP norm de-duplication is diagnostic-only until per-parameter
        # ownership is proven against the Native FSDP2 layout.  In particular,
        # DistributedOptimizer may already partition a replicated dense
        # parameter across the full stats group; blindly dividing every dense
        # entry by EP underestimates the norm and changes clipping.  Keep the
        # historical MCore norm as the default parity path.
        if (
            expert_parallel_size > 1
            and os.environ.get("FLUXVLA_EP_NORM_DEDUP", "0") == "1"
        ):
            # ``stats_group`` is the one-instance DistOpt group.  Reduce the
            # two ownership classes independently before taking the norm.
            dense_squared = torch.zeros((), dtype=torch.float32, device=trace_device)
            expert_squared = torch.zeros_like(dense_squared)
            if owned_gradients:
                squared = torch.stack(
                    [
                        norm.float().square()
                        for norm in torch._foreach_norm(owned_gradients, 2.0)
                    ]
                )
                mask = torch.tensor(expert_entry_mask, device=trace_device, dtype=torch.bool)
                if (~mask).any():
                    dense_squared = squared[~mask].sum()
                if mask.any():
                    expert_squared = squared[mask].sum()
            totals = torch.stack((dense_squared, expert_squared))
            torch.distributed.all_reduce(
                totals,
                op=torch.distributed.ReduceOp.SUM,
                group=stats_group,
            )
            # MCore's ordinary DP group includes the EP dimension.  A local
            # routed expert is replicated only across the expert-data group,
            # whose size is DP/EP; using the ordinary DP size would divide its
            # norm twice and under-estimate the clipping denominator.
            dense_dp_size = parallel_state.get_expert_data_parallel_world_size()
            totals[0].div_(float(expert_parallel_size))
            totals[1].div_(float(dense_dp_size))
            total_norm = totals.sum().sqrt()
            runtime_trace = os.environ.get("FLUXVLA_FSDP2_NORM_RUNTIME_TRACE")
            if runtime_trace:
                Path(runtime_trace).parent.mkdir(parents=True, exist_ok=True)
                with Path(runtime_trace).open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "optimizer_class": type(chained_optimizer).__name__,
                                "dense_squared": float(dense_squared.item()),
                                "expert_squared": float(expert_squared.item()),
                                "expert_parallel_size": expert_parallel_size,
                                "expert_data_parallel_size": dense_dp_size,
                                "total_norm": float(total_norm.item()),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        else:
            torch.distributed.all_reduce(
                local_squared,
                op=torch.distributed.ReduceOp.SUM,
                group=stats_group,
            )
            total_norm = local_squared.sum().sqrt()
        # Keep the squared-norm values in FP32 through the final scalar
        # reduction. Taking an elementwise sqrt and then a second norm changes
        # the last ULP relative to FSDP2's norm(stack(per-parameter norms)).
        return float(total_norm.item())

    optimizer.get_grad_norm = types.MethodType(get_grad_norm_fsdp2, optimizer)


def _fsdp2_clip_coefficient(
    max_norm: float,
    total_norm: float | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build PyTorch FSDP2's coefficient, including NormPartial semantics."""
    denominator = _FSDP2_PARTIAL_CLIP_DENOMINATOR
    if denominator is None:
        # Match Native FSDP2's clip_grad_norm_ epsilon.  The historical MCore
        # path used 2e-6 here; that changes the coefficient by several FP32
        # ULPs and can flip BF16 parameter updates at the first optimizer step.
        norm = torch.as_tensor(total_norm, dtype=torch.float32, device=device)
        denominator = norm + 1.0e-6
    else:
        denominator = denominator.to(device=device, dtype=torch.float32)
    return (float(max_norm) / denominator).clamp(max=1.0)


def _install_fsdp2_foreach_clip() -> None:
    """Use the same CUDA FP32 coefficient and foreach scale as FSDP2."""
    import megatron.core.optimizer.optimizer as optimizer_module
    from megatron.core.optimizer.clip_grads import to_local_if_dtensor

    original = optimizer_module.clip_grad_by_total_norm_fp32
    if getattr(original, "_fluxvla_lingbot_fsdp2_clip", False):
        return

    @torch.no_grad()
    def clip_grad_by_total_norm_fsdp2(
        parameters,
        max_norm,
        total_norm,
        use_decoupled_grad=False,
    ):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        gradients = [
            to_local_if_dtensor(gradient).detach()
            for parameter in parameters
            if (gradient := getattr(parameter, grad_attr, None)) is not None
        ]
        if not gradients:
            return
        coefficient = _fsdp2_clip_coefficient(
            max_norm,
            total_norm,
            gradients[0].device,
        )
        global _LAST_FSDP2_CLIP_COEFFICIENT
        _LAST_FSDP2_CLIP_COEFFICIENT = coefficient.detach().clone()
        torch._foreach_mul_(gradients, coefficient)

    clip_grad_by_total_norm_fsdp2._fluxvla_lingbot_fsdp2_clip = True
    optimizer_module.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fsdp2


@torch.no_grad()
def _sum_then_postdivide_grad_sync(bucket_group, original_start_grad_sync) -> None:
    """Match the active PyTorch FSDP2 gradient reduction contract.

    PyTorch 2.7 FSDP2 uses NCCL AVG directly for FP32/BF16 reductions. MCore's
    equivalent ``average_in_collective`` path must pass through unchanged; a
    pre-scaled SUM is mathematically equivalent but changes FP32 rounding and
    can flip BF16 parameter values after the optimizer update. The legacy SUM
    branch remains for explicitly configured older-runtime diagnostics.
    """
    config = bucket_group.ddp_config
    if not config.use_distributed_optimizer:
        raise RuntimeError(
            "LingBot FSDP2 reduction ordering requires MCore DistributedOptimizer"
        )
    if config.overlap_grad_reduce:
        raise RuntimeError(
            "LingBot FSDP2 reduction ordering requires synchronous gradient reduction"
        )
    if config.num_distributed_optimizer_instances != 1:
        raise RuntimeError(
            "LingBot FSDP2 reduction ordering requires one DistributedOptimizer instance"
        )
    reduction_states = [
        bool(getattr(parameter, "_fluxvla_grad_already_reduced", False))
        for bucket in bucket_group.buckets
        for parameter in getattr(bucket, "params_list", ())
    ]
    if any(reduction_states):
        if not all(reduction_states):
            raise RuntimeError(
                "LingBot already-reduced gradients cannot share a bucket with "
                "ordinary gradients"
            )
        # Gradients are already the full-DP FSDP2 average, but remain in the
        # ordinary DistOpt buffer so optimizer ownership and parameter
        # all-gather are unchanged. With synchronous reduction there is no
        # communication handle for finish_grad_sync to wait on.
        bucket_group.grad_reduce_handle = None
        return

    if config.average_in_collective:
        return original_start_grad_sync(bucket_group)

    world_size = bucket_group.intra_distributed_optimizer_instance_size
    rank = bucket_group.intra_distributed_optimizer_instance_rank
    factors = [bucket.gradient_scaling_factor for bucket in bucket_group.buckets]
    if not factors:
        return original_start_grad_sync(bucket_group)
    factor = factors[0]
    if factor <= 0.0 or any(abs(item - factor) > 1e-12 for item in factors):
        raise RuntimeError(
            "LingBot FSDP2 reduction ordering requires one positive gradient "
            f"scaling factor per bucket group, got {factors}"
        )
    reduction_divisor = round(1.0 / factor)
    if (
        not math.isclose(factor * reduction_divisor, 1.0, abs_tol=1e-12)
        or reduction_divisor < world_size
        or reduction_divisor % world_size
    ):
        raise RuntimeError(
            "LingBot FSDP2 reduction ordering expected a reciprocal global-DP "
            f"scaling factor compatible with communication world size {world_size}, "
            f"got {factor}"
        )

    from torch.distributed.algorithms._comm_hooks.default_hooks import DefaultState

    predivide = DefaultState._get_gradient_predivide_factor(reduction_divisor)
    postdivide = reduction_divisor / predivide

    try:
        for bucket in bucket_group.buckets:
            bucket.gradient_scaling_factor = 1.0 / predivide
        original_start_grad_sync(bucket_group)
    finally:
        for bucket, factor in zip(bucket_group.buckets, factors, strict=True):
            bucket.gradient_scaling_factor = factor

    if bucket_group.grad_reduce_handle is not None:
        raise RuntimeError(
            "LingBot FSDP2 reduction ordering cannot post-divide an asynchronous collective"
        )
    for bucket in bucket_group.buckets:
        shards = bucket.grad_data.chunk(world_size)
        if len(shards) != world_size:
            raise RuntimeError("LingBot gradient buffer cannot be evenly sharded over DP")
        # Expert buckets may communicate over expert-DP while retaining the
        # dense global-DP divisor, so derive both factors from the recorded
        # gradient scaling factor rather than the communication world size.
        if postdivide > 1.0:
            shards[rank].div_(postdivide)


def install_lingbot_fsdp2_reduction_order() -> None:
    """Install LingBot's opt-in FSDP2-compatible DP reduction ordering."""
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
    from megatron.training import get_args

    original = _ParamAndGradBucketGroup.start_grad_sync
    if getattr(original, "_fluxvla_lingbot_fsdp2_order", False):
        return

    def start_grad_sync(bucket_group):
        args = get_args()
        if not getattr(args, "lingbot_fsdp2_reduction_order", True):
            return original(bucket_group)
        return _sum_then_postdivide_grad_sync(bucket_group, original)

    start_grad_sync._fluxvla_lingbot_fsdp2_order = True
    _ParamAndGradBucketGroup.start_grad_sync = start_grad_sync


def install_lingbot_optimizer_builder() -> None:
    """Add native LingBot's routed-expert LR multiplier without vendoring MCore."""
    from megatron.training import get_args
    from megatron.training import training as megatron_training
    import megatron.core.optimizer as mcore_optimizer

    _install_fsdp2_foreach_clip()
    _install_strict_fsdp2_dp_reduction()
    _install_strict_fsdp2_microbatch_reduction()

    get_param_groups = mcore_optimizer._get_param_groups
    if not getattr(get_param_groups, "_fluxvla_lingbot_chunked", False):
        def get_chunked_param_groups(*args, **kwargs):
            groups = get_param_groups(*args, **kwargs)
            runtime_args = get_args()
            if not getattr(runtime_args, "lingbot_low_memory_adam", True):
                return groups
            return _chunk_parameter_groups(
                groups,
                int(getattr(runtime_args, "lingbot_adam_foreach_chunk_size", 1)),
            )

        get_chunked_param_groups._fluxvla_lingbot_chunked = True
        mcore_optimizer._get_param_groups = get_chunked_param_groups

    original = megatron_training.setup_model_and_optimizer
    if getattr(original, "_fluxvla_lingbot_optimizer", False):
        return

    def setup_model_and_optimizer(
        model_provider_func,
        model_type,
        no_wd_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0,
        checkpointing_context=None,
    ):
        if scale_lr_cond is not None:
            raise ValueError("LingBot MCore owns scale_lr_cond for routed experts")
        args = get_args()
        if getattr(args, "optimizer_cpu_offload", False) and getattr(
            args, "use_precision_aware_optimizer", False
        ):
            # Megatron's read-only optimizer_config compares a missing TE
            # version with packaging.Version.  Keep the backend usable with
            # its torch hybrid optimizer without modifying Megatron-LM.
            from megatron.core import utils as mcore_utils
            from packaging.version import Version

            if mcore_utils.get_te_version() is None:
                def _missing_te_version() -> Version:
                    return Version("0.0.0")

                mcore_utils.get_te_version = _missing_te_version
        multiplier = float(getattr(args, "lingbot_expert_lr_multiplier", 1.0))
        result = original(
            model_provider_func,
            model_type,
            no_wd_decay_cond=no_wd_decay_cond,
            scale_lr_cond=is_routed_expert_parameter,
            # ``lr_mult`` is the scheduler-wide multiplier and would scale
            # every parameter group.  Native LingBot applies the multiplier
            # only to routed-expert parameters; the predicate above already
            # provides that per-group scale, so keep the global factor 1.0.
            lr_mult=1.0,
            checkpointing_context=checkpointing_context,
        )
        _, optimizer, _ = result
        scheduler = result[2]
        # Native FluxVLA advances its scheduler once per optimizer update.
        # Megatron's default scheduler uses consumed samples instead, so its
        # ``increment`` is ``GB`` and the LR curve changes when GB changes.
        # Keep the MCore backend's public LR knobs step-based and ignore that
        # sample-count increment at this boundary.  This is an instance-level
        # adapter; Megatron-LM remains read-only.
        decay_steps = getattr(args, "lr_decay_iters", None)
        if decay_steps is None:
            decay_steps = getattr(args, "train_iters", None)
        if decay_steps is not None:
            scheduler.lr_decay_steps = int(decay_steps)
            scheduler.wd_incr_steps = int(
                getattr(args, "train_iters", decay_steps)
            )
            original_scheduler_step = scheduler.step

            def step_by_optimizer_update(increment=1, *, _step=original_scheduler_step):
                del increment
                return _step(1)

            scheduler.step = step_by_optimizer_update

        # FluxVLA Native calls ``prepare_step`` before the first optimizer
        # update; Megatron advances its scheduler after the update.  Prime a
        # fresh run to schedule(step=1); checkpoint resume keeps its cursor.
        if (
            int(getattr(scheduler, "num_steps", 0)) == 0
            and int(getattr(args, "iteration", 0)) == 0
        ):
            scheduler.step(1)
        restored = _restore_fp32_master_parameters(optimizer)
        # In the Native-FP32 master path the model parameters themselves are
        # already FP32, so Megatron has no pre-cast ``_fluxvla_fp32_init``
        # snapshots to restore. Its optimizer main shards are initialized from
        # those parameters directly; only the historical BF16 model path needs
        # the explicit restoration guard.
        preserve_fp32 = os.environ.get("FLUXVLA_MCORE_PRESERVE_FP32_PARAMS") == "1"
        precision_aware = bool(
            getattr(args, "use_precision_aware_optimizer", False)
        )
        if (
            getattr(args, "use_distributed_optimizer", False)
            and restored == 0
            and not preserve_fp32
            and not precision_aware
        ):
            raise RuntimeError("LingBot MCore did not restore any FP32 master shards")
        if getattr(args, "lingbot_low_memory_adam", True):
            for group in optimizer.param_groups:
                group["foreach"] = True
        _install_fsdp2_parameterwise_grad_norm(result[0], optimizer)
        return result

    setup_model_and_optimizer._fluxvla_lingbot_optimizer = True
    megatron_training.setup_model_and_optimizer = setup_model_and_optimizer


__all__ = [
    "install_lingbot_fsdp2_reduction_order",
    "install_lingbot_optimizer_builder",
    "is_routed_expert_parameter",
    "register_lingbot_fsdp2_reduction_units",
]
