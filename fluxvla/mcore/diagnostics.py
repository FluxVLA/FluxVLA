"""Opt-in diagnostics for native MCore optimizer alignment."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

_REGISTERED_PARAMETERS: list[tuple[str, torch.nn.Parameter]] = []
_REGISTERED_MODEL_PARAMETERS: list[torch.nn.Parameter] = []
_REGISTERED_MODEL_NAMED_PARAMETERS: list[tuple[str, torch.nn.Parameter]] = []
_REGISTERED_GRAD_NORM_ENTRIES: list[tuple[str, torch.nn.Parameter | None]] = []
_REGISTERED_GRAD_REDUCE_ENTRIES: list[tuple[str, torch.nn.Parameter | None, int]] = []


def _fsdp_gradient_divide_factors(world_size: int) -> tuple[float, float]:
    """Return the exact pre/post divide factors selected by PyTorch FSDP."""
    from torch.distributed.algorithms._comm_hooks.default_hooks import (
        DefaultState,
    )

    predivide = DefaultState._get_gradient_predivide_factor(world_size)
    return predivide, world_size / predivide


def _ordered_grad_reduce_entries(
    trainable_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    grad_order: Callable[[str], tuple],
    zero_segments: Iterable[tuple[tuple, int]],
) -> list[tuple[str, torch.nn.Parameter | None, int]]:
    """Build the logical FSDP flat-buffer layout, including omitted zero tensors."""
    entries = [
        (grad_order(name), name, parameter, parameter.numel())
        for name, parameter in trainable_parameters
        if getattr(parameter, "allreduce", True)
    ]
    entries.extend(
        (key, f"__zero_reduce_segment_{index}", None, numel)
        for index, (key, numel) in enumerate(zero_segments)
    )
    entries.sort(key=lambda item: item[0])
    return [(name, parameter, numel) for _, name, parameter, numel in entries]


def _fsdp_no_shard_total_norm(gradients: list[torch.Tensor]) -> torch.Tensor:
    """Use the same foreach-capable norm path as FSDP ``NO_SHARD``."""
    get_total_norm = getattr(torch.nn.utils, "get_total_norm", None)
    if get_total_norm is None:
        from torch.nn.utils.clip_grad import _get_total_norm as get_total_norm
    return get_total_norm(gradients, 2.0, foreach=None)


def _tp_aware_fsdp_total_norm(
    entries: Iterable[tuple[str, torch.nn.Parameter | None]],
) -> torch.Tensor | None:
    """Reconstruct FSDP's ordered per-tensor norm from TP parameter shards."""
    from megatron.core import parallel_state

    entries = list(entries)
    real_gradients = [
        grad.detach()
        for _, parameter in entries
        if parameter is not None
        and (grad := getattr(parameter, "main_grad", parameter.grad)) is not None
    ]
    if not real_gradients:
        return None

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = torch.distributed.get_world_size(tp_group)
    zero = torch.zeros((), dtype=torch.float32, device=real_gradients[0].device)
    logical_norms = []
    for _, parameter in entries:
        if parameter is None:
            logical_norms.append(zero)
            continue
        grad = getattr(parameter, "main_grad", parameter.grad)
        if grad is None:
            continue
        grad = grad.detach().contiguous()
        is_tp_sharded = bool(
            getattr(parameter, "tensor_model_parallel", False)
            or getattr(parameter, "_tp_sharded", False)
        )
        if is_tp_sharded and tp_world_size > 1:
            shards = [torch.empty_like(grad) for _ in range(tp_world_size)]
            torch.distributed.all_gather(shards, grad, group=tp_group)
            partition_dim = int(getattr(parameter, "partition_dim", 0))
            grad = torch.cat(shards, dim=partition_dim)
        logical_norms.append(torch._foreach_norm([grad], 2.0)[0])
    if not logical_norms:
        return None
    return torch.linalg.vector_norm(torch.stack(logical_norms), 2.0)


def _write_grad_norm_input_trace(
    *,
    backend: str,
    step: int,
    entries: Iterable[tuple[str, torch.nn.Parameter | None]],
) -> None:
    """Record the exact ordered inputs to the global grad-norm reduction."""
    root_value = os.environ.get("FLUXVLA_GRAD_NORM_INPUT_TRACE_DIR")
    if not root_value:
        return
    rank = int(os.environ.get("RANK", "0"))
    path = Path(root_value) / f"{backend}_rank{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for index, (name, parameter) in enumerate(entries):
            grad = (
                None
                if parameter is None
                else getattr(parameter, "main_grad", parameter.grad)
            )
            row = {
                "backend": backend,
                "step": step,
                "index": index,
                "name": name,
                "shape": None if parameter is None else list(parameter.shape),
                "grad_norm": None,
            }
            if grad is not None:
                row["grad_norm"] = float(
                    torch.linalg.vector_norm(
                        grad.detach(), 2.0, dtype=torch.float32
                    ).item()
                )
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _write_post_reduce_gradient_trace(step: int) -> None:
    root_value = os.environ.get("FLUXVLA_POST_REDUCE_GRAD_TRACE_DIR")
    if not root_value:
        return
    rank = int(os.environ.get("RANK", "0"))
    path = Path(root_value) / f"native_rank{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for name, parameter in _REGISTERED_MODEL_NAMED_PARAMETERS:
            # DistributedOptimizer may retain a ``main_grad`` attribute while
            # clearing it between accumulation windows.  ``getattr`` alone
            # does not fall back when that attribute is present but None.
            # The wrapped model parameter's ``.grad`` is the accumulated
            # (pre-DistOpt-normalization) gradient.  ``main_grad`` is the
            # optimizer-owned buffer and may already be divided by the
            # micro-batch/data-parallel factors on later iterations.
            grad = parameter.grad
            if grad is None:
                grad = getattr(parameter, "main_grad", None)
            if grad is None:
                row = {
                    "step": step,
                    "name": name,
                    "shape": list(parameter.shape),
                    "grad": None,
                }
            else:
                value = grad.detach().float()
                row = {
                    "step": step,
                    "name": name,
                    "shape": list(value.shape),
                    "grad_norm": float(value.norm().item()),
                    "grad_mean": float(value.mean().item()),
                    "grad_abs_max": float(value.abs().max().item()),
                    "grad_sum": float(value.sum().item()),
                }
            stream.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def _apply_bf16_grad_roundtrip(parameters) -> None:
    """Match FSDP ``reduce_dtype=bf16`` for FP32 gradient buffers."""
    for parameter in parameters:
        grad = getattr(parameter, "main_grad", parameter.grad)
        if grad is not None and grad.is_floating_point():
            grad.copy_(grad.to(torch.bfloat16).to(grad.dtype))


def _install_fsdp_bf16_grad_communication() -> None:
    """Reproduce FSDP's BF16 collective with FP32 model parameters.

    MCore selects its communication buffer dtype from the parameter dtype. The
    PI0.5 alignment contract deliberately keeps parameters in FP32, while the
    FluxVLA FSDP reference reduces one flat gradient in BF16. This opt-in
    adapter compacts gradients in model parameter order before the collective
    and copies the result back into MCore's buffers. Compaction is important:
    DistributedOptimizer inserts alignment padding, and changing the NCCL
    message layout can change BF16 summation results. The adapter is restricted
    to the synchronous, single-instance DP configuration used for golden tests.
    """
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )

    original = DistributedDataParallel.finish_grad_sync
    if getattr(original, "_fluxvla_fsdp_bf16_contract", False):
        return

    @torch.no_grad()
    def fsdp_bf16_finish_grad_sync(ddp):
        config = ddp.ddp_config
        if (
            config.overlap_grad_reduce
            or config.num_distributed_optimizer_instances != 1
        ):
            raise RuntimeError(
                "PI05_FSDP_GRAD_REDUCE_CONTRACT requires synchronous gradient "
                "reduction and one distributed optimizer instance"
            )
        entries = _REGISTERED_GRAD_REDUCE_ENTRIES or [
            (str(index), parameter, parameter.numel())
            for index, parameter in enumerate(ddp.module.parameters())
            if parameter.requires_grad and getattr(parameter, "allreduce", True)
        ]
        gradients = [
            getattr(parameter, "main_grad", parameter.grad)
            for _, parameter, _ in entries
            if parameter is not None
        ]
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("FSDP gradient contract found a missing dense gradient")
        group = ddp.dp_cp_group
        world_size = torch.distributed.get_world_size(group)
        total_numel = sum(numel for _, _, numel in entries)
        compact = torch.zeros(
            total_numel,
            dtype=torch.bfloat16,
            device=gradients[0].device,
        )
        offset = 0
        for _, parameter, numel in entries:
            end = offset + numel
            if parameter is not None:
                gradient = getattr(parameter, "main_grad", parameter.grad)
                compact[offset:end].copy_(gradient.view(-1))
            offset = end
        # FSDP does not always divide by the full DP size before the
        # collective. For DP4 it uses /2 -> all-reduce -> /2; reproducing
        # that BF16 rounding order is required for long-run alignment.
        predivide, postdivide = _fsdp_gradient_divide_factors(world_size)
        if predivide > 1:
            compact.div_(predivide)
        torch.distributed.all_reduce(
            compact,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        if postdivide > 1:
            compact.div_(postdivide)
        offset = 0
        for _, parameter, numel in entries:
            end = offset + numel
            if parameter is not None:
                gradient = getattr(parameter, "main_grad", parameter.grad)
                gradient.copy_(compact[offset:end].view_as(gradient))
            offset = end
        for bucket_group in ddp.bucket_groups + ddp.expert_parallel_bucket_groups:
            bucket_group.grad_reduce_handle = None

    fsdp_bf16_finish_grad_sync._fluxvla_fsdp_bf16_contract = True
    DistributedDataParallel.finish_grad_sync = fsdp_bf16_finish_grad_sync


def _install_fsdp_microbatch_grad_roundtrip() -> None:
    """Round the first of two accumulated micro-batch gradients to BF16.

    FSDP DP4/MB1 rounds each rank-local gradient at its BF16 collective
    boundary. MCore DP2/TP2/MB1 instead accumulates two micro-batches in FP32
    before its DP collective. This accuracy-only adapter inserts the missing
    rounding boundary after the first backward. Synchronous MCore DDP does not
    install ``config.no_sync_func``, so this must wrap the scheduler's
    ``backward_step`` rather than ``DistributedDataParallel.no_sync``.
    """
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.pipeline_parallel import schedules

    original = schedules.backward_step
    if getattr(original, "_fluxvla_fsdp_microbatch_contract", False):
        return

    state = {"backward_count": 0, "roundtrip_count": 0}

    def fsdp_microbatch_backward_step(*args, **kwargs):
        num_microbatches = get_num_microbatches()
        if num_microbatches != 2:
            raise RuntimeError(
                "PI05_FSDP_MICROBATCH_GRAD_CONTRACT requires exactly two "
                f"micro-batches, got {num_microbatches}"
            )
        result = original(*args, **kwargs)
        microbatch_index = state["backward_count"] % num_microbatches
        state["backward_count"] += 1
        if microbatch_index == 0:
            _apply_bf16_grad_roundtrip(_REGISTERED_MODEL_PARAMETERS)
            state["roundtrip_count"] += 1
            trace_path = os.environ.get("FLUXVLA_MICROBATCH_GRAD_TRACE")
            if trace_path and int(os.environ.get("RANK", "0")) == 0:
                path = Path(trace_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "backward_count": state["backward_count"],
                                "roundtrip_count": state["roundtrip_count"],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        return result

    fsdp_microbatch_backward_step._fluxvla_fsdp_microbatch_contract = True
    schedules.backward_step = fsdp_microbatch_backward_step


def install_fsdp_grad_reduce_contract() -> None:
    """Optionally reproduce FSDP's BF16 gradient-reduction boundary."""
    if os.environ.get("PI05_FSDP_GRAD_REDUCE_CONTRACT") != "1":
        return

    _install_fsdp_bf16_grad_communication()
    if os.environ.get("PI05_FSDP_MICROBATCH_GRAD_CONTRACT") == "1":
        _install_fsdp_microbatch_grad_roundtrip()

    from megatron.core.optimizer.optimizer import ChainedOptimizer

    original = ChainedOptimizer.step
    if getattr(original, "_fluxvla_fsdp_reduce_contract", False):
        return

    state = {"step": 0}

    @torch.no_grad()
    def fsdp_bf16_reduce_step(optimizer):
        state["step"] += 1
        _apply_bf16_grad_roundtrip(_REGISTERED_MODEL_PARAMETERS)
        _write_post_reduce_gradient_trace(state["step"])
        return original(optimizer)

    fsdp_bf16_reduce_step._fluxvla_fsdp_reduce_contract = True
    ChainedOptimizer.step = fsdp_bf16_reduce_step


def install_fsdp_grad_norm_contract() -> None:
    """Optionally match PyTorch ``clip_grad_norm_`` reduction and scaling."""
    if (os.environ.get("PI05_FSDP_GRAD_NORM_CONTRACT") != "1"
            and os.environ.get("FLUXVLA_FSDP_GRAD_NORM_CONTRACT") != "1"):
        return

    import megatron.core.optimizer.optimizer as optimizer_module
    from megatron.core.optimizer.clip_grads import to_local_if_dtensor
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    original = ChainedOptimizer.get_grad_norm
    if getattr(original, "_fluxvla_fsdp_contract", False):
        return

    state = {"step": 0}

    @torch.no_grad()
    def fsdp_ordered_grad_norm(optimizer):
        if not optimizer.grads_states_parallel_group_is_shared():
            return original(optimizer)
        state["step"] += 1
        entries = _REGISTERED_GRAD_NORM_ENTRIES or [
            (name, parameter) for name, parameter in _REGISTERED_MODEL_NAMED_PARAMETERS
        ]
        _write_grad_norm_input_trace(
            backend="mcore", step=state["step"], entries=entries
        )
        gradients = []
        for _, parameter in entries:
            if parameter is None:
                gradients.append(None)
                continue
            grad = getattr(parameter, "main_grad", parameter.grad)
            if grad is not None:
                gradients.append(grad)
        if not gradients:
            for child in optimizer.chained_optimizers:
                gradients.extend(child.get_main_grads_for_grad_norm())
        if not gradients:
            return 0.0
        real_gradients = [grad for grad in gradients if grad is not None]
        if not real_gradients:
            return 0.0
        from megatron.core import parallel_state

        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            total_norm = _tp_aware_fsdp_total_norm(entries)
            if total_norm is not None:
                return float(total_norm.item())
        zero = torch.zeros((), dtype=torch.float32, device=real_gradients[0].device)
        norm_inputs = [zero if grad is None else grad.detach() for grad in gradients]
        total_norm = _fsdp_no_shard_total_norm(norm_inputs)
        return float(total_norm.item())

    fsdp_ordered_grad_norm._fluxvla_fsdp_contract = True
    ChainedOptimizer.get_grad_norm = fsdp_ordered_grad_norm

    original_clip = optimizer_module.clip_grad_by_total_norm_fp32
    if getattr(original_clip, "_fluxvla_fsdp_contract", False):
        return

    @torch.no_grad()
    def fsdp_tensor_clip(
        parameters,
        max_norm,
        total_norm,
        use_decoupled_grad=False,
    ):
        """Scale gradients with PyTorch's CUDA-FP32 coefficient semantics."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        gradients = [
            to_local_if_dtensor(grad).detach()
            for parameter in parameters
            if (grad := getattr(parameter, grad_attr, None)) is not None
        ]
        if not gradients:
            return
        device = gradients[0].device
        norm = torch.as_tensor(total_norm, dtype=torch.float32, device=device)
        coefficient = (
            torch.as_tensor(max_norm, dtype=torch.float32, device=device)
            / (norm + 1.0e-6)
        ).clamp(max=1.0)
        torch._foreach_mul_(gradients, coefficient)

    fsdp_tensor_clip._fluxvla_fsdp_contract = True
    optimizer_module.clip_grad_by_total_norm_fp32 = fsdp_tensor_clip


def register_alignment_parameters(
    model: torch.nn.Module,
    *,
    grad_norm_order: Callable[[str], tuple] | None = None,
    grad_norm_zero_keys: Iterable[tuple] = (),
    grad_reduce_zero_segments: Iterable[tuple[tuple, int]] = (),
) -> None:
    """Register selected model parameters before MCore wraps the model."""
    trace_parameters = os.environ.get("FLUXVLA_PARAM_TRACE") == "1"
    match_fsdp_norm = (
        os.environ.get("PI05_FSDP_GRAD_NORM_CONTRACT") == "1"
        or os.environ.get("FLUXVLA_FSDP_GRAD_NORM_CONTRACT") == "1"
    )
    match_fsdp_reduce = os.environ.get("PI05_FSDP_GRAD_REDUCE_CONTRACT") == "1"
    trace_post_reduce = bool(os.environ.get("FLUXVLA_POST_REDUCE_GRAD_TRACE_DIR"))
    if (
        not trace_parameters
        and not match_fsdp_norm
        and not match_fsdp_reduce
        and not trace_post_reduce
    ):
        return
    named_parameters = list(model.named_parameters())
    _REGISTERED_MODEL_PARAMETERS.clear()
    _REGISTERED_MODEL_NAMED_PARAMETERS.clear()
    _REGISTERED_GRAD_NORM_ENTRIES.clear()
    _REGISTERED_GRAD_REDUCE_ENTRIES.clear()
    trainable_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    _REGISTERED_MODEL_NAMED_PARAMETERS.extend(trainable_parameters)
    _REGISTERED_MODEL_PARAMETERS.extend(
        parameter for _, parameter in trainable_parameters
    )
    if grad_norm_order is not None:
        ordered_entries = [
            (grad_norm_order(name), name, parameter)
            for name, parameter in trainable_parameters
        ]
        ordered_entries.extend(
            (key, f"__zero_slot_{index}", None)
            for index, key in enumerate(grad_norm_zero_keys)
        )
        ordered_entries.sort(key=lambda item: item[0])
        _REGISTERED_GRAD_NORM_ENTRIES.extend(
            (name, parameter) for _, name, parameter in ordered_entries
        )
        _REGISTERED_GRAD_REDUCE_ENTRIES.extend(
            _ordered_grad_reduce_entries(
                trainable_parameters,
                grad_order=grad_norm_order,
                zero_segments=grad_reduce_zero_segments,
            )
        )
    if not trace_parameters:
        return
    pattern = re.compile(
        os.environ.get(
            "FLUXVLA_PARAM_TRACE_PATTERN",
            r"(?:^|\.)(action_out_proj\.weight|suffix_embedder\.action_in_proj\.weight|"
            r"layers\.0\.attn\.(vlm_q|exp_q)\.weight)$",
        )
    )
    _REGISTERED_PARAMETERS.clear()
    _REGISTERED_PARAMETERS.extend(
        (name, parameter)
        for name, parameter in named_parameters
        if pattern.search(name)
    )


def _optimizer_tensors(
    optimizer: Any, model_parameter: torch.nn.Parameter
) -> dict | None:
    """Return locally owned optimizer tensors for one model parameter."""
    for child in getattr(optimizer, "chained_optimizers", (optimizer,)):
        model_map = getattr(child, "model_param_gbuf_map", None)
        group_map = getattr(child, "model_param_group_index_map", None)
        if model_map is not None and model_parameter in model_map:
            if group_map is not None and model_parameter not in group_map:
                continue
            # Megatron's precision-aware optimizer stores the low-precision
            # main parameter plus BF16 Adam states and, depending on TE/config,
            # may not allocate a ``master_param`` state at all.  Its generic
            # accessor unconditionally pops that key, so use the owning
            # optimizer parameter directly for diagnostics in that case.
            try:
                tensors = child._get_main_param_and_optimizer_states(model_parameter)
            except KeyError as error:
                if error.args != ("master_param",):
                    raise
                group_index, group_order = group_map[model_parameter]
                base_optimizer = getattr(child, "optimizer", None)
                if base_optimizer is None:
                    continue
                sharded_parameter = base_optimizer.param_groups[group_index]["params"][
                    group_order
                ]
                state = base_optimizer.state.get(sharded_parameter, {})
                tensors = {"param": sharded_parameter}
                for state_name in ("exp_avg", "exp_avg_sq"):
                    if state_name not in state:
                        continue
                    getter = getattr(base_optimizer, "get_unscaled_state", None)
                    tensors[state_name] = (
                        getter(sharded_parameter, state_name)
                        if getter is not None
                        else state[state_name]
                    )
            model_grad = getattr(model_parameter, "main_grad", model_parameter.grad)
            if model_grad is not None and group_map is not None:
                param_range = child._get_model_param_range_map(model_parameter)["param"]
                model_grad = model_grad.view(-1)[param_range.start : param_range.end]
            else:
                param_range = None
            return {
                **tensors,
                "model_grad": model_grad,
                "model_range": None
                if param_range is None
                else (param_range.start, param_range.end),
            }

        main_parameter = getattr(model_parameter, "main_param", model_parameter)
        base_optimizer = getattr(child, "optimizer", None)
        if base_optimizer is None:
            continue
        if any(
            candidate is main_parameter
            for group in base_optimizer.param_groups
            for candidate in group["params"]
        ):
            return {
                "param": main_parameter,
                "model_grad": getattr(
                    model_parameter, "main_grad", model_parameter.grad
                ),
                **base_optimizer.state[main_parameter],
            }
    return None


def _parameter_lr(optimizer: Any, master_parameter: torch.Tensor) -> float | None:
    for child in getattr(optimizer, "chained_optimizers", (optimizer,)):
        base_optimizer = getattr(child, "optimizer", None)
        if base_optimizer is None:
            continue
        for group in base_optimizer.param_groups:
            if any(parameter is master_parameter for parameter in group["params"]):
                return float(group["lr"])
    return None


def _optimizer_state_for_master(
    optimizer: Any, master_parameter: torch.Tensor
) -> dict[str, Any] | None:
    """Find DistOpt state after the first step by master-parameter identity."""
    for child in getattr(optimizer, "chained_optimizers", (optimizer,)):
        for owner in (child, getattr(child, "optimizer", None)):
            if owner is None:
                continue
            state = getattr(owner, "state", None)
            if state is not None and master_parameter in state:
                return state[master_parameter]
    return None


def _norm(value: Any) -> float | None:
    if not isinstance(value, torch.Tensor):
        return None
    return float(value.detach().float().norm().item())


def _cpu_tensor(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().float().cpu().clone()


def _write_optimizer_tensor_trace(
    *,
    phase: str,
    step: int,
    rank: int,
    snapshots: list[dict[str, Any]],
    optimizer: Any,
) -> None:
    root_value = os.environ.get("FLUXVLA_OPTIMIZER_TENSOR_TRACE_DIR")
    if not root_value:
        return

    parameters = []
    for snapshot in snapshots:
        tensors = snapshot.get("optimizer_tensors")
        if tensors is None:
            tensors = _optimizer_tensors(optimizer, snapshot["model_parameter"])
        master_parameter = (
            snapshot["model_parameter"] if tensors is None else tensors["param"]
        )
        optimizer_state = _optimizer_state_for_master(
            optimizer, master_parameter
        )
        if optimizer_state is not None:
            tensors = {
                **({} if tensors is None else tensors),
                **optimizer_state,
            }
        parameters.append(
            {
                "name": snapshot["name"],
                "model_shape": list(snapshot["model_parameter"].shape),
                "master_shape": list(master_parameter.shape),
                "model_range": None if tensors is None else tensors.get("model_range"),
                "parameter_before": snapshot["before_cpu"],
                "raw_grad": snapshot["raw_grad_cpu"],
                "post_clip_grad": _cpu_tensor(master_parameter.grad),
                "parameter_after": _cpu_tensor(master_parameter),
                "exp_avg": _cpu_tensor(tensors.get("exp_avg")),
                "exp_avg_sq": _cpu_tensor(tensors.get("exp_avg_sq")),
            }
        )

    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": "mcore",
            "phase": phase,
            "step": step,
            "rank": rank,
            "parameters": parameters,
        },
        root / f"mcore_step_{step:06d}_rank_{rank:02d}.pt",
    )


def install_optimizer_alignment_trace() -> None:
    """Wrap MCore's optimizer step without modifying Megatron-LM source."""
    if os.environ.get("FLUXVLA_PARAM_TRACE") != "1":
        return

    from megatron.core.optimizer.optimizer import ChainedOptimizer

    original_step = ChainedOptimizer.step
    if getattr(original_step, "_fluxvla_alignment_trace", False):
        return

    state = {"step": None}

    @torch.no_grad()
    def traced_step(optimizer):
        if state["step"] is None:
            from megatron.training import get_args

            state["step"] = int(getattr(get_args(), "iteration", 0))
        state["step"] += 1
        target_step = int(os.environ.get("FLUXVLA_PARAM_TRACE_STEP", "0"))
        if target_step and state["step"] != target_step:
            return original_step(optimizer)
        summary_only = os.environ.get("FLUXVLA_PARAM_SUMMARY_ONLY") == "1"
        tensor_trace = bool(os.environ.get("FLUXVLA_OPTIMIZER_TENSOR_TRACE_DIR"))
        snapshots = []
        for name, model_parameter in _REGISTERED_PARAMETERS:
            tensors = _optimizer_tensors(optimizer, model_parameter)
            if tensors is None and model_parameter.grad is None:
                continue
            master_parameter = model_parameter if tensors is None else tensors["param"]
            model_grad = model_parameter.grad if tensors is None else tensors["model_grad"]
            snapshots.append(
                {
                    "name": name,
                    "model_parameter": model_parameter,
                    "before": (
                        None
                        if summary_only
                        else master_parameter.detach().float().clone()
                    ),
                    "before_cpu": (
                        _cpu_tensor(master_parameter) if tensor_trace else None
                    ),
                    "raw_grad_cpu": _cpu_tensor(model_grad) if tensor_trace else None,
                    "raw_grad_norm": _norm(model_grad),
                    "raw_grad_sum": (
                        None
                        if model_grad is None
                        else float(model_grad.detach().float().sum().item())
                    ),
                    "raw_grad_abs_max": (
                        None
                        if model_grad is None
                        else float(model_grad.detach().float().abs().max().item())
                    ),
                    "lr": _parameter_lr(optimizer, master_parameter),
                    # DistOpt may replace/gather its model-parameter mapping
                    # during ``step``. Keep the pre-step lookup alive so the
                    # tensor trace remains useful after the optimizer update.
                    "optimizer_tensors": tensors,
                    "master_parameter": master_parameter,
                }
            )

        rank = int(os.environ.get("RANK", "0"))
        if tensor_trace:
            # Persist the optimizer boundary before AdamW lazily allocates its
            # state. Besides making OOMs diagnosable, this provides a bounded
            # correctness gate for environments that cannot fit first-step
            # optimizer state. A successful update overwrites this file with
            # the complete post-update trace below.
            _write_optimizer_tensor_trace(
                phase="pre_update",
                step=state["step"],
                rank=rank,
                snapshots=snapshots,
                optimizer=optimizer,
            )
        if os.environ.get("FLUXVLA_PARITY_STOP_BEFORE_OPTIMIZER") == "1":
            raise SystemExit(0)

        result = original_step(optimizer)
        success, grad_norm, _ = result
        if os.environ.get("FLUXVLA_MODEL_REPLICA_TRACE") == "1":
            replica_root = Path(
                os.environ.get(
                    "FLUXVLA_MODEL_REPLICA_TRACE_DIR", "work_dirs/model_replica_trace"
                )
            )
            replica_root.mkdir(parents=True, exist_ok=True)
            replica_path = replica_root / f"rank{rank}.jsonl"
            replica_pattern = os.environ.get("FLUXVLA_PARAM_TRACE_PATTERN")
            replica_regex = re.compile(replica_pattern) if replica_pattern else None
            with replica_path.open("a", encoding="utf-8") as stream:
                for name, parameter in _REGISTERED_PARAMETERS:
                    if replica_regex is not None and not replica_regex.search(name):
                        continue
                    value = parameter.detach().float()
                    stream.write(
                        json.dumps(
                            {
                                "step": state["step"],
                                "name": name,
                                "shape": list(parameter.shape),
                                "sum": float(value.sum().item()),
                                "norm": float(value.norm().item()),
                                "abs_max": float(value.abs().max().item()),
                            }
                        )
                        + "\n"
                    )
        grad_norm_value = None if grad_norm is None else float(grad_norm)
        clip_grad = float(optimizer.config.clip_grad)
        clip_coefficient = None
        if grad_norm_value is not None:
            clip_coefficient = min(1.0, clip_grad / (grad_norm_value + 1e-6))
            try:
                from .lingbot_vla_v2 import optimizer as lingbot_optimizer

                actual_coefficient = (
                    lingbot_optimizer._LAST_FSDP2_CLIP_COEFFICIENT
                )
                if actual_coefficient is not None:
                    clip_coefficient = float(actual_coefficient.item())
            except ImportError:
                pass
        _write_optimizer_tensor_trace(
            phase="post_update",
            step=state["step"],
            rank=rank,
            snapshots=snapshots,
            optimizer=optimizer,
        )
        root = Path(os.environ.get("FLUXVLA_PARAM_TRACE_DIR", "work_dirs/param_trace"))
        path = root / f"native_rank{rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as stream:
            for snapshot in snapshots:
                tensors = _optimizer_tensors(optimizer, snapshot["model_parameter"])
                master_parameter = (
                    snapshot["model_parameter"] if tensors is None else tensors["param"]
                )
                optimizer_state = (
                    None
                    if tensors is None
                    else _optimizer_state_for_master(optimizer, master_parameter)
                )
                if optimizer_state is not None:
                    tensors = {**tensors, **optimizer_state}
                elif tensors is None:
                    tensors = {}
                # Native FSDP2 retains FP32 parameter storage but casts managed
                # weights to BF16 for compute. DistOpt similarly copies each
                # FP32 master shard into a BF16 model parameter. Project the
                # master value to expose updates that cross a BF16 boundary,
                # rather than harmless FP32 ULP differences.
                parameter_bf16 = master_parameter.detach().to(
                    dtype=torch.bfloat16
                ).float()
                parameter_bf16_bits = (
                    parameter_bf16.to(torch.bfloat16)
                    .contiguous()
                    .view(torch.int16)
                    .to(torch.int64)
                    .bitwise_and(0xFFFF)
                )
                row = {
                    "step": state["step"],
                    "name": snapshot["name"],
                    "shape": list(snapshot["model_parameter"].shape),
                    "master_shape": list(master_parameter.shape),
                    "success": bool(success),
                    "lr": snapshot["lr"],
                    "grad_norm": grad_norm_value,
                    "raw_grad_norm": snapshot["raw_grad_norm"],
                    "raw_grad_sum": snapshot["raw_grad_sum"],
                    "raw_grad_abs_max": snapshot["raw_grad_abs_max"],
                    "post_clip_grad_norm": _norm(master_parameter.grad),
                    "clip_coefficient": clip_coefficient,
                    "parameter_norm": _norm(master_parameter),
                    "parameter_sum": float(master_parameter.detach().float().sum().item()),
                    "parameter_abs_max": float(
                        master_parameter.detach().float().abs().max().item()
                    ),
                    "parameter_bf16_norm": float(parameter_bf16.norm().item()),
                    "parameter_bf16_sum": float(parameter_bf16.sum().item()),
                    "parameter_bf16_abs_max": float(
                        parameter_bf16.abs().max().item()
                    ),
                    "parameter_bf16_bits_sum": int(
                        parameter_bf16_bits.sum().item()
                    ),
                    "parameter_delta_norm": (
                        None
                        if snapshot["before"] is None
                        else float(
                            (master_parameter.detach().float() - snapshot["before"])
                            .norm()
                            .item()
                        )
                    ),
                    "exp_avg_norm": _norm(tensors.get("exp_avg")),
                    "exp_avg_sq_norm": _norm(tensors.get("exp_avg_sq")),
                }
                for state_name in ("exp_avg", "exp_avg_sq"):
                    value = tensors.get(state_name)
                    row[f"{state_name}_sum"] = (
                        None
                        if not isinstance(value, torch.Tensor)
                        else float(value.detach().float().sum().item())
                    )
                    row[f"{state_name}_abs_max"] = (
                        None
                        if not isinstance(value, torch.Tensor)
                        else float(value.detach().float().abs().max().item())
                    )
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        return result

    traced_step._fluxvla_alignment_trace = True
    ChainedOptimizer.step = traced_step


__all__ = [
    "install_fsdp_grad_norm_contract",
    "install_fsdp_grad_reduce_contract",
    "install_optimizer_alignment_trace",
    "register_alignment_parameters",
]
