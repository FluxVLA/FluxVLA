"""Compare native DDP and FluxVLA FSDP reduction on a frozen LingBot replay.

This is an accuracy gate, not a throughput benchmark.  The 6.376B backbone is
frozen and only ``action_out_proj`` is optimized so the comparison can run on
partially occupied A100s.  Both backends emit the same tensor-level trace for
offline comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2


ROOT = Path("/ceph/teamGolden/andyzeng/VLA")
# Keep the diagnostic default identical to the FluxVLA MCore production
# launcher.  An alternate checkpoint remains selectable only via an explicit
# environment override, but must never be selected implicitly.
CHECKPOINT = Path(
    os.environ.get(
        "LINGBOT_ALIGNMENT_CHECKPOINT",
        "/ceph/teamGolden/andyzeng/VLA/models/lingbot-vla-v2-6b",
    )
)
TOKENIZER = Path(
    os.environ.get(
        "LINGBOT_ALIGNMENT_TOKENIZER",
        "/ceph/teamGolden/andyzeng/VLA/models/Qwen3-VL-4B-Instruct",
    )
)
SCALAR_LOSS_KEYS = (
    "loss",
    "loss_vla",
    "loss_depth",
    "loss_future_depth",
    "loss_future_video",
    "loss_moe_aux",
)


def _fp32_chunked_all_reduce(
    value: torch.Tensor,
    chunk_mib: int,
    *,
    process_group: Any = None,
) -> torch.Tensor:
    """Average a bucket in FP32 without materializing a full FP32 bucket."""
    if value.dtype == torch.float32:
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=process_group)
        value.div_(dist.get_world_size(process_group))
        return value
    chunk_numel = max(1, chunk_mib * 1024 * 1024 // value.element_size())
    reduced = torch.empty(min(chunk_numel, value.numel()), device=value.device, dtype=torch.float32)
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), chunk_numel):
        end = min(start + chunk_numel, flat.numel())
        chunk = reduced[: end - start]
        chunk.copy_(flat[start:end])
        dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=process_group)
        chunk.div_(dist.get_world_size(process_group))
        flat[start:end].copy_(chunk)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ddp", "fsdp"), required=True)
    parser.add_argument(
        "--fsdp-sharding",
        choices=("no-shard", "full-shard"),
        default="no-shard",
    )
    parser.add_argument(
        "--fsdp-scope",
        choices=("action-head", "model"),
        default="action-head",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--train-scope",
        choices=("action-head", "action-path", "action-expert", "alignment", "qwen", "all"),
        default="action-head",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--lr-decay-iters", type=int, default=None)
    parser.add_argument(
        "--fp32-master-optimizer",
        action="store_true",
        help=(
            "Update FP32 master parameters and copy them back to the BF16 model, "
            "matching the MCore DistOpt mixed-precision contract. DDP diagnostics only."
        ),
    )
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument(
        "--replay-order",
        choices=("manifest", "official_step_rank"),
        default="manifest",
        help="Ordering contract for teacher-cache entries.",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--same-rank-seed",
        action="store_true",
        help="Use one model RNG seed on every DP rank for backend parity replay.",
    )
    parser.add_argument(
        "--distributed-replay",
        action="store_true",
        help="Use DistributedSampler order so each rank consumes its own replay samples.",
    )
    parser.add_argument(
        "--logical-fsdp-grad-norm",
        action="store_true",
        help="Compute FSDP clipping from logical shard gradients for DDP alignment.",
    )
    parser.add_argument(
        "--flat-fsdp-grad-norm",
        action="store_true",
        help="Reconstruct each FlatParameter gradient with shard padding metadata (diagnostic).",
    )
    parser.add_argument(
        "--deterministic-grad-norm",
        action="store_true",
        help="Accumulate gradient squares in FP64 for identical DDP/FSDP clipping diagnostics.",
    )
    parser.add_argument(
        "--fp32-grad-reduce",
        action="store_true",
        help="Reduce data-parallel gradients in FP32 in both DDP and FSDP.",
    )
    parser.add_argument(
        "--fp32-reduce-chunk-mb",
        type=int,
        default=256,
        help="Temporary FP32 DDP reduction chunk size in MiB.",
    )
    parser.add_argument(
        "--parameter-summary",
        action="store_true",
        help="Collect component parameter summaries (slow and memory intensive).",
    )
    parser.add_argument(
        "--fsdp-gradient-breakdown",
        action="store_true",
        help="Collect per-FSDP-handle gradient breakdown (diagnostic only).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-save-final-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-adamw-foreach",
        action="store_true",
        help="Disable AdamW foreach temporaries to reduce optimizer peak memory.",
    )
    parser.add_argument(
        "--probe-parameter",
        action="append",
        default=[],
        help="Exact name or unique suffix of a trainable parameter to trace (repeatable).",
    )
    return parser.parse_args()


def _resolve_parameter_probes(
    module: torch.nn.Module,
    queries: list[str],
) -> dict[str, tuple[torch.nn.Parameter, torch.Size]]:
    named_parameters = dict(module.named_parameters())
    probes: dict[str, tuple[torch.nn.Parameter, torch.Size]] = {}
    for query in queries:
        matches = [
            (name, parameter)
            for name, parameter in named_parameters.items()
            if name == query or name.endswith(query)
        ]
        if len(matches) != 1:
            names = [name for name, _ in matches]
            raise ValueError(f"probe {query!r} matched {len(matches)} parameters: {names[:5]}")
        name, parameter = matches[0]
        if not parameter.requires_grad:
            raise ValueError(f"probe parameter {name!r} is frozen")
        probes[name] = (parameter, parameter.shape)
    return probes


def _build_batch(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = torch.load(path, map_location="cpu", weights_only=False)
    batch.pop("source_metadata", None)
    # Older replays store views as [num_views, patches, hidden] and need the
    # 224-grid conversion.  Current real-batch replays already carry the
    # native [batch, views, patches, hidden] contract and its grid/masks;
    # preserve those fields verbatim for parity-sensitive runs.
    if batch["images"].ndim == 3:
        batch["images"] = torch.nn.functional.interpolate(
            batch["images"].float().unsqueeze(0),
            size=(192, 1536),
            mode="bilinear",
            align_corners=False,
        )
        batch["image_grid_thw"] = torch.tensor(
            [[[1, 12, 16]] * 3], dtype=torch.long
        )
        batch["img_masks"] = torch.ones((1, 3), dtype=torch.bool)
    elif batch["images"].ndim != 4:
        raise ValueError(
            "replay must contain images [num_views, patches, hidden] or "
            "[batch, num_views, patches, hidden]"
        )
    if batch["joint_mask"].ndim == 2:
        batch["joint_mask"] = batch["joint_mask"].unsqueeze(0)
    if batch["action_is_pad"].ndim == 1:
        batch["action_is_pad"] = batch["action_is_pad"].unsqueeze(0)
    if batch["state_joint_mask"].ndim == 1:
        batch["state_joint_mask"] = batch["state_joint_mask"].unsqueeze(0)
    batch["noise"] = torch.zeros_like(batch["actions"])
    batch["time"] = torch.full((1,), 0.5)
    for key in ("depth_targets", "future_depth_targets"):
        batch[key] = torch.nn.functional.pad(batch[key], (0, 0, 0, 4))
    return {
        key: value.to(
            device=device,
            dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
        )
        for key, value in batch.items()
    }


def _build_manifest_batches(
    path: Path,
    device: torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
    distributed_replay: bool = False,
    replay_order: str = "manifest",
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    from fluxvla.datasets.lingbot_teacher_cache import LingBotTeacherCacheDataset

    dataset = LingBotTeacherCacheDataset(
        str(path),
        cache_in_memory=True,
        validate_targets=True,
        replay_order=replay_order,
    )
    batches = []
    identities = []
    if distributed_replay:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0, drop_last=False
        )
        indices = list(iter(sampler))
    else:
        indices = list(range(len(dataset)))
    for index in indices:
        sample = dataset[index]
        batches.append(
            {
                key: value.unsqueeze(0).to(
                    device=device,
                    dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
                )
                for key, value in sample.items()
            }
        )
        identities.append(dataset.identity(index))
    return batches, identities


def _build_model(
    device: torch.device,
    train_scope: str,
    *,
    preserve_fp32_master: bool = False,
) -> tuple[LingBotVLAV2, torch.nn.Module]:
    production_parity = os.environ.get("LINGBOT_ALIGNMENT_PRODUCTION") == "1"
    if production_parity:
        # The production replay owns the auxiliary teacher heads and uses the
        # same BF16 action-flow, eager action/MoE, and post-training contract as
        # the native MCore provider.  Keep this opt-in so the historical
        # VLA-only diagnostic remains reproducible.
        model_config = {
            "attention_implementation": "eager",
            "vit_attn_implementation": os.environ.get(
                "LINGBOT_ALIGNMENT_VIT_ATTN", "eager"
            ),
            "moe_implementation": os.environ.get(
                "LINGBOT_ALIGNMENT_MOE", "eager"
            ),
            "action_fp32": os.environ.get("LINGBOT_ALIGNMENT_ACTION_FP32", "0") == "1",
            "post_training": True,
        }
    else:
        model_config = {
            "align_params": {},
            "attention_implementation": "eager",
            "moe_implementation": "fused",
            "action_fp32": True,
            "post_training": False,
        }
    model = LingBotVLAV2(
        tokenizer_path=str(TOKENIZER),
        pretrained_name_or_path=str(CHECKPOINT),
        native_config_path=str(ROOT / "FluxVLA/work_dirs/robotwin_reduced_v2.yaml"),
        low_cpu_mem_usage=(
            os.environ.get("LINGBOT_ALIGNMENT_LOW_CPU", "1") == "1"
        ),
        # This alignment harness certifies the VLA action path.  Disable the
        # optional depth/video teacher heads so replay batches with either the
        # 256-token or 260-token cached target contract remain comparable.
        # Match the official deterministic smoke contract exactly: the
        # reference launcher uses eager attention/MoE, FP32 flow-matching
        # action arithmetic, and no post-training auxiliary heads.
        model_config=model_config,
    )
    model.from_pretrained()
    model.model.config.action_fp32 = model_config["action_fp32"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    action_head = model.model.model.action_out_proj
    if model_config["action_fp32"]:
        action_head.float()
    native_core = model.model.model
    if train_scope in ("action-head", "action-path", "action-expert", "alignment", "qwen", "all"):
        action_head.requires_grad_(True)
    if train_scope == "action-path":
        for name, parameter in model.named_parameters():
            if name.endswith(("state_proj.weight", "state_proj.bias", "action_in_proj.weight", "action_in_proj.bias")):
                parameter.requires_grad_(True)
    if train_scope in ("action-expert", "alignment", "qwen", "all"):
        for name, parameter in model.named_parameters():
            if "qwenvl_with_expert.qwen_expert" in name:
                parameter.requires_grad_(True)
    if train_scope in ("alignment", "qwen", "all"):
        alignment_tokens = ("align_head", "shared_task_proj", "video_align", "depth_align")
        for name, parameter in model.named_parameters():
            if any(token in name for token in alignment_tokens):
                parameter.requires_grad_(True)
    if train_scope in ("qwen", "all"):
        for name, parameter in model.named_parameters():
            if "qwenvl_with_expert.qwenvl" in name:
                parameter.requires_grad_(True)
    if train_scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    if preserve_fp32_master:
        if train_scope != "action-head":
            raise ValueError(
                "FP32-master diagnostics currently preserve only the action-head initial state"
            )
        for parameter in action_head.parameters():
            parameter._alignment_fp32_init = parameter.detach().float().clone()
    model.to(device=device, dtype=torch.bfloat16)
    if model_config["action_fp32"]:
        action_head.float()
    model.train()
    return model, action_head


def _mean_scalar(value: torch.Tensor) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced)


def _clone_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().float().cpu().clone()


def _gather_sharded_tensor(
    tensor: torch.Tensor | None,
    *,
    expected_shape: torch.Size,
    move_to_cpu: bool = True,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    flat = tensor.detach().float().reshape(-1)
    local_size = torch.tensor([flat.numel()], device=flat.device, dtype=torch.int64)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    max_size = max(int(size.item()) for size in sizes)
    padded = torch.zeros(max_size, device=flat.device, dtype=flat.dtype)
    padded[: flat.numel()].copy_(flat)
    shards = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(shards, padded)
    full = torch.cat(
        [shard[: int(size.item())] for shard, size in zip(shards, sizes, strict=True)]
    )
    expected_numel = expected_shape.numel()
    if full.numel() != expected_numel:
        raise RuntimeError(
            f"Reconstructed shard has {full.numel()} elements, expected {expected_numel}"
        )
    full = full.reshape(expected_shape)
    return full.cpu() if move_to_cpu else full


def _state_tensor(optimizer: torch.optim.Optimizer, parameter: torch.nn.Parameter, key: str) -> torch.Tensor:
    value = optimizer.state[parameter].get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"AdamW state {key!r} was not initialized")
    return value


def _cosine_lr(*, step: int, maximum: float, minimum: float, decay_iters: int) -> float:
    """Return Megatron's post-step cosine LR for the next optimizer update."""
    progress = min(max(step, 0), decay_iters) / decay_iters
    return minimum + 0.5 * (maximum - minimum) * (math.cos(math.pi * progress) + 1.0)


def _gradient_breakdown(module: torch.nn.Module) -> dict[str, float]:
    """Return FP32 squared-gradient totals by model component."""
    totals: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None or not parameter.requires_grad:
            continue
        if "action_out_proj" in name:
            bucket = "action_head"
        elif "qwenvl_with_expert.qwenvl" in name:
            bucket = "qwen"
        elif "qwenvl_with_expert.qwen_expert" in name:
            bucket = "action_expert"
        elif any(token in name for token in ("align_head", "video_align", "depth_align", "shared_task_proj")):
            bucket = "alignment"
        else:
            bucket = "other"
        value = float(parameter.grad.detach().float().square().sum())
        totals[bucket] = totals.get(bucket, 0.0) + value
    totals["__total__"] = sum(totals.values())
    return totals


def _fsdp_gradient_breakdown(root: FSDP) -> dict[str, float]:
    """Summarize FSDP handle gradients by their logical parameter names.

    ``use_orig_params`` may expose gradients only through FlatParameter on
    nested handles.  Classifying each handle from ``_fqns`` keeps this
    diagnostic robust without treating missing logical views as zero.
    """
    handle_totals: dict[str, torch.Tensor] = {}
    sharded_params: set[torch.nn.Parameter] = set()
    for module_name, module in root.named_modules():
        if not isinstance(module, FSDP):
            continue
        handle = object.__getattribute__(module, "_handle")
        if handle is None or handle.flat_param.grad is None:
            continue
        names = " ".join(getattr(handle.flat_param, "_fqns", ()))
        classification_name = f"{module_name} {names}"
        if "qwenvl_with_expert.qwenvl" in classification_name:
            bucket = "qwen"
        elif "qwenvl_with_expert.qwen_expert" in classification_name:
            bucket = "action_expert"
        elif any(token in classification_name for token in ("align_head", "video_align", "depth_align", "shared_task_proj")):
            bucket = "alignment"
        elif "action_out_proj" in classification_name:
            bucket = "action_head"
        else:
            bucket = "other"
        value = handle.flat_param.grad.detach().float().square().sum()
        if not handle.uses_sharded_strategy:
            value /= dist.get_world_size()
        handle_totals[bucket] = handle_totals.get(bucket, torch.zeros_like(value)) + value

    # Nested FSDP handles can expose original parameter views only through the
    # wrapped module.  Use those views when they carry a gradient, which lets
    # diagnostic accounting include alignment/action-expert parameters that do
    # not own a directly visible FlatParameter.grad on the root traversal.
    wrapped = getattr(root, "module", None)
    wrapped_totals: dict[str, torch.Tensor] = {}
    if wrapped is not None:
        for module in root.modules():
            if not isinstance(module, FSDP):
                continue
            handle = object.__getattribute__(module, "_handle")
            if handle is not None and handle.uses_sharded_strategy:
                sharded_params.update(handle.flat_param._params)
        for parameter_name, parameter in wrapped.named_parameters():
            if parameter.grad is None or not parameter.requires_grad:
                continue
            if "qwenvl_with_expert.qwenvl" in parameter_name:
                bucket = "qwen"
            elif "qwenvl_with_expert.qwen_expert" in parameter_name:
                bucket = "action_expert"
            elif any(token in parameter_name for token in ("align_head", "video_align", "depth_align", "shared_task_proj")):
                bucket = "alignment"
            elif "action_out_proj" in parameter_name:
                bucket = "action_head"
            else:
                bucket = "other"
            value = parameter.grad.detach().float().square().sum()
            if parameter not in sharded_params:
                value /= dist.get_world_size()
            wrapped_totals[bucket] = wrapped_totals.get(bucket, torch.zeros_like(value)) + value
    totals = wrapped_totals if wrapped_totals else handle_totals
    for value in totals.values():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in totals.items()}


def _parameter_summary(
    module: torch.nn.Module,
    *,
    backend: str,
    fsdp_sharding: str,
) -> dict[str, dict[str, float]]:
    """Summarize parameter values without materializing a full 6B state dict.

    ``use_orig_params`` exposes the same logical names under FSDP.  Full-shard
    parameters are disjoint across ranks, while DDP/NO_SHARD parameters are
    replicated; normalize the latter by world size before reporting totals.
    """
    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if "action_out_proj" in name:
            bucket = "action_head"
        elif "qwenvl_with_expert.qwenvl" in name:
            bucket = "qwen"
        elif "qwenvl_with_expert.qwen_expert" in name:
            bucket = "action_expert"
        elif any(token in name for token in ("align_head", "video_align", "depth_align", "shared_task_proj")):
            bucket = "alignment"
        else:
            bucket = "other"
        grouped.setdefault(bucket, []).append(parameter)

    replicated = backend == "ddp" or fsdp_sharding == "no-shard"
    result: dict[str, dict[str, float]] = {}
    for bucket, tensors in grouped.items():
        # Accumulate per parameter to avoid a transient FP32 copy of all 6B
        # parameters (which would invalidate the memory diagnostic itself).
        stats = torch.zeros(4, device=tensors[0].device, dtype=torch.float64)
        for parameter in tensors:
            tensor = parameter.detach().float().reshape(-1)
            stats[0] += tensor.sum(dtype=torch.float64)
            stats[1] += tensor.square().sum(dtype=torch.float64)
            stats[2] = torch.maximum(stats[2], tensor.abs().max().double())
            stats[3] += tensor.numel()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if replicated:
            stats[:3] /= dist.get_world_size()
            stats[3] /= dist.get_world_size()
        result[bucket] = {
            "sum": float(stats[0]),
            "sq_sum": float(stats[1]),
            "abs_max_sum": float(stats[2]),
            "numel": float(stats[3]),
        }
    return result


def _logical_fsdp_clip_grad_norm(
    fsdp_root: FSDP,
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Clip local FSDP shards using one logical global L2 norm.

    FSDP's nested ``clip_grad_norm_`` may reduce per-handle norms in a
    different order than DDP. Summing local shard squares once over the data
    parallel group gives the same logical parameter norm while retaining
    sharded gradients and optimizer state.
    """
    # FSDP can leave parameters outside a sharded handle (for example an
    # ignored FP32 action head subsequently wrapped as a child FSDP unit).
    # Such gradients are replicated within the current process group and must
    # not be all-reduced a second time.  Mirror FSDP's sharded/non-sharded
    # partition rather than reducing every ``model.parameters()`` entry.
    sharded_params: set[torch.nn.Parameter] = set()
    for module in fsdp_root.modules():
        if not isinstance(module, FSDP):
            continue
        handle = object.__getattribute__(module, "_handle")
        if handle is None:
            continue
        if not handle._use_orig_params:
            sharded_params.add(handle.flat_param)
        elif handle.uses_sharded_strategy:
            sharded_params.update(handle.flat_param._params)

    sharded_sq = torch.zeros((), device=next(p for p in parameters if p.grad is not None).device,
                             dtype=torch.float64)
    nonsharded_sq = torch.zeros_like(sharded_sq)
    grouped_grads: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    grouped_params: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter in parameters:
        if parameter.grad is None:
            continue
        key = (parameter.grad.device, parameter.grad.dtype)
        grouped_grads.setdefault(key, []).append(parameter.grad.detach())
        grouped_params.setdefault(key, []).append(parameter)
    # Match torch.nn.utils.get_total_norm's foreach path.  In particular, do
    # not cast every BF16 gradient to FP32 before the per-tensor norm.
    for key, grads in grouped_grads.items():
        norms = torch._foreach_norm(grads, 2.0)
        for parameter, parameter_norm in zip(grouped_params[key], norms, strict=True):
            if parameter in sharded_params:
                sharded_sq += parameter_norm.double().square()
            else:
                nonsharded_sq += parameter_norm.double().square()
    dist.all_reduce(sharded_sq, op=dist.ReduceOp.SUM)
    # Non-sharded parameters are replicated, so only rank-local contribution
    # is added after the collective, matching FSDP.clip_grad_norm_.
    grad_norm = (sharded_sq + nonsharded_sq).sqrt().to(dtype=torch.float32)
    clip = torch.clamp(float(max_norm) / (grad_norm + 1e-6), max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(clip.to(device=parameter.grad.device, dtype=parameter.grad.dtype))
    return grad_norm


def _deterministic_clip_grad_norm(
    fsdp_root: FSDP | None,
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Clip with the same FP64 elementwise accumulation on DDP and FSDP.

    FSDP contributes only local shards to the collective; replicated
    parameters are counted once per process.  Avoiding foreach norm kernels
    removes dtype-dependent reduction order from the parity gate.
    """
    sharded_params: set[torch.nn.Parameter] = set()
    if fsdp_root is not None:
        for module in fsdp_root.modules():
            if not isinstance(module, FSDP):
                continue
            handle = object.__getattribute__(module, "_handle")
            if handle is None or not handle.uses_sharded_strategy:
                continue
            if handle._use_orig_params:
                sharded_params.update(handle.flat_param._params)
    local_sq = torch.zeros((), device=next(p for p in parameters if p.grad is not None).device, dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        contribution = value.square().sum(dtype=torch.float64)
        if parameter in sharded_params:
            local_sq += contribution
        elif fsdp_root is not None:
            # Replicated FSDP parameters are already present on every rank.
            local_sq += contribution / dist.get_world_size()
        else:
            local_sq += contribution
    if fsdp_root is not None:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    grad_norm = local_sq.sqrt().float()
    clip = torch.clamp(float(max_norm) / (grad_norm + 1e-6), max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(clip.to(parameter.grad.device, parameter.grad.dtype))
    return grad_norm


def _flat_fsdp_clip_grad_norm(
    fsdp_root: FSDP,
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Compute norm from complete FlatParameter gradients.

    ``use_orig_params`` exposes local views, but their boundaries are not the
    global parameter boundaries.  Gathering ``flat_param.grad`` and removing
    only the documented tail padding reconstructs the exact logical vector.
    This is a diagnostic implementation; it favors correctness over bandwidth.
    """
    sharded_params: set[torch.nn.Parameter] = set()
    norms: list[torch.Tensor] = []
    for module in fsdp_root.modules():
        if not isinstance(module, FSDP):
            continue
        handle = object.__getattribute__(module, "_handle")
        if handle is None or not handle.uses_sharded_strategy:
            continue
        flat_param = handle.flat_param
        if handle._use_orig_params:
            sharded_params.update(flat_param._params)
        grad = flat_param.grad
        if grad is None:
            continue
        flat = grad.detach().reshape(-1)
        sizes = [torch.zeros(1, device=flat.device, dtype=torch.int64)
                 for _ in range(dist.get_world_size())]
        size = torch.tensor([flat.numel()], device=flat.device, dtype=torch.int64)
        dist.all_gather(sizes, size)
        max_size = max(int(item.item()) for item in sizes)
        padded = torch.zeros(max_size, device=flat.device, dtype=flat.dtype)
        padded[:flat.numel()].copy_(flat)
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
        full = torch.cat([part[:int(item.item())]
                          for part, item in zip(gathered, sizes, strict=True)])
        unpadded = int(flat_param._unpadded_unsharded_size.numel())
        logical_flat = full[:unpadded]
        offset = 0
        for numel in flat_param._numels:
            numel = int(numel)
            norms.append(logical_flat[offset:offset + numel].view(-1))
            offset += numel

    for parameter in parameters:
        if parameter.grad is not None and parameter not in sharded_params:
            norms.append(parameter.grad.detach())
    if norms:
        grouped = {}
        for tensor in norms:
            grouped.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        scalar_norms = []
        for (device, _dtype), tensors in grouped.items():
            scalar_norms.extend(torch._foreach_norm(tensors, 2.0))
        grad_norm = torch.linalg.vector_norm(
            torch.stack([value.to(norms[0].device) for value in scalar_norms]), 2.0)
    else:
        grad_norm = torch.zeros((), device=next(iter(parameters)).device)
    clip = torch.clamp(float(max_norm) / (grad_norm + 1e-6), max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(clip.to(parameter.grad.device, parameter.grad.dtype))
    return grad_norm


def _write_summary(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def _checkpoint_path(directory: Path, rank: int) -> Path:
    return directory / f"rank{rank:04d}.pt"


def _save_checkpoint(
    directory: Path,
    rank: int,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    parameters = [
        parameter.detach().cpu().clone()
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    torch.save(
        {
            "step": step,
            "parameters": parameters,
            "optimizer": optimizer.state_dict(),
            "world_size": dist.get_world_size(),
        },
        _checkpoint_path(directory, rank),
    )


def _load_checkpoint(
    directory: Path,
    rank: int,
    optimizer: torch.optim.Optimizer,
) -> int:
    path = _checkpoint_path(directory, rank)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("world_size") != dist.get_world_size():
        raise RuntimeError("local-shard checkpoint resume requires the same world size")
    current_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    saved_parameters = checkpoint["parameters"]
    if len(current_parameters) != len(saved_parameters):
        raise RuntimeError("checkpoint optimizer parameter count does not match")
    with torch.no_grad():
        for index, (parameter, saved) in enumerate(
            zip(current_parameters, saved_parameters, strict=True)
        ):
            if parameter.shape != saved.shape:
                raise RuntimeError(
                    f"checkpoint parameter {index} shape {tuple(saved.shape)} does not match "
                    f"current shape {tuple(parameter.shape)}"
                )
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def main() -> None:
    args = _parse_args()
    if args.fp32_master_optimizer and args.backend != "ddp":
        raise ValueError("--fp32-master-optimizer currently supports DDP diagnostics only")
    if args.fp32_master_optimizer and (args.resume or args.checkpoint_dir is not None):
        raise ValueError(
            "FP32-master checkpoint save/resume needs an explicit model/master mapping; "
            "it is not supported by this diagnostic mode"
        )
    if (args.min_lr is None) != (args.lr_decay_iters is None):
        raise ValueError("--min-lr and --lr-decay-iters must be specified together")
    if args.lr_decay_iters is not None and args.lr_decay_iters <= 0:
        raise ValueError("--lr-decay-iters must be positive")
    if args.batch is not None and args.batch_manifest is not None:
        raise ValueError("--batch and --batch-manifest are mutually exclusive")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    seed = args.seed if args.same_rank_seed else args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model, action_head = _build_model(
        device,
        args.train_scope,
        preserve_fp32_master=args.fp32_master_optimizer,
    )
    parameter_probes = _resolve_parameter_probes(model, args.probe_parameter)
    trace_parameter_name = next(
        name for name, parameter in model.named_parameters() if parameter is action_head.weight
    )
    action_head_parameter_ids = {id(parameter) for parameter in action_head.parameters()}
    debug_fsdp = os.environ.get("FLUXVLA_DEBUG_FSDP") == "1"
    if debug_fsdp and rank == 0:
        print("[debug-fsdp] model-ready", flush=True)
    if args.gradient_checkpointing:
        enable_checkpointing = getattr(model.model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise RuntimeError("LingBot policy does not expose gradient_checkpointing_enable")
        enable_checkpointing()
    action_head_weight_shape = action_head.weight.shape
    action_head_bias_shape = action_head.bias.shape
    if args.backend == "ddp":
        # VLA-only replay intentionally disables optional alignment heads, so
        # full-scope DDP must account for parameters that are not touched by
        # the selected loss.  This is required for multi-step reducer reuse;
        # action-head/expert scopes retain the cheaper default.
        find_unused = args.train_scope in ("alignment", "qwen", "all")
        train_model: torch.nn.Module = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused,
        )
        if args.fp32_grad_reduce:
            def fp32_comm_hook(_state, bucket):
                value = bucket.buffer()
                _fp32_chunked_all_reduce(value, args.fp32_reduce_chunk_mb)
                future = torch.futures.Future()
                future.set_result(value)
                return future

            train_model.register_comm_hook(None, fp32_comm_hook)
        train_head = action_head
    else:
        if debug_fsdp and rank == 0:
            print("[debug-fsdp] before-wrap", flush=True)
        sharding_strategy = {
            "no-shard": ShardingStrategy.NO_SHARD,
            "full-shard": ShardingStrategy.FULL_SHARD,
        }[args.fsdp_sharding]
        mixed_precision = (
            MixedPrecision(
                # Keep the model's native parameter dtype for parity.  Only
                # the data-parallel reduction is promoted to FP32; casting
                # the FP32 action head to BF16 changes the forward/gradient
                # path before any collective runs.
                param_dtype=None,
                reduce_dtype=torch.float32,
                buffer_dtype=None,
            )
            if args.fp32_grad_reduce
            else None
        )
        if args.fsdp_scope == "model":
            train_model = FSDP(
                model,
                auto_wrap_policy=model.get_fsdp_wrapping_policy(),
                ignored_modules=[action_head],
                sharding_strategy=sharding_strategy,
                device_id=device,
                use_orig_params=True,
                limit_all_gathers=True,
                mixed_precision=mixed_precision,
            )
        else:
            train_model = model
        wrapped_head = FSDP(
            action_head,
            sharding_strategy=sharding_strategy,
            device_id=device,
            use_orig_params=True,
            mixed_precision=mixed_precision,
        )
        if debug_fsdp and rank == 0:
            print("[debug-fsdp] after-wrap", flush=True)
        native_core = model.model.model
        native_core.action_out_proj = wrapped_head
        if args.fsdp_sharding == "full-shard":
            native_fp32_linear = native_core._fp32_linear

            def fsdp_aware_fp32_linear(self, linear, hidden_states):
                if isinstance(linear, FSDP):
                    return linear(hidden_states.float())
                return native_fp32_linear(linear, hidden_states)

            native_core._fp32_linear = types.MethodType(fsdp_aware_fp32_linear, native_core)
        train_head = wrapped_head

    trainable = [parameter for parameter in train_model.parameters() if parameter.requires_grad]
    if len(trainable) < 2:
        raise RuntimeError(f"Expected trainable parameters, got {len(trainable)} tensors")
    trace_weight = next(parameter for parameter in train_head.parameters() if parameter.requires_grad)
    weight = trace_weight
    weight_shape = action_head_weight_shape
    trace_weight_sharded = (
        args.backend == "fsdp"
        and args.fsdp_sharding == "full-shard"
    )
    action_head_only_full_shard = trace_weight_sharded and args.train_scope == "action-head"

    def snapshot(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if trace_weight_sharded:
            return _gather_sharded_tensor(tensor, expected_shape=weight_shape)
        return _clone_cpu(tensor)

    def snapshot_probe(parameter: torch.nn.Parameter, shape: torch.Size, *, gradient: bool):
        tensor = parameter.grad if gradient else parameter
        parameter_sharded = (
            args.backend == "fsdp"
            and args.fsdp_sharding == "full-shard"
            and id(parameter) not in action_head_parameter_ids
        )
        if parameter_sharded:
            if tensor is None:
                tensor = torch.empty(0, device=device, dtype=parameter.dtype)
            return _gather_sharded_tensor(tensor, expected_shape=shape)
        return _clone_cpu(tensor)

    master_parameters = (
        [
            torch.nn.Parameter(
                getattr(parameter, "_alignment_fp32_init").to(device=parameter.device)
            )
            for parameter in trainable
        ]
        if args.fp32_master_optimizer
        else trainable
    )
    if args.fp32_master_optimizer:
        for parameter in trainable:
            del parameter._alignment_fp32_init
    model_to_master = dict(zip(trainable, master_parameters, strict=True))
    trace_optimizer_weight = model_to_master[weight]
    optimizer = torch.optim.AdamW(
        master_parameters,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=0.0,
        foreach=False if args.no_adamw_foreach else None,
    )
    if debug_fsdp and rank == 0:
        print(f"[debug-fsdp] optimizer-ready trainable={len(trainable)}", flush=True)
    start_step = 0
    if args.resume:
        if args.checkpoint_dir is None:
            raise ValueError("--resume requires --checkpoint-dir")
        start_step = _load_checkpoint(args.checkpoint_dir, rank, optimizer)
    if args.batch_manifest is not None:
        batches, sample_identities = _build_manifest_batches(
            args.batch_manifest,
            device,
            rank=rank,
            world_size=dist.get_world_size(),
            distributed_replay=args.distributed_replay,
            replay_order=args.replay_order,
        )
    else:
        batch_path = args.batch or Path("/tmp/lingbot_real_batch.pt")
        batches = [_build_batch(batch_path, device)]
        sample_identities = [{"path": str(batch_path)}]
    traces: list[dict[str, Any]] = []
    trace_parameter = {
        "name": trace_parameter_name,
        "shape": list(action_head_weight_shape),
        "numel": action_head_weight_shape.numel(),
    }

    for step in range(start_step + 1, args.steps + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        sample_index = (step - 1) % len(batches)
        batch = batches[sample_index]
        optimizer.zero_grad(set_to_none=True)
        if args.fp32_master_optimizer:
            # The optimizer owns detached FP32 masters, so its zero_grad()
            # cannot clear gradients accumulated on the BF16 model tensors.
            for parameter in trainable:
                parameter.grad = None
        parameter_before = snapshot(trace_optimizer_weight)
        probe_before = {
            name: snapshot_probe(parameter, shape, gradient=False)
            for name, (parameter, shape) in parameter_probes.items()
        }
        output = train_model(**batch)
        if debug_fsdp and rank == 0:
            print(f"[debug-fsdp] forward-step={step}", flush=True)
        output["loss"].backward()
        if debug_fsdp and rank == 0:
            print(f"[debug-fsdp] backward-step={step}", flush=True)
        raw_grad = snapshot(weight.grad)
        probe_raw_grad = {
            name: snapshot_probe(parameter, shape, gradient=True)
            for name, (parameter, shape) in parameter_probes.items()
        }
        if args.backend == "fsdp" and not args.fsdp_gradient_breakdown:
            gradient_breakdown = {}
        elif args.backend == "fsdp" and args.fsdp_scope == "model":
            gradient_breakdown = _fsdp_gradient_breakdown(train_model)
        else:
            gradient_breakdown = _gradient_breakdown(train_model)
        if args.fp32_master_optimizer:
            for parameter, master_parameter in model_to_master.items():
                master_parameter.grad = (
                    None if parameter.grad is None else parameter.grad.detach().float().clone()
                )
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                master_parameters, args.max_grad_norm
            )
        elif args.deterministic_grad_norm:
            grad_norm_tensor = _deterministic_clip_grad_norm(
                train_model if args.backend == "fsdp" else None,
                trainable,
                args.max_grad_norm,
            )
        elif action_head_only_full_shard:
            logical_grads = [
                _gather_sharded_tensor(
                    parameter.grad
                    if parameter.grad is not None
                    else torch.empty(0, device=device, dtype=parameter.dtype),
                    expected_shape=shape,
                    move_to_cpu=False,
                )
                for parameter, shape in zip(
                    trainable,
                    (action_head_weight_shape, action_head_bias_shape),
                    strict=True,
                )
            ]
            grad_norm_tensor = torch.nn.utils.get_total_norm(logical_grads)
            clip_tensor = torch.clamp(args.max_grad_norm / (grad_norm_tensor + 1e-6), max=1.0)
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_tensor.to(parameter.grad.device))
        elif args.backend == "fsdp" and args.flat_fsdp_grad_norm:
            grad_norm_tensor = _flat_fsdp_clip_grad_norm(train_model, trainable, args.max_grad_norm)
        elif args.backend == "fsdp" and args.logical_fsdp_grad_norm:
            grad_norm_tensor = _logical_fsdp_clip_grad_norm(train_model, trainable, args.max_grad_norm)
        elif args.backend == "fsdp":
            # The native adapter is not itself an FSDP root; only the selected
            # action head may be wrapped.  Clip the logical trainable list so
            # this harness also works with the FluxVLA-owned model core.
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        else:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        grad_norm = float(torch.as_tensor(grad_norm_tensor).float())
        clip_coefficient = min(1.0, args.max_grad_norm / (grad_norm + 1e-6))
        optimizer_weight = trace_optimizer_weight
        post_clip_grad = snapshot(
            optimizer_weight.grad if args.fp32_master_optimizer else weight.grad
        )
        applied_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if args.fp32_master_optimizer:
            with torch.no_grad():
                for parameter, master_parameter in model_to_master.items():
                    parameter.copy_(master_parameter.to(dtype=parameter.dtype))
        if args.lr_decay_iters is not None:
            next_lr = _cosine_lr(
                step=step,
                maximum=args.lr,
                minimum=args.min_lr,
                decay_iters=args.lr_decay_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = next_lr
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - step_started) * 1000.0
        parameter_summary = (
            _parameter_summary(
                train_model,
                backend=args.backend,
                fsdp_sharding=args.fsdp_sharding,
            )
            if args.parameter_summary
            else None
        )

        if args.checkpoint_dir is not None and (
            not args.checkpoint_save_final_only or step == args.steps
        ):
            _save_checkpoint(args.checkpoint_dir, rank, step, optimizer)

        trace = {
            "step": step,
            "sample_index": sample_index,
            "sample_identity": sample_identities[sample_index],
            "losses": {
                key: _mean_scalar(output[key])
                for key in SCALAR_LOSS_KEYS
                if isinstance(output.get(key), torch.Tensor) and output[key].numel() == 1
            },
            "grad_norm": grad_norm,
            "clip_coefficient": clip_coefficient,
            "lr": applied_lr,
            "gradient_breakdown": gradient_breakdown,
            "parameter_before": parameter_before,
            "raw_grad": raw_grad,
            "post_clip_grad": post_clip_grad,
            "parameter_after": snapshot(trace_optimizer_weight),
            "exp_avg": snapshot(_state_tensor(optimizer, optimizer_weight, "exp_avg")),
            "exp_avg_sq": snapshot(_state_tensor(optimizer, optimizer_weight, "exp_avg_sq")),
            "parameter_probes": {
                name: {
                    "parameter_before": probe_before[name],
                    "raw_grad": probe_raw_grad[name],
                    "parameter_after": snapshot_probe(parameter, shape, gradient=False),
                }
                for name, (parameter, shape) in parameter_probes.items()
            },
        }
        if parameter_summary is not None:
            trace["parameter_summary"] = parameter_summary
        if rank == 0:
            traces.append(trace)
            print(
                f"{args.backend.upper()} STEP {step:03d} "
                f"LOSS {trace['losses']['loss']:.10f} GNORM {grad_norm:.10f} "
                f"CLIP {clip_coefficient:.10f} LR {applied_lr:.10e} "
                f"ELAPSED_MS {elapsed_ms:.3f}",
                flush=True,
            )

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": args.backend,
                "fsdp_sharding": args.fsdp_sharding if args.backend == "fsdp" else None,
                "fsdp_scope": args.fsdp_scope if args.backend == "fsdp" else None,
                "train_scope": args.train_scope,
                "logical_fsdp_grad_norm": args.logical_fsdp_grad_norm,
                "deterministic_grad_norm": args.deterministic_grad_norm,
                "adam_eps": args.adam_eps,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "world_size": dist.get_world_size(),
                "steps": args.steps,
                "seed": args.seed,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "lr_decay_iters": args.lr_decay_iters,
                "fp32_master_optimizer": args.fp32_master_optimizer,
                "max_grad_norm": args.max_grad_norm,
                "traces": traces,
                "trace_parameter": trace_parameter,
                "parameter_probes": [
                    {"name": name, "shape": list(shape), "numel": shape.numel()}
                    for name, (_, shape) in parameter_probes.items()
                ],
            },
            args.output,
        )
        _write_summary(
            args.output,
            {
                "backend": args.backend,
                "fsdp_sharding": args.fsdp_sharding if args.backend == "fsdp" else None,
                "fsdp_scope": args.fsdp_scope if args.backend == "fsdp" else None,
                "world_size": dist.get_world_size(),
                "steps": args.steps,
                "seed": args.seed,
                "logical_fsdp_grad_norm": args.logical_fsdp_grad_norm,
                "deterministic_grad_norm": args.deterministic_grad_norm,
                "losses": [trace["losses"]["loss"] for trace in traces],
                "grad_norms": [trace["grad_norm"] for trace in traces],
                "clip_coefficients": [trace["clip_coefficient"] for trace in traces],
                "trace_parameter": trace_parameter,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
