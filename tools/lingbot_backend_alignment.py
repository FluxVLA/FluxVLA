"""Compare native DDP and FluxVLA FSDP reduction on a frozen LingBot replay.

This is an accuracy gate, not a throughput benchmark.  The 6.376B backbone is
frozen and only ``action_out_proj`` is optimized so the comparison can run on
partially occupied A100s.  Both backends emit the same tensor-level trace for
offline comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2


ROOT = Path("/ceph/teamGolden/andyzeng/VLA")
SHARED_ROOT = Path("/ceph/teamGolden/sjkang/rl/rlinf_adapt")
CHECKPOINT = SHARED_ROOT / "weights/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
SCALAR_LOSS_KEYS = (
    "loss",
    "loss_vla",
    "loss_depth",
    "loss_future_depth",
    "loss_future_video",
    "loss_moe_aux",
)


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
        choices=("action-head", "action-expert", "alignment", "qwen", "all"),
        default="action-head",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--logical-fsdp-grad-norm",
        action="store_true",
        help="Compute FSDP clipping from logical shard gradients for DDP alignment.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-save-final-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _build_batch(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = torch.load(path, map_location="cpu", weights_only=False)
    batch.pop("source_metadata", None)
    # The replay stores the three views as [3, 1200, 1536].  The native Qwen
    # visual path used by this checkpoint consumes the 224-grid contract:
    # [1, 3, 192, 1536] paired with one [1, 12, 16] grid per view.
    if batch["images"].ndim != 3:
        raise ValueError(
            "replay must contain images [num_views, patches, hidden]"
        )
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
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    from fluxvla.datasets.lingbot_teacher_cache import LingBotTeacherCacheDataset

    dataset = LingBotTeacherCacheDataset(
        str(path),
        cache_in_memory=True,
        validate_targets=True,
    )
    batches = []
    identities = []
    for index in range(len(dataset)):
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


def _build_model(device: torch.device, train_scope: str) -> tuple[LingBotVLAV2, torch.nn.Module]:
    model = LingBotVLAV2(
        source_root=str(ROOT / "lingbot-vla-v2"),
        tokenizer_path=str(SHARED_ROOT / "weights/Qwen3-VL-4B-Instruct"),
        pretrained_name_or_path=str(CHECKPOINT),
        native_config_path=str(ROOT / "lingbot-vla-v2/golden_artifacts/robotwin_reduced.yaml"),
    )
    model.from_pretrained()
    model.to(device=device, dtype=torch.bfloat16)
    model.model.config.action_fp32 = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    action_head = model.model.model.action_out_proj
    action_head.float()
    native_core = model.model.model
    if train_scope in ("action-head", "action-expert", "alignment", "qwen", "all"):
        action_head.requires_grad_(True)
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


def _logical_fsdp_clip_grad_norm(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Clip local FSDP shards using one logical global L2 norm.

    FSDP's nested ``clip_grad_norm_`` may reduce per-handle norms in a
    different order than DDP. Summing local shard squares once over the data
    parallel group gives the same logical parameter norm while retaining
    sharded gradients and optimizer state.
    """
    local_sq = torch.zeros(
        (),
        device=next(p for p in parameters if p.grad is not None).device,
        dtype=torch.float64,
    )
    for parameter in parameters:
        if parameter.grad is not None:
            # Match torch.nn.utils.get_total_norm: compute each parameter's
            # float32 vector norm first, then accumulate squared norms.
            parameter_norm = torch.linalg.vector_norm(parameter.grad.detach().float())
            local_sq += parameter_norm.double().square()
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    grad_norm = local_sq.sqrt().to(dtype=torch.float32)
    clip = torch.clamp(float(max_norm) / (grad_norm + 1e-6), max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(clip.to(device=parameter.grad.device, dtype=parameter.grad.dtype))
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
    if args.batch is not None and args.batch_manifest is not None:
        raise ValueError("--batch and --batch-manifest are mutually exclusive")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    from lingbotvla.distributed.parallel_state import init_parallel_state

    init_parallel_state(dp_size=dist.get_world_size(), dp_mode="ddp")
    model, action_head = _build_model(device, args.train_scope)
    if args.gradient_checkpointing:
        enable_checkpointing = getattr(model.model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise RuntimeError("LingBot policy does not expose gradient_checkpointing_enable")
        enable_checkpointing()
    action_head_weight_shape = action_head.weight.shape
    action_head_bias_shape = action_head.bias.shape
    if args.backend == "ddp":
        train_model: torch.nn.Module = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        train_head = action_head
    else:
        if args.fsdp_scope == "model" and args.fsdp_sharding != "full-shard":
            raise ValueError("--fsdp-scope model requires --fsdp-sharding full-shard")
        sharding_strategy = {
            "no-shard": ShardingStrategy.NO_SHARD,
            "full-shard": ShardingStrategy.FULL_SHARD,
        }[args.fsdp_sharding]
        if args.fsdp_scope == "model":
            train_model = FSDP(
                model,
                auto_wrap_policy=model.get_fsdp_wrapping_policy(),
                ignored_modules=[action_head],
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=device,
                use_orig_params=True,
                limit_all_gathers=True,
            )
        else:
            train_model = model
        wrapped_head = FSDP(
            action_head,
            sharding_strategy=sharding_strategy,
            device_id=device,
            use_orig_params=True,
        )
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

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    start_step = 0
    if args.resume:
        if args.checkpoint_dir is None:
            raise ValueError("--resume requires --checkpoint-dir")
        start_step = _load_checkpoint(args.checkpoint_dir, rank, optimizer)
    if args.batch_manifest is not None:
        batches, sample_identities = _build_manifest_batches(args.batch_manifest, device)
    else:
        batch_path = args.batch or Path("/tmp/lingbot_real_batch.pt")
        batches = [_build_batch(batch_path, device)]
        sample_identities = [{"path": str(batch_path)}]
    traces: list[dict[str, Any]] = []

    for step in range(start_step + 1, args.steps + 1):
        sample_index = (step - 1) % len(batches)
        batch = batches[sample_index]
        optimizer.zero_grad(set_to_none=True)
        parameter_before = snapshot(weight)
        output = train_model(**batch)
        output["loss"].backward()
        raw_grad = snapshot(weight.grad)
        if action_head_only_full_shard:
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
        elif args.backend == "fsdp" and args.logical_fsdp_grad_norm:
            grad_norm_tensor = _logical_fsdp_clip_grad_norm(trainable, args.max_grad_norm)
        elif args.backend == "fsdp":
            grad_norm_tensor = train_model.clip_grad_norm_(args.max_grad_norm)
        else:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        grad_norm = float(torch.as_tensor(grad_norm_tensor).float())
        clip_coefficient = min(1.0, args.max_grad_norm / (grad_norm + 1e-6))
        post_clip_grad = snapshot(weight.grad)
        optimizer.step()

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
            "parameter_before": parameter_before,
            "raw_grad": raw_grad,
            "post_clip_grad": post_clip_grad,
            "parameter_after": snapshot(weight),
            "exp_avg": snapshot(_state_tensor(optimizer, weight, "exp_avg")),
            "exp_avg_sq": snapshot(_state_tensor(optimizer, weight, "exp_avg_sq")),
        }
        if rank == 0:
            traces.append(trace)
            print(
                f"{args.backend.upper()} STEP {step:03d} "
                f"LOSS {trace['losses']['loss']:.10f} GNORM {grad_norm:.10f} "
                f"CLIP {clip_coefficient:.10f}",
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
                "world_size": dist.get_world_size(),
                "steps": args.steps,
                "seed": args.seed,
                "lr": args.lr,
                "max_grad_norm": args.max_grad_norm,
                "traces": traces,
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
                "losses": [trace["losses"]["loss"] for trace in traces],
                "grad_norms": [trace["grad_norm"] for trace in traces],
                "clip_coefficients": [trace["clip_coefficient"] for trace in traces],
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
