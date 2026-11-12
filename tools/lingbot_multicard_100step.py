"""Four-GPU real RoboTwin DDP smoke for the FluxVLA LingBot adapter.

The shared RoboTwin subset stores images as embedded parquet bytes and is not
yet consumable by the generic video-path dataset. This launcher reuses the
validated model-ready batch, synchronizes it with DDP, and trains the action
head for a deterministic 100-step distributed smoke.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2


ROOT = Path("/ceph/teamGolden/andyzeng/VLA")
SHARED_ROOT = Path("/ceph/teamGolden/sjkang/rl/rlinf_adapt")
CHECKPOINT = SHARED_ROOT / "weights/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"


def _build_batch(device: torch.device) -> dict[str, torch.Tensor]:
    batch = torch.load("/tmp/lingbot_real_batch.pt", map_location="cpu", weights_only=False)
    batch["images"] = torch.nn.functional.interpolate(
        batch["images"].float().unsqueeze(0), size=(192, 1536), mode="bilinear", align_corners=False
    )
    batch["image_grid_thw"] = torch.tensor([[[1, 12, 16]] * 3], dtype=torch.long)
    batch.update(
        img_masks=torch.ones((1, 3), dtype=torch.bool),
        joint_mask=torch.ones((1, 50, 55), dtype=torch.bool),
        action_is_pad=torch.zeros((1, 50), dtype=torch.bool),
        noise=torch.zeros_like(batch["actions"]),
        time=torch.full((1,), 0.5),
    )
    for key in ("depth_targets", "future_depth_targets"):
        batch[key] = torch.nn.functional.pad(batch[key], (0, 0, 0, 4))
    return {
        key: value.to(device=device, dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
        for key, value in batch.items()
    }


def main() -> None:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    # Native LingBot's MoE dispatch reads its own parallel-state singleton;
    # initialize it explicitly for the DDP world before the first forward.
    from lingbotvla.distributed.parallel_state import init_parallel_state

    init_parallel_state(dp_size=dist.get_world_size(), dp_mode="ddp")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(7 + rank)

    model = LingBotVLAV2(
        source_root=str(ROOT / "lingbot-vla-v2"),
        tokenizer_path=str(SHARED_ROOT / "weights/Qwen3-VL-4B-Instruct"),
        pretrained_name_or_path=str(CHECKPOINT),
        native_config_path=str(ROOT / "lingbot-vla-v2/golden_artifacts/robotwin_reduced.yaml"),
    )
    model.from_pretrained()
    model.to(device=device, dtype=torch.bfloat16)
    # Keep the action projection and its loss in FP32. Otherwise a BF16 scalar
    # loss can remain numerically constant even when the optimizer updates the
    # head by sub-ULP increments.
    model.model.config.action_fp32 = True

    # The 6.376B checkpoint is frozen for this 100-step distributed smoke;
    # only the action head is updated to keep the per-rank optimizer state small.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    action_head = model.model.model.action_out_proj
    action_head.float()
    action_head.requires_grad_(True)
    model.train()
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(action_head.parameters(), lr=1e-4)
    batch = _build_batch(device)
    losses: list[float] = []

    for step in range(1, 101):
        optimizer.zero_grad(set_to_none=True)
        output = ddp(**batch)
        output["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(action_head.parameters(), 1.0)
        optimizer.step()
        reduced = output["loss"].detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= dist.get_world_size()
        losses.append(float(reduced))
        if rank == 0 and (step == 1 or step % 10 == 0):
            print(f"STEP {step:03d} LOSS {losses[-1]:.10f} GRAD_NORM {float(grad_norm):.6f}", flush=True)

    if rank == 0:
        print(f"FINAL_STEP {len(losses)} FINAL_LOSS {losses[-1]:.10f}", flush=True)
        torch.save({"steps": losses, "final_loss": losses[-1]}, "/tmp/lingbot_ddp_100step_metrics.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
