"""Run a low-memory LingBot real-batch forward smoke in a detached process."""

from __future__ import annotations

import torch
import os

from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2


ROOT = "/ceph/teamGolden/andyzeng/VLA"
SHARED_ROOT = "/ceph/teamGolden/sjkang/rl/rlinf_adapt"
CHECKPOINT = f"{SHARED_ROOT}/weights/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"


def main() -> None:
    batch = torch.load("/tmp/lingbot_real_batch.pt", map_location="cpu", weights_only=False)
    # The native model accepts [num_views, patches, width]. Reduce the raw
    # real-image patches to the 224-level Qwen grid before the smoke forward.
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

    model = LingBotVLAV2(
        source_root=f"{ROOT}/lingbot-vla-v2",
        tokenizer_path=f"{SHARED_ROOT}/weights/Qwen3-VL-4B-Instruct",
        pretrained_name_or_path=CHECKPOINT,
        native_config_path=f"{ROOT}/lingbot-vla-v2/golden_artifacts/robotwin_reduced.yaml",
    )
    model.from_pretrained()
    train_step = os.environ.get("LINGBOT_TRAIN_STEP", "0") == "1"
    model.to(device="cuda", dtype=torch.bfloat16)
    if train_step:
        model.train()
        model.model.gradient_checkpointing_enable()
    else:
        model.eval()
    for key in ("images", "state", "actions", "noise", "time"):
        batch[key] = batch[key].cuda().bfloat16()
    for key in ("depth_targets", "future_depth_targets", "future_video_targets", "future_video_current_patch"):
        batch[key] = batch[key].cuda().bfloat16()
    for key in ("image_grid_thw", "img_masks", "joint_mask", "action_is_pad", "lang_tokens", "lang_masks"):
        batch[key] = batch[key].cuda()
    print("FORWARD_START", flush=True)
    if train_step:
        output = model(**batch)
        output["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        action_head = model.model.model.action_out_proj
        optimizer = torch.optim.AdamW(action_head.parameters(), lr=1e-6)
        optimizer.step()
        print("GRAD_NORM", float(grad_norm), "ACTION_OPTIMIZER_STEP", flush=True)
    else:
        with torch.inference_mode():
            output = model(**batch)
    print("LOSS", float(output["loss"]), float(output["loss_vla"]), flush=True)


if __name__ == "__main__":
    main()
