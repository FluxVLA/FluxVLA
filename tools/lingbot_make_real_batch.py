"""Materialize a real RoboTwin parquet sample for LingBot alignment replays."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image


IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _pack_robotwin(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map RoboTwin's 14D interleaved arms into native feature groups."""
    if value.shape[-1] != 14:
        raise ValueError(f"RoboTwin state/action must be 14D, got {tuple(value.shape)}")
    arm = torch.cat((value[..., :6], value[..., 7:13]), dim=-1)
    effector = torch.stack((value[..., 6], value[..., 13]), dim=-1)
    return arm, effector


def _normalize_robotwin(value: torch.Tensor, stats: dict, prefix: str) -> torch.Tensor:
    """Apply native normalization and place features in the native 55D layout."""
    arm, effector = _pack_robotwin(value)
    groups = []
    for group_name, group in (("arm.position", arm), ("effector.position", effector)):
        item = stats[f"{prefix}.{group_name}"]
        low = torch.as_tensor(item["q01"], dtype=group.dtype, device=group.device)
        high = torch.as_tensor(item["q99"], dtype=group.dtype, device=group.device)
        groups.append((group - low) / (high - low + 1e-6) * 2.0 - 1.0)
    # Native config reserves 14 slots for arm.position and end.position even
    # though RoboTwin supplies 12 arm joints.  Effector starts at offset 28.
    packed = torch.zeros((*value.shape[:-1], 55), dtype=value.dtype, device=value.device)
    packed[..., :12] = groups[0]
    packed[..., 28:30] = groups[1]
    return packed


def _decode_image(value: dict) -> Image.Image:
    if value.get("bytes") is None:
        raise ValueError("RoboTwin parquet image does not contain embedded PNG bytes")
    return Image.open(io.BytesIO(value["bytes"])).convert("RGB")


def _find_episode(dataset: Path, episode_index: int) -> tuple[Path, object]:
    for path in sorted((dataset / "data").glob("**/*.parquet")):
        table = pq.read_table(path)
        indices = table["episode_index"].to_pylist()
        rows = [index for index, value in enumerate(indices) if int(value) == episode_index]
        if rows:
            return path, table.take(rows)
    raise IndexError(f"episode {episode_index} not found under {dataset}")


def main() -> None:
    from transformers import AutoTokenizer

    from fluxvla.transforms.lingbot_vla_v2 import LingBotPad55D, LingBotQwen3VLImageTransform

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path("../lingbot-vla-v2/assets/norm_stats/robotwin.json"),
        help="Native LingBot norm_stats JSON (bounds_99_woclip).",
    )
    args = parser.parse_args()

    path, table = _find_episode(args.dataset, args.episode_index)
    frame = min(args.frame_index, table.num_rows - 1)
    row = table.slice(frame, 1).to_pylist()[0]
    end = min(frame + args.horizon, table.num_rows)
    action_values = table["action"].slice(frame, end - frame).to_pylist()
    actions = torch.tensor(action_values, dtype=torch.float32)
    valid_action_steps = actions.shape[0]
    if actions.shape[0] < args.horizon:
        actions = torch.cat(
            [actions, torch.zeros(args.horizon - actions.shape[0], actions.shape[-1])], dim=0
        )

    norm_payload = json.loads(args.norm_stats.read_text())
    norm_stats = norm_payload.get("norm_stats", norm_payload)
    raw_state = torch.tensor(row["observation.state"], dtype=torch.float32)
    raw_actions = actions
    sample = {
        # Official VLADataset applies torchvision Resize((img_size, img_size))
        # before the Qwen processor; RoboTwin frames are 4:3, so relying on
        # the processor's aspect-ratio-preserving pixel budget yields a
        # different [H, W] grid and a different action-expert sequence.
        "images": [
            _decode_image(row[key]).resize((256, 256), Image.Resampling.BILINEAR)
            for key in IMAGE_KEYS
        ],
        "state": _normalize_robotwin(raw_state, norm_stats, "observation.state"),
        "actions": _normalize_robotwin(raw_actions, norm_stats, "action"),
    }
    # Official RobotWin production config sets img_size=256 and yields a
    # Qwen3-VL grid of [1, 16, 16] per view.  Keep replay preprocessing on
    # that contract; the transform default (224) is intended for other VLA
    # recipes and silently changes the action-expert token sequence.
    sample = LingBotQwen3VLImageTransform(str(args.tokenizer), image_size=256)(sample)
    # The normalizer already emits the native 55D layout.  Keep this call for
    # validation/padding compatibility with older invocations.
    sample = LingBotPad55D()(sample)
    sample["state"] = sample["state"].unsqueeze(0)
    sample["actions"] = sample["actions"].unsqueeze(0)
    sample["action_is_pad"] = torch.arange(args.horizon) >= valid_action_steps

    task_rows = pq.read_table(args.dataset / "meta/tasks.parquet").to_pylist()
    task_by_index = {int(item["task_index"]): item["__index_level_0__"] for item in task_rows}
    task = task_by_index[int(row["task_index"])]
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=True
    )
    language = tokenizer(
        task,
        max_length=72,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    sample["lang_tokens"] = language["input_ids"]
    sample["lang_masks"] = language["attention_mask"].bool()

    # Auxiliary teachers are intentionally placeholders in this artifact.
    # The batch is valid for real image/language/state/action main-loss gates;
    # it must not be reported as real depth/video supervision.
    sample["depth_targets"] = torch.zeros(1, 252, 1024, dtype=torch.bfloat16)
    sample["future_depth_targets"] = torch.zeros(1, 252, 1024, dtype=torch.bfloat16)
    sample["future_video_targets"] = torch.zeros(1, 256, 1024, dtype=torch.bfloat16)
    sample["future_video_current_patch"] = torch.zeros(1, 256, 1024, dtype=torch.bfloat16)
    sample["source_metadata"] = {
        "dataset": str(args.dataset),
        "parquet": str(path),
        "episode_index": args.episode_index,
        "frame_index": frame,
        "action_horizon": args.horizon,
        "future_frame_index": min(frame + args.horizon - 1, table.num_rows - 1),
        "task": task,
        "auxiliary_targets": "zero_placeholders",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample, args.output)
    print(f"saved {args.output}")
    for key, value in sample.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
        print(f"{key}: {shape}")


if __name__ == "__main__":
    main()
