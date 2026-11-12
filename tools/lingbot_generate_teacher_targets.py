"""Materialize native LingBot depth and video teacher targets for a replay batch."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image


IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--future-offset", type=int)
    parser.add_argument("--augmentation-seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def _decode_rgb(value: dict[str, Any]) -> torch.Tensor:
    data = value.get("bytes")
    if data is None:
        raise ValueError("RoboTwin parquet image does not contain embedded PNG bytes")
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


def _load_teacher_images(
    parquet_path: Path,
    episode_index: int,
    frame_index: int,
    future_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    table = pq.read_table(parquet_path)
    rows = [
        index
        for index, value in enumerate(table["episode_index"].to_pylist())
        if int(value) == episode_index
    ]
    if not rows:
        raise ValueError(f"episode {episode_index} is absent from {parquet_path}")
    if frame_index < 0 or frame_index >= len(rows):
        raise IndexError(
            f"frame {frame_index} is outside episode {episode_index} with {len(rows)} frames"
        )
    future_frame = min(frame_index + future_offset, len(rows) - 1)
    current_row = table.slice(rows[frame_index], 1).to_pylist()[0]
    future_row = table.slice(rows[future_frame], 1).to_pylist()[0]
    current = torch.stack([_decode_rgb(current_row[key]) for key in IMAGE_KEYS])
    future = torch.stack([_decode_rgb(future_row[key]) for key in IMAGE_KEYS])
    return current.unsqueeze(0), future.unsqueeze(0), future_frame


def _validate_target(name: str, value: torch.Tensor, expected_tokens: tuple[int, ...]) -> None:
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != 1024:
        raise ValueError(f"{name} has invalid shape {tuple(value.shape)}")
    if value.shape[1] not in expected_tokens:
        raise ValueError(f"{name} has invalid token count {value.shape[1]}")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} contains non-finite values")
    if torch.count_nonzero(value).item() == 0:
        raise ValueError(f"{name} is all zero")


def main() -> None:
    args = _parse_args()
    batch = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = batch.get("source_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("input replay does not contain source_metadata")
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    align_config = config["train"]["align_params"]
    future_offset = args.future_offset
    if future_offset is None:
        future_offset = int(batch["actions"].shape[-2]) - 1

    current, future, future_frame = _load_teacher_images(
        Path(metadata["parquet"]),
        int(metadata["episode_index"]),
        int(metadata["frame_index"]),
        future_offset,
    )
    device = torch.device(args.device)
    current = current.to(device=device, non_blocking=True)
    future = future.to(device=device, non_blocking=True)

    # This command is an explicitly offline preprocessing tool.  The official
    # implementation is allowed here as a read-only teacher provider, while
    # FluxVLA training never imports it.  Keep the boundary explicit by using
    # the source tree directly instead of a runtime import shim.
    from fluxvla.models.vlas.lingbot_vla_v2 import _install_transformers_compat

    source_root = str(args.source_root.resolve())
    sys.path.insert(0, source_root)
    _install_transformers_compat()
    from lingbotvla.models.vla.vision_models.module_utils import (
        build_depth_model,
        build_video_model,
        get_depth_target,
        get_video_target,
    )

    depth_models = build_depth_model(align_config)
    video_teacher = build_video_model(align_config)
    depth_type = align_config["depth"]["model_type"]
    effective_fps = align_config["video"].get("effective_fps")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        depth_targets, _ = get_depth_target(depth_type, depth_models, current)
        future_depth_targets, _ = get_depth_target(depth_type, depth_models, future)
        video_bundle = get_video_target(
            video_teacher,
            current,
            future,
            align_config["video"],
            effective_fps=effective_fps,
        )
    if not isinstance(video_bundle, dict):
        raise ValueError("configuration must enable current video patch targets")
    targets = {
        "depth_targets": depth_targets,
        "future_depth_targets": future_depth_targets,
        "future_video_targets": video_bundle["patch"],
        "future_video_current_patch": video_bundle["current_patch"],
    }
    for name, value in targets.items():
        expected_tokens = (252, 256) if "depth" in name else (256,)
        _validate_target(name, value, expected_tokens=expected_tokens)
        batch[name] = value.detach().cpu()

    metadata.update(
        auxiliary_targets="native_moge_morgbd_dino",
        future_frame_index=future_frame,
        future_offset=future_offset,
        teacher_config=str(args.config),
        target_shapes={name: list(value.shape) for name, value in targets.items()},
    )
    cache_identity = {
        "dataset": metadata.get("dataset"),
        "parquet": metadata.get("parquet"),
        "episode_index": int(metadata["episode_index"]),
        "frame_index": int(metadata["frame_index"]),
        "future_frame_index": future_frame,
        "future_offset": future_offset,
        "augmentation_seed": args.augmentation_seed,
        "teacher_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata["teacher_cache_key"] = cache_key
    metadata["teacher_cache_identity"] = cache_identity
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, args.output)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        if args.manifest.exists():
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        else:
            manifest = {"format": "fluxvla.lingbot_teacher_cache.v1", "entries": []}
        if manifest.get("format") != "fluxvla.lingbot_teacher_cache.v1":
            raise ValueError(f"Unsupported cache manifest format in {args.manifest}")
        relative_path = os.path.relpath(args.output.resolve(), args.manifest.parent.resolve())
        entry = {**cache_identity, "path": relative_path, "cache_key": cache_key}
        manifest["entries"] = [
            item for item in manifest.get("entries", []) if item.get("cache_key") != cache_key
        ]
        manifest["entries"].append(entry)
        temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.manifest)
    print(f"saved {args.output}")
    for name, value in targets.items():
        print(
            f"{name}: shape={tuple(value.shape)} dtype={value.dtype} "
            f"abs_mean={value.float().abs().mean().item():.8f}"
        )


if __name__ == "__main__":
    main()
