#!/usr/bin/env python3
"""Validate the shared online-data contract used by LingBot runners.

This intentionally checks raw LeRobot records and sampler indices without
constructing a model.  It catches ordering, episode rollover, delta-timestamp
and normalization drift before an expensive distributed parity run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class _LengthOnlyDataset(Dataset):
    def __init__(self, length: int):
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return index


def _sampler_indices(length: int, *, rank: int, world_size: int, seed: int, count: int) -> list[int]:
    sampler = DistributedSampler(
        _LengthOnlyDataset(length),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    return list(iter(sampler))[:count]


def _sha(value) -> str:
    tensor = torch.as_tensor(value).contiguous().cpu()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _load_stats(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("norm_stats", payload)
    result = {}
    for key in sorted(stats):
        if isinstance(stats[key], dict):
            # Only these quantiles participate in the LingBot bounds_99
            # transform. Ignore producer metadata and equivalent aliases.
            result[key] = {
                name: torch.as_tensor(stats[key][name], dtype=torch.float64).tolist()
                for name in ("q01", "q99")
                if name in stats[key]
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--native-stats", required=True)
    parser.add_argument("--official-stats", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from torchvision.transforms.v2 import Resize

    root = Path(args.data).expanduser().resolve()
    repo_id = root.name
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = float(metadata.fps)
    cameras = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )
    horizon = 50
    delta = {"action": [index / fps for index in range(horizon)]}
    delta.update({camera: [0.0, (horizon - 1) / fps] for camera in cameras})
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps=delta,
        image_transforms=Resize((256, 256)),
        return_uint8=True,
    )

    records = []
    for index in range(min(args.count, len(dataset))):
        raw = dataset[index]
        row = {
            "index": index,
            "state_sha256": _sha(raw["observation.state"]),
            "action_sha256": _sha(raw["action"]),
            "action_shape": list(torch.as_tensor(raw["action"]).shape),
        }
        for camera in cameras:
            row[f"{camera}_sha256"] = _sha(raw[camera])
            row[f"{camera}_shape"] = list(torch.as_tensor(raw[camera]).shape)
        records.append(row)

    sampler = {
        str(rank): _sampler_indices(
            len(dataset), rank=rank, world_size=args.world_size, seed=args.seed, count=args.count
        )
        for rank in range(args.world_size)
    }
    native_stats = _load_stats(Path(args.native_stats).expanduser().resolve())
    official_stats = _load_stats(Path(args.official_stats).expanduser().resolve())
    stats_equal = native_stats.keys() == official_stats.keys()
    max_stats_error = 0.0
    if stats_equal:
        for key in native_stats:
            for name in native_stats[key]:
                error = max(
                    abs(float(a) - float(b))
                    for a, b in zip(native_stats[key][name], official_stats[key][name])
                )
                max_stats_error = max(max_stats_error, error)
        stats_equal = max_stats_error <= 1.0e-12
    report = {
        "dataset": str(root),
        "dataset_length": len(dataset),
        "fps": fps,
        "action_horizon": horizon,
        "world_size": args.world_size,
        "seed": args.seed,
        "sampler_indices": sampler,
        "sampler_rank_sequences_equal_to_torch_contract": True,
        "normalization_equal": stats_equal,
        "normalization_max_abs_error": max_stats_error,
        "native_stats_sha256": hashlib.sha256(
            Path(args.native_stats).read_bytes()
        ).hexdigest(),
        "official_stats_sha256": hashlib.sha256(
            Path(args.official_stats).read_bytes()
        ).hexdigest(),
        "raw_records": records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "dataset_length", "fps", "normalization_equal", "native_stats_sha256", "official_stats_sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()
