"""Convert native per-rank LingBot batch dumps into a FluxVLA cache manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


_UNBATCHED_CACHE_KEYS = (
    "images",
    "image_grid_thw",
    "img_masks",
    "state_joint_mask",
    "joint_mask",
    "action_is_pad",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-pattern", required=True, help="Pattern containing {rank}")
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--distributed-sampler-seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.world_size < 1:
        raise ValueError("steps and world-size must both be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = args.steps * args.world_size
    generator = torch.Generator().manual_seed(args.distributed_sampler_seed)
    sampler_order = torch.randperm(num_samples, generator=generator).tolist()
    entries = [None] * num_samples
    for step in range(1, args.steps + 1):
        for rank in range(args.world_size):
            source = Path(args.dump_pattern.format(rank=rank, step=step))
            sample = torch.load(source, map_location="cpu", weights_only=False)
            for key in _UNBATCHED_CACHE_KEYS:
                value = sample[key]
                if value.shape[0] != 1:
                    raise ValueError(
                        f"{source}: expected leading singleton for {key}, got {value.shape}"
                    )
                sample[key] = value.squeeze(0)
            cache_key = (
                f"native-gb{args.world_size}-seed{args.seed}-step{step}-rank{rank}"
            )
            sample["source_metadata"] = {
                "auxiliary_targets": "native_moge_morgbd_dino",
                "source": str(source),
                "teacher_cache_key": cache_key,
                "native_rank": rank,
                "native_step": step,
                "seed": args.seed,
            }
            filename = f"cache_native_step{step:04d}_rank{rank}.pt"
            torch.save(sample, args.output_dir / filename)
            sampler_position = (step - 1) * args.world_size + rank
            manifest_index = sampler_order[sampler_position]
            entries[manifest_index] = {
                "path": filename,
                "cache_key": cache_key,
                "native_rank": rank,
                "native_step": step,
            }

    manifest = {"format": "fluxvla.lingbot_teacher_cache.v1", "entries": entries}
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
