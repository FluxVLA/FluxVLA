"""Reduce per-rank native LingBot raw outputs into a step-wise JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pattern", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for step in range(1, args.steps + 1):
        rank_outputs = [
            torch.load(
                args.input_pattern.format(step=step, rank=rank),
                map_location="cpu",
                weights_only=False,
            )
            for rank in range(args.world_size)
        ]
        scalar_keys = set.intersection(
            *[
                {key for key, value in output.items() if isinstance(value, torch.Tensor) and value.numel() == 1}
                for output in rank_outputs
            ]
        )
        row = {"step": step}
        for key in sorted(scalar_keys):
            row[key] = sum(float(output[key]) for output in rank_outputs) / args.world_size
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
