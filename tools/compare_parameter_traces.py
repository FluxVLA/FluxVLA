#!/usr/bin/env python3
"""Compare Native/FSDP and MCore optimizer alignment JSONL traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "lr",
    "grad_norm",
    "raw_grad_norm",
    "post_clip_grad_norm",
    "clip_coefficient",
    "parameter_delta_norm",
    "exp_avg_norm",
    "exp_avg_sq_norm",
)


def _read(root: Path) -> dict[tuple[int, str], dict]:
    rows = {}
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(int(row["step"]), row["name"])] = row
    return rows


def _diff(left: float | None, right: float | None) -> tuple[float | None, float | None]:
    if left is None or right is None:
        return None, None
    delta = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)), 1e-30)
    return delta, delta / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference = _read(args.reference)
    candidate = _read(args.candidate)
    keys = sorted(set(reference) & set(candidate))
    if not keys:
        raise SystemExit("no matching (step, parameter) rows")
    print(f"matched_rows={len(keys)} reference_rows={len(reference)} candidate_rows={len(candidate)}")
    for field in FIELDS:
        values = [_diff(reference[key].get(field), candidate[key].get(field)) for key in keys]
        values = [item for item in values if item[0] is not None]
        if not values:
            continue
        max_abs = max(item[0] for item in values)
        max_rel = max(item[1] for item in values)
        print(f"{field}: max_abs={max_abs:.8e} max_rel={max_rel:.8e}")


if __name__ == "__main__":
    main()
