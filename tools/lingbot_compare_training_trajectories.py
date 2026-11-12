"""Compare native LingBot and FluxVLA JSONL training trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "loss" not in row and "VLA Train/Loss (Raw)" in row:
                    row = {
                        "step": int(row["VLA Train/Step"]),
                        "loss": float(row["VLA Train/Loss (Raw)"]),
                        "grad_norm": float(row["VLA Train/Grad Norm"]),
                    }
                if "loss" in row:
                    rows.append(row)
    return rows


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--flux", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Compare only the first N rows from each trajectory.",
    )
    args = parser.parse_args()

    native = _load_rows(args.native)
    flux = _load_rows(args.flux)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        native = native[: args.max_steps]
        flux = flux[: args.max_steps]
    if len(native) != len(flux):
        raise ValueError(f"trajectory lengths differ: native={len(native)}, flux={len(flux)}")

    comparisons = []
    for index, (native_row, flux_row) in enumerate(zip(native, flux), start=1):
        native_loss = float(native_row["loss"])
        flux_loss = float(flux_row["loss"])
        absolute_error = abs(native_loss - flux_loss)
        comparisons.append(
            {
                "step": index,
                "native_loss": native_loss,
                "flux_loss": flux_loss,
                "absolute_error": absolute_error,
                "relative_error": absolute_error / max(abs(native_loss), 1e-12),
            }
        )

    absolute = [row["absolute_error"] for row in comparisons]
    relative = [row["relative_error"] for row in comparisons]
    native_losses = [row["native_loss"] for row in comparisons]
    flux_losses = [row["flux_loss"] for row in comparisons]
    first_bad = next(
        (row["step"] for row in comparisons if row["absolute_error"] >= args.threshold),
        None,
    )
    max_error_row = max(comparisons, key=lambda row: row["absolute_error"])
    threshold_counts = {
        f"ge_{threshold:g}": sum(error >= threshold for error in absolute)
        for threshold in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2)
    }
    result = {
        "steps": len(comparisons),
        "threshold": args.threshold,
        "first_bad_step": first_bad,
        "native_mean_loss": sum(native_losses) / len(native_losses),
        "flux_mean_loss": sum(flux_losses) / len(flux_losses),
        "mean_absolute_error": sum(absolute) / len(absolute),
        "root_mean_square_error": math.sqrt(
            sum(error * error for error in absolute) / len(absolute)
        ),
        "p95_absolute_error": _percentile(absolute, 0.95),
        "max_absolute_error": max(absolute),
        "max_absolute_error_step": max_error_row["step"],
        "mean_relative_error": sum(relative) / len(relative),
        "final_step_absolute_error": comparisons[-1]["absolute_error"],
        "cumulative_absolute_error": sum(absolute),
        "threshold_counts": threshold_counts,
        "per_step": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_step"}, indent=2))


if __name__ == "__main__":
    main()
