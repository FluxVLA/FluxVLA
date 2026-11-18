"""Summarize LingBot Native and MCore GB64 training logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


MCORE_STEP = re.compile(
    r"iteration\s+(?P<step>\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+).*?"
    r"learning rate:\s*(?P<lr>[0-9.Ee+-]+).*?"
    r"global batch size:\s*(?P<gb>\d+).*?"
    r"loss:\s*(?P<loss>[0-9.Ee+-]+).*?"
    r"loss_vla:\s*(?P<loss_vla>[0-9.Ee+-]+).*?"
    r"grad norm:\s*(?P<grad_norm>[0-9.Ee+\-infnaINFNA.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--baseline", default="native_fsdp4")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_native(path: Path) -> list[dict]:
    candidates = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    rows = []
    for candidate in candidates:
        if candidate.name == "run-metrics.jsonl":
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "VLA Train/Step" not in row:
                continue
            rows.append(
                {
                    "step": int(row["VLA Train/Step"]),
                    "step_time_s": float(row["VLA Train/Step Time"]),
                    "global_batch_size": 64,
                    "loss": float(row["VLA Train/Loss"]),
                    "loss_vla": float(row["VLA Train/Loss Vla"]),
                    "grad_norm": float(row["VLA Train/Grad Norm"]),
                    "lr": float(row["VLA Train/Learning Rate"]),
                }
            )
    return sorted(rows, key=lambda row: row["step"])


def parse_mcore(path: Path) -> list[dict]:
    rows = []
    for match in MCORE_STEP.finditer(path.read_text(encoding="utf-8", errors="replace")):
        value = match.groupdict()
        rows.append(
            {
                "step": int(value["step"]),
                "step_time_s": float(value["ms"]) / 1000.0,
                "global_batch_size": int(value["gb"]),
                "lr": float(value["lr"]),
                "loss": float(value["loss"]),
                "loss_vla": float(value["loss_vla"]),
                "grad_norm": float(value["grad_norm"]),
            }
        )
    return rows


def summarize(rows: list[dict], warmup_steps: int, expected_steps: int) -> dict:
    if not rows:
        return {"status": "NO_DATA", "steps": 0}
    measured = [row for row in rows if row["step"] > warmup_steps]
    times = [row["step_time_s"] for row in measured]
    gb = rows[0]["global_batch_size"]
    result = {
        "status": "PASS" if len(rows) == expected_steps else "PARTIAL",
        "steps": len(rows),
        "global_batch_size": gb,
        "first_loss": rows[0]["loss"],
        "last_loss": rows[-1]["loss"],
        "first_loss_vla": rows[0]["loss_vla"],
        "last_loss_vla": rows[-1]["loss_vla"],
        "first_logged_lr": rows[0].get("lr"),
        "last_logged_lr": rows[-1].get("lr"),
        "mean_step_time_s": statistics.mean(times) if times else None,
        "p50_step_time_s": statistics.median(times) if times else None,
        "p95_step_time_s": quantile(times, 0.95) if times else None,
        "throughput_samples_s": gb / statistics.mean(times) if times else None,
        "rows": rows,
    }
    return result


def quantile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated quantile without a NumPy dependency."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_peak_gpu_memory(path: Path) -> float | None:
    """Read the largest MiB value from the lightweight nvidia-smi monitor."""
    if not path.is_file():
        return None
    peak = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        try:
            used_mib = float(fields[1])
        except ValueError:
            continue
        peak = used_mib if peak is None else max(peak, used_mib)
    return peak


def compare_rows(baseline: list[dict], candidate: list[dict]) -> dict:
    baseline_by_step = {row["step"]: row for row in baseline}
    candidate_by_step = {row["step"]: row for row in candidate}
    steps = sorted(baseline_by_step.keys() & candidate_by_step.keys())
    result = {"common_steps": len(steps)}
    for key in ("loss", "loss_vla", "grad_norm"):
        pairs = [
            (baseline_by_step[step].get(key), candidate_by_step[step].get(key))
            for step in steps
        ]
        differences = [abs(left - right) for left, right in pairs if left is not None and right is not None]
        if differences:
            result[f"{key}_mean_abs_diff"] = statistics.mean(differences)
            result[f"{key}_max_abs_diff"] = max(differences)
            result[f"{key}_max_abs_diff_step"] = steps[differences.index(max(differences))]
            result[f"{key}_last_abs_diff"] = differences[-1]
            if len(differences) > 1:
                left_values = [left for left, right in pairs if left is not None and right is not None]
                right_values = [right for left, right in pairs if left is not None and right is not None]
                left_mean = statistics.mean(left_values)
                right_mean = statistics.mean(right_values)
                numerator = sum(
                    (left - left_mean) * (right - right_mean)
                    for left, right in zip(left_values, right_values, strict=True)
                )
                denominator = math.sqrt(
                    sum((value - left_mean) ** 2 for value in left_values)
                    * sum((value - right_mean) ** 2 for value in right_values)
                )
                result[f"{key}_pearson"] = numerator / denominator if denominator else None
    result["accuracy_gate"] = (
        "PASS"
        if result.get("loss_max_abs_diff", math.inf) < 1e-3
        and result.get("loss_vla_max_abs_diff", math.inf) < 1e-3
        else "FAIL"
    )
    return result


def main() -> None:
    args = parse_args()
    result = {"warmup_steps": args.warmup_steps, "cases": {}}
    for spec in args.case:
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        rows = parse_native(path) if path.is_dir() or path.suffix == ".jsonl" else parse_mcore(path)
        summary = summarize(rows, args.warmup_steps, args.expected_steps)
        gpu_path = path / "gpu.csv" if path.is_dir() else path.parent / "gpu.csv"
        summary["peak_gpu_memory_mib"] = parse_peak_gpu_memory(gpu_path)
        result["cases"][name] = summary
    baseline = result["cases"].get(args.baseline)
    result["baseline"] = args.baseline
    result["comparisons"] = {}
    if baseline:
        for name, case in result["cases"].items():
            if name != args.baseline:
                result["comparisons"][name] = compare_rows(baseline["rows"], case["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in case.items() if k != "rows"} for name, case in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
