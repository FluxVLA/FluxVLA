"""Summarize a PyTorch profiler Chrome trace without Kineto re-parsing."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _top(events, category: str, limit: int):
    totals = defaultdict(lambda: [0, 0.0])
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == category:
            row = totals[event.get("name", "unknown")]
            row[0] += 1
            row[1] += float(event.get("dur", 0.0))
    rows = [
        {"name": name, "calls": count, "total_us": duration}
        for name, (count, duration) in totals.items()
    ]
    return sorted(rows, key=lambda row: row["total_us"], reverse=True)[:limit]


def analyze(trace_path: Path, limit: int):
    events = json.loads(trace_path.read_text(encoding="utf-8"))["traceEvents"]
    stages = defaultdict(lambda: [0, 0.0])
    shape_variants = defaultdict(set)
    peak_allocated = defaultdict(int)
    communication_us = 0.0
    communication_calls = 0

    for event in events:
        name = event.get("name", "")
        if (event.get("ph") == "X" and event.get("cat") == "user_annotation"
                and name.startswith("PI05/")):
            row = stages[name]
            row[0] += 1
            row[1] += float(event.get("dur", 0.0))
        if event.get("cat") == "cpu_op":
            dims = event.get("args", {}).get("Input Dims")
            if dims is not None and len(shape_variants[name]) < 20:
                shape_variants[name].add(json.dumps(dims, separators=(",", ":")))
        if name == "[memory]":
            args = event.get("args", {})
            if args.get("Device Type") == 1:
                device = int(args.get("Device Id", -1))
                peak_allocated[device] = max(
                    peak_allocated[device], int(args.get("Total Allocated", 0))
                )
        if event.get("cat") == "kernel" and any(
            token in name.lower() for token in ("nccl", "allreduce", "allgather")
        ):
            communication_calls += 1
            communication_us += float(event.get("dur", 0.0))

    stage_rows = [
        {"name": name, "calls": count, "total_us": duration}
        for name, (count, duration) in stages.items()
    ]
    stage_rows.sort(key=lambda row: row["total_us"], reverse=True)
    top_operators = _top(events, "cpu_op", limit)
    return {
        "trace": str(trace_path),
        "event_count": len(events),
        "python_function_events": sum(
            event.get("cat") == "python_function" for event in events
        ),
        "stage_wall_time": stage_rows,
        "kernel_hotspots": _top(events, "kernel", limit),
        "operator_hotspots": top_operators,
        "hot_operator_input_shapes": {
            row["name"]: sorted(shape_variants[row["name"]])
            for row in top_operators
        },
        "peak_allocated_bytes": dict(sorted(peak_allocated.items())),
        "communication": {
            "kernel_calls": communication_calls,
            "kernel_total_us": communication_us,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    summary = analyze(args.trace, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
