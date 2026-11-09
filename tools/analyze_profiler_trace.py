#!/usr/bin/env python3
"""Summarize a PyTorch Chrome profiler trace without profiler UI dependencies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def _event_group(event: dict[str, Any]) -> str:
    category = str(event.get("cat", "")).lower()
    name = str(event.get("name", ""))
    lowered_name = name.lower()
    if "nccl" in lowered_name:
        return "nccl_kernel" if category == "kernel" else "nccl_cpu"
    if category == "kernel":
        return "cuda_kernel"
    if "memcpy" in category or "memcpy" in lowered_name:
        return "memcpy"
    if "memset" in category or "memset" in lowered_name:
        return "memset"
    if category == "cpu_op":
        return "cpu_op"
    if category == "user_annotation":
        return "user_annotation"
    return category or "uncategorized"


def _iter_trace_events(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Incrementally decode the traceEvents array from a large Chrome trace."""
    decoder = json.JSONDecoder()
    buffer = ""
    in_events = False
    position = 0
    with path.open("r", encoding="utf-8") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if chunk:
                if position:
                    buffer = buffer[position:]
                    position = 0
                buffer += chunk
            elif position >= len(buffer):
                return

            if not in_events:
                key_index = buffer.find('"traceEvents"')
                if key_index < 0:
                    if not chunk:
                        raise ValueError(f"traceEvents not found in {path}")
                    buffer = buffer[-len('"traceEvents"') :]
                    continue
                array_index = buffer.find("[", key_index)
                if array_index < 0:
                    if not chunk:
                        raise ValueError(f"traceEvents array not found in {path}")
                    buffer = buffer[key_index:]
                    continue
                buffer = buffer[array_index + 1 :]
                in_events = True

            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n,":
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    event, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if isinstance(event, dict):
                    yield event
                position = end

            if not chunk:
                raise ValueError(f"incomplete traceEvents array in {path}")


def _summarize(events: Iterable[dict[str, Any]], top: int) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    group_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        duration_us = float(event["dur"])
        group = _event_group(event)
        name = str(event.get("name", "<unnamed>"))
        grouped[group][name][0] += duration_us
        grouped[group][name][1] += 1
        group_totals[group][0] += duration_us
        group_totals[group][1] += 1

    result: dict[str, Any] = {"groups": {}}
    for group, (duration_us, count) in sorted(
        group_totals.items(), key=lambda item: item[1][0], reverse=True
    ):
        top_events = sorted(
            grouped[group].items(), key=lambda item: item[1][0], reverse=True
        )[:top]
        result["groups"][group] = {
            "total_duration_ms": duration_us / 1_000.0,
            "count": int(count),
            "top": [
                {
                    "name": name,
                    "total_duration_ms": values[0] / 1_000.0,
                    "count": int(values[1]),
                    "mean_duration_us": values[0] / values[1],
                }
                for name, values in top_events
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = _summarize(_iter_trace_events(args.trace), args.top)
    summary["trace"] = str(args.trace)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
