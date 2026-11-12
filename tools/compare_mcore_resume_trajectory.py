#!/usr/bin/env python3
"""Verify that a split MCore save/resume run matches a continuous run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

TENSOR_FIELDS = (
    "parameter_before",
    "raw_grad",
    "post_clip_grad",
    "parameter_after",
    "exp_avg",
    "exp_avg_sq",
)
SCALAR_FIELDS = (
    "lr",
    "grad_norm",
    "clip_coefficient",
    "raw_grad_norm",
    "post_clip_grad_norm",
    "parameter_norm",
    "parameter_delta_norm",
    "exp_avg_norm",
    "exp_avg_sq_norm",
)
LOG_PATTERN = re.compile(
    r"iteration\s+(?P<step>\d+)/\s*\d+.*?"
    r"learning rate:\s*(?P<lr>[0-9.Ee+-]+).*?"
    r"lm_loss:\s*(?P<loss>[0-9.Ee+-]+).*?"
    r"grad norm:\s*(?P<grad_norm>[0-9.Ee+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-dir", type=Path, required=True)
    parser.add_argument("--phase-dir", type=Path, action="append", required=True)
    parser.add_argument("--require-steps", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_log(path: Path) -> dict[int, dict[str, float]]:
    result = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = LOG_PATTERN.search(line)
            if match:
                result[int(match.group("step"))] = {
                    field: float(match.group(field))
                    for field in ("lr", "loss", "grad_norm")
                }
    return result


def _merge_unique(mappings: list[dict[Any, Any]], label: str) -> dict[Any, Any]:
    result = {}
    for mapping in mappings:
        overlap = result.keys() & mapping.keys()
        if overlap:
            raise ValueError(f"duplicate {label}: {sorted(overlap)[:8]}")
        result.update(mapping)
    return result


def _load_trace(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (int(row["step"]), row["name"])
            if key in result:
                raise ValueError(f"duplicate parameter trace row {key} in {path}")
            result[key] = row
    return result


def _empty_field_result() -> dict[str, Any]:
    return {"all_exact": True, "compared": 0, "first_bad": None, "max_abs": 0.0}


def _compare_value(
    result: dict[str, Any],
    lhs: float,
    rhs: float,
    location: dict[str, Any],
) -> None:
    difference = abs(lhs - rhs)
    result["compared"] += 1
    result["max_abs"] = max(result["max_abs"], difference)
    if lhs != rhs:
        result["all_exact"] = False
        if result["first_bad"] is None:
            result["first_bad"] = {**location, "continuous": lhs, "resumed": rhs}


def _find_phase_tensor(phase_dirs: list[Path], step: int, rank: int) -> Path:
    name = f"mcore_step_{step:06d}_rank_{rank:02d}.pt"
    matches = [path / "optimizer_tensors" / name for path in phase_dirs]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected one resumed tensor artifact for step={step}, rank={rank}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _parameters(path: Path) -> dict[str, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return {row["name"]: row for row in artifact["parameters"]}


def compare(
    continuous_dir: Path,
    phase_dirs: list[Path],
    require_steps: int,
    world_size: int,
) -> dict[str, Any]:
    expected_steps = set(range(1, require_steps + 1))
    continuous_log = _load_log(continuous_dir / "launch.log")
    resumed_log = _merge_unique(
        [_load_log(path / "launch.log") for path in phase_dirs], "log steps"
    )
    if continuous_log.keys() != expected_steps or resumed_log.keys() != expected_steps:
        raise ValueError(
            "logs do not contain exactly the required steps: "
            f"continuous={sorted(continuous_log)}, resumed={sorted(resumed_log)}"
        )

    log_fields = {field: _empty_field_result() for field in ("loss", "lr", "grad_norm")}
    scalar_fields = {field: _empty_field_result() for field in SCALAR_FIELDS}
    tensor_fields = {field: _empty_field_result() for field in TENSOR_FIELDS}

    for step in sorted(expected_steps):
        for field, field_result in log_fields.items():
            _compare_value(
                field_result,
                continuous_log[step][field],
                resumed_log[step][field],
                {"step": step},
            )

    for rank in range(world_size):
        continuous_trace = _load_trace(
            continuous_dir / "param_trace" / f"native_rank{rank}.jsonl"
        )
        resumed_trace = _merge_unique(
            [
                _load_trace(path / "param_trace" / f"native_rank{rank}.jsonl")
                for path in phase_dirs
            ],
            f"parameter trace rows for rank {rank}",
        )
        if continuous_trace.keys() != resumed_trace.keys():
            raise ValueError(f"parameter trace keys differ for rank {rank}")
        if {step for step, _ in continuous_trace} != expected_steps:
            raise ValueError(
                f"parameter trace does not cover all steps for rank {rank}"
            )
        for (step, name), lhs_row in continuous_trace.items():
            rhs_row = resumed_trace[(step, name)]
            for field, field_result in scalar_fields.items():
                _compare_value(
                    field_result,
                    float(lhs_row[field]),
                    float(rhs_row[field]),
                    {"step": step, "rank": rank, "name": name},
                )

        for step in sorted(expected_steps):
            name = f"mcore_step_{step:06d}_rank_{rank:02d}.pt"
            lhs_parameters = _parameters(continuous_dir / "optimizer_tensors" / name)
            rhs_parameters = _parameters(_find_phase_tensor(phase_dirs, step, rank))
            if lhs_parameters.keys() != rhs_parameters.keys():
                raise ValueError(
                    f"optimizer parameter keys differ at step={step}, rank={rank}"
                )
            for parameter_name, lhs_row in lhs_parameters.items():
                rhs_row = rhs_parameters[parameter_name]
                for field, field_result in tensor_fields.items():
                    lhs = lhs_row.get(field)
                    rhs = rhs_row.get(field)
                    if (lhs is None) != (rhs is None):
                        raise ValueError(
                            f"unpaired {field} at step={step}, rank={rank}, "
                            f"parameter={parameter_name}"
                        )
                    if lhs is None:
                        continue
                    if lhs.shape != rhs.shape:
                        raise ValueError(
                            f"{field} shape differs at step={step}, rank={rank}, "
                            f"parameter={parameter_name}: {tuple(lhs.shape)} != {tuple(rhs.shape)}"
                        )
                    exact = torch.equal(lhs, rhs)
                    difference = float((lhs - rhs).abs().max().item())
                    field_result["compared"] += 1
                    field_result["max_abs"] = max(field_result["max_abs"], difference)
                    if not exact:
                        field_result["all_exact"] = False
                        if field_result["first_bad"] is None:
                            field_result["first_bad"] = {
                                "step": step,
                                "rank": rank,
                                "name": parameter_name,
                                "max_abs": difference,
                            }

    all_fields = [
        *log_fields.values(),
        *scalar_fields.values(),
        *tensor_fields.values(),
    ]
    return {
        "status": "PASS" if all(field["all_exact"] for field in all_fields) else "FAIL",
        "required_steps": require_steps,
        "world_size": world_size,
        "phase_dirs": [str(path) for path in phase_dirs],
        "log_scalars": log_fields,
        "optimizer_scalars": scalar_fields,
        "optimizer_tensors": tensor_fields,
    }


def main() -> None:
    args = parse_args()
    result = compare(
        args.continuous_dir,
        args.phase_dir,
        args.require_steps,
        args.world_size,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
