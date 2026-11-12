#!/usr/bin/env python3
"""Compare fixed-replay PI0.5 FSDP and Native MCore accuracy artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

MCORE_LOSS_PATTERN = re.compile(
    r"iteration\s+(?P<step>\d+)/\s*\d+.*?lm_loss:\s*(?P<loss>[0-9.Ee+-]+)"
)
TENSOR_FIELDS = (
    "parameter_before",
    "raw_grad",
    "post_clip_grad",
    "parameter_after",
    "exp_avg",
    "exp_avg_sq",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsdp-metrics", type=Path, required=True)
    parser.add_argument("--fsdp-param-trace", type=Path, required=True)
    parser.add_argument("--mcore-log", type=Path, required=True)
    parser.add_argument("--mcore-param-trace", type=Path, required=True)
    parser.add_argument("--fsdp-tensors", type=Path)
    parser.add_argument("--mcore-tensors", type=Path)
    parser.add_argument("--parameter-name", default="action_out_proj.weight")
    parser.add_argument("--skip-tensors", action="store_true")
    parser.add_argument("--require-steps", type=int, required=True)
    parser.add_argument("--loss-atol", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_fsdp_losses(path: Path) -> dict[int, float]:
    losses = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            step = int(row["VLA Train/Step"])
            losses[step] = float(row["VLA Train/Loss (Raw)"])
    return losses


def load_mcore_losses(path: Path) -> dict[int, float]:
    losses = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = MCORE_LOSS_PATTERN.search(line)
            if match:
                losses[int(match.group("step"))] = float(match.group("loss"))
    return losses


def canonical_name(name: str) -> str:
    name = name.removeprefix("_fsdp_wrapped_module.")
    name = name.replace("action_out_proj.projector.weight", "action_out_proj.weight")
    return name.replace(".layer.", ".")


def load_mcore_scalars(
    path: Path, parameter_name: str
) -> dict[int, dict[str, float]]:
    scalars = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row.get("name")
            if name is not None and canonical_name(name) != parameter_name:
                continue
            step = int(row["step"])
            scalars.setdefault(step, row)
    return scalars


def scalar_comparison(
    fsdp_metrics: Path,
    fsdp_trace: Path,
    mcore_trace: Path,
    steps: list[int],
    parameter_name: str,
) -> dict[str, dict[str, float]]:
    fsdp = {}
    with fsdp_metrics.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            fsdp[int(row["VLA Train/Step"])] = {
                "grad_norm": float(row["VLA Train/Grad Norm"]),
                "clip_coefficient": float(row["VLA Train/Clip Coefficient"]),
            }
    mcore = load_mcore_scalars(mcore_trace, parameter_name)
    with fsdp_trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row.get("name")
            if name is not None and canonical_name(name) != parameter_name:
                continue
            step = int(row["step"])
            phase = row["phase"]
            target = fsdp.setdefault(step, {})
            if phase == "pre_clip":
                target["raw_grad_norm"] = row["grad_norm"]
            elif phase == "post_update":
                target.update(
                    {
                        "post_clip_grad_norm": row["grad_norm"],
                        "parameter_norm": row["parameter_norm"],
                        "parameter_delta_norm": row["parameter_delta_norm"],
                        "exp_avg_norm": row["exp_avg_norm"],
                        "exp_avg_sq_norm": row["exp_avg_sq_norm"],
                    }
                )
    result = {}
    fields = (
        "grad_norm",
        "clip_coefficient",
        "raw_grad_norm",
        "post_clip_grad_norm",
        "parameter_norm",
        "parameter_delta_norm",
        "exp_avg_norm",
        "exp_avg_sq_norm",
    )
    for field in fields:
        differences = [abs(fsdp[step][field] - mcore[step][field]) for step in steps]
        result[field] = {
            "max_abs": max(differences),
            "max_abs_step": steps[differences.index(max(differences))],
            "mae": math.fsum(differences) / len(differences),
        }
    return result


def load_parameters(path: Path) -> dict[str, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    return {canonical_name(row["name"]): row for row in artifact["parameters"]}


def tensor_comparison(
    fsdp_root: Path, mcore_root: Path, steps: list[int]
) -> dict[str, Any]:
    fields = {
        name: {"max_abs": 0.0, "first_bad_step": None, "all_exact": True}
        for name in TENSOR_FIELDS
    }
    compared_parameters = 0
    for step in steps:
        fsdp_path = fsdp_root / f"fsdp_step_{step:06d}_rank_00.pt"
        mcore_path = mcore_root / f"mcore_step_{step:06d}_rank_00.pt"
        if not fsdp_path.is_file() or not mcore_path.is_file():
            raise FileNotFoundError(f"missing paired optimizer trace at step {step}")
        fsdp_parameters = load_parameters(fsdp_path)
        mcore_parameters = load_parameters(mcore_path)
        common_names = sorted(fsdp_parameters.keys() & mcore_parameters.keys())
        if not common_names:
            raise ValueError(f"no common traced parameters at step {step}")
        compared_parameters += len(common_names)
        for name in common_names:
            for field in TENSOR_FIELDS:
                lhs = fsdp_parameters[name].get(field)
                rhs = mcore_parameters[name].get(field)
                if lhs is None or rhs is None:
                    if lhs is not rhs:
                        raise ValueError(
                            f"unpaired {field} tensor for {name} at step {step}"
                        )
                    continue
                lhs = lhs.reshape(-1)
                rhs = rhs.reshape(-1)
                if lhs.shape != rhs.shape:
                    raise ValueError(
                        f"{field} shape mismatch for {name} at step {step}: "
                        f"{tuple(lhs.shape)} != {tuple(rhs.shape)}"
                    )
                exact = torch.equal(lhs, rhs)
                max_abs = float((lhs - rhs).abs().max().item())
                fields[field]["max_abs"] = max(fields[field]["max_abs"], max_abs)
                if not exact:
                    fields[field]["all_exact"] = False
                    if fields[field]["first_bad_step"] is None:
                        fields[field]["first_bad_step"] = step
    return {"compared_parameter_steps": compared_parameters, "fields": fields}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    fsdp_losses = load_fsdp_losses(args.fsdp_metrics)
    mcore_losses = load_mcore_losses(args.mcore_log)
    paired_steps = fsdp_losses.keys() & mcore_losses.keys()
    steps = list(range(1, args.require_steps + 1))
    missing_steps = [step for step in steps if step not in paired_steps]
    if missing_steps:
        raise ValueError(
            f"expected steps 1..{args.require_steps}; missing paired steps {missing_steps}"
        )
    differences = [abs(fsdp_losses[step] - mcore_losses[step]) for step in steps]
    fsdp_values = torch.tensor(
        [fsdp_losses[step] for step in steps], dtype=torch.float64
    )
    mcore_values = torch.tensor(
        [mcore_losses[step] for step in steps], dtype=torch.float64
    )
    pearson = (
        1.0
        if len(steps) == 1 and differences[0] == 0.0
        else float(
            torch.corrcoef(torch.stack((fsdp_values, mcore_values)))[0, 1].item()
        )
    )
    max_abs = max(differences)
    result = {
        "status": "PASS" if max_abs < args.loss_atol else "FAIL",
        "required_steps": args.require_steps,
        "loss_atol": args.loss_atol,
        "loss": {
            "mae": math.fsum(differences) / len(differences),
            "max_abs": max_abs,
            "max_abs_step": steps[differences.index(max_abs)],
            "pearson": pearson,
            "steps_ge_atol": [
                step
                for step, difference in zip(steps, differences)
                if difference >= args.loss_atol
            ],
        },
        "optimizer_scalars": scalar_comparison(
            args.fsdp_metrics,
            args.fsdp_param_trace,
            args.mcore_param_trace,
            steps,
            getattr(args, "parameter_name", "action_out_proj.weight"),
        ),
    }
    if not args.skip_tensors:
        if args.fsdp_tensors is None or args.mcore_tensors is None:
            raise ValueError(
                "tensor directories are required unless --skip-tensors is set"
            )
        result["optimizer_tensors"] = tensor_comparison(
            args.fsdp_tensors, args.mcore_tensors, steps
        )
    return result


def main() -> None:
    args = parse_args()
    result = compare(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
