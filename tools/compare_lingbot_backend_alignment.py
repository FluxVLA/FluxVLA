"""Compare tensor-level traces emitted by ``lingbot_backend_alignment.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TENSOR_FIELDS = (
    "parameter_before",
    "raw_grad",
    "post_clip_grad",
    "parameter_after",
    "exp_avg",
    "exp_avg_sq",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    parser.add_argument("--tensor-atol", type=float, default=0.0)
    parser.add_argument("--step-start", type=int)
    parser.add_argument("--step-end", type=int)
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)

    def selected_traces(payload):
        return [
            trace
            for trace in payload["traces"]
            if (args.step_start is None or trace["step"] >= args.step_start)
            and (args.step_end is None or trace["step"] <= args.step_end)
        ]

    reference_traces = selected_traces(reference)
    candidate_traces = selected_traces(candidate)
    if not reference_traces:
        raise RuntimeError("No traces selected")
    if len(reference_traces) != len(candidate_traces):
        raise RuntimeError("Trace lengths differ")

    result = {
        "reference": reference["backend"],
        "candidate": candidate["backend"],
        "steps": [trace["step"] for trace in reference_traces],
        "max_loss_abs_diff": 0.0,
        "max_grad_norm_abs_diff": 0.0,
        "max_clip_coefficient_abs_diff": 0.0,
        "sample_sequence_equal": True,
        "tensor_fields": {field: {"max_abs_diff": 0.0, "bitwise_equal": True} for field in TENSOR_FIELDS},
    }
    for reference_step, candidate_step in zip(reference_traces, candidate_traces, strict=True):
        if reference_step["step"] != candidate_step["step"]:
            raise RuntimeError("Step numbers differ")
        result["sample_sequence_equal"] &= (
            reference_step.get("sample_index") == candidate_step.get("sample_index")
            and reference_step.get("sample_identity") == candidate_step.get("sample_identity")
        )
        reference_loss_keys = set(reference_step["losses"])
        candidate_loss_keys = set(candidate_step["losses"])
        if reference_loss_keys != candidate_loss_keys:
            raise RuntimeError(
                f"Loss keys differ: {sorted(reference_loss_keys)} vs {sorted(candidate_loss_keys)}"
            )
        for key in reference_loss_keys:
            result["max_loss_abs_diff"] = max(
                result["max_loss_abs_diff"],
                abs(reference_step["losses"][key] - candidate_step["losses"][key]),
            )
        result["max_grad_norm_abs_diff"] = max(
            result["max_grad_norm_abs_diff"],
            abs(reference_step["grad_norm"] - candidate_step["grad_norm"]),
        )
        result["max_clip_coefficient_abs_diff"] = max(
            result["max_clip_coefficient_abs_diff"],
            abs(reference_step["clip_coefficient"] - candidate_step["clip_coefficient"]),
        )
        for field in TENSOR_FIELDS:
            lhs, rhs = reference_step[field], candidate_step[field]
            field_result = result["tensor_fields"][field]
            field_result["bitwise_equal"] &= torch.equal(lhs, rhs)
            field_result["max_abs_diff"] = max(
                field_result["max_abs_diff"],
                float((lhs - rhs).abs().max()),
            )

    tensors_pass = all(
        value["max_abs_diff"] <= args.tensor_atol
        for value in result["tensor_fields"].values()
    )
    result["passed"] = (
        result["max_loss_abs_diff"] <= args.loss_atol
        and tensors_pass
        and result["sample_sequence_equal"]
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
