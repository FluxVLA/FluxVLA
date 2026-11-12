"""Compare tensor-level traces emitted by ``lingbot_backend_alignment.py``."""

from __future__ import annotations

import argparse
import json
import math
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


def _canonical_parameter_name(name: str) -> str:
    prefixes = ("module.", "_fsdp_wrapped_module.")
    while any(name.startswith(prefix) for prefix in prefixes):
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
    return name


def _trace_parameter(payload: dict[str, Any], tensor: torch.Tensor) -> dict[str, Any]:
    metadata = payload.get("trace_parameter")
    if metadata is None:
        return {
            "name": None,
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "metadata_available": False,
        }
    result = dict(metadata)
    result["name"] = _canonical_parameter_name(str(result["name"]))
    result["shape"] = list(result["shape"])
    result["numel"] = int(result["numel"])
    result["metadata_available"] = True
    if result["numel"] != tensor.numel() or result["shape"] != list(tensor.shape):
        raise RuntimeError(
            "trace_parameter metadata does not match tensor: "
            f"metadata={result}, tensor_shape={list(tensor.shape)}"
        )
    return result


def _difference_location(
    difference: torch.Tensor,
    *,
    step: int,
    parameter_name: str | None,
) -> dict[str, Any]:
    flat = difference.reshape(-1)
    flat_index = int(flat.argmax())
    index = list(torch.unravel_index(torch.tensor(flat_index), difference.shape))
    return {
        "step": step,
        "max_abs_diff": float(flat[flat_index]),
        "flat_index": flat_index,
        "parameter_name": parameter_name,
        "parameter_offset": flat_index,
        "tensor_index": [int(value) for value in index],
    }


def _probe_tensors(trace: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        _canonical_parameter_name(name): tensors
        for name, tensors in trace.get("parameter_probes", {}).items()
    }


def compare_payloads(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    loss_atol: float,
    tensor_atol: float,
    first_bad_atol: float | None = None,
    step_start: int | None = None,
    step_end: int | None = None,
) -> dict[str, Any]:
    if first_bad_atol is None:
        first_bad_atol = tensor_atol

    def selected_traces(payload):
        return [
            trace
            for trace in payload["traces"]
            if (step_start is None or trace["step"] >= step_start)
            and (step_end is None or trace["step"] <= step_end)
        ]

    reference_traces = selected_traces(reference)
    candidate_traces = selected_traces(candidate)
    if not reference_traces:
        raise RuntimeError("No traces selected")
    if len(reference_traces) != len(candidate_traces):
        raise RuntimeError("Trace lengths differ")

    reference_parameter = _trace_parameter(reference, reference_traces[0][TENSOR_FIELDS[0]])
    candidate_parameter = _trace_parameter(candidate, candidate_traces[0][TENSOR_FIELDS[0]])
    if reference_parameter["shape"] != candidate_parameter["shape"]:
        raise RuntimeError("Trace parameter shapes differ")
    if (
        reference_parameter["name"] is not None
        and candidate_parameter["name"] is not None
        and reference_parameter["name"] != candidate_parameter["name"]
    ):
        raise RuntimeError(
            "Trace parameter names differ: "
            f"{reference_parameter['name']} vs {candidate_parameter['name']}"
        )
    parameter_name = reference_parameter["name"] or candidate_parameter["name"]
    loss_series: dict[str, dict[str, list[float]]] = {}

    result = {
        "reference": reference["backend"],
        "candidate": candidate["backend"],
        "steps": [trace["step"] for trace in reference_traces],
        "trace_parameter": {
            "name": parameter_name,
            "shape": reference_parameter["shape"],
            "numel": reference_parameter["numel"],
            "metadata_available": reference_parameter["metadata_available"]
            and candidate_parameter["metadata_available"],
        },
        "first_bad_atol": first_bad_atol,
        "max_loss_abs_diff": 0.0,
        "loss_fields": {},
        "max_grad_norm_abs_diff": 0.0,
        "max_clip_coefficient_abs_diff": 0.0,
        "sample_sequence_equal": True,
        "tensor_fields": {
            field: {
                "max_abs_diff": 0.0,
                "bitwise_equal": True,
                "first_threshold_breach": None,
                "max_difference": None,
            }
            for field in TENSOR_FIELDS
        },
        "parameter_probes": {},
    }
    for reference_step, candidate_step in zip(reference_traces, candidate_traces, strict=True):
        if reference_step["step"] != candidate_step["step"]:
            raise RuntimeError("Step numbers differ")
        step = int(reference_step["step"])
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
            reference_value = float(reference_step["losses"][key])
            candidate_value = float(candidate_step["losses"][key])
            series = loss_series.setdefault(
                key,
                {"reference": [], "candidate": [], "abs_diff": []},
            )
            series["reference"].append(reference_value)
            series["candidate"].append(candidate_value)
            series["abs_diff"].append(abs(reference_value - candidate_value))
            result["max_loss_abs_diff"] = max(
                result["max_loss_abs_diff"],
                series["abs_diff"][-1],
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
            if lhs.shape != rhs.shape:
                raise RuntimeError(f"{field} tensor shapes differ at step {step}")
            difference = (lhs - rhs).abs()
            location = _difference_location(
                difference,
                step=step,
                parameter_name=parameter_name,
            )
            field_result = result["tensor_fields"][field]
            field_result["bitwise_equal"] &= torch.equal(lhs, rhs)
            if location["max_abs_diff"] > field_result["max_abs_diff"]:
                field_result["max_abs_diff"] = location["max_abs_diff"]
                field_result["max_difference"] = location
            if (
                field_result["first_threshold_breach"] is None
                and location["max_abs_diff"] > first_bad_atol
            ):
                field_result["first_threshold_breach"] = location

        reference_probes = _probe_tensors(reference_step)
        candidate_probes = _probe_tensors(candidate_step)
        if set(reference_probes) != set(candidate_probes):
            raise RuntimeError(
                "Parameter probe names differ: "
                f"{sorted(reference_probes)} vs {sorted(candidate_probes)}"
            )
        for name in sorted(reference_probes):
            probe_result = result["parameter_probes"].setdefault(name, {})
            reference_fields = reference_probes[name]
            candidate_fields = candidate_probes[name]
            if set(reference_fields) != set(candidate_fields):
                raise RuntimeError(f"Parameter probe fields differ for {name}")
            for field in sorted(reference_fields):
                lhs, rhs = reference_fields[field], candidate_fields[field]
                if lhs.shape != rhs.shape:
                    raise RuntimeError(f"Probe {name} {field} shapes differ at step {step}")
                difference = (lhs - rhs).abs()
                location = _difference_location(
                    difference,
                    step=step,
                    parameter_name=name,
                )
                field_result = probe_result.setdefault(
                    field,
                    {
                        "max_abs_diff": 0.0,
                        "bitwise_equal": True,
                        "first_threshold_breach": None,
                        "max_difference": None,
                    },
                )
                field_result["bitwise_equal"] &= torch.equal(lhs, rhs)
                if location["max_abs_diff"] > field_result["max_abs_diff"]:
                    field_result["max_abs_diff"] = location["max_abs_diff"]
                    field_result["max_difference"] = location
                if (
                    field_result["first_threshold_breach"] is None
                    and location["max_abs_diff"] > first_bad_atol
                ):
                    field_result["first_threshold_breach"] = location

    for key, series in sorted(loss_series.items()):
        differences = series["abs_diff"]
        maximum_index = max(range(len(differences)), key=differences.__getitem__)
        ordered = sorted(differences)
        reference_mean = sum(series["reference"]) / len(series["reference"])
        candidate_mean = sum(series["candidate"]) / len(series["candidate"])
        covariance = sum(
            (reference - reference_mean) * (candidate - candidate_mean)
            for reference, candidate in zip(
                series["reference"], series["candidate"], strict=True
            )
        )
        variance_product = sum(
            (value - reference_mean) ** 2 for value in series["reference"]
        ) * sum((value - candidate_mean) ** 2 for value in series["candidate"])
        result["loss_fields"][key] = {
            "max_abs_diff": differences[maximum_index],
            "max_abs_diff_step": result["steps"][maximum_index],
            "mean_abs_diff": sum(differences) / len(differences),
            "median_abs_diff": ordered[len(ordered) // 2]
            if len(ordered) % 2
            else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2,
            "p95_abs_diff": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "rmse": math.sqrt(sum(value**2 for value in differences) / len(differences)),
            "correlation": covariance / math.sqrt(variance_product)
            if variance_product > 0
            else None,
            "first_threshold_breach_step": next(
                (
                    result["steps"][index]
                    for index, value in enumerate(differences)
                    if value > loss_atol
                ),
                None,
            ),
            "threshold_breach_count": sum(value > loss_atol for value in differences),
            "final_abs_diff": differences[-1],
            "reference_final_10_mean": sum(series["reference"][-10:])
            / len(series["reference"][-10:]),
            "candidate_final_10_mean": sum(series["candidate"][-10:])
            / len(series["candidate"][-10:]),
        }

    tensors_pass = all(
        value["max_abs_diff"] <= tensor_atol for value in result["tensor_fields"].values()
    )
    probes_pass = all(
        field["max_abs_diff"] <= tensor_atol
        for probe in result["parameter_probes"].values()
        for field in probe.values()
    )
    result["passed"] = (
        result["max_loss_abs_diff"] <= loss_atol
        and tensors_pass
        and probes_pass
        and result["sample_sequence_equal"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    parser.add_argument("--tensor-atol", type=float, default=0.0)
    parser.add_argument("--first-bad-atol", type=float)
    parser.add_argument("--step-start", type=int)
    parser.add_argument("--step-end", type=int)
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)

    result = compare_payloads(
        reference,
        candidate,
        loss_atol=args.loss_atol,
        tensor_atol=args.tensor_atol,
        first_bad_atol=args.first_bad_atol,
        step_start=args.step_start,
        step_end=args.step_end,
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
