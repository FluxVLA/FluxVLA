import importlib.util
from pathlib import Path

import pytest
import torch


def _load_comparator():
    path = Path(__file__).parents[1] / "tools" / "compare_lingbot_backend_alignment.py"
    spec = importlib.util.spec_from_file_location("compare_lingbot_backend_alignment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload(backend: str, name: str | None, values: list[torch.Tensor]):
    traces = []
    for step, value in enumerate(values, 1):
        traces.append(
            {
                "step": step,
                "sample_index": step - 1,
                "sample_identity": {"sample": step - 1},
                "losses": {"loss": float(step)},
                "grad_norm": float(step),
                "clip_coefficient": 1.0,
                **{
                    field: value.clone()
                    for field in (
                        "parameter_before",
                        "raw_grad",
                        "post_clip_grad",
                        "parameter_after",
                        "exp_avg",
                        "exp_avg_sq",
                    )
                },
            }
        )
    payload = {"backend": backend, "traces": traces}
    if name is not None:
        payload["trace_parameter"] = {
            "name": name,
            "shape": list(values[0].shape),
            "numel": values[0].numel(),
        }
    return payload


def test_compare_reports_first_threshold_breach_and_parameter_offset():
    comparator = _load_comparator()
    reference = _payload(
        "ddp",
        "module.model.action_out_proj.weight",
        [torch.zeros(2, 2), torch.zeros(2, 2)],
    )
    candidate_values = [torch.zeros(2, 2), torch.zeros(2, 2)]
    candidate_values[0][0, 1] = 5e-4
    candidate_values[1][1, 0] = 2e-3
    candidate = _payload(
        "fsdp",
        "_fsdp_wrapped_module.model.action_out_proj.weight",
        candidate_values,
    )

    result = comparator.compare_payloads(
        reference,
        candidate,
        loss_atol=0.0,
        tensor_atol=1e-3,
        first_bad_atol=1e-3,
    )

    raw_grad = result["tensor_fields"]["raw_grad"]
    assert result["trace_parameter"]["name"] == "model.action_out_proj.weight"
    first_bad = raw_grad["first_threshold_breach"]
    assert first_bad == {
        "step": 2,
        "max_abs_diff": first_bad["max_abs_diff"],
        "flat_index": 2,
        "parameter_name": "model.action_out_proj.weight",
        "parameter_offset": 2,
        "tensor_index": [1, 0],
    }
    assert first_bad["max_abs_diff"] == pytest.approx(2e-3)
    assert raw_grad["max_difference"] == raw_grad["first_threshold_breach"]
    assert result["loss_fields"]["loss"]["max_abs_diff"] == 0.0
    assert result["loss_fields"]["loss"]["threshold_breach_count"] == 0
    assert result["loss_fields"]["loss"]["correlation"] == pytest.approx(1.0)
    assert not result["passed"]


def test_compare_marks_legacy_trace_metadata_as_unavailable():
    comparator = _load_comparator()
    value = torch.zeros(2, 2)
    result = comparator.compare_payloads(
        _payload("ddp", None, [value]),
        _payload("fsdp", None, [value]),
        loss_atol=0.0,
        tensor_atol=0.0,
    )

    assert result["trace_parameter"] == {
        "name": None,
        "shape": [2, 2],
        "numel": 4,
        "metadata_available": False,
    }
    assert result["passed"]


def test_compare_reports_parameter_probe_difference():
    comparator = _load_comparator()
    value = torch.zeros(2)
    reference = _payload("ddp", None, [value])
    candidate = _payload("fsdp", None, [value])
    reference["traces"][0]["parameter_probes"] = {
        "module.model.norm.weight": {
            "parameter_before": value.clone(),
            "raw_grad": value.clone(),
            "parameter_after": value.clone(),
        }
    }
    candidate["traces"][0]["parameter_probes"] = {
        "_fsdp_wrapped_module.model.norm.weight": {
            "parameter_before": value.clone(),
            "raw_grad": torch.tensor([0.0, 0.25]),
            "parameter_after": value.clone(),
        }
    }

    result = comparator.compare_payloads(
        reference,
        candidate,
        loss_atol=0.0,
        tensor_atol=1e-3,
    )

    first_bad = result["parameter_probes"]["model.norm.weight"]["raw_grad"][
        "first_threshold_breach"
    ]
    assert first_bad["step"] == 1
    assert first_bad["parameter_name"] == "model.norm.weight"
    assert first_bad["parameter_offset"] == 1
    assert first_bad["max_abs_diff"] == 0.25
    assert not result["passed"]
