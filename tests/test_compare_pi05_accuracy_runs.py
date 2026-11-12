import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import torch


def _load_comparator():
    path = Path(__file__).parents[1] / "tools" / "compare_pi05_accuracy_runs.py"
    spec = importlib.util.spec_from_file_location("compare_pi05_accuracy_runs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_fixed_replay_accuracy_artifacts(tmp_path):
    comparator = _load_comparator()
    fsdp_tensors = tmp_path / "fsdp_tensors"
    mcore_tensors = tmp_path / "mcore_tensors"
    fsdp_tensors.mkdir()
    mcore_tensors.mkdir()
    metrics = tmp_path / "fsdp.jsonl"
    mcore_log = tmp_path / "mcore.log"
    mcore_trace = tmp_path / "mcore_trace.jsonl"
    fsdp_trace = tmp_path / "fsdp_trace.jsonl"

    metrics.write_text(
        json.dumps(
            {
                "VLA Train/Step": 1,
                "VLA Train/Loss (Raw)": 0.125,
                "VLA Train/Grad Norm": 2.0,
                "VLA Train/Clip Coefficient": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mcore_log.write_text(
        "iteration 1/ 1 | lm_loss: 1.250000E-01 | grad norm: 2.000\n",
        encoding="utf-8",
    )
    mcore_trace.write_text(
        json.dumps(
            {
                "step": 1,
                "grad_norm": 2.0,
                "clip_coefficient": 0.5,
                "raw_grad_norm": 3.0,
                "post_clip_grad_norm": 1.5,
                "parameter_norm": 4.0,
                "parameter_delta_norm": 0.1,
                "exp_avg_norm": 0.2,
                "exp_avg_sq_norm": 0.3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fsdp_trace.write_text(
        "\n".join(
            (
                json.dumps({"step": 1, "phase": "pre_clip", "grad_norm": 3.0}),
                json.dumps(
                    {
                        "step": 1,
                        "phase": "post_update",
                        "grad_norm": 1.5,
                        "parameter_norm": 4.0,
                        "parameter_delta_norm": 0.1,
                        "exp_avg_norm": 0.2,
                        "exp_avg_sq_norm": 0.3,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    fields = {name: tensor.clone() for name in comparator.TENSOR_FIELDS}
    torch.save(
        {
            "parameters": [
                {
                    "name": "_fsdp_wrapped_module.action_out_proj.projector.weight",
                    **fields,
                }
            ]
        },
        fsdp_tensors / "fsdp_step_000001_rank_00.pt",
    )
    torch.save(
        {
            "parameters": [
                {
                    "name": "action_out_proj.weight",
                    **{name: value.reshape(-1) for name, value in fields.items()},
                }
            ]
        },
        mcore_tensors / "mcore_step_000001_rank_00.pt",
    )

    result = comparator.compare(
        Namespace(
            fsdp_metrics=metrics,
            fsdp_param_trace=fsdp_trace,
            mcore_log=mcore_log,
            mcore_param_trace=mcore_trace,
            fsdp_tensors=fsdp_tensors,
            mcore_tensors=mcore_tensors,
            skip_tensors=False,
            require_steps=1,
            loss_atol=1.0e-3,
        )
    )

    assert result["status"] == "PASS"
    assert result["loss"]["max_abs"] == 0.0
    assert result["optimizer_scalars"]["grad_norm"]["max_abs"] == 0.0
    assert all(
        field["all_exact"] for field in result["optimizer_tensors"]["fields"].values()
    )


def test_scalar_comparison_selects_the_requested_parameter(tmp_path):
    comparator = _load_comparator()
    metrics = tmp_path / "fsdp.jsonl"
    fsdp_trace = tmp_path / "fsdp_trace.jsonl"
    mcore_trace = tmp_path / "mcore_trace.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "VLA Train/Step": 1,
                "VLA Train/Grad Norm": 2.0,
                "VLA Train/Clip Coefficient": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fsdp_rows = []
    mcore_rows = []
    for name, value in (("action_out_proj.projector.weight", 3.0), ("action_out_proj.projector.bias", 9.0)):
        fsdp_rows.extend(
            (
                {"step": 1, "phase": "pre_clip", "name": name, "grad_norm": value},
                {
                    "step": 1,
                    "phase": "post_update",
                    "name": name,
                    "grad_norm": value / 2,
                    "parameter_norm": 4.0,
                    "parameter_delta_norm": 0.1,
                    "exp_avg_norm": 0.2,
                    "exp_avg_sq_norm": 0.3,
                },
            )
        )
        mcore_rows.append(
            {
                "step": 1,
                "name": name.replace(".projector", ""),
                "grad_norm": 2.0,
                "clip_coefficient": 0.5,
                "raw_grad_norm": value,
                "post_clip_grad_norm": value / 2,
                "parameter_norm": 4.0,
                "parameter_delta_norm": 0.1,
                "exp_avg_norm": 0.2,
                "exp_avg_sq_norm": 0.3,
            }
        )
    fsdp_trace.write_text(
        "\n".join(json.dumps(row) for row in fsdp_rows) + "\n", encoding="utf-8"
    )
    mcore_trace.write_text(
        "\n".join(json.dumps(row) for row in reversed(mcore_rows)) + "\n",
        encoding="utf-8",
    )

    result = comparator.scalar_comparison(
        metrics,
        fsdp_trace,
        mcore_trace,
        [1],
        "action_out_proj.weight",
    )

    assert all(field["max_abs"] == 0.0 for field in result.values())
