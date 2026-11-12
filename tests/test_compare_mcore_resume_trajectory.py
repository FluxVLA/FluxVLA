import importlib.util
import json
from pathlib import Path

import pytest
import torch


def _load_comparator():
    path = Path(__file__).parents[1] / "tools" / "compare_mcore_resume_trajectory.py"
    spec = importlib.util.spec_from_file_location(
        "compare_mcore_resume_trajectory", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_step(root: Path, step: int, value: float) -> None:
    (root / "optimizer_tensors").mkdir(parents=True, exist_ok=True)
    (root / "param_trace").mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor([value])
    fields = {
        name: tensor.clone()
        for name in (
            "parameter_before",
            "raw_grad",
            "post_clip_grad",
            "parameter_after",
            "exp_avg",
            "exp_avg_sq",
        )
    }
    torch.save(
        {"parameters": [{"name": "weight", **fields}]},
        root / "optimizer_tensors" / f"mcore_step_{step:06d}_rank_00.pt",
    )
    scalar_row = {
        "step": step,
        "name": "weight",
        **{
            name: value
            for name in (
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
        },
    }
    with (root / "param_trace" / "native_rank0.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(scalar_row) + "\n")
    with (root / "launch.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"iteration {step}/ 2 | learning rate: {value:.6E} | "
            f"lm_loss: {value:.6E} | grad norm: {value:.6E} |\n"
        )


def test_resume_trajectory_is_exact_across_phases(tmp_path):
    comparator = _load_comparator()
    continuous = tmp_path / "continuous"
    phase1 = tmp_path / "phase1"
    phase2 = tmp_path / "phase2"
    for step in (1, 2):
        _write_step(continuous, step, float(step))
    _write_step(phase1, 1, 1.0)
    _write_step(phase2, 2, 2.0)

    result = comparator.compare(continuous, [phase1, phase2], 2, 1)

    assert result["status"] == "PASS"
    assert all(field["all_exact"] for field in result["log_scalars"].values())
    assert all(field["all_exact"] for field in result["optimizer_scalars"].values())
    assert all(field["all_exact"] for field in result["optimizer_tensors"].values())


def test_resume_trajectory_rejects_duplicate_phase_steps(tmp_path):
    comparator = _load_comparator()
    continuous = tmp_path / "continuous"
    phase1 = tmp_path / "phase1"
    phase2 = tmp_path / "phase2"
    _write_step(continuous, 1, 1.0)
    _write_step(phase1, 1, 1.0)
    _write_step(phase2, 1, 1.0)

    with pytest.raises(ValueError, match="duplicate log steps"):
        comparator.compare(continuous, [phase1, phase2], 1, 1)


def test_resume_trajectory_reports_first_tensor_difference(tmp_path):
    comparator = _load_comparator()
    continuous = tmp_path / "continuous"
    phase = tmp_path / "phase"
    _write_step(continuous, 1, 1.0)
    _write_step(phase, 1, 1.0)
    path = phase / "optimizer_tensors" / "mcore_step_000001_rank_00.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    artifact["parameters"][0]["exp_avg"].add_(0.25)
    torch.save(artifact, path)

    result = comparator.compare(continuous, [phase], 1, 1)

    assert result["status"] == "FAIL"
    assert result["optimizer_tensors"]["exp_avg"]["max_abs"] == 0.25
    assert result["optimizer_tensors"]["exp_avg"]["first_bad"]["step"] == 1
