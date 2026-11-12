import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).parents[1] / "tools" / "validate_pi05_rl_contract.py"
SPEC = importlib.util.spec_from_file_location("validate_pi05_rl_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_bundle(root: Path, *, tensor_shape: tuple[int, ...] = (2, 2)) -> Path:
    root.mkdir()
    save_file({"weight": torch.ones(tensor_shape)}, root / "model.safetensors")
    config = {
        "input_features": {
            "observation.images.base": {"type": "VISUAL", "shape": [3, 16, 16]},
            "observation.state": {"type": "STATE", "shape": [32]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [32]}},
        "chunk_size": 10,
        "n_action_steps": 10,
        "max_state_dim": 32,
        "max_action_dim": 32,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return root


def _write_norm_stats(path: Path) -> Path:
    state_q01 = [float(index) for index in range(8)] + [0.0] * 24
    action_q01 = [float(index) for index in range(7)] + [0.0] * 25
    payload = {
        "norm_stats": {
            "state": {"q01": state_q01, "q99": state_q01, "mean": state_q01, "std": state_q01},
            "actions": {
                "q01": action_q01,
                "q99": action_q01,
                "mean": action_q01,
                "std": action_q01,
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(bundle: Path, template: Path, norm_stats: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=bundle,
        template=template,
        norm_stats=norm_stats,
        env_state_dim=8,
        env_action_dim=7,
        model_state_dim=32,
        model_action_dim=32,
        require_processors=False,
        report=None,
    )


def test_matching_contract_and_zero_padding_pass(tmp_path):
    bundle = _write_bundle(tmp_path / "bundle")
    template = _write_bundle(tmp_path / "template") / "model.safetensors"
    norm_stats = _write_norm_stats(tmp_path / "norm_stats.json")

    report = MODULE.validate(_args(bundle, template, norm_stats))

    assert report["overall"] == "pass"
    assert report["model_tensor_count"] == 1


def test_tensor_shape_mismatch_fails(tmp_path):
    bundle = _write_bundle(tmp_path / "bundle")
    template = _write_bundle(tmp_path / "template", tensor_shape=(3, 2)) / "model.safetensors"
    norm_stats = _write_norm_stats(tmp_path / "norm_stats.json")

    report = MODULE.validate(_args(bundle, template, norm_stats))

    assert report["overall"] == "fail"
    check = next(item for item in report["checks"] if item["name"] == "template_tensor_contract")
    assert check["status"] == "fail"
