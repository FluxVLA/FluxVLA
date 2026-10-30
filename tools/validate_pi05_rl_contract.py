"""Validate a Pi0.5 checkpoint bundle before RL simulation training."""

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _tensor_contract(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    contract = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():  # noqa: SIM118 - safe_open is not a Mapping
            tensor_slice = tensors.get_slice(key)
            contract[key] = (tuple(tensor_slice.get_shape()), str(tensor_slice.get_dtype()))
    return contract


def _feature_summary(config: dict[str, Any]) -> dict[str, Any]:
    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    image_features = {
        key: value.get("shape")
        for key, value in input_features.items()
        if value.get("type") == "VISUAL"
    }
    state_features = {
        key: value.get("shape")
        for key, value in input_features.items()
        if value.get("type") == "STATE"
    }
    action_features = {
        key: value.get("shape")
        for key, value in output_features.items()
        if value.get("type") == "ACTION"
    }
    return {
        "images": image_features,
        "states": state_features,
        "actions": action_features,
        "chunk_size": config.get("chunk_size"),
        "n_action_steps": config.get("n_action_steps"),
        "max_state_dim": config.get("max_state_dim"),
        "max_action_dim": config.get("max_action_dim"),
    }


def _processor_normalization_summary(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    steps = []
    for step in config.get("steps", []):
        if step.get("registry_name") not in {"normalizer_processor", "unnormalizer_processor"}:
            continue
        step_config = step.get("config", {})
        state_file = step.get("state_file")
        steps.append(
            {
                "registry_name": step.get("registry_name"),
                "feature_keys": sorted(step_config.get("features", {})),
                "norm_map": step_config.get("norm_map", {}),
                "state_file": state_file,
                "state_file_exists": bool(state_file and (path.parent / state_file).is_file()),
            }
        )
    return {"path": str(path), "normalization_steps": steps}


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str, *, warning: bool = False
) -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if passed else ("warning" if warning else "fail"),
            "detail": detail,
        }
    )


def _validate_norm_stats(
    norm_stats_path: Path,
    max_state_dim: int,
    max_action_dim: int,
    env_state_dim: int,
    env_action_dim: int,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _load_json(norm_stats_path)
    stats = payload.get("norm_stats", payload)
    state = stats.get("state", stats.get("proprio", {}))
    actions = stats.get("actions", stats.get("action", {}))

    state_length = len(state.get("q01", state.get("mean", [])))
    action_length = len(actions.get("q01", actions.get("mean", [])))
    _add_check(
        checks,
        "norm_stats_state_dim",
        state_length == max_state_dim and env_state_dim <= state_length,
        f"stats={state_length}, model={max_state_dim}, env={env_state_dim}",
    )
    _add_check(
        checks,
        "norm_stats_action_dim",
        action_length == max_action_dim and env_action_dim <= action_length,
        f"stats={action_length}, model={max_action_dim}, env={env_action_dim}",
    )

    def tail_is_zero(feature_stats: dict[str, Any], start: int) -> bool:
        arrays = [feature_stats.get(key, []) for key in ("mean", "std", "q01", "q99")]
        return all(all(float(value) == 0.0 for value in array[start:]) for array in arrays if array)

    state_padding_zero = tail_is_zero(state, env_state_dim)
    action_padding_zero = tail_is_zero(actions, env_action_dim)
    _add_check(
        checks,
        "norm_stats_padding",
        state_padding_zero and action_padding_zero,
        f"state_tail_zero={state_padding_zero}, action_tail_zero={action_padding_zero}",
    )
    return {
        "path": str(norm_stats_path),
        "state_length": state_length,
        "action_length": action_length,
        "env_state_dim": env_state_dim,
        "env_action_dim": env_action_dim,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_dir = args.checkpoint_dir.resolve()
    model_path = checkpoint_dir / "model.safetensors"
    config_path = checkpoint_dir / "config.json"
    preprocessor_path = checkpoint_dir / "policy_preprocessor.json"
    postprocessor_path = checkpoint_dir / "policy_postprocessor.json"
    checks: list[dict[str, Any]] = []

    for name, path in (
        ("model", model_path),
        ("config", config_path),
        ("preprocessor", preprocessor_path),
        ("postprocessor", postprocessor_path),
    ):
        required = name in {"model", "config"} or args.require_processors
        _add_check(
            checks,
            f"checkpoint_{name}_exists",
            path.is_file(),
            str(path),
            warning=not required,
        )

    if not model_path.is_file() or not config_path.is_file():
        return {"overall": "fail", "checks": checks}

    config = _load_json(config_path)
    features = _feature_summary(config)
    max_state_dim = int(features.get("max_state_dim") or args.model_state_dim)
    max_action_dim = int(features.get("max_action_dim") or args.model_action_dim)
    model_contract = _tensor_contract(model_path)
    _add_check(checks, "model_has_tensors", bool(model_contract), f"tensor_count={len(model_contract)}")

    template_summary = None
    if args.template is not None:
        template_contract = _tensor_contract(args.template.resolve())
        missing = sorted(set(template_contract) - set(model_contract))
        unexpected = sorted(set(model_contract) - set(template_contract))
        mismatched = sorted(
            key
            for key in set(model_contract) & set(template_contract)
            if model_contract[key] != template_contract[key]
        )
        _add_check(
            checks,
            "template_tensor_contract",
            not missing and not unexpected and not mismatched,
            f"missing={len(missing)}, unexpected={len(unexpected)}, shape_or_dtype={len(mismatched)}",
        )
        template_summary = {
            "path": str(args.template.resolve()),
            "tensor_count": len(template_contract),
            "missing": missing,
            "unexpected": unexpected,
            "shape_or_dtype_mismatches": mismatched,
        }

    processor_summaries = []
    for path in (preprocessor_path, postprocessor_path):
        if not path.is_file():
            continue
        summary = _processor_normalization_summary(path)
        processor_summaries.append(summary)
        for step in summary["normalization_steps"]:
            has_features = bool(step["feature_keys"])
            has_state = step["state_file_exists"]
            _add_check(
                checks,
                f'{step["registry_name"]}_has_embedded_stats',
                has_features and has_state,
                f"features={step['feature_keys']}, state_file={step['state_file']}",
                warning=args.norm_stats is not None,
            )

    norm_stats_summary = None
    if args.norm_stats is not None:
        norm_stats_summary = _validate_norm_stats(
            args.norm_stats.resolve(),
            max_state_dim,
            max_action_dim,
            args.env_state_dim,
            args.env_action_dim,
            checks,
        )

    overall = "fail" if any(check["status"] == "fail" for check in checks) else "pass"
    return {
        "overall": overall,
        "checkpoint_dir": str(checkpoint_dir),
        "model_tensor_count": len(model_contract),
        "features": features,
        "template": template_summary,
        "processors": processor_summaries,
        "norm_stats": norm_stats_summary,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--env-state-dim", type=int, default=8)
    parser.add_argument("--env-action-dim", type=int, default=7)
    parser.add_argument("--model-state-dim", type=int, default=32)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--require-processors", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(args)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
