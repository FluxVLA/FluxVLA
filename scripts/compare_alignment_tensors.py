"""Compare DDP tensors with reconstructed tensor-parallel captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_manifest(path: Path) -> tuple[Path, dict[tuple[str, str, str, int], dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    events = {}
    for event in manifest["events"]:
        if "tensor_file" not in event or event["name"].endswith(".layer"):
            continue
        key = (event["stage"], event["name"], event["leaf"], event["call"])
        events[key] = event
    return path.parent, events


def _load_tensor(root: Path, event: dict) -> torch.Tensor:
    return torch.load(root / event["tensor_file"], map_location="cpu", weights_only=True)


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_float = reference.float().reshape(-1)
    candidate_float = candidate.float().reshape(-1)
    difference = (candidate_float - reference_float).abs()
    cosine = 1.0
    if reference_float.numel() and (reference_float.norm() != 0 or candidate_float.norm() != 0):
        cosine = float(F.cosine_similarity(reference_float, candidate_float, dim=0).item())
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "cosine": cosine,
        "mean_abs_error": float(difference.mean().item()) if difference.numel() else 0.0,
        "max_abs_error": float(difference.max().item()) if difference.numel() else 0.0,
    }


def _is_partitioned_intermediate(key: tuple[str, str, str, int]) -> bool:
    stage, name, _leaf, _call = key
    return (
        stage == "forward_output" and name.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.act_fn"))
    ) or (stage == "forward_input" and name.endswith("mlp.act_fn"))


def _is_row_input(key: tuple[str, str, str, int]) -> bool:
    stage, name, _leaf, _call = key
    return stage == "forward_input" and name.endswith("mlp.down_proj")


def compare(reference_path: Path, tensor_parallel_paths: list[Path]) -> dict:
    reference_root, reference_events = _load_manifest(reference_path)
    tp_manifests = [_load_manifest(path) for path in tensor_parallel_paths]
    rows = []
    missing = []

    for key, reference_event in reference_events.items():
        tp_events = [events.get(key) for _root, events in tp_manifests]
        if any(event is None for event in tp_events):
            missing.append({"key": list(key), "side": "tensor_parallel"})
            continue

        reference = _load_tensor(reference_root, reference_event)
        shards = [
            _load_tensor(root, event)
            for (root, _events), event in zip(tp_manifests, tp_events, strict=True)
        ]
        reconstruction = "replicated_rank0"
        candidate = shards[0]
        if _is_partitioned_intermediate(key) or _is_row_input(key):
            candidate = torch.cat(shards, dim=-1)
            reconstruction = "concatenate_last_dimension"

        row = {
            "stage": key[0],
            "name": key[1],
            "leaf": key[2],
            "call": key[3],
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "reconstruction": reconstruction,
            "tp_ranks_replicated": all(torch.equal(shards[0], shard) for shard in shards[1:]),
        }
        if reference.shape == candidate.shape:
            row.update(_metrics(reference, candidate))
        else:
            row.update(exact=False, cosine=None, mean_abs_error=None, max_abs_error=None)
        rows.append(row)

    first_non_exact = next((row for row in rows if not row["exact"]), None)
    return {
        "reference_manifest": str(reference_path),
        "tensor_parallel_manifests": [str(path) for path in tensor_parallel_paths],
        "compared_tensors": len(rows),
        "exact_tensors": sum(row["exact"] for row in rows),
        "missing": missing,
        "first_non_exact": first_non_exact,
        "comparisons": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tensor-parallel", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare(args.reference, args.tensor_parallel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
