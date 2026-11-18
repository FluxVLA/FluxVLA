"""Create and compare immutable LingBot checkpoint identity records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: str | Path, *, hash_weights: bool = False) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    metadata: dict[str, Any] = {"checkpoint": str(root)}
    for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        file_path = root / name
        if file_path.is_file():
            metadata[f"{name}_sha256"] = _sha256(file_path)
    files = []
    for file_path in sorted(root.glob("*.safetensors")):
        item: dict[str, Any] = {"name": file_path.name, "size": file_path.stat().st_size}
        if hash_weights:
            item["sha256"] = _sha256(file_path)
        files.append(item)
    if not files:
        raise ValueError(f"checkpoint has no safetensors files: {root}")
    metadata["weight_files"] = files
    metadata["weight_file_count"] = len(files)
    metadata["weight_total_bytes"] = sum(item["size"] for item in files)
    metadata["identity_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return metadata


def assert_same_checkpoint(left: dict[str, Any], right: dict[str, Any]) -> None:
    comparable = ("config.json_sha256", "model.safetensors.index.json_sha256", "weight_files")
    mismatches = [key for key in comparable if left.get(key) != right.get(key)]
    if mismatches:
        raise ValueError(
            "checkpoint identity mismatch in " + ", ".join(mismatches) +
            "; refuse parity comparison\n"
            f"left={left.get('checkpoint')} right={right.get('checkpoint')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+")
    parser.add_argument("--hash-weights", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [checkpoint_identity(path, hash_weights=args.hash_weights) for path in args.checkpoint]
    if len(records) == 2:
        assert_same_checkpoint(records[0], records[1])
    rendered = json.dumps(records[0] if len(records) == 1 else records, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
