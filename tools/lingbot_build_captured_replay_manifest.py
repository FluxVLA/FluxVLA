"""Build an exact replay manifest from official LingBot captured batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(source_manifest: Path, capture_root: Path) -> dict:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    entries = []
    for entry in source.get("entries", []):
        rank = int(entry["official_rank"])
        step = int(entry["official_step"])
        batch_path = (capture_root / f"batch_rank{rank}_step{step}.pt").resolve()
        flow_path = (capture_root / f"flow_rank{rank}_step{step - 1}.pt").resolve()
        if not batch_path.is_file():
            raise FileNotFoundError(f"Captured batch does not exist: {batch_path}")
        if not flow_path.is_file():
            raise FileNotFoundError(f"Captured flow does not exist: {flow_path}")
        converted = dict(entry)
        converted.update(
            path=str(batch_path),
            flow_path=str(flow_path),
            captured_batch=True,
        )
        entries.append(converted)
    if not entries:
        raise ValueError(f"Source manifest has no entries: {source_manifest}")
    return {
        **{key: value for key, value in source.items() if key != "entries"},
        "capture_root": str(capture_root.resolve()),
        "captured_batch_replay": True,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.source_manifest.resolve(), args.capture_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
