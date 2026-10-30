"""Build a topology-independent LingBot replay for GB64 parity runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6400)
    parser.add_argument("--logical-world-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.logical_world_size <= 0:
        raise ValueError("samples and logical-world-size must be positive")
    if args.samples % args.logical_world_size:
        raise ValueError("samples must be divisible by logical-world-size")

    source_path = args.source.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    entries = source.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest has no entries")
    if any(not entry.get("flow_path") for entry in entries):
        raise ValueError("every source entry must contain a captured flow_path")
    ordered = sorted(
        entries,
        key=lambda entry: (
            int(entry["official_step"]),
            int(entry["official_rank"]),
        ),
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flow_root = output_path.parent / f"{output_path.stem}_flow_aliases"
    flow_root.mkdir(parents=True, exist_ok=True)

    expanded = []
    for position in range(args.samples):
        original = ordered[position % len(ordered)]
        source_flow = Path(original["flow_path"]).expanduser().resolve()
        if not source_flow.is_file():
            raise FileNotFoundError(source_flow)
        flow_alias = flow_root / f"flow_{position:06d}.pt"
        if flow_alias.is_symlink():
            if flow_alias.resolve() != source_flow:
                flow_alias.unlink()
        elif flow_alias.exists():
            raise FileExistsError(flow_alias)
        if not flow_alias.exists():
            os.symlink(source_flow, flow_alias)

        entry = dict(original)
        entry.update(
            cache_key=f"gb64-{position:06d}-{original['cache_key']}",
            flow_path=str(flow_alias),
            official_step=position // args.logical_world_size,
            official_rank=position % args.logical_world_size,
            sampler_position=position,
            source_cache_key=original["cache_key"],
            source_official_step=original["official_step"],
            source_official_rank=original["official_rank"],
        )
        expanded.append(entry)

    manifest = {
        "format": source["format"],
        "description": "Deterministic 100-step GB64 replay for four-GPU parity",
        "source_manifest": str(source_path),
        "source_entry_count": len(ordered),
        "samples": args.samples,
        "logical_world_size": args.logical_world_size,
        "global_batch_size": 64,
        "optimizer_steps": args.samples // 64,
        "sampler_seed": 42,
        "captured_batch_replay": True,
        "flow_capture_complete": True,
        "entries": expanded,
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(expanded)} entries to {output_path}; "
        f"flow aliases: {flow_root}"
    )


if __name__ == "__main__":
    main()
