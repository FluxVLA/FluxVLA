"""Compare native/Flux LingBot attention traces, including sequence segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    parser.add_argument("flux", type=Path)
    args = parser.parse_args()
    native = torch.load(args.native, map_location="cpu", weights_only=False)
    flux = torch.load(args.flux, map_location="cpu", weights_only=False)
    keys = (
        "attention.position_ids",
        "attention.query_pre_rope",
        "attention.key_pre_rope",
        "attention.query_post_rope",
        "attention.key_post_rope",
        "attention.value",
        "attention.output",
    )
    for key in keys:
        if key not in native or key not in flux:
            continue
        a, b = native[key].float(), flux[key].float()
        if a.shape != b.shape:
            print(f"{key}: shape native={tuple(a.shape)} flux={tuple(b.shape)}")
            continue
        diff = (a - b).abs()
        print(f"{key}: full max={diff.max().item():.6g} mean={diff.mean().item():.6g}")
        if a.ndim >= 3 and a.shape[1] > 64:
            split = a.shape[1] - 51
            for name, sl in (("vlm", slice(0, split)), ("action", slice(split, None))):
                segment_diff = (a[:, sl] - b[:, sl]).abs()
                print(
                    f"  {name} max={segment_diff.max().item():.6g} "
                    f"mean={segment_diff.mean().item():.6g} shape={tuple(a[:, sl].shape)}"
                )
            if key in {"attention.query_pre_rope", "attention.key_pre_rope", "attention.value"}:
                for native_offset in (0, split):
                    for flux_offset in (0, split):
                        d = (a[:, native_offset:native_offset + 51] - b[:, flux_offset:flux_offset + 51]).abs()
                        print(f"  seg native[{native_offset}] vs flux[{flux_offset}] max={d.max().item():.6g} mean={d.mean().item():.6g}")


if __name__ == "__main__":
    main()
