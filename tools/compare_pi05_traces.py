"""Compare two PI0.5 alignment trace directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    names = sorted(p.name for p in args.left.glob("*.pt"))
    for name in names:
        right_name = name
        if name.startswith("fluxvla_pre_layer_"):
            layer = name.removeprefix("fluxvla_pre_layer_")
            right_name = (
                "pmp_embed.pt"
                if layer == "00.pt"
                else f"pmp_layer_{int(layer[:-3]) - 1:02d}.pt"
            )
        elif name == "fluxvla_layer_00_rope.pt":
            right_name = name.replace("fluxvla_", "pmp_")
        elif name.startswith("fluxvla_layer_"):
            right_name = name.replace("fluxvla_", "pmp_")
        if not (args.right / right_name).exists():
            continue
        left = torch.load(args.left / name, map_location="cpu", weights_only=True)
        right = torch.load(args.right / right_name, map_location="cpu", weights_only=True)
        print(f"{name} <-> {right_name}")
        for key, value in left.items():
            if not isinstance(value, torch.Tensor) or key not in right:
                continue
            other = right[key]
            if value.shape != other.shape:
                print(f"  {key}: shape {tuple(value.shape)} != {tuple(other.shape)}")
                continue
            lhs = value.float()
            rhs = other.float()
            rel = (lhs - rhs).norm() / lhs.norm().clamp_min(1e-30)
            print(
                f"  {key}: max_abs={(lhs - rhs).abs().max().item():.6e} "
                f"rel={rel.item():.6e}"
            )


if __name__ == "__main__":
    main()
