"""Report segment-wise differences for PI0.5 embedding traces."""

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    args = p.parse_args()
    left = torch.load(args.left, map_location="cpu", weights_only=True)
    right = torch.load(args.right, map_location="cpu", weights_only=True)
    for key in ("prefix", "suffix"):
        a, b = left[key].float(), right[key].float()
        print(key, tuple(a.shape))
        for lo, hi in ((0, 256), (256, 512), (512, a.shape[1]), (0, a.shape[1])):
            x, y = a[:, lo:hi], b[:, lo:hi]
            if x.numel() == 0:
                continue
            print(
                lo,
                hi,
                (x - y).abs().max().item(),
                ((x - y).norm() / x.norm()).item(),
                (x - y).abs().mean().item(),
            )


if __name__ == "__main__":
    main()
