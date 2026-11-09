"""Dedicated Megatron-Core training entry for FluxVLA PI0.5 recipes.

The regular ``scripts/train.py`` remains available for legacy runners.  This
entry validates the MCore topology before dispatching to the same dataset and
metric construction path, so launchers can unambiguously select the native
backend.
"""

from __future__ import annotations

import os

from mmengine import Config

from scripts.train import parse_args, train


if __name__ == "__main__":
    os.environ.setdefault("FLUXVLA_BACKEND", "mcore")
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    train(args, cfg)
