"""Minimal logging helper for the vendored FastWAM core.

Ported from ``fastwam.utils.logging_config`` so the vendored FastWAM modules
stay self-contained within FluxVLA without depending on the upstream
``fastwam`` package.
"""
import logging
import os

import torch.distributed as dist


def _is_main_process() -> bool:
    """Best-effort main-process check without any synchronization."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")

    return True


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Drop-in replacement for ``fastwam.utils.logging_config.get_logger``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _is_main_process():
        logger.propagate = False
        logger.disabled = True

    return logger
