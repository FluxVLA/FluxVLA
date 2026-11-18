"""Run the normal FluxVLA training entrypoint with PyTorch profiling enabled."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from fluxvla.engines.runners.base_train_runner import BaseTrainRunner
from fluxvla.engines.runners.fsdp_train_runner import FSDPTrainRunner


PROFILE_DIR = Path(os.environ.get("FLUXVLA_PROFILE_DIR", "profiling/pi05_libero"))
PROFILE_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_RANK = int(os.environ.get("FLUXVLA_PROFILE_RANK", "0"))

_profiler = None
_original_setup = FSDPTrainRunner.run_setup
_original_run = BaseTrainRunner.run
_original_step = BaseTrainRunner._training_step


def _profiled_setup(self, *args, **kwargs):
    result = _original_setup(self, *args, **kwargs)
    global _profiler
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != PROFILE_RANK:
        return result
    _profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(PROFILE_DIR)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )
    _profiler.start()
    return result


def _profiled_step(self, batch, should_step=True):
    with record_function("FluxVLA.TrainingStep"):
        result = _original_step(self, batch, should_step=should_step)
    if _profiler is not None and should_step:
        _profiler.step()
    return result


def _profiled_run(self, *args, **kwargs):
    try:
        return _original_run(self, *args, **kwargs)
    finally:
        if _profiler is not None:
            _profiler.stop()
            if os.environ.get("FLUXVLA_PROFILE_SUMMARY", "0") == "1":
                (PROFILE_DIR / "profiler_summary.txt").write_text(
                    _profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=100
                    ),
                    encoding="utf-8",
                )


FSDPTrainRunner.run_setup = _profiled_setup
BaseTrainRunner._training_step = _profiled_step
BaseTrainRunner.run = _profiled_run

if __name__ == "__main__":
    # Execute the production launcher so argument parsing/config resolution stay identical.
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
