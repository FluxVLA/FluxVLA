"""Megatron forward step and loss closure for LingBot-VLA-v2."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path

import torch
from megatron.core import parallel_state


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _dp_mean_metrics(metrics: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce scalar logging metrics with one collective per microbatch."""
    if not metrics:
        return {}
    names = list(metrics)
    reduced = torch.stack(
        [metrics[name].detach().float().reshape(()) for name in names]
    )
    if torch.distributed.is_initialized():
        group = parallel_state.get_data_parallel_group()
        torch.distributed.all_reduce(reduced, group=group)
        reduced /= parallel_state.get_data_parallel_world_size()
    return dict(zip(names, reduced.unbind(), strict=True))


def forward_step(data_iterator, model):
    batch = _move(next(data_iterator), torch.cuda.current_device())
    if os.environ.get("FLUXVLA_BATCH_TRACE") == "1":
        sample_index = batch.get("sample_index")
        if isinstance(sample_index, torch.Tensor):
            root = Path(os.environ.get("FLUXVLA_BATCH_TRACE_DIR", "work_dirs/batch_trace"))
            root.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            with (root / f"batch_rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"sample_index": sample_index.detach().cpu().tolist()}) + "\n")
    output = model(**batch)
    if not isinstance(output, Mapping) or "loss" not in output:
        raise TypeError("LingBot MCore model must return a mapping containing loss")

    def loss_func(model_output):
        loss = model_output["loss"]
        local_metrics = {}
        for name, value in model_output.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                local_metrics[name.replace("/", "_")] = value
        local_metrics.setdefault("lm loss", loss)
        metrics = _dp_mean_metrics(local_metrics)
        return loss, metrics

    return output, loss_func


__all__ = ["forward_step"]
