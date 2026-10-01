"""RoboTwin seed partitioning and validation-only checkpoint selection."""

import json
import math
import os
import time
from pathlib import Path


def partition_seeds(seeds, *, seed, rank, world_size):
    """Partition seeds without rounding down to the environment count."""
    import torch
    values = torch.as_tensor(seeds, dtype=torch.long)
    if values.numel() != torch.unique(values).numel():
        raise ValueError('Duplicate seeds are not allowed')
    indices = torch.randperm(
        len(values), generator=torch.Generator().manual_seed(seed))
    partitions = torch.tensor_split(values[indices], world_size)
    if not 0 <= rank < world_size or not len(partitions[rank]):
        raise ValueError('Empty/invalid seed partition')
    return partitions[rank]


def choose_best(records):
    if not records:
        raise ValueError('No validation checkpoints')
    if any(record['split'] != 'validation' for record in records):
        raise ValueError(
            'Checkpoint selection accepts validation results only')
    if any(not math.isfinite(record['success_rate']) for record in records):
        raise ValueError('Non-finite validation score')
    return min(records, key=lambda row: (-row['success_rate'], row['step']))


def check_phase_deadline():
    path = os.environ.get('FLUX_RL_BUDGET_FILE')
    if path:
        budget = json.loads(Path(path).read_text())
        if time.time() >= budget['training_deadline']:
            raise TimeoutError(
                'Six-hour preflight/baseline/training allocation exhausted')


def validate_config(cfg, phase):
    if (cfg.actor.model.fluxvla.rollout_micro_batch_size !=
            cfg.actor.micro_batch_size):
        raise ValueError('RoboTwin rollout compute microbatch must match '
                         'actor microbatch for probability parity')
    if cfg.env[phase].task_config.task_name != 'adjust_bottle':
        raise ValueError('RoboTwin currently supports adjust_bottle only')
