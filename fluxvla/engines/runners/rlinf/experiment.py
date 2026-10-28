"""Small controls that prohibit selection on held-out test data."""

import json
import math
import os
import time
from pathlib import Path


def require_wandb():
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError(
            'WANDB_API_KEY is missing; formal PPO must use online W&B')
    if os.environ.get('WANDB_MODE', 'online') != 'online':
        raise RuntimeError('Formal PPO requires WANDB_MODE=online')
    import wandb

    # Authenticate without wandb.login(), which can persist a key to .netrc.
    # MetricLogger owns the run, and the key stays only in the environment.
    if not wandb.Api(api_key=os.environ['WANDB_API_KEY'], timeout=30).viewer:
        raise RuntimeError('W&B authentication failed')


def partition_seeds(seeds, *, seed, rank, world_size):
    """Create disjoint worker slices without backend truncation."""
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
            raise TimeoutError('Configured training deadline exhausted')
