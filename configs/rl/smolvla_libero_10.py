# Copyright 2026 Limx Dynamics
"""SmolVLA LIBERO-10 preprocessing defaults for the RL bridge only.

Preserve the existing RL normalization without changing the legacy SFT
recipe's episode-stat aggregation. These PI0.5 action-window statistics are
not equivalent to SmolVLA's automatically aggregated SFT statistics.
For a checkpoint trained with different statistics, pass its own JSON via
``simulator.norm_stats_path`` in the RL configuration.
"""

from mmengine.config import read_base

with read_base():
    from ..pi05.pi05_paligemma_libero_10_full_finetune import \
        _PI05_LIBERO_10_STATS as _LIBERO_10_STATS
    from ..smolvla.smolvla_libero_10_finetune import *  # noqa: F401,F403

# Builtin dict copies work during MMEngine's lazy config parsing; imported
# functions such as deepcopy become LazyObjects and cannot be called here.
train_dataloader = dict(train_dataloader)  # noqa: F405
train_dataloader['dataset'] = dict(train_dataloader['dataset'])
train_dataloader['dataset']['dataset_statistics'] = _LIBERO_10_STATS
del _LIBERO_10_STATS
