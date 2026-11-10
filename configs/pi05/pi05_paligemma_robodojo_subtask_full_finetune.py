# Copyright 2026 Limx Dynamics
"""PI05 full fine-tuning on RoboDojo v3's stored dense subtask instructions.

No planner/API calls are made during training. The current frame's annotation
replaces the episode instruction before PreparePromptWithState/ProcessPrompts.
Intervals are [start, end). Chunks stop at each subtask boundary and repeat the
last valid target with loss mask zero; they never supervise the next subtask.
Missing, empty, overlapping or incomplete annotations fail explicitly.

Keep the original RoboDojo 3-camera, 14D joint-delta/absolute-gripper contract,
50-step horizon, and PI05 base checkpoint (not the RoboCasa checkpoint).
Recompute q01/q99 over valid subtask-local action chunks at startup,
rather than reusing the whole-episode RoboDojo or raw absolute-action stats.

Defaults: 2 GPUs, 2 samples/GPU, accumulation 1 => global batch 4,
60k optimizer updates. Full sharding and no EMA reduce memory requirements.
For another GPU count, adjust accumulation to keep the desired global batch.
Only local JSONL logging is enabled, so no WandB/API credentials are needed.

Run from the FluxVLA repository with the fluxvla Python environment:
    torchrun --standalone --nproc_per_node=2 scripts/train.py \\
        --config \\
        configs/pi05/pi05_paligemma_robodojo_subtask_full_finetune.py \\
        --work-dir work_dirs/pi05_robodojo_subtask

This is a training-only recipe: whole-task evaluation is intentionally
disabled. Deployment must provide subtask text, e.g. through
HierarchicalPolicy.
"""

import os

_base_ = './pi05_paligemma_robodojo_full_finetune.py'

data_root = os.environ.get(
    'ROBODOJO_SUBTASK_DATA_ROOT',
    '/mnt/data/cpfs/users/neri/robodojo/RoboDojo_lerobot_v3')

train_dataloader = _base_.train_dataloader
train_dataloader.per_device_batch_size = 2
train_dataloader.per_device_num_workers = 4
train_dataloader.dataset.dataset_statistics = None
train_dataloader.dataset.auto_compute_statistics = dict(
    state_key='observation.state',
    action_key='action',
    delta_mask=[True] * 6 + [False] + [True] * 6 + [False])

_source = train_dataloader.dataset.datasets[0]
_source.type = 'ParquetDatasetV3'
_source.data_root_path = [data_root]
_source.instruction_source = 'subtask'
_source.subtask_prefix = 'dense'
_source.pop('supervise_terminal_padding')
# Retain provenance for inspection without feeding the episode instruction
# into the tokenizer. The model still consumes only task_description=subtask.
_subtask_meta = [
    'full_task_description', 'subtask', 'subtask_index', 'subtask_start_frame',
    'subtask_end_frame'
]
_source.transforms[0].parquet_keys.extend(_subtask_meta)

runner = dict(
    grad_accumulation_steps=1,
    sharding_strategy='full-shard',
    ema_decay=None,
    collator=dict(meta_keys=_base_.runner.collator.meta_keys + _subtask_meta),
    metric=dict(active_trackers=('jsonl', )))

eval = None
themis = None
