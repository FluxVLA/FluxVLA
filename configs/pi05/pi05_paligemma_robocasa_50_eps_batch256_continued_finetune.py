"""Conservative PI0.5 continuation on the first 50 RoboCasa episodes.

This stage starts from the strongest measured 30-episode policy (31.58%).
It broadens every task uniformly to episodes 0--49 from the V2.1 dataset;
unlike the unsuccessful hard-task controls, no task root is repeated. The
short 3k-step schedule uses a 1e-6 backbone LR and 5e-6 expert/projection LR
to limit forgetting while an OpenPI-sized effective batch reduces update
noise.

Example for two GPUs:
    torchrun --nproc_per_node=2 scripts/train.py \
        --config \
        configs/pi05/\
pi05_paligemma_robocasa_50_eps_batch256_continued_finetune.py \
        --work-dir work_dirs/\
pi05_paligemma_robocasa_50_eps_batch256_continued_finetune
"""

from mmengine.config import read_base

# yapf: disable
with read_base():
    from .pi05_paligemma_robocasa_30_eps_full_finetune import \
        _ROBOCASA_TASK_DIRS
    from .pi05_paligemma_robocasa_30_eps_full_finetune import \
        eval as _base_eval
    from .pi05_paligemma_robocasa_30_eps_full_finetune import (
        model, runner, train_dataloader)
# yapf: enable

eval = _base_eval

_CONTINUATION_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'

# Continue native FluxVLA weights from the best 1,200-episode evaluation.
# Retain the checkpoint's 16-step action contract end to end.
model['pretrained_name_or_path'] = (
    './work_dirs/pi05_paligemma_robocasa_30_eps_full_finetune_'
    '3edfd4923_bs256/checkpoints/'
    'step-030000-epoch-10-loss=0.0079.safetensors')
model['name_mapping'] = None
model['strict_mapping'] = True
model['n_action_steps'] = 16

# Each V2.1 task directory contains exactly 1,000 episodes. ParquetDataset
# applies train_episode_fraction independently to each root while preserving
# episode order, so 0.05 selects episodes 0--49 for all 24 tasks. List each
# root once to keep the mixture balanced instead of repeating hard tasks.
train_dataloader['per_device_batch_size'] = 4
train_dataloader['dataset']['reshuffle_each_epoch'] = True
train_dataloader['dataset']['datasets']['data_root_path'] = [
    f'{_CONTINUATION_DATA_ROOT}/{task_dir}' for task_dir in _ROBOCASA_TASK_DIRS
]
train_dataloader['dataset']['datasets']['train_episode_fraction'] = 0.05
train_dataloader['dataset']['datasets']['repeat_to_full_length'] = False
train_dataloader['dataset']['datasets']['action_window_size'] = 16

# Keep the base config's fixed first-30 q01/q99 statistics asset and statistic
# name. Recomputing them on first50 would change both the checkpoint's input
# distribution and eval denormalization semantics.

# 4 samples/device * 2 devices * 32 micro-batches = 256 samples/update.
# Full-shard keeps FP32 master parameters in FSDP while executing BF16 mixed
# precision forward/backward. This is a fresh optimizer continuation, not an
# optimizer-state resume, so warm it up for 500 of the 3,000 updates.
runner['max_steps'] = 3000
runner['grad_accumulation_steps'] = 32
runner['optimizer'] = dict(
    # Stage 4 showed that a uniform 1e-5 continuation can forget useful
    # behavior. Keep both backbones at the conservative default and give the
    # expert plus small projection modules enough LR to absorb the extra data.
    lr=1e-6,
    type='AdamW',
    betas=(0.9, 0.95),
    weight_decay=0.0,
    paramwise_learning_rate={
        'llm_expert': 5e-6,
        'projector': 5e-6,
        'action_in_proj': 5e-6,
        'action_out_proj': 5e-6,
        'time_mlp_in': 5e-6,
        'time_mlp_out': 5e-6,
    },
)
runner['save_iter_interval'] = 1000
runner['max_keep_ckpts'] = 4
runner['sharding_strategy'] = 'full-shard'
runner['lr_scheduler'] = dict(
    type='linear-warmup+constant',
    warmup_steps=500,
)
