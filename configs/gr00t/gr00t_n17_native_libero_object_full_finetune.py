# Copyright 2026 Limx Dynamics
"""Native GR00T N1.7 LIBERO-object parquet fine-tuning config."""

import os
from ast import literal_eval


_SUITE = 'libero_object'
_DATASET_NAME = 'libero_object_no_noops_lerobotv2.1'
_STATISTIC_NAME = 'libero_object_no_noops_native'

_LIBERO_DATA_ROOT = os.environ.get(
    'LIBERO_DATA_ROOT',
    f'./datasets/{_DATASET_NAME}')
_N17_INIT_CKPT = os.environ.get(
    'N17_INIT_CKPT',
    './checkpoints/GR00T-N1.7-3B')
_N17_PROCESSOR_META = os.environ.get(
    'N17_PROCESSOR_META',
    f'./checkpoints/'
    f'GR00T-N1.7-LIBERO/{_SUITE}')
_BACKBONE_MODEL_PATH = os.environ.get(
    'N17_BACKBONE_MODEL_PATH',
    './checkpoints/'
    'nvidia/Cosmos-Reason2-2B')


def _parse_active_trackers(value):
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    if isinstance(parsed, str) and parsed.strip().startswith(('[', '(')):
        parsed = literal_eval(parsed)
    if isinstance(parsed, str):
        parsed = tuple(
            item.strip() for item in parsed.split(',') if item.strip())
    return tuple(parsed)


_ACTIVE_TRACKERS = _parse_active_trackers(
    os.environ.get('N17_ACTIVE_TRACKERS', "('jsonl',)"))

_PROCESSOR_KWARGS = dict(
    model_name=_BACKBONE_MODEL_PATH,
    state_dropout_prob=float(os.environ.get('N17_STATE_DROPOUT_PROB', '0.2')),
    color_jitter_params=dict(
        brightness=float(os.environ.get('N17_COLOR_JITTER_BRIGHTNESS', '0.3')),
        contrast=float(os.environ.get('N17_COLOR_JITTER_CONTRAST', '0.4')),
        saturation=float(os.environ.get('N17_COLOR_JITTER_SATURATION', '0.5')),
        hue=float(os.environ.get('N17_COLOR_JITTER_HUE', '0.08')),
    ),
    transformers_loading_kwargs=dict(
        local_files_only=True,
        trust_remote_code=True,
    ),
)

model = dict(
    type='GrootN17VLA',
    model_path=_N17_INIT_CKPT,
    processor_path=_N17_PROCESSOR_META,
    backbone_model_path=_BACKBONE_MODEL_PATH,
    embodiment_tag='LIBERO_PANDA',
    load_metadata=True,
    use_flash_attention=False,
    load_mode='native_safe',
    qwen3_runtime='compat_457',
    processor_runtime='native',
    assembly_runtime='native',
)

train_dataloader = dict(
    per_device_batch_size=int(os.environ.get('N17_PER_DEVICE_BATCH_SIZE', '80')),
    per_device_num_workers=int(os.environ.get('N17_NUM_WORKERS', '4')),
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['states'],
            'action': ['actions'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        shuffle=True,
        seed=int(os.environ.get('N17_DATASET_SEED', '42')),
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_LIBERO_DATA_ROOT,
                statistic_name=_STATISTIC_NAME,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                train_episode_fraction=float(
                    os.environ.get('N17_TRAIN_EPISODE_FRACTION', '1.0')),
                repeat_to_full_length=bool(
                    int(os.environ.get('N17_REPEAT_TO_FULL_LENGTH', '0'))),
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.state',
                            'timestamp',
                            'actions',
                            'info',
                            'stats',
                            'action_masks',
                        ],
                        video_keys=[
                            'observation.images.image',
                            'observation.images.wrist_image',
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions'],
                        },
                    ),
                ],
                action_window_size=16,
            ),
        ],
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=int(os.environ.get('N17_MAX_STEPS', '20000')),
    learning_rate=float(os.environ.get('N17_LR', '1e-4')),
    weight_decay=float(os.environ.get('N17_WEIGHT_DECAY', '1e-5')),
    max_grad_norm=float(os.environ.get('N17_MAX_GRAD_NORM', '1.0')),
    grad_accumulation_steps=int(
        os.environ.get('N17_GRAD_ACCUM_STEPS', '1')),
    sampler=None,
    save_iter_interval=int(os.environ.get('N17_SAVE_ITER_INTERVAL', '1000')),
    save_epoch_interval=1,
    max_keep_ckpts=int(os.environ.get('N17_MAX_KEEP_CKPTS', '5')),
    save_full_model=True,
    collator=dict(
        type='GrootN17NativeCollator',
        processor_path=_N17_PROCESSOR_META,
        embodiment_tag='LIBERO_PANDA',
        flat_layout='auto',
        train_mode=True,
        processor_kwargs=_PROCESSOR_KWARGS,
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=tuple(_ACTIVE_TRACKERS),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type=os.environ.get(
        'N17_LR_SCHEDULER_TYPE', 'linear-warmup+cosine-decay'),
    warmup_ratio=float(os.environ.get('N17_WARMUP_RATIO', '0.05')),
    enable_gradient_checkpointing=bool(
        int(os.environ.get('N17_ENABLE_GRAD_CKPT', '0'))),
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy=os.environ.get('N17_SHARDING_STRATEGY', 'no-shard'),
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    model_family='groot_n17_native',
    task_suite_name=_SUITE,
    num_trials_per_task=int(os.environ.get('N17_AUTO_EVAL_TRIALS', '50')),
    eval_chunk_size=8,
    max_steps=720,
    seed=int(os.environ.get('N17_AUTO_EVAL_SEED', '7')),
    result_output_dir=os.environ.get(
        'N17_AUTO_EVAL_OUTPUT_DIR',
        'work_dirs/'
        f'n17_native_{_SUITE}_posttrain_auto_eval'),
    dataset=dict(type='LiberoN17EvalDataset', replay_key='video.image'),
    denormalize_action=dict(),
)
