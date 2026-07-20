# Copyright 2026 Limx Dynamics
"""Native GR00T N1.7 LIBERO-goal parquet fine-tuning config."""

_SUITE = 'libero_goal'
_DATASET_NAME = 'libero_goal_no_noops_lerobotv2.1'
_STATISTIC_NAME = 'libero_goal_no_noops_native'

_LIBERO_DATA_ROOT = f'./datasets/{_DATASET_NAME}'
_N17_INIT_CKPT = './checkpoints/GR00T-N1.7-3B'
_N17_PROCESSOR_META = f'./checkpoints/GR00T-N1.7-LIBERO/{_SUITE}'
_BACKBONE_MODEL_PATH = './checkpoints/nvidia/Cosmos-Reason2-2B'
_ACTIVE_TRACKERS = ('jsonl',)

_PROCESSOR_KWARGS = dict(
    model_name=_BACKBONE_MODEL_PATH,
    state_dropout_prob=0.2,
    color_jitter_params=dict(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
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
    per_device_batch_size=80,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['states'],
            'action': ['actions'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        shuffle=True,
        seed=42,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_LIBERO_DATA_ROOT,
                statistic_name=_STATISTIC_NAME,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                train_episode_fraction=1.0,
                repeat_to_full_length=False,
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
                    dict(
                        type='ProcessGrootN17NativeInputs',
                        processor_path=_N17_PROCESSOR_META,
                        embodiment_tag='LIBERO_PANDA',
                        flat_layout='auto',
                        train_mode=True,
                        processor_kwargs=_PROCESSOR_KWARGS,
                    ),
                ],
                action_window_size=16,
            ),
        ],
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=20000,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=1e-5,
    ),
    max_grad_norm=1.0,
    grad_accumulation_steps=1,
    sampler=None,
    save_iter_interval=1000,
    save_epoch_interval=1,
    max_keep_ckpts=5,
    collator=dict(
        type='GrootN17DataCollator',
        model_name=_BACKBONE_MODEL_PATH,
        transformers_loading_kwargs=dict(
            local_files_only=True,
            trust_remote_code=True,
        ),
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=_ACTIVE_TRACKERS,
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='no-shard',
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    model_family='groot_n17_native',
    task_suite_name=_SUITE,
    num_trials_per_task=50,
    eval_chunk_size=8,
    max_steps=720,
    seed=7,
    result_output_dir=(
        'work_dirs/'
        f'n17_native_{_SUITE}_posttrain_auto_eval'),
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[dict(type='ProcessGrootN17LiberoEvalInputs')],
        extra_batch_keys=['n17_observation', 'n17_task']),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        denorm_action=False,
        requires_norm_stats=False),
)
