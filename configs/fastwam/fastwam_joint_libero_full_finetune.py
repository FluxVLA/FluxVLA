# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FastWAM joint model trained on all LIBERO suites with P0-A FSDP.

seed = 42

_ckpt_root = './checkpoints'
_tokenizer = _ckpt_root + '/fastwam_base_full/tokenizer'
_text_prompt_template = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')

_frame_window_size = 9
_action_window_size = 32
_frame_sample_stride = 4

# Train jointly on every no-noops LIBERO suite.
_data_root_paths = [
    'datasets/libero_spatial_no_noops_lerobotv2.1',
    'datasets/libero_object_no_noops_lerobotv2.1',
    'datasets/libero_goal_no_noops_lerobotv2.1',
    'datasets/libero_10_no_noops_lerobotv2.1',
]
_statistic_name = 'libero_all_no_noops'

model = dict(
    type='FastWAMVLA',
    pretrained_name_or_path=  # noqa: E251
    _ckpt_root + '/fastwam_base_full/fastwam_base_full.safetensors',
    torch_dtype='bf16',
    num_views=2,
    frame_window_size=_frame_window_size,
    proprio_dim=8,
    action_horizon=_action_window_size,
    mot_checkpoint_mixed_attn=True,
    vlm_backbone=dict(
        type='Wan22Backbone',
        text_embed_cache_context_len=128,
        text_embed_cache_size=256,
        text_embed_cache_device='cpu',
    ),
    vla_head=dict(
        type='FastWAMJointHead',
        video_dit_config=dict(
            has_image_input=False,
            patch_size=[1, 2, 2],
            in_dim=48,
            hidden_dim=3072,
            ffn_dim=14336,
            freq_dim=256,
            text_dim=4096,
            out_dim=48,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            eps=1.0e-06,
            seperated_timestep=True,
            require_clip_embedding=False,
            require_vae_embedding=False,
            fuse_vae_embedding_in_latents=True,
            video_attention_mask_mode='first_frame_causal',
            action_conditioned=False,
            action_dim=7,
            action_group_causal_mask_mode='group_diagonal',
            use_gradient_checkpointing=True,
        ),
        action_dit_config=dict(
            action_dim=7,
            hidden_dim=1024,
            ffn_dim=4096,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            text_dim=4096,
            freq_dim=256,
            eps=1.0e-06,
            use_gradient_checkpointing=True,
        ),
        video_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        action_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        loss=dict(lambda_video=1.0, lambda_action=1.0),
    ),
)

# Training and evaluation encode unseen prompts online, then reuse a
# per-process CPU-memory LRU cache without reading or writing cache files.
inference_model = model.copy()

train_dataloader = dict(
    # Global batch: 8 microbatch x 16 GPUs x 1 accumulation = 128.
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        reshuffle_each_epoch=True,
        seed=42,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_data_root_paths,
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
                    video_backend='torchcodec',
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=0,
                ),
                dict(
                    type='ResizeImages',
                    height=224,
                    width=224,
                    backend='torchvision',
                    scale_to_unit_interval=True,
                ),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=7,
                    state_dim=8,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max',
                    pad_invalid_action_delta_dims=True,
                    delta_action_dim_mask=[
                        True, True, True, True, True, True, False
                    ],
                ),
                dict(
                    type='PrepareVideo',
                    num_views=2,
                    frame_window_size=_frame_window_size,
                    tile_direction='horizontal',
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    tokenizer=dict(
                        type='PretrainedTokenizer', model_path=_tokenizer),
                    max_len=128,
                    use_conversation=False,
                    prompt_template=_text_prompt_template,
                ),
            ],
            action_window_size=_action_window_size,
            action_key='action',
            use_delta=False,
            statistic_name=_statistic_name,
            window_start_idx=0,
            frame_window_size=_frame_window_size,
            frame_sample_stride=_frame_sample_stride,
        ),
    ),
)

# Optional in-training eval dataset for the FastWAM-style wandb eval metrics.
# Leave both as None to match the FastWAM LIBERO source config, where
# val_dataset falls back to train_dataset when no validation split is defined.
val_dataloader = None
eval_dataset = None

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=10,
    max_steps=None,
    save_epoch_interval=1,
    save_iter_interval=10000,
    max_keep_ckpts=10,
    optimizer=dict(lr=1e-4, type='AdamW', weight_decay=1e-2),
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['task_description', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay-min-lr',
        warmup_ratio=0.05,
        min_lr_ratio=0.01,
        betas=(0.9, 0.95),
        weight_decay_style='uniform',
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    reduce_in_full_precision=True,
    grad_accumulation_steps=1,
    mixed_precision_dtype='bf16',
    dataset_sharding_strategy='blockwise',
    sharding_strategy='shard-grad-op',
    evaluator=dict(
        type='training-eval',
        eval_every=1000,
        num_inference_steps=10,
        seed=42,
        save_video=True,
        video_fps=8,
    ),
)

# Evaluate every LIBERO suite using the combined training statistics.
eval = dict(
    runner=dict(
        type='LiberoEvalRunner',
        task_suite_name=[
            'libero_spatial',
            'libero_object',
            'libero_goal',
            'libero_10',
        ],
        model_family='fastwam',
        task_ids=None,
        norm_stats_key=_statistic_name,
        eval_chunk_size=10,
        eval_shard_strategy='episode',
        preprocess_every_step=False,
        num_inference_steps=10,
        max_steps=dict(
            libero_10=700,
            libero_goal=400,
            libero_object=400,
            libero_spatial=400,
        ),
        inference_seed=42,
        resize_size=224,
        num_trials_per_task=50,
        num_steps_wait=30,
        seed=42,
        enable_mixed_precision_training=True,
        mixed_precision_dtype='bf16',
        model_build_device='cuda',
        model_build_dtype='bf16',
        save_rollout_videos=True,
        save_multi_view_rollout_videos=True,
        dataset=dict(
            type='LiberoParquetEvalDataset',
            img_buffer_len=1,
            transforms=[
                dict(
                    type='ProcessLiberoEvalInputs',
                    img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                ),
                dict(
                    type='TransformImage',
                    image_resize_strategy='resize-naive',
                    input_sizes=[[3, 224, 224], [3, 224, 224]],
                    means=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                    stds=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                ),
                dict(
                    type='LiberoProprioFromInputs',
                    norm_type='min_max',
                    out_key='states',
                    stat_key='proprio',
                    state_dim=8,
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_tokenizer,
                    ),
                    max_len=128,
                    use_conversation=False,
                    prompt_template=(
                        "A video recorded from a robot's point of view "
                        'executing the following instruction: {task}'),
                ),
                dict(
                    type='PrepareVideo',
                    num_views=2,
                    frame_window_size=1,
                    tile_direction='horizontal',
                ),
            ],
        ),
        denormalize_action=dict(
            type='DenormalizeLiberoAction',
            norm_type='min_max',
            action_dim=7,
        ),
    ),
    manager=dict(
        num_gpus=8,
        max_tasks_per_gpu=2,
        master_port_base=29690,
        monitor_interval=5,
        status_interval=30,
        launch_delay=0.5),
)
