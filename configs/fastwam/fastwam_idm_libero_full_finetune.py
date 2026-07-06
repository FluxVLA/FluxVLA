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
# FastWAM world-action model (idm, teacher-forcing) jointly trained on all
# four LIBERO suites (spatial + object + goal + 10).

_ckpt_root = './checkpoints'
_tokenizer = _ckpt_root + '/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'

_frame_window_size = 9
_action_window_size = 32
_frame_sample_stride = 4

# This 30Hz statefix export keeps the same four LIBERO suites in LeRobot v2.1
# format, but uses 256x256 H.264 videos and different state statistics from
# the MuJoCo 3.3.2 release. Use a separate statistic key so checkpoints do not
# silently reuse the old 20Hz `libero_no_noops` normalization.
_libero_root = (
    '/mnt/data/cpfs/mnt/data/yanis/datasets/'
    'libero_4suite_unified_lerobotv2_30hz_statefix_fullfintuneeval/')
_data_root_path = [
    _libero_root + 'libero_spatial_lerobotv2.1',
    _libero_root + 'libero_object_lerobotv2.1',
    _libero_root + 'libero_goal_lerobotv2.1',
    _libero_root + 'libero_10_lerobotv2.1',
]
_statistic_name = 'libero_30hz_statefix_no_noops'

model = dict(
    type='FastWAMVLA',
    pretrained_name_or_path=None,
    num_views=2,
    frame_window_size=_frame_window_size,
    proprio_dim=8,
    action_horizon=_action_window_size,
    mot_checkpoint_mixed_attn=True,
    vlm_backbone=dict(
        type='Wan22Backbone',
        model_id='Wan-AI/Wan2.2-TI2V-5B',
        tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B',
        tokenizer_max_len=128,
        load_text_encoder=False,
        redirect_common_files=False,
    ),
    vla_head=dict(
        type='FastWAMIDMHead',
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
        action_dit_pretrained_path=(
            _ckpt_root +
            '/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'),
        skip_dit_load_from_pretrain=False,
        video_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        action_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        loss=dict(lambda_video=1.0, lambda_action=1.0),
    ),
)

# Evaluation loads the T5 text encoder so prompts are encoded on the fly,
# while training keeps the cached-context path (load_text_encoder=False).
inference_model = model.copy()
inference_model['vlm_backbone'] = dict(
    model['vlm_backbone'], load_text_encoder=True)

train_dataloader = dict(
    per_device_batch_size=16,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_data_root_path,
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
                    type='LoadCachedTextEmbedding',
                    cache_dir=('/root/projects/ryanhu/data/'
                               'text_embeds_cache/libero'),
                    context_len=128,
                    enc_id='wan22ti2v5b',
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

runner = dict(
    type='DDPTrainRunner',
    max_epochs=10,
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
            'context',
            'context_mask',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
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
    grad_accumulation_steps=1,
    mixed_precision_dtype='bf16',
    evaluator=dict(
        type='training-eval',
        eval_every=200,
        num_inference_steps=10,
        seed=42,
        save_video=True,
        video_fps=8,
    ),
)

# Default full-suite rollout. eval.py runs each listed suite in sequence.
# norm_stats_key keeps the merged 30Hz statefix statistics carried by the
# checkpoint.
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
        allowed_missing_key_prefixes=('vlm_backbone.text_encoder.', ),
        norm_stats_key=_statistic_name,
        eval_chunk_size=10,
        eval_shard_strategy='task',
        preprocess_every_step=False,
        num_inference_steps=10,
        max_steps=dict(
            libero_spatial=400,
            libero_object=400,
            libero_goal=400,
            libero_10=700,
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
