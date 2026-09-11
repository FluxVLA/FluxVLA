"""DiT4DiT training-time RTC and Oli RTC inference for Loco-Mani task 1."""

import os

_repo_root = os.path.abspath(os.environ.get('FLUXVLA_ROOT', '.'))
_cosmos_base_model = os.path.join(_repo_root, 'checkpoints',
                                  'Cosmos-Predict2.5-2B')
_cosmos_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path=os.path.join(_cosmos_base_model, 'tokenizer'),
    model_max_length=512)
_data_roots = [
    '/mnt/data/cpfs/users/jikun/vcube_data/'
    'wbt_done_dim_0609_0630_task1'
]
_statistic_name = 'hud04_locomani_task1'
_task = ('Turn around and move back to the first box. Bend down, grasp the '
         'first box with both hands, and lift it. Carry the first box to the '
         'second box located in front of you. Place the first box on top of '
         'the second box.')
_action_dim, _output_action_dim, _state_dim = 64, 43, 64
_action_horizon, _frame_window_size, _image_size = 32, 9, 224
seed = 42

model = dict(
    type='DiT4DiTVLA',
    repeated_diffusion_steps=4,
    vlm_backbone=dict(
        type='Cosmos25Backbone',
        base_model=_cosmos_base_model,
        revision='diffusers/base/post-trained',
        torch_dtype='bf16',
        local_files_only=True,
        extract_layer=17,
        trainable=True,
        frozen_submodules=['text_encoder', 'vae'],
        split_future_frames=True,
        num_frames_out=_frame_window_size,
        fixed_seed=None,
        num_inference_steps=1,
        conditional_frame_timestep=0.0001,
        future_loss_type='flow_matching',
        detach_hidden_states=True,
        flow_matching_time_distribution='uniform',
        flow_matching_high_sigma_ratio=None,
        flow_matching_high_sigma_min=None,
        fsdp_min_num_params=0),
    vla_head=dict(
        type='DiT4DiTActionHead',
        action_model_type='DiT-B',
        hidden_size=2560,
        add_pos_embed=True,
        max_seq_len=1024,
        action_dim=_action_dim,
        ori_action_dim=_output_action_dim,
        state_dim=_state_dim,
        action_horizon=_action_horizon,
        future_action_window_size=_action_horizon - 1,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        num_inference_timesteps=4,
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',
            temperature=1.0),
        diffusion_model_cfg=dict(
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_layers=16,
            output_dim=2560,
            positional_embeddings=None)),
    freeze_vlm_backbone=False)
inference_model = dict(
    model,
    init_empty_weights=True,
    vlm_backbone=dict(model['vlm_backbone'], load_pretrained_weights=False))


def _pipeline(frame_window_size):
    return [
        dict(
            type='ProcessParquetInputs',
            parquet_keys=[
                'observation.state', 'timestamp', 'actions', 'info', 'stats',
                'action_masks'
            ],
            video_keys=[
                'observation.images.head', 'observation.images.left_wrist'
            ],
            name_mappings={
                'observation.state': ['states'],
                'actions': ['actions']
            }),
        dict(
            type='ProcessCosmos25Prompt',
            tokenizer=_cosmos_tokenizer,
            input_key='task_description',
            remove_input_key=True),
        dict(
            type='ResizeImages',
            height=_image_size,
            width=_image_size,
            backend='torch',
            scale_divisor=255.0,
            output_layout='flattened_chw'),
        dict(
            type='PrepareVideo',
            num_views=2,
            frame_window_size=frame_window_size,
            tile_direction='vertical',
            combine_view_masks=True),
        dict(
            type='NormalizeStatesAndActions',
            action_dim=_action_dim,
            state_dim=_state_dim,
            state_key='proprio',
            action_key='action',
            norm_type='mean_std',
            output_dtype='float16'),
        dict(
            type='PrepareStateActionTargets',
            state_history_length=1,
            action_horizon=_action_horizon,
            valid_action_dim=_output_action_dim,
            dtype='float16'),
    ]


train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_data_roots,
            transforms=_pipeline(_frame_window_size),
            action_window_size=_action_horizon,
            action_key='action',
            use_delta=False,
            statistic_name=_statistic_name,
            window_start_idx=0,
            frame_window_size=_frame_window_size,
            frame_sample_stride=4)))
runner = dict(
    type='FSDPTrainRunner',
    max_epochs=10,
    grad_accumulation_steps=1,
    optimizer=dict(
        lr=1e-4, type='AdamW', weight_decay=1e-2, betas=(0.9, 0.95)),
    max_grad_norm=1.0,
    save_epoch_interval=1,
    max_keep_ckpts=2,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'actions',
            'action_masks', 'frame_masks', 'lang_tokens', 'lang_masks'
        ],
        meta_keys=['info', 'stats']),
    tokenizer=_cosmos_tokenizer,
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.05),
    sharding_strategy='global-shard-grad-op',
    pre_fsdp_param_dtype='fp32',
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    reduce_in_full_precision=False,
    change_key_name=False)

inference = dict(
    type='OliRTCInferenceRunner',
    task_suite_name=_statistic_name,
    task_descriptions={'1': _task},
    seed=7,
    state_dim=_state_dim,
    action_chunk=_action_horizon,
    publish_rate=30,
    max_publish_step=10000,
    # Seven 30 Hz steps cover about 233 ms, leaving two steps of margin over
    # the estimated 150 ms inference latency. Start the next request with nine
    # executable steps remaining so inference finishes before the handoff.
    execute_horizon=16,
    async_remaining_actions_threshold=9,
    rtc_config=dict(enabled=True, method='prefix', prefix_len=7),
    interactive=True,
    default_prompt_id='1',
    default_execution_count=1000,
    mixed_precision_dtype='bf16',
    low_cpu_mem_usage=True,
    camera_names=['head', 'left_wrist'],
    dataset=dict(
        type='PrivateInferenceDataset',
        statistic_name=_statistic_name,
        embodiment_id=0,
        inject_model_path=False,
        img_keys=['head', 'left_wrist'],
        transforms=[
            dict(
                type='ProcessCosmos25Prompt',
                tokenizer=_cosmos_tokenizer,
                input_key='task_description',
                remove_input_key=True),
            dict(
                type='ResizeImages',
                height=_image_size,
                width=_image_size,
                backend='torch',
                scale_divisor=255.0,
                output_layout='flattened_chw'),
            dict(
                type='PrepareVideo',
                num_views=2,
                frame_window_size=1,
                tile_direction='vertical',
                combine_view_masks=True),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=_action_dim,
                state_dim=_state_dim,
                state_key='proprio',
                action_key=None,
                norm_type='mean_std',
                statistics_key='norm_stats',
                output_dtype='float32'),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=_action_horizon,
                valid_action_dim=_output_action_dim,
                dtype='float32')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_statistic_name,
        norm_type='mean_std',
        action_dim=_output_action_dim),
    operator=dict(
        type='OliOperator',
        control_backend='mros',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic='/left_wrist_camera/color/image_raw/compressed',
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        teleop_wbt_topic='/teleop_cmd_WBT',
        finger_force_levels=(2.0, 2.0)))
