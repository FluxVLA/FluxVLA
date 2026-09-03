"""SmolVLA RoboDojo ARX-X5 multi-task finetuning config.

Unlike the official single-task leaderboard recipe, this config trains one
policy on the complete 3,500-episode RoboDojo dataset. It uses absolute 14D
joint actions, mean/std normalization, a 50-step action horizon, a global
batch size of 512, and 100K optimizer updates. Unlike the released expert-only
recipe, the VLM backbone is also finetuned.

The batch settings target eight GPUs:
    32 samples/GPU * 8 GPUs * 2 accumulation steps = 512.

Example:
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/smolvla/smolvla_robodojo_full_finetune.py \
        --work-dir work_dirs/smolvla_robodojo_full_finetune
"""

_ROBODOJO_DATA_ROOT = './datasets/RoboDojo_lerobot_v21_video'
_ROBODOJO_GENERALIZATION_BASE_TASKS = ('stack_bowls', 'push_T',
                                       'pack_objects_into_box', 'fold_clothes',
                                       'hang_mugs', 'sweep_blocks',
                                       'pour_liquid_into_cup', 'make_toast',
                                       'arrange_largest_number',
                                       'sort_nesting_dolls_by_size',
                                       'store_laptop_and_headphones',
                                       'stack_blocks')
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

model = dict(
    type='SmolVLAFlowMatching',
    vlm_backbone=dict(
        type='SmolVLMBackbone',
        vision_config=dict(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=512,
            patch_size=16,
            intermediate_size=3072,
            hidden_act='gelu_pytorch_tanh',
            layer_norm_eps=1e-6,
        ),
        text_config=dict(
            hidden_size=960,
            num_hidden_layers=32,
            num_attention_heads=15,
            num_key_value_heads=5,
            head_dim=64,
            intermediate_size=2560,
            vocab_size=49280,
            rms_norm_eps=1e-5,
            hidden_act='silu',
            max_position_embeddings=8192,
        ),
        scale_factor=4,
        num_vlm_layers=16,
        torch_dtype='bfloat16',
    ),
    llm_expert=dict(
        type='SmolVLMExpert',
        hidden_size=720,
        num_hidden_layers=16,
        num_attention_heads=15,
        num_key_value_heads=5,
        head_dim=64,
        intermediate_size=-1,
        vocab_size=49280,
        attention_bias=False,
        rms_norm_eps=1e-5,
        hidden_act='silu',
        max_position_embeddings=8192,
        attention_mode='cross_attn',
        vlm_kv_dim=320,
        self_attn_every_n_layers=2,
        torch_dtype='bfloat16',
    ),
    state_proj=dict(type='LinearProjector', in_dim=32, out_dim=960),
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=720),
    action_out_proj=dict(type='LinearProjector', in_dim=720, out_dim=32),
    action_time_mlp_in=dict(type='LinearProjector', in_dim=1440, out_dim=720),
    action_time_mlp_out=dict(type='LinearProjector', in_dim=720, out_dim=720),
    freeze_vlm_backbone=False,
    max_action_dim=32,
    ori_action_dim=14,
    chunk_size=50,
    num_steps=10,
    add_image_special_tokens=False,
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/smolvla_base/model.safetensors',
    name_mapping={
        'vlm_backbone.vlm': 'model.vlm_with_expert.vlm.model',
        'llm_expert.expert': 'model.vlm_with_expert.lm_expert',
        'state_proj.projector': 'model.state_proj',
        'action_in_proj.projector': 'model.action_in_proj',
        'action_out_proj.projector': 'model.action_out_proj',
        'action_time_mlp_in.projector': 'model.action_time_mlp_in',
        'action_time_mlp_out.projector': 'model.action_time_mlp_out',
    })

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=32,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=0,
        reshuffle_each_epoch=True,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action', 'timestamp'],
        statistic_name='robodojo_arx_x5',
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=[_ROBODOJO_DATA_ROOT],
                action_key='action',
                use_delta=False,
                statistic_name='robodojo_arx_x5',
                window_start_idx=0,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist',
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions'],
                        },
                        video_backend='pyav'),
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=32,
                        state_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std'),
                    dict(
                        type='ParquetPrompter',
                        use_conversation=False,
                        add_new_line=True),
                    dict(
                        type='ProcessPrompts',
                        max_len=48,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path='./checkpoints/'
                            'SmolVLM2-500M-Video-Instruct')),
                    dict(
                        type='ResizeImagesWithPad',
                        height=512,
                        width=512,
                        pad_direction='top-left'),
                    dict(type='SimpleNormalizeImages'),
                ],
                action_window_size=50)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    grad_accumulation_steps=2,
    deterministic_algorithms=True,
    seed=0,
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True,
    ),
    max_grad_norm=10.0,
    sharding_strategy='full-shard',
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'lang_tokens',
            'lang_masks',
            'actions',
            'action_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path='./checkpoints/SmolVLM2-500M-Video-Instruct'),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_steps=1000,
        decay_steps=30000,
        min_lr=2.5e-6,
    ),
    save_iter_interval=10000,
    max_keep_ckpts=10,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='smolvla',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    dataset=dict(
        type='RoboDojoEvalDataset',
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist']),
            dict(
                type='StateFromInputs',
                stat_key='proprio',
                norm_type='mean_std',
                state_dim=32),
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                add_new_line=True),
            dict(
                type='ProcessPrompts',
                max_len=48,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path='./checkpoints/'
                    'SmolVLM2-500M-Video-Instruct')),
            dict(
                type='TransformImage',
                image_resize_strategy='letterbox',
                input_sizes=[[3, 512, 512]] * 3,
                means=[[127.5, 127.5, 127.5]] * 3,
                stds=[[127.5, 127.5, 127.5]] * 3,
                letterbox_fill=[0, 0, 0],
                letterbox_pad_position='top-left'),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=14,
        statistic_name='robodojo_arx_x5'),
)

themis = dict(
    transport=dict(
        host='127.0.0.1',
        port=5555,
        timeout_s=30.0,
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key='robodojo_arx_x5',
        image_encoding='rgb8',
        report_service_name='/fluxvla/report_evaluation',
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root='/root/projects/RoboDojo',
            device_id=1,
            action_mode='joint',
            headless=True,
            save_videos=False),
        model_client=dict(type='FluxVLAZMQModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        execute_horizon=50,
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/fluxthemis',
    ),
    ros_server=dict(
        dataset_section='eval',
        evaluation_reporting=dict(result_output_dir='work_dirs/fluxthemis'),
        device='cuda:0',
        workers=dict(
            startup_timeout_s=900.0,
            request_timeout_s=120.0,
            lease_timeout_s=900.0,
        ),
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context=dict(task_suite_name='robodojo_arx_x5'),
    ),
)
