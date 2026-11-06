"""FastWAM full fine-tuning and evaluation on RoboDojo ARX-X5.

This standalone config reproduces the evaluated six-epoch run: three cameras,
absolute 14D joint actions, a 32-step action horizon sampled at stride four,
and full-shard FSDP optimization at a fixed 1e-6 learning rate.
"""

_ROBODOJO_DATA_ROOT = './datasets/RoboDojo_lerobot_v21_video'
_ROBODOJO_DATA_ROOTS = [_ROBODOJO_DATA_ROOT]
_STATISTIC_NAME = 'robodojo_arx_x5'
_FASTWAM_CHECKPOINT = (
    './checkpoints/fastwam_base_full/fastwam_base_full.safetensors')
_FASTWAM_TOKENIZER = './checkpoints/fastwam_base_full/tokenizer'
_TEXT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')
_EXECUTE_HORIZON = 16

_ROBODOJO_GENERALIZATION_BASE_TASKS = (
    'stack_bowls',
    'push_T',
    'pack_objects_into_box',
    'fold_clothes',
    'hang_mugs',
    'sweep_blocks',
    'pour_liquid_into_cup',
    'make_toast',
    'arrange_largest_number',
    'sort_nesting_dolls_by_size',
    'store_laptop_and_headphones',
    'stack_blocks',
)
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

# Exact frame-level statistics from all 1,859,602 RoboDojo samples.
_ROBODOJO_STATS = {
    'robodojo_arx_x5': {
        'action': {
            'count':
            1859602,
            'max': [
                1.7537026405334473, 3.0546703338623047, 3.8525187969207764,
                2.0974574089050293, 1.8103265762329102, 3.1366536617279053,
                1.0, 1.6732014417648315, 3.3797459602355957, 4.252756595611572,
                2.5287232398986816, 3.1440579891204834, 3.0381109714508057, 1.0
            ],
            'mean': [
                -0.20032191276550293, 0.925260603427887, 0.6879700422286987,
                -0.33993399143218994, 0.06580745428800583,
                0.003984309732913971, 0.771653950214386, 0.17281471192836761,
                0.8216421008110046, 0.6193687319755554, -0.3603578805923462,
                -0.06270533800125122, 0.0036061492282897234, 0.7855709195137024
            ],
            'min': [
                -1.4863648414611816, -0.2586335241794586, -0.06850092858076096,
                -2.2642147541046143, -1.4876768589019775, -3.0168392658233643,
                -3.212450869184004e-17, -1.4924354553222656,
                -0.27809593081474304, -0.18774886429309845,
                -2.5354628562927246, -3.0663280487060547, -3.0486762523651123,
                -3.212450869184004e-17
            ],
            'q01': [
                -1.051365569829941, -3.5348887174584814e-14,
                1.741881830823392e-16, -1.5984010362625123,
                -0.6003574305772781, -1.6678147149085998, 0.0,
                -0.4509398394823074, -2.5859282299029243e-14,
                1.234041714549172e-16, -1.64055180311203, -1.2633746123313903,
                -1.7645285475254058, 0.0
            ],
            'q99': [
                0.5431358617544174, 2.495765209197998, 2.492974226474762,
                1.3241519677639007, 1.2496635341644287, 1.7392305016517635,
                1.0, 1.0814965963363647, 2.4167031812667843,
                2.3470158338546754, 1.141731116771698, 0.5208872479200363,
                1.489715996980667, 1.0
            ],
            'std': [
                0.34360724687576294, 0.8902974724769592, 0.7318998575210571,
                0.6505846977233887, 0.29849204421043396, 0.5843351483345032,
                0.34186822175979614, 0.3206902742385864, 0.8718543648719788,
                0.7120647430419922, 0.6016045212745667, 0.28415030241012573,
                0.5370385050773621, 0.34119123220443726
            ]
        },
        'proprio': {
            'count':
            1859602,
            'max': [
                1.7537026405334473, 3.0546703338623047, 3.8525187969207764,
                2.0974574089050293, 1.8103265762329102, 3.1366536617279053,
                1.0, 1.6732014417648315, 3.3797459602355957, 4.252756595611572,
                2.5287232398986816, 3.1440579891204834, 3.0381109714508057, 1.0
            ],
            'mean': [
                -0.20028550922870636, 0.9250829815864563, 0.6878653764724731,
                -0.3399182856082916, 0.06579338014125824, 0.004018673673272133,
                0.7717168927192688, 0.17282654345035553, 0.8213191032409668,
                0.6190997958183289, -0.3601549565792084, -0.06264610588550568,
                0.0037324042059481144, 0.7856565713882446
            ],
            'min': [
                -1.4863648414611816, -0.2586335241794586, -0.06850092858076096,
                -2.2642147541046143, -1.4876768589019775, -3.0168392658233643,
                -3.212450869184004e-17, -1.4924354553222656,
                -0.27809593081474304, -0.18774886429309845,
                -2.5354628562927246, -3.0663280487060547, -3.0486762523651123,
                -3.212450869184004e-17
            ],
            'q01': [
                -1.051365569829941, -3.52302449637288e-14,
                1.6996893991849812e-16, -1.5984010362625123,
                -0.6003412520885467, -1.6678147149085998, 0.0,
                -0.4500492787361145, -2.5859282299029243e-14,
                1.194697284288627e-16, -1.6398817586898804, -1.263367258310318,
                -1.760098042488098, 0.0
            ],
            'q99': [
                0.5431358617544174, 2.495765209197998, 2.492974226474762,
                1.323737324476242, 1.2496635341644287, 1.7391636216640465, 1.0,
                1.0814965963363647, 2.416655488014221, 2.346989154815674,
                1.1415718960762022, 0.5208872479200363, 1.4889542925357817, 1.0
            ],
            'std': [
                0.34361061453819275, 0.8903892636299133, 0.7319481372833252,
                0.6505074501037598, 0.29847919940948486, 0.5842824578285217,
                0.34185686707496643, 0.3206428587436676, 0.8718962669372559,
                0.7120621204376221, 0.6014657616615295, 0.2840825021266937,
                0.5366978049278259, 0.3415590524673462
            ]
        }
    }
}

model = dict(
    type='FastWAMVLA',
    pretrained_name_or_path=_FASTWAM_CHECKPOINT,
    pretrained_skip_prefixes=[
        'vla_head.mot.mixtures.action.action_encoder.',
        'vla_head.mot.mixtures.action.head.',
        'vla_head.proprio_encoder.',
    ],
    torch_dtype='bf16',
    num_views=3,
    frame_window_size=9,
    proprio_dim=14,
    action_horizon=32,
    num_inference_steps=10,
    action_norm_type='min_max',
    mot_checkpoint_mixed_attn=True,
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='Wan22Backbone',
        text_embed_cache_context_len=128,
        text_embed_cache_device='cpu',
        text_embed_cache_size=256,
    ),
    vla_head=dict(
        type='FastWAMHead',
        action_dit_config=dict(
            action_dim=14,
            attn_head_dim=128,
            eps=1e-06,
            ffn_dim=4096,
            freq_dim=256,
            hidden_dim=1024,
            num_heads=24,
            num_layers=30,
            text_dim=4096,
            use_gradient_checkpointing=True,
        ),
        action_scheduler=dict(
            infer_shift=5.0,
            num_train_timesteps=1000,
            train_shift=5.0,
        ),
        loss=dict(
            lambda_action=1.0,
            lambda_video=1.0,
        ),
        video_dit_config=dict(
            action_conditioned=False,
            action_dim=14,
            action_group_causal_mask_mode='group_diagonal',
            attn_head_dim=128,
            eps=1e-06,
            ffn_dim=14336,
            freq_dim=256,
            fuse_vae_embedding_in_latents=True,
            has_image_input=False,
            hidden_dim=3072,
            in_dim=48,
            num_heads=24,
            num_layers=30,
            out_dim=48,
            patch_size=[1, 2, 2],
            require_clip_embedding=False,
            require_vae_embedding=False,
            seperated_timestep=True,
            text_dim=4096,
            use_gradient_checkpointing=True,
            video_attention_mask_mode='first_frame_causal',
        ),
        video_scheduler=dict(
            infer_shift=5.0,
            num_train_timesteps=1000,
            train_shift=5.0,
        ),
    ),
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=2,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        dataset_statistics=_ROBODOJO_STATS,
        datasets=dict(
            type='ParquetDataset',
            action_key='action',
            action_window_size=32,
            data_root_path=_ROBODOJO_DATA_ROOTS,
            frame_sample_stride=4,
            frame_window_size=9,
            statistic_name=_STATISTIC_NAME,
            supervise_terminal_padding=True,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=0,
                    name_mappings={
                        'actions': ['actions'],
                        'observation.state': ['states'],
                    },
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_backend='torchcodec',
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                ),
                dict(
                    type='ResizeImages',
                    backend='torchvision',
                    height=224,
                    scale_to_unit_interval=True,
                    width=224,
                ),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=14,
                    action_key='action',
                    norm_type='min_max',
                    output_dtype='float32',
                    state_dim=14,
                    state_key='proprio',
                ),
                dict(
                    type='PrepareVideo',
                    combine_view_masks=True,
                    frame_window_size=9,
                    num_views=3,
                    tile_direction='horizontal',
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    max_len=128,
                    prompt_template=_TEXT_PROMPT_TEMPLATE,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_FASTWAM_TOKENIZER,
                    ),
                    use_conversation=False,
                ),
            ],
            use_delta=False,
            window_start_idx=0,
        ),
        name_mappings={
            'action': ['action'],
            'observation.state': ['proprio'],
        },
        reshuffle_each_epoch=True,
        seed=42,
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
    ),
)

runner = dict(
    type='FSDPTrainRunner',
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
    dataset_sharding_strategy='blockwise',
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    evaluator=dict(
        type='training-eval',
        eval_every=1000,
        num_inference_steps=10,
        save_video=True,
        seed=42,
        video_fps=8,
    ),
    fsdp_wrap_policy='execution-block',
    grad_accumulation_steps=1,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay-min-lr',
        betas=[0.9, 0.95],
        min_lr_ratio=1.0,
        warmup_ratio=0.0,
        weight_decay_style='uniform',
    ),
    max_epochs=6,
    max_grad_norm=1.0,
    max_keep_ckpts=6,
    max_steps=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=['jsonl', 'wandb'],
        grad_accumulation_steps=1,
        run_dir='work_dirs',
        window_size=1,
    ),
    mixed_precision_dtype='bf16',
    optimizer=dict(
        type='AdamW',
        fused=True,
        lr=1e-06,
        weight_decay=0.01,
    ),
    reduce_in_full_precision=True,
    sampler=None,
    save_epoch_interval=1,
    save_iter_interval=10000,
    sharding_strategy='full-shard',
)

eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='fastwam',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    dataset=dict(
        type='RoboDojoEvalDataset',
        state_dtype='fp32',
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[
                    [3, 224, 224],
                    [3, 224, 224],
                    [3, 224, 224],
                ],
                means=[
                    [127.5, 127.5, 127.5],
                    [127.5, 127.5, 127.5],
                    [127.5, 127.5, 127.5],
                ],
                stds=[
                    [127.5, 127.5, 127.5],
                    [127.5, 127.5, 127.5],
                    [127.5, 127.5, 127.5],
                ],
            ),
            dict(
                type='StateFromInputs',
                norm_type='min_max',
                stat_key='proprio',
                state_dim=14,
            ),
            dict(
                type='LiberoPromptFromInputs',
                max_len=128,
                prompt_template=_TEXT_PROMPT_TEMPLATE,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_FASTWAM_TOKENIZER,
                ),
                use_conversation=False,
            ),
            dict(
                type='PrepareVideo',
                combine_view_masks=True,
                frame_window_size=1,
                num_views=3,
                tile_direction='horizontal',
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        action_dim=14,
        norm_type='min_max',
        statistic_name=_STATISTIC_NAME,
    ),
)

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key=_STATISTIC_NAME,
    ),
    evaluation=dict(
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root='/root/projects/RoboDojo',
            device_id=1,
            action_mode='joint',
            headless=True,
            save_videos=False,
        ),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        base_seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        execute_horizon=_EXECUTE_HORIZON,
        stop_on_success=True,
    ),
    server=dict(
        runner=dict(type='FluxVLAServingRunner'),
        evaluation_reporting=dict(enabled=True),
        denormalize_context=dict(task_suite_name=_STATISTIC_NAME),
    ),
)

seed = 42
eval_dataset = None
val_dataloader = None
