"""SmolVLA full-data fine-tuning and evaluation on RoboTwin.

The data combines clean and randomized LeRobot v2.1 datasets for all 50
RoboTwin tasks (27,500 episodes), with three 480x640 (H x W) RGB cameras.
The policy uses 14-D absolute joint/gripper targets and mean/std statistics
shared by training and closed-loop evaluation.

The model follows the RoboDojo config: full VLM finetuning and a 50-step
action horizon. Training runs for 5 epochs (59,330 optimizer updates on
16 GPUs), with 3% LR warmup and cosine decay over the remaining steps.
The batch settings target sixteen GPUs:
    32 samples/GPU * 16 GPUs * 1 accumulation step = 512.

Example:
    # Run on both nodes with NODE_RANK=0/1 and the same MASTER_ADDR.
    torchrun --nproc_per_node=8 --nnodes=2 \
        --node-rank=${NODE_RANK} --master-addr=${MASTER_ADDR} \
        --master-port=29500 scripts/train.py \
        --config configs/smolvla/smolvla_robotwin_all_data_full_finetune.py \
        --work-dir work_dirs/smolvla_robotwin_all_data_full_finetune

Evaluation defaults to all 50 tasks in both clean and random suites.
"""

_ROBOTWIN_DATA_ROOTS = [
    'datasets/robotwin_lerobotv2.1/robotwin_clean_lerobotv2.1',
    'datasets/robotwin_lerobotv2.1/robotwin_randomized_lerobotv2.1',
]

# All 50 RoboTwin tasks, matching the upstream RoboTwin benchmark.
_ROBOTWIN_TASK_LIST = [
    'adjust_bottle',
    'beat_block_hammer',
    'blocks_ranking_rgb',
    'blocks_ranking_size',
    'click_alarmclock',
    'click_bell',
    'dump_bin_bigbin',
    'grab_roller',
    'handover_block',
    'handover_mic',
    'hanging_mug',
    'lift_pot',
    'move_can_pot',
    'move_pillbottle_pad',
    'move_playingcard_away',
    'move_stapler_pad',
    'open_laptop',
    'open_microwave',
    'pick_diverse_bottles',
    'pick_dual_bottles',
    'place_a2b_left',
    'place_a2b_right',
    'place_bread_basket',
    'place_bread_skillet',
    'place_burger_fries',
    'place_can_basket',
    'place_cans_plasticbox',
    'place_container_plate',
    'place_dual_shoes',
    'place_empty_cup',
    'place_fan',
    'place_mouse_pad',
    'place_object_basket',
    'place_object_scale',
    'place_object_stand',
    'place_phone_stand',
    'place_shoe',
    'press_stapler',
    'put_bottles_dustbin',
    'put_object_cabinet',
    'rotate_qrcode',
    'scan_object',
    'shake_bottle',
    'shake_bottle_horizontally',
    'stack_blocks_three',
    'stack_blocks_two',
    'stack_bowls_three',
    'stack_bowls_two',
    'stamp_seal',
    'turn_switch',
]

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

inference_model = dict(model, pretrained_name_or_path=None)

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
        statistic_name='robotwin_all',
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_ROBOTWIN_DATA_ROOTS,
                action_key='action',
                use_delta=False,
                statistic_name='robotwin_all',
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
    max_epochs=5,
    max_steps=None,
    grad_accumulation_steps=1,
    deterministic_algorithms=True,
    seed=0,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=0.01,
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
        warmup_ratio=0.03,
        min_lr=2.5e-6,
    ),
    save_iter_interval=10000,
    max_keep_ckpts=10,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    type='RobotwinEvalRunner',
    task_suite_name=['clean', 'random'],
    model_family='smolvla',
    task_list=_ROBOTWIN_TASK_LIST,
    eval_chunk_size=50,
    num_trials_per_task=100,
    seed=0,
    unnorm_key='robotwin_all',
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
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
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=14,
        statistic_name='robotwin_all'),
)
