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

# PI0.5 RoboTwin finetune aligned with the official RoboTwin OpenPI recipe
# (openpi TrainConfig `pi05_aloha_full_base`, adapt_to_pi=False):
#   - delta joint actions with absolute grippers (make_bool_mask(6,-1,6,-1))
#   - q01/q99 quantile normalization computed over the transformed (delta)
#     action space, native-dimension discrete state prompt, padding after
#     tokenization
#   - AdamW(2.5e-5, betas=(0.9, 0.95), eps=1e-8, wd=1e-10 on all params),
#     linear warmup (3%) -> cosine decay to 2.5e-6, grad clip 1.0
#   - EMA 0.99 (checkpoints store EMA weights for evaluation), seed 42
#   - batch/duration follow the in-house RoboTwin convention instead of the
#     official recipe: global batch = 8 x GPU count and 5 training epochs
#     (official: global batch 64, 20k steps, warmup 1000, decay 30000)
# Unlike configs/pi05/pi05_paligemma_aloha_full_finetune.py there is no
# JointSignTransform / gripper coordinate conversion: the official RoboTwin
# configs pass adapt_to_pi=False, which leaves state/actions untouched.

# Joint dims are deltas w.r.t. the current state; grippers stay absolute.
_ALOHA_DELTA_MASK = [True] * 6 + [False] + [True] * 6 + [False]

model = dict(
    type='PI05FlowMatching',
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=50,
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    # Match the OpenPI flow-matching objective.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/pi05_base/model.safetensors',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=14,
    # Supervise all padded model dimensions, as in OpenPI.
    loss_action_dim=32,
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=42,
        reshuffle_each_epoch=True,
        # Keep state and action statistics separate: action statistics are
        # computed over delta actions and must not reuse observation.state.
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        # Compute q01/q99/mean/std over the TRANSFORMED action space (deltas
        # for joints, absolute grippers) on rank zero at startup, exactly like
        # openpi scripts/compute_norm_stats.py. The result is written to
        # <work_dir>/dataset_statistics.json, which RobotwinEvalRunner also
        # reads at evaluation time.
        auto_compute_statistics=dict(
            profile='absolute',
            delta_mask=_ALOHA_DELTA_MASK,
        ),
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=[
                    'datasets/robotwin_clean_lerobotv2.1',
                    'datasets/robotwin_randomized_lerobotv2.1',
                ],
                # Official RoboTwin repacks {"actions": "action"}: use the
                # recorded action column, with the chunk starting at the
                # current frame (OpenPI delta_timestamps convention).
                action_key='action',
                action_window_size=50,
                window_start_idx=0,
                # OpenPI/LeRobot supervises repeated terminal hold actions.
                supervise_terminal_padding=True,
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
                            'observation.images.cam_right_wrist'
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        },
                        # Pin the decoder so an optional torchcodec install
                        # cannot silently change training inputs.
                        video_backend='pyav'),
                    dict(type='RelativeActions', mask=_ALOHA_DELTA_MASK),
                    # Normalize at native dimension; padding happens after
                    # prompt tokenization, matching OpenPI.
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=None,
                        state_dim=None,
                        state_key='proprio',
                        action_key='action',
                        norm_type='quantile',
                        output_dtype='float32'),
                    dict(type='PreparePromptWithState'),
                    dict(
                        type='ProcessPrompts',
                        max_len=200,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'checkpoints/pi05_base',  # noqa: E501
                        )),
                    dict(type='PadStatesAndActions', model_action_dim=32),
                    dict(
                        type='ResizeImagesWithPad',
                        height=224,
                        width=224,
                        backend='pil'),
                    dict(type='SimpleNormalizeImages'),
                    dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
                ])
        ]))

runner = dict(
    type='FSDPTrainRunner',
    # In-house RoboTwin convention: global batch = 8 x GPU count (e.g. 16
    # GPUs -> 128) and epoch-based duration. The official OpenPI recipe is
    # global batch 64 / max_steps=20_000 (warmup_steps=1000,
    # decay_steps=30000) with grad accumulation sized to hold batch 64.
    max_epochs=3,
    ema_decay=0.99,
    seed=42,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    # BF16 compute with FP32 sharded master parameters and reductions.
    sharding_strategy='global-shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        # Ratio-based warmup and full-run decay so the schedule follows the
        # epoch-derived total step count on any GPU configuration.
        warmup_ratio=0.03,
        min_lr=2.5e-6),
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'checkpoints/pi05_base',  # noqa: E501
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        # OpenPI reports the mean of each 100-step logging interval.
        window_size=100),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)

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

eval = dict(
    type='RobotwinEvalRunner',
    model_family='pi05',
    task_list=_ROBOTWIN_TASK_LIST,
    eval_chunk_size=10,
    num_trials_per_task=100,
    seed=7,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        # Mirror the training pipeline exactly (minus augmentation).
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                action_dim=None,
                state_dim=None,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(type='PreparePromptWithState'),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    'checkpoints/pi05_base',
                )),
            dict(type='PadStatesAndActions', model_action_dim=32),
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages'),
        ]),
    # Quantile-denormalize the delta actions, then add the current raw state
    # back onto the joint dimensions (grippers remain absolute).
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        delta_action_mask=_ALOHA_DELTA_MASK,
        norm_type='quantile',
        action_dim=14,
    ),
)
