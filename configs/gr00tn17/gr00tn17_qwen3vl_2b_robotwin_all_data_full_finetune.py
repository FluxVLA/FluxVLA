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
"""GR00T N1.7 full-data fine-tuning and evaluation on RoboTwin.

The data combines clean and randomized LeRobot v2.1 datasets for all 50
RoboTwin tasks (27,500 episodes), with three 480x640 (H x W) RGB cameras.
The policy uses 14-D absolute joint/gripper targets and mean/std statistics
shared by training and closed-loop evaluation.

Usage (16 GPUs):
    # Run on both nodes with NODE_RANK=0/1 and the same MASTER_ADDR.
    torchrun --nproc-per-node=8 --nnodes=2 \
        --node-rank=${NODE_RANK} --master-addr=${MASTER_ADDR} \
        --master-port=29500 scripts/train.py \
        --config \
        configs/gr00tn17/gr00tn17_qwen3vl_2b_robotwin_all_data_full_finetune.py \
        --work-dir work_dirs/gr00tn17_qwen3vl_2b_robotwin_all_data_full_finetune

Evaluation:
    torchrun --nproc-per-node=1 scripts/eval.py \
        --config \
        configs/gr00tn17/gr00tn17_qwen3vl_2b_robotwin_all_data_full_finetune.py \
        --ckpt-path <checkpoint.safetensors>

Evaluation defaults to all 50 tasks in both clean and random suites.
"""

_STATISTIC_NAME = 'robotwin_all'
_N17_INIT_CKPT = './checkpoints/GR00T-N1.7-3B'
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'

_QWEN3_VL_CONFIG = dict(
    architectures=['Qwen3VLForConditionalGeneration'],
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    vision_end_token_id=151653,
    tie_word_embeddings=False,
    text_config=dict(
        model_type='qwen3_vl_text',
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act='silu',
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters=dict(
            rope_type='default',
            rope_theta=5000000.0,
            mrope_section=[24, 20, 20],
            mrope_interleaved=True,
        ),
    ),
    vision_config=dict(
        model_type='qwen3_vl_vision',
        depth=24,
        hidden_size=1024,
        hidden_act='gelu_pytorch_tanh',
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2048,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[5, 11, 17],
        initializer_range=0.02,
    ),
)

_ACTIVE_TRACKERS = ('jsonl', )

_TASK_NAMES = [
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
_DATA_PATHS = [
    'datasets/robotwin_clean_lerobotv2.1',
    'datasets/robotwin_randomized_lerobotv2.1',
]

_N17_MODALITY_CONFIGS = dict(
    robotwin=dict(
        video=dict(
            delta_indices=[0],
            modality_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        ),
        state=dict(
            delta_indices=[0],
            modality_keys=['joints'],
        ),
        action=dict(
            delta_indices=list(range(40)),
            modality_keys=['joints'],
        ),
    ),
)

# State/action normalization uses the merged training dataset statistics.
# This processor metadata is consumed by image augmentation only.
_PROCESSOR_KWARGS = dict(
    modality_configs=_N17_MODALITY_CONFIGS,
    statistics=dict(robotwin={}),
    embodiment_id_mapping=dict(robotwin=0),
    max_state_dim=132,
    max_action_dim=132,
    max_action_horizon=40,
    use_percentiles=False,
    clip_outliers=False,
    use_relative_action=False,
    apply_sincos_state_encoding=False,
    formalize_language=True,
    use_albumentations=True,
    shortest_image_edge=None,
    crop_fraction=None,
    image_target_size=(256, 256),
    image_crop_size=(230, 230),
    state_dropout_prob=0.2,
    color_jitter_params=dict(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
    ),
    use_mean_std=True,
)

# Fine-tune embodiment slot 0, following the existing RoboTwin convention.
model = dict(
    type='GrootN17VLA',
    model_path=_N17_INIT_CKPT,
    embodiment_tag='robotwin',
    processor_kwargs=_PROCESSOR_KWARGS,
    state_dropout_prob=0.2,
    load_metadata=True,
    qwen3_runtime='compat_457',
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='GrootN17Qwen3Backbone',
        model_config=_QWEN3_VL_CONFIG,
        select_layer=16,
        reproject_vision=False,
        use_flash_attention=True,
        load_bf16=False,
        qwen3_runtime='compat_457',
    ),
    vla_head=dict(type='GrootN17ActionHead'),
    use_relative_action=False,
)

# Evaluation restores the complete FluxVLA checkpoint directly.
inference_model = model.copy()
inference_model.update(
    model_path=None,
    load_metadata=False,
    load_pretrained_weights=False,
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_DATA_PATHS,
                statistic_name=_STATISTIC_NAME,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                train_episode_fraction=1.0,
                repeat_to_full_length=False,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=0,
                        parquet_keys=[
                            'observation.state',
                            'timestamp',
                            'actions',
                            'info',
                            'stats',
                            'action_masks',
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
                        video_backend='torchcodec',
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_key='proprio',
                        action_key='action',
                        state_dim=132,
                        action_dim=132,
                        norm_type='mean_std',
                        clip_norm=False,
                        normalization_epsilon=1e-6,
                        preserve_input_dtype=True,
                    ),
                    dict(
                        type='PrepareStateActionTargets',
                        state_history_length=1,
                        action_horizon=40,
                        valid_action_dim=14,
                        state_dropout_prob=0.2,
                    ),
                    dict(
                        type='GrootN17ImageAugmentation',
                        embodiment_tag='robotwin',
                        image_key='images',
                        output_image_key='images',
                        train_mode=True,
                        processor_kwargs=_PROCESSOR_KWARGS,
                    ),
                    dict(
                        type='QWen2VLImageTransform',
                        img_key='images',
                        size=dict(
                            shortest_edge=65536,
                            longest_edge=16777216,
                        ),
                        patch_size=16,
                        temporal_patch_size=2,
                        merge_size=2,
                        image_mean=[0.5, 0.5, 0.5],
                        image_std=[0.5, 0.5, 0.5],
                        to_tensor=True,
                    ),
                    dict(
                        type='ProcessPromptsWithImage',
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=_QWEN_TOKENIZER_PATH,
                            padding_side='left',
                            trust_remote_code=False,
                        ),
                        max_len=512,
                        add_system=False,
                        add_assistant_stub=False,
                        task_pos='after_images',
                        image_tag_template='',
                        img_start='<|vision_start|>',
                        img_end='<|vision_end|>',
                        img_context_token='<|image_pad|>',
                        img_tokens_source='from_image_grid_thw',
                        image_grid_thw_key='image_grid_thw',
                        image_merge_size=2,
                        padding_side='left',
                        use_eos_as_pad=False,
                        truncate=False,
                        lowercase_task_description=True,
                        strip_task_punctuation=True,
                        attention_mask_dtype='int64',
                        output_keys=[
                            'lang_tokens',
                            'lang_masks',
                            'images',
                            'image_grid_thw',
                            'states',
                            'actions',
                            'action_masks',
                            'embodiment_ids',
                        ],
                    ),
                ],
                action_window_size=40,
                require_full_window=True,
            ),
        ],
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=None,
    max_epochs=5,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=1e-5,
    ),
    max_grad_norm=1.0,
    grad_accumulation_steps=2,
    sampler=None,
    save_iter_interval=5000,
    save_epoch_interval=1,
    max_keep_ckpts=10,
    collator=dict(
        type='DictCollator',
        keys=[
            'lang_tokens',
            'lang_masks',
            'images',
            'image_grid_thw',
            'states',
            'actions',
            'action_masks',
            'embodiment_ids',
        ],
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=_ACTIVE_TRACKERS,
        run_dir='work_dirs',
        grad_accumulation_steps=2,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='shard-grad-op',
    change_key_name=False,
)

eval = dict(
    type='RobotwinEvalRunner',
    task_suite_name=['clean', 'random'],
    model_family='groot_n17',
    task_list=_TASK_NAMES,
    instruction_type='unseen',
    eval_chunk_size=40,
    num_trials_per_task=100,
    seed=7,
    unnorm_key=_STATISTIC_NAME,
    mixed_precision_dtype='bf16',
    save_video=False,
    dataset=dict(
        type='PrivateInferenceDataset',
        inject_model_path=False,
        embodiment_id=0,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_key='proprio',
                action_key=None,
                state_dim=132,
                action_dim=132,
                norm_type='mean_std',
                clip_norm=False,
                normalization_epsilon=1e-6,
                preserve_input_dtype=True,
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=40,
                valid_action_dim=14,
                state_dropout_prob=0.0,
            ),
            dict(
                type='GrootN17ImageAugmentation',
                embodiment_tag='robotwin',
                image_key='images',
                output_image_key='images',
                train_mode=False,
                processor_kwargs=_PROCESSOR_KWARGS,
            ),
            dict(
                type='QWen2VLImageTransform',
                img_key='images',
                size=dict(
                    shortest_edge=65536,
                    longest_edge=16777216,
                ),
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
                to_tensor=True,
            ),
            dict(
                type='ProcessPromptsWithImage',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_QWEN_TOKENIZER_PATH,
                    padding_side='left',
                    trust_remote_code=False,
                ),
                max_len=512,
                add_system=False,
                add_assistant_stub=False,
                task_pos='after_images',
                image_tag_template='',
                img_start='<|vision_start|>',
                img_end='<|vision_end|>',
                img_context_token='<|image_pad|>',
                img_tokens_source='from_image_grid_thw',
                image_grid_thw_key='image_grid_thw',
                image_merge_size=2,
                padding_side='left',
                use_eos_as_pad=False,
                truncate=False,
                lowercase_task_description=True,
                strip_task_punctuation=True,
                attention_mask_dtype='int64',
                output_keys=[
                    'lang_tokens',
                    'lang_masks',
                    'images',
                    'image_grid_thw',
                    'states',
                    'embodiment_ids',
                ],
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=14,
        normalize_gripper_action=False,
        invert_gripper_action=False,
    ),
)
