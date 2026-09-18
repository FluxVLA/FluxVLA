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
"""Candy GR00T N1.7 training and Oli RTC inference.

This is a standalone port of
``gr00t_n17_candy_full1783_direct_rtc32_state43_action52_30ep_fluxvla.py``.
The source recipe already stores base x/y/yaw as deltas and trains joint and
finger targets directly, so inference uses ordinary min/max denormalization;
it must not apply the PI0.5 joint-relative postprocessor.
"""

_DATA_ROOTS = [
    '/mnt/data/cpfs/limx_embmc/VLA_Data/Fixed-Feet-Mani/hf_cache/lerobot/'
    f'lerobot/{name}' for name in [
        '0611_2_candy_subtask_delta_base_relabel_2_v21',
        '0612_candy_subtask_delta_base_relabel_2_v21',
        '0616_candy_subtask_delta_base_relabel_2_v21',
        '0618_candy_subtask_delta_base_relabel_2_v21',
        '0622_candy_subtask_delta_base_relabel_2_v21',
        '0623_candy_subtask_delta_base_relabel_2_v21',
        '0624_candy_subtask_delta_base_relabel_2_v21',
        '0709_candy_subtask_delta_base_relabel_2_v21',
    ]
]
_N17_INIT_CKPT = './checkpoints/GR00T-N1.7-3B'
_STATISTIC_NAME = 'candy_0611_0709_full1783_direct_state43_action52'
_EMBODIMENT_KEY = 'new_embodiment'
_EMBODIMENT_ID = 10
_STATE_DIM = 43
_ACTION_DIM = 52
_MODEL_DIM = 132
_ACTION_HORIZON = 32
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'
_VALID_WINDOWS = 363271
_GLOBAL_BATCH_SIZE = 256
_MAX_STEPS = (_VALID_WINDOWS // _GLOBAL_BATCH_SIZE) * 30
_CHECKPOINT_STEPS = [
    5676,
    11352,
    17028,
    22704,
    28380,
    34056,
    39732,
    42570,
]

assert _MAX_STEPS == 42570

_MODEL_ACTION_DIM_MASK = [True] * _ACTION_DIM + [False] * (
    _MODEL_DIM - _ACTION_DIM)

_TASK_PROMPTS = {
    '1': ('pick up the white candy and place it in the left section of the '
          'snack tray with left arm'),
    '2': ('pick up the purple candy and place it in the right section of the '
          'snack tray with left arm'),
    '3': ('pick up the red candy and place it in the middle section of the '
          'snack tray with left arm'),
}

_PREPARE_POSE = [
    -0.0376718,
    0.0743988,
    0.0287065,
    -0.00910288,
    -0.0216001,
    -0.0656817,
    -0.0658258,
    -0.101149,
    -0.178495,
    0.0896175,
    -0.0522503,
    0.112336,
    -0.0160254,
    -0.00427001,
    0.00303777,
    -0.067359,
    0.433261,
    0.0380359,
    0.355374,
    -0.471247,
    -1.27736,
    0.191936,
    -0.771603,
    0.118538,
    0.336419,
    -0.266369,
    0.264269,
    -1.5451,
    -0.0991593,
    -0.592179,
    -0.182942,
] + [0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

_MODALITY_CONFIGS = {
    _EMBODIMENT_KEY:
    dict(
        video=dict(
            delta_indices=[0],
            modality_keys=['head', 'left_wrist'],
        ),
        state=dict(
            delta_indices=[0],
            modality_keys=['joint_positions', 'brainco_hand_qpos'],
        ),
        action=dict(
            delta_indices=list(range(_ACTION_HORIZON)),
            modality_keys=[
                'joint_positions',
                'base_position_command',
                'base_rotation_command',
                'brainco_hand_qpos',
            ],
            action_dim_mask=[float(value) for value in _MODEL_ACTION_DIM_MASK],
        ),
    ),
}

_PROCESSOR_KWARGS = dict(
    modality_configs=_MODALITY_CONFIGS,
    statistics={_EMBODIMENT_KEY: {}},
    embodiment_id_mapping={_EMBODIMENT_KEY: _EMBODIMENT_ID},
    max_state_dim=_MODEL_DIM,
    max_action_dim=_MODEL_DIM,
    max_action_horizon=_ACTION_HORIZON,
    use_percentiles=False,
    clip_outliers=True,
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
)

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
        rms_norm_eps=1e-6,
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

# The source config's ``action_horizon_override=32`` belongs to its older model
# wrapper. The native FluxVLA runtime reads this value from model_config.
model = dict(
    type='GrootN17VLA',
    model_path=_N17_INIT_CKPT,
    model_config=dict(
        action_horizon=_ACTION_HORIZON,
        state_dropout_prob=0.2,
    ),
    embodiment_tag=_EMBODIMENT_KEY,
    processor_kwargs=_PROCESSOR_KWARGS,
    load_metadata=True,
    load_pretrained_weights=True,
    qwen3_runtime='compat_457',
    freeze_vision_backbone=False,
    freeze_llm_backbone=False,
    freeze_vlm_backbone=False,
    freeze_projector=False,
    vlm_backbone=dict(
        type='GrootN17Qwen3Backbone',
        model_config=_QWEN3_VL_CONFIG,
        select_layer=16,
        reproject_vision=False,
        use_flash_attention=True,
        load_bf16=False,
        qwen3_runtime='compat_457',
    ),
    vla_head=dict(
        type='GrootN17ActionHead',
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',
        ),
    ),
)

# A saved FluxVLA checkpoint is self-contained; inference must build the same
# architecture on meta without reopening the original N1.7 initializer.
inference_model = model.copy()
inference_model.update(
    model_path=None,
    load_metadata=False,
    load_pretrained_weights=False,
)

_TRAIN_TRANSFORMS = [
    dict(
        type='ProcessParquetInputs',
        embodiment_id=_EMBODIMENT_ID,
        parquet_keys=[
            'observation.state',
            'timestamp',
            'actions',
            'info',
            'stats',
            'action_masks',
        ],
        video_keys=[
            'observation.images.head',
            'observation.images.left_wrist',
        ],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        },
    ),
    dict(
        type='NormalizeStatesAndActions',
        state_key='proprio',
        action_key='action',
        state_dim=_MODEL_DIM,
        action_dim=_MODEL_DIM,
        norm_type='min_max',
        action_norm_mask=[True] * _ACTION_DIM,
        clip_norm=True,
        normalization_epsilon=0.0,
        preserve_input_dtype=True,
    ),
    dict(
        type='PrepareStateActionTargets',
        state_history_length=1,
        action_horizon=_ACTION_HORIZON,
        valid_action_dim=_ACTION_DIM,
        state_dropout_prob=0.2,
    ),
    dict(
        type='GrootN17ImageAugmentation',
        embodiment_tag=_EMBODIMENT_KEY,
        image_key='images',
        output_image_key='images',
        train_mode=True,
        processor_kwargs=_PROCESSOR_KWARGS,
    ),
    dict(
        type='QWen2VLImageTransform',
        img_key='images',
        size=dict(shortest_edge=65536, longest_edge=16777216),
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
        max_len=180,
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
            'sample_weight',
        ],
    ),
]

train_dataloader = dict(
    per_device_batch_size=32,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        statistic_name=_STATISTIC_NAME,
        auto_compute_statistics=dict(profile='absolute'),
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action'],
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        datasets=dict(
            type='ParquetDatasetV3',
            data_root_path=_DATA_ROOTS,
            statistic_name=_STATISTIC_NAME,
            action_key='action',
            use_delta=False,
            window_start_idx=0,
            transforms=_TRAIN_TRANSFORMS,
            action_window_size=_ACTION_HORIZON,
            require_full_window=True,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=_MAX_STEPS,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-5),
    max_grad_norm=1.0,
    grad_accumulation_steps=1,
    sampler=None,
    save_iter_interval=5676,
    save_step_milestones=_CHECKPOINT_STEPS,
    save_model_only=True,
    save_epoch_interval=30,
    max_keep_ckpts=2,
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
            'sample_weight',
        ],
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.05),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='shard-grad-op',
    change_key_name=False,
    seed=42,
)

inference = dict(
    type='OliRTCInferenceRunner',
    seed=7,
    state_dim=_STATE_DIM,
    action_chunk=_ACTION_HORIZON,
    publish_rate=30,
    max_publish_step=10000,
    execute_horizon=16,
    async_remaining_actions_threshold=9,
    rtc_config=dict(enabled=True, method='prefix', prefix_len=7),
    interactive=True,
    default_prompt_id='1',
    default_execution_count=1000,
    prepare_pose=_PREPARE_POSE,
    prepare_pose_duration_sec=4.0,
    prepare_pose_prompt_id='0',
    apply_jpeg_compression=True,
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    camera_names=['head', 'left_wrist'],
    task_descriptions=_TASK_PROMPTS,
    dataset=dict(
        type='PrivateInferenceDataset',
        inject_model_path=False,
        statistic_name=_STATISTIC_NAME,
        embodiment_id=_EMBODIMENT_ID,
        img_keys=['head', 'left_wrist'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_key='proprio',
                action_key=None,
                state_dim=_MODEL_DIM,
                norm_type='min_max',
                clip_norm=True,
                normalization_epsilon=0.0,
                preserve_input_dtype=True,
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=_ACTION_HORIZON,
                valid_action_dim=_ACTION_DIM,
                state_dropout_prob=0.0,
            ),
            dict(
                type='GrootN17ImageAugmentation',
                embodiment_tag=_EMBODIMENT_KEY,
                image_key='images',
                output_image_key='images',
                train_mode=False,
                processor_kwargs=_PROCESSOR_KWARGS,
            ),
            dict(
                type='QWen2VLImageTransform',
                img_key='images',
                size=dict(shortest_edge=65536, longest_edge=16777216),
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
                max_len=180,
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
        statistic_name=_STATISTIC_NAME,
        action_dim=_ACTION_DIM,
        norm_type='min_max',
    ),
    operator=dict(
        type='OliOperator',
        control_backend='mros',
        hand_mode='finger',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic='/left_wrist_camera/color/image_raw/compressed',
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        finger_force_levels=(2.0, 2.0),
        teleop_wbt_topic='/teleop_cmd_WBT',
    ),
)
