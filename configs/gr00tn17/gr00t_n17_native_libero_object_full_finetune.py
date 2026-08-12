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
"""Native GR00T N1.7 LIBERO-object parquet fine-tuning config."""

_SUITE = 'libero_object'
_DATASET_NAME = 'libero_object_no_noops_lerobotv2.1'
_STATISTIC_NAME = 'libero_object_no_noops_native'

_LIBERO_DATA_ROOT = f'./datasets/{_DATASET_NAME}'
_N17_INIT_CKPT = './checkpoints/GR00T-N1.7-3B'
_QWEN_TOKENIZER_PATH = ('fluxvla/models/third_party_models/qwen3_tokenizer')
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
_ACTIVE_TRACKERS = ('jsonl', )

_N17_STATE_LAYOUT = (
    ('x', 1),
    ('y', 1),
    ('z', 1),
    ('roll', 1),
    ('pitch', 1),
    ('yaw', 1),
    ('gripper', 2),
)
_N17_ACTION_LAYOUT = (
    ('x', 1),
    ('y', 1),
    ('z', 1),
    ('roll', 1),
    ('pitch', 1),
    ('yaw', 1),
    ('gripper', 1),
)


def _split_n17_statistics(flat_statistics, layout):
    statistics = {}
    offset = 0
    for key, dim in layout:
        statistics[key] = {
            name: values[offset:offset + dim]
            for name, values in flat_statistics.items()
        }
        offset += dim
    return statistics


# Statistics computed from the matching no-noops training parquet data.
_N17_FLAT_STATISTICS = {
    'state': {
        'min': [
            -0.17652566730976105, -0.28169912099838257, 0.008128181099891663,
            2.2889645099639893, -1.8830392360687256, -1.0428133010864258,
            0.0006782737909816206, -0.041782498359680176
        ],
        'max': [
            0.14580604434013367, 0.3322660028934479, 0.38492220640182495,
            3.4003379344940186, 0.7954724431037903, 0.6642153263092041,
            0.04104502499103546, -0.0005043679848313332
        ],
        'mean': [
            -0.029590097511622324, -0.007297964757377373, 0.20263691224329605,
            3.107347627230808, -0.21535565470621534, -0.11427672217768083,
            0.029502615037539585, -0.030672144202923204
        ],
        'std': [
            0.06715783727495737, 0.17582774832397174, 0.07806197322485103,
            0.08837916027138361, 0.33845984937441376, 0.2067143161337991,
            0.00947681687068548, 0.009059544368504133
        ],
        'q01': [
            -0.149256848692894, -0.2592130792140961, 0.009932418912649154,
            2.7467152214050294, -1.401689453125, -0.6867975068092346,
            0.008699759356677533, -0.04015225619077682
        ],
        'q99': [
            0.09154553085565566, 0.2911862111091613, 0.3361880838871002,
            3.260209331512451, 0.3196117353439328, 0.39586576938629137,
            0.03989199593663215, -0.010282181538641466
        ]
    },
    'action': {
        'min': [
            -0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448,
            -0.29035714268684387, -0.32892856001853943, 0.0
        ],
        'max': [
            0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488,
            0.35035714507102966, 0.1810714304447174, 1.0
        ],
        'mean': [
            0.07127183159196328, 0.13518370972847993, -0.046194642259443225,
            0.0011625276173484506, 0.007098307809942585, -0.015082184949062672,
            0.46515324845117295
        ],
        'std': [
            0.26873661352251504, 0.4385525563650988, 0.4474788640468074,
            0.024271765379373398, 0.04932243328601301, 0.04218624160534833,
            0.4987842257994289
        ],
        'q01': [
            -0.5410714149475098, -0.8758928775787354, -0.9375,
            -0.06964285671710968, -0.11678571254014969, -0.1607142835855484,
            0.0
        ],
        'q99': [
            0.8464285731315613, 0.84375, 0.9375, 0.08142857253551483,
            0.14884285390376858, 0.08571428805589676, 1.0
        ]
    }
}
_N17_STATISTICS = dict(
    libero_sim=dict(
        state=_split_n17_statistics(_N17_FLAT_STATISTICS['state'],
                                    _N17_STATE_LAYOUT),
        action=_split_n17_statistics(_N17_FLAT_STATISTICS['action'],
                                     _N17_ACTION_LAYOUT),
    ))
_N17_MODALITY_CONFIGS = dict(
    libero_sim=dict(
        video=dict(
            delta_indices=[0],
            modality_keys=['image', 'wrist_image'],
        ),
        state=dict(
            delta_indices=[0],
            modality_keys=[key for key, _ in _N17_STATE_LAYOUT],
        ),
        action=dict(
            delta_indices=list(range(16)),
            modality_keys=[key for key, _ in _N17_ACTION_LAYOUT],
        ),
    ))

_PROCESSOR_KWARGS = dict(
    modality_configs=_N17_MODALITY_CONFIGS,
    statistics=_N17_STATISTICS,
    embodiment_id_mapping=dict(libero_sim=2),
    max_state_dim=132,
    max_action_dim=132,
    max_action_horizon=40,
    use_percentiles=True,
    clip_outliers=True,
    use_relative_action=False,
    apply_sincos_state_encoding=False,
    formalize_language=True,
    use_albumentations=True,
    shortest_image_edge=256,
    crop_fraction=0.95,
    image_target_size=(256, 256),
    image_crop_size=(256, 256),
    state_dropout_prob=0.2,
    color_jitter_params=dict(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
    ),
)

model = dict(
    type='GrootN17VLA',
    model_path=_N17_INIT_CKPT,
    embodiment_tag='LIBERO_PANDA',
    processor_kwargs=_PROCESSOR_KWARGS,
    load_metadata=True,
    qwen3_runtime='compat_457',
    freeze_vlm_backbone=True,
    freeze_projector=False,
    vlm_backbone=dict(
        type='GrootN17Qwen3Backbone',
        model_config=_QWEN3_VL_CONFIG,
        select_layer=16,
        reproject_vision=True,
        use_flash_attention=False,
        load_bf16=False,
        tune_top_llm_layers=0,
        trainable_params_fp32=False,
        qwen3_runtime='compat_457',
    ),
    vla_head=dict(
        type='GrootN17ActionHead',
        tune_diffusion_model=True,
        tune_vlln=True,
    ),
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
                        type='BuildModalityStateActionTargets',
                        embodiment_tag='LIBERO_PANDA',
                        flat_layout='auto',
                        train_mode=True,
                        processor_kwargs=_PROCESSOR_KWARGS,
                    ),
                    dict(
                        type='BuildQwenVLChatImageContent',
                        embodiment_tag='LIBERO_PANDA',
                        image_key='images',
                        output_image_key='images',
                        text_key='task_description',
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
                        type='QwenVLImageTokenExpandAndTokenize',
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=_QWEN_TOKENIZER_PATH,
                            padding_side='left',
                            trust_remote_code=False,
                        ),
                        input_ids_key='lang_tokens',
                        attention_mask_key='lang_masks',
                        merge_size=2,
                        padding='max_length',
                        tokenizer_kwargs=dict(
                            max_length=180,
                            truncation=False,
                        ),
                        squeeze_batch=True,
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
    inference_seed=42,
    num_steps_wait=10,
    eval_shard_strategy='episode',
    preprocess_every_step=True,
    save_rollout_videos=False,
    save_failed_rollout_videos=False,
    save_multi_view_rollout_videos=False,
    result_output_dir=('work_dirs/'
                       f'n17_native_{_SUITE}_posttrain_auto_eval'),
    norm_stats_key=_STATISTIC_NAME,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=[
                    'agentview_image',
                    'robot0_eye_in_hand_image',
                ],
            ),
            dict(
                type='LiberoProprioFromInputs',
                norm_type=None,
                out_key='states',
                modality_keys=[
                    'x',
                    'y',
                    'z',
                    'roll',
                    'pitch',
                    'yaw',
                    'gripper',
                ],
            ),
            dict(
                type='BuildModalityStateActionTargets',
                embodiment_tag='LIBERO_PANDA',
                state_key='states',
                flat_layout='auto',
                train_mode=False,
                processor_kwargs=_PROCESSOR_KWARGS,
            ),
            dict(
                type='BuildQwenVLChatImageContent',
                embodiment_tag='LIBERO_PANDA',
                image_key='pixel_values',
                output_image_key='pixel_values',
                text_key='task_description',
                train_mode=False,
                processor_kwargs=_PROCESSOR_KWARGS,
            ),
            dict(
                type='QWen2VLImageTransform',
                img_key='pixel_values',
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
                type='QwenVLImageTokenExpandAndTokenize',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_QWEN_TOKENIZER_PATH,
                    padding_side='left',
                    trust_remote_code=False,
                ),
                input_ids_key='lang_tokens',
                attention_mask_key='lang_masks',
                merge_size=2,
                padding='max_length',
                tokenizer_kwargs=dict(
                    max_length=180,
                    truncation=False,
                ),
                squeeze_batch=True,
                output_keys=[
                    'lang_tokens',
                    'lang_masks',
                    'pixel_values',
                    'img_masks',
                    'image_grid_thw',
                    'states',
                    'embodiment_ids',
                    'replay_img',
                ],
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoAction', denorm_action=False, action_dim=7),
)
