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
"""Native GR00T N1.7 LIBERO-10 parquet fine-tuning config.

Defaults follow the official GR00T N1.7 LIBERO fine-tuning recipe.
LIBERO modality metadata and normalization statistics are self-contained
in this config.
"""

_SUITE = 'libero_10'
_DATASET_NAME = 'libero_10_no_noops_lerobotv2.1'
_STATISTIC_NAME = 'libero_10_no_noops_native'

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
            -0.48278069496154785, -0.3309336006641388, 0.44550687074661255,
            1.1323540210723877, -3.6312508583068848, -1.842738389968872,
            -0.005453015677630901, -0.04112039878964424
        ],
        'max': [
            0.2103137969970703, 0.38887521624565125, 1.333192229270935,
            3.7248642444610596, 3.5618896484375, 1.3863215446472168,
            0.041575800627470016, 0.0013126095291227102
        ],
        'mean': [
            -0.041913267292894754, 0.03459178937442461, 0.826588200639446,
            2.9025952235853074, -0.5570652394455817, -0.16592166707651643,
            0.0284503134250701, -0.028802363005983177
        ],
        'std': [
            0.1062499279379542, 0.14401688696973244, 0.2575997325122509,
            0.3486750480333318, 1.2496987319182473, 0.35329866207723215,
            0.013186505225440641, 0.013033613397826261
        ],
        'q01': [
            -0.3865206345915794, -0.2835936737060547, 0.4480444353818893,
            1.8793639504909516, -2.928461148738861, -1.1567491829395293,
            0.002069159597158432, -0.040017270520329475
        ],
        'q99': [
            0.1524330329895019, 0.3259116277098653, 1.2536243999004364,
            3.296849384307861, 2.7515456867217982, 0.6876976984739301,
            0.040040221586823466, -0.0018127537204418339
        ]
    },
    'action': {
        'min': [
            -0.9375, -0.9375, -0.9375, -0.23642857372760773,
            -0.3053571283817291, -0.3642857074737549, 0.0
        ],
        'max': [
            0.9375, 0.9375, 0.9375, 0.32892856001853943, 0.36964285373687744,
            0.375, 1.0
        ],
        'mean': [
            0.019056566441900073, 0.056724760591172235, -0.05623928876675204,
            0.004756678449741797, 0.0027974923231023534, -0.007146069658086462,
            0.5459915611814345
        ],
        'std': [
            0.28014137458397925, 0.3585648567836422, 0.36740624604286787,
            0.03793317388007877, 0.053935862618483994, 0.0881014089030479,
            0.4978802831004402
        ],
        'q01': [
            -0.6160714030265808, -0.7746696585416795, -0.7607142925262451,
            -0.09749999642372131, -0.14678572118282318, -0.2742857038974762,
            0.0
        ],
        'q99': [
            0.7714285850524902, 0.8464285731315613, 0.9375, 0.1403571367263794,
            0.15857142210006714, 0.335357129573822, 1.0
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
    use_relative_action=True,
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

model = dict(
    type='GrootN17VLA',
    model_path=_N17_INIT_CKPT,
    embodiment_tag='LIBERO_PANDA',
    processor_kwargs=_PROCESSOR_KWARGS,
    state_dropout_prob=0.2,
    load_metadata=True,
    qwen3_runtime='compat_457',
    freeze_vlm_backbone=True,
    freeze_vla_head=False,
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
        dataset_statistics={
            _STATISTIC_NAME: {
                'states': _N17_FLAT_STATISTICS['state'],
                'actions': _N17_FLAT_STATISTICS['action'],
            },
        },
        shuffle=True,
        reshuffle_each_epoch=True,
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
                        embodiment_id=2,
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
                        type='NormalizeStatesAndActions',
                        state_key='states',
                        action_key='actions',
                        state_dim=132,
                        action_dim=132,
                        norm_type='quantile',
                        clip_norm=True,
                        normalization_epsilon=0.0,
                        preserve_input_dtype=True,
                    ),
                    dict(
                        type='PrepareStateActionTargets',
                        state_history_length=1,
                        action_horizon=40,
                        valid_action_dim=7,
                        state_dropout_prob=0.2,
                    ),
                    dict(
                        type='GrootN17ImageAugmentation',
                        embodiment_tag='LIBERO_PANDA',
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
                ],
                action_window_size=16,
                drop_incomplete_action_windows=True,
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
    grad_accumulation_steps=2,
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
        grad_accumulation_steps=2,
        window_size=1),
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
    type='LiberoEvalRunner',
    model_family='groot_n17_native',
    task_suite_name=_SUITE,
    num_trials_per_task=20,
    eval_chunk_size=8,
    max_steps=720,
    seed=7,
    inference_seed=None,
    num_steps_wait=0,
    eval_shard_strategy='episode',
    preprocess_every_step=False,
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
                embodiment_id=2,
            ),
            dict(
                type='LiberoProprioFromInputs',
                norm_type=None,
                out_key='states',
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_key='states',
                action_key=None,
                state_dim=132,
                norm_type='quantile',
                clip_norm=True,
                normalization_epsilon=0.0,
                preserve_input_dtype=True,
                statistics_key='norm_stats',
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=40,
                valid_action_dim=7,
                state_dropout_prob=0.0,
            ),
            dict(
                type='GrootN17ImageAugmentation',
                embodiment_tag='LIBERO_PANDA',
                image_key='pixel_values',
                output_image_key='pixel_values',
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
        type='DenormalizeLiberoAction',
        denorm_action=True,
        norm_type='quantile',
        action_dim=7,
    ),
)
