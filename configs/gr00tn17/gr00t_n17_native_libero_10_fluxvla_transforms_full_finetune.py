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
"""N1.7 LIBERO-10 recipe composed from FluxVLA transforms.

This inherits the model, runner, and evaluation settings from the native
recipe and only replaces the training parquet preprocessing pipeline.
"""

_base_ = ['./gr00t_n17_native_libero_10_full_finetune.py']

_STATISTIC_NAME = 'libero_10_no_noops_native'
_LIBERO_DATA_ROOT = './datasets/libero_10_no_noops_lerobotv2.1'
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'

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

_PROCESSOR_KWARGS = dict(
    modality_configs=dict(
        libero_sim=dict(
            video=dict(
                delta_indices=[0],
                modality_keys=['image', 'wrist_image'],
            ), )),
    statistics=dict(libero_sim=dict()),
    embodiment_id_mapping=dict(libero_sim=2),
    use_albumentations=True,
    shortest_image_edge=None,
    crop_fraction=None,
    image_target_size=(256, 256),
    image_crop_size=(230, 230),
    color_jitter_params=dict(
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
    ),
)

_OUTPUT_KEYS = [
    'lang_tokens',
    'lang_masks',
    'images',
    'image_grid_thw',
    'states',
    'actions',
    'action_masks',
    'embodiment_ids',
    'sample_weight',
]

_EVAL_OUTPUT_KEYS = [
    'lang_tokens',
    'lang_masks',
    'pixel_values',
    'img_masks',
    'image_grid_thw',
    'states',
    'embodiment_ids',
    'replay_img',
]

# The official N1.7 fine-tuning launcher overrides the checkpoint value for
# both processor-side and action-head-side state dropout. Make the model-side
# override explicit instead of depending on the value saved in config.json.
model = dict(state_dropout_prob=0.2)

train_dataloader = dict(
    dataset=dict(
        dataset_statistics={
            _STATISTIC_NAME: {
                'states': _N17_FLAT_STATISTICS['state'],
                'actions': _N17_FLAT_STATISTICS['action'],
            },
        },
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
                        output_keys=_OUTPUT_KEYS,
                    ),
                ],
                action_window_size=16,
                drop_incomplete_action_windows=True,
            ),
        ],
    ), )

# Keep online LIBERO evaluation on the same composable FluxVLA transforms as
# training. ProcessLiberoEvalInputs and LiberoProprioFromInputs only adapt the
# simulator observation; normalization, state shaping, image processing, and
# prompt tokenization are shared with the parquet training pipeline.
eval = dict(
    dataset=dict(
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
                output_keys=_EVAL_OUTPUT_KEYS,
            ),
        ],
    ),
)
