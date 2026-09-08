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
"""Native GR00T N1.7 dual-Franka qpos training and real-robot inference."""

_base_ = ['./gr00t_n17_native_libero_10_full_finetune.py']

_DATASET_ROOT = (
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_STATISTIC_NAME = 'private'
_EMBODIMENT_KEY = 'franka_dual'
# Reuse N1.7's released Panda embodiment embedding for the dual-Panda setup.
_EMBODIMENT_ID = 13
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'
_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')

# Exact quantiles from the matching PI0.5 20260519 dual-Franka config.  The
# action statistics are computed after converting the 14 arm joints to
# state-relative deltas; the two gripper widths remain absolute.
_FRANKA_STATE_STATS = dict(
    q01=[
        -0.03914828851819038,
        -0.8123818564414979,
        -0.29229462444782256,
        -2.357604742050171,
        -1.6529088115692139,
        1.5588371849060059,
        0.48917633056640625,
        0.00039071665378287435,
        0.0001337525379494764,
        -0.7974990296363831,
        -0.2607133948802948,
        -2.4808950853347778,
        -0.23938447266817092,
        1.5709584951400757,
        0.08988710731267929,
        0.0015195267042145133,
    ],
    q99=[
        0.030357560664415358,
        0.7884451413154607,
        0.2536210554838183,
        -1.3048278450965878,
        0.09813202053308487,
        2.3545701169967654,
        1.2594540476799012,
        0.0808599665760994,
        0.07816524744033813,
        0.6832878422737122,
        0.24506871283054354,
        -1.538973252773283,
        1.7701099276542667,
        2.732522406578064,
        1.081895608901978,
        0.08082187920808792,
    ],
)
_FRANKA_ACTION_STATS = dict(
    q01=[
        -0.03508321955800056,
        -0.927380074262619,
        -0.3236503484845162,
        -0.706730649471283,
        -1.1979312765598298,
        -0.3072397756576538,
        -0.4018883168697357,
        0.0003920299932360649,
        -0.02795792240649462,
        -0.7779699826240539,
        -0.2855138486623764,
        -0.3657342553138733,
        -1.4835465264320373,
        -0.47738827466964723,
        -0.638928741812706,
        0.0015195267042145133,
    ],
    q99=[
        0.02903852557763444,
        0.6866954135894772,
        0.2221186743676661,
        0.3931978082656826,
        1.3720208597183223,
        0.2697093939781179,
        0.3257311290502548,
        0.0808599665760994,
        0.026975513361394386,
        0.7459093928337084,
        0.35851511627435645,
        0.3525113666057582,
        1.343698980808258,
        0.5126444828510281,
        0.5077816033363334,
        0.08082187920808792,
    ],
)


def _slice_statistics(statistics, start, end):
    return {name: values[start:end] for name, values in statistics.items()}


_MODALITY_KEYS = [
    'left_arm',
    'left_gripper',
    'right_arm',
    'right_gripper',
]
_RELATIVE_JOINT_ACTION = dict(
    rep='RELATIVE',
    type='NON_EEF',
    format='DEFAULT',
    state_key=None,
)
_ABSOLUTE_GRIPPER_ACTION = dict(
    rep='ABSOLUTE',
    type='NON_EEF',
    format='DEFAULT',
    state_key=None,
)
_N17_MODALITY_CONFIGS = {
    _EMBODIMENT_KEY:
    dict(
        video=dict(
            delta_indices=[0],
            modality_keys=[
                'cam_front',
                'cam_wrist_left',
                'cam_wrist_right',
            ],
        ),
        state=dict(
            delta_indices=[0],
            modality_keys=_MODALITY_KEYS,
        ),
        action=dict(
            delta_indices=list(range(40)),
            modality_keys=_MODALITY_KEYS,
            action_configs=[
                _RELATIVE_JOINT_ACTION,
                _ABSOLUTE_GRIPPER_ACTION,
                _RELATIVE_JOINT_ACTION,
                _ABSOLUTE_GRIPPER_ACTION,
            ],
        ),
        language=dict(
            delta_indices=[0],
            modality_keys=['task'],
        ),
    ),
}
_N17_STATISTICS = {
    _EMBODIMENT_KEY:
    dict(
        state=dict(
            left_arm=_slice_statistics(_FRANKA_STATE_STATS, 0, 7),
            left_gripper=_slice_statistics(_FRANKA_STATE_STATS, 7, 8),
            right_arm=_slice_statistics(_FRANKA_STATE_STATS, 8, 15),
            right_gripper=_slice_statistics(_FRANKA_STATE_STATS, 15, 16),
        ),
        action=dict(
            left_arm=_slice_statistics(_FRANKA_ACTION_STATS, 0, 7),
            left_gripper=_slice_statistics(_FRANKA_ACTION_STATS, 7, 8),
            right_arm=_slice_statistics(_FRANKA_ACTION_STATS, 8, 15),
            right_gripper=_slice_statistics(_FRANKA_ACTION_STATS, 15, 16),
        ),
    ),
}
_DATASET_STATISTICS = {
    _STATISTIC_NAME: {
        'states': _FRANKA_STATE_STATS,
        'actions': _FRANKA_ACTION_STATS,
    },
}
_PROCESSOR_KWARGS = dict(
    modality_configs=_N17_MODALITY_CONFIGS,
    statistics=_N17_STATISTICS,
    embodiment_id_mapping={_EMBODIMENT_KEY: _EMBODIMENT_ID},
    max_state_dim=132,
    max_action_dim=132,
    max_action_horizon=40,
    use_percentiles=True,
    clip_outliers=True,
    use_relative_action=True,
    apply_sincos_state_encoding=False,
    formalize_language=True,
    use_albumentations=True,
    shortest_image_edge=256,
    crop_fraction=0.95,
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
    embodiment_tag=_EMBODIMENT_KEY,
    processor_kwargs=dict(
        _delete_=True,
        **_PROCESSOR_KWARGS,
    ),
    state_dropout_prob=0.2,
)

# Rebuild the inference config from the Franka-specific model overrides rather
# than inheriting the LIBERO_PANDA inference config from the base file.
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
        name_mappings={
            'observation.state': ['states', 'actions'],
        },
        statistic_keys=['observation.state', 'timestamp'],
        statistic_name=_STATISTIC_NAME,
        dataset_statistics=_DATASET_STATISTICS,
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_DATASET_ROOT,
                statistic_name=_STATISTIC_NAME,
                action_key='observation.state',
                use_delta=False,
                window_start_idx=0,
                train_episode_fraction=1.0,
                repeat_to_full_length=False,
                transforms=[
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
                            'observation.images.cam_front',
                            'observation.images.cam_wrist_left',
                            'observation.images.cam_wrist_right',
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions'],
                        },
                    ),
                    dict(
                        type='RelativeActions',
                        mask=[True] * 7 + [False] + [True] * 7 + [False],
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
                        valid_action_dim=16,
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
                        max_len=256,
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
    max_steps=3625,
    grad_accumulation_steps=1,
    save_iter_interval=450,
    max_keep_ckpts=8,
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
    metric=dict(grad_accumulation_steps=1, ),
    lr_scheduler=dict(
        _delete_=True,
        type='linear-warmup+cosine-decay',
        warmup_steps=180,
        decay_steps=3625,
    ),
    sharding_strategy='shard-grad-op',
    keep_params_fp32=True,
)

inference = dict(
    type='FrankaInferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    task_descriptions={'1': _TASK_DESCRIPTION},
    seed=7,
    action_mode='joint',
    active_arms=('left', 'right'),
    async_execution=False,
    execute_horizon=40,
    prepare_pose=None,
    dataset=dict(
        type='PrivateInferenceDataset',
        inject_model_path=False,
        embodiment_id=_EMBODIMENT_ID,
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_key='states',
                action_key=None,
                state_dim=132,
                norm_type='quantile',
                clip_norm=True,
                normalization_epsilon=0.0,
                preserve_input_dtype=True,
            ),
            dict(
                type='PrepareStateActionTargets',
                state_history_length=1,
                action_horizon=40,
                valid_action_dim=16,
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
                max_len=256,
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
        type='DenormalizeDeltaAction',
        statistic_name=_STATISTIC_NAME,
        norm_type='quantile',
        action_dim=16,
        delta_action_mask=[True] * 7 + [False] + [True] * 7 + [False],
    ),
    action_chunk=40,
    operator=dict(
        type='FrankaDualOperator',
        image_encoding='rgb8',
        command_mode='joint',
        img_left_topic='/camera_left_wrist/color/image_raw',
        img_right_topic='/camera_right_wrist/color/image_raw',
        img_front_topic='/camera_front/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_arm_right_topic='/right_arm/joint_states',
        puppet_franka_state_left_topic=(
            '/left_arm/franka_state_controller/franka_states'),
        puppet_franka_state_right_topic=(
            '/right_arm/franka_state_controller/franka_states'),
        sync_warning_enabled=True,
        cartesian_cmd_left_topic=(
            '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
        cartesian_cmd_right_topic=(
            '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
        joint_cmd_left_topic=(
            '/left_arm/ruckig_joint_impedance_controller/target_joint_state'),
        joint_cmd_right_topic=(
            '/right_arm/ruckig_joint_impedance_controller/target_joint_state'),
        gripper_left_topic='/left_arm/franka_gripper/move/goal',
        gripper_right_topic='/right_arm/franka_gripper/move/goal',
        gripper_control_mode='grasp',
    ),
)
