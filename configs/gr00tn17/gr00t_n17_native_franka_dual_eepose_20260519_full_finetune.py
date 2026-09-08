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
"""Native GR00T N1.7 dual-Franka EE-pose training and robot inference.

Each arm uses ``[x, y, z, qx, qy, qz, qw, gripper_width]``.  The Cartesian
pose and gripper targets are absolute, matching the PI0.5 Franka EE-pose
recipe and the 20260519 LeRobot dataset.
"""

_base_ = ['./gr00t_n17_native_libero_10_full_finetune.py']

_DATASET_ROOT = (
    '/mnt/data/cpfs/mnt/data/liyinhao/datasets/'
    'RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop')
_STATISTIC_NAME = 'private'
_EMBODIMENT_KEY = 'franka_dual_eepose'
# Reuse N1.7's released Panda embodiment embedding for dual-Panda control.
_EMBODIMENT_ID = 13
_QWEN_TOKENIZER_PATH = 'fluxvla/models/third_party_models/qwen3_tokenizer'
_TASK_DESCRIPTION = (
    'The right arm picks up the shuttlecock bucket, hands it to the left arm, '
    'and places it on the plate.')

# Computed from all 99 episodes with the repository's ``franka-eepose``
# profile.  Actions follow the PI0.5 convention: absolute EE poses with
# terminal padding included in the statistics computation.
_FRANKA_STATE_STATS = dict(
    q01=[
        0.3037597620487213,
        -0.35402622520923616,
        0.14794699043035509,
        0.7140382289886474,
        -0.09163575455546379,
        -0.0202855958789587,
        -0.029231580384075643,
        0.00039071665378287435,
        0.3069322407245636,
        -0.05707312546670437,
        0.14529650568962096,
        -0.7987435495853424,
        -0.0965876829624176,
        -0.1802465319633484,
        -0.3971003895998001,
        0.0015195267042145133,
    ],
    q99=[
        0.7443822014331818,
        0.008142047487199306,
        0.6667602038383484,
        0.9999997615814209,
        0.10866434335708625,
        0.14994624316692357,
        0.6941786932945252,
        0.0808599665760994,
        0.7438275456428529,
        0.3428739887475968,
        0.5934124124050141,
        0.9999999403953552,
        0.07118354380130769,
        0.177644230425358,
        0.702909963130951,
        0.08082187920808792,
    ],
)
_FRANKA_ACTION_STATS = dict(
    q01=[
        0.3037613332271576,
        -0.35402998328208923,
        0.1479463279247284,
        0.714038074016571,
        -0.09164456278085709,
        -0.0215742364525795,
        -0.02923443913459778,
        0.0003920299932360649,
        0.3069392442703247,
        -0.0570814348757267,
        0.1452961415052414,
        -0.7987663149833679,
        -0.0965888723731041,
        -0.1802748143672943,
        -0.39711296558380127,
        0.0015195267042145133,
    ],
    q99=[
        0.7443833947181702,
        0.008147988468408585,
        0.6667618155479431,
        0.9999997019767761,
        0.10866738855838776,
        0.14994816482067108,
        0.6941827535629272,
        0.0808599665760994,
        0.7438315749168396,
        0.34287577867507935,
        0.5934155583381653,
        0.9999998807907104,
        0.07118412107229233,
        0.1776532381772995,
        0.7029126286506653,
        0.08082187920808792,
    ],
)


def _slice_statistics(statistics, start, end):
    return {name: values[start:end] for name, values in statistics.items()}


_MODALITY_KEYS = [
    'left_eef_pose',
    'left_gripper',
    'right_eef_pose',
    'right_gripper',
]
# Official N1.7 EEF conversion expects relative XYZ+rotation-6D actions.  This
# dataset instead stores absolute XYZ+quaternion targets, so keep each 7D pose
# as a generic absolute vector and do not apply EEF/delta conversion.
_ABSOLUTE_POSE_ACTION = dict(
    rep='ABSOLUTE',
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
                _ABSOLUTE_POSE_ACTION,
                _ABSOLUTE_GRIPPER_ACTION,
                _ABSOLUTE_POSE_ACTION,
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
            left_eef_pose=_slice_statistics(_FRANKA_STATE_STATS, 0, 7),
            left_gripper=_slice_statistics(_FRANKA_STATE_STATS, 7, 8),
            right_eef_pose=_slice_statistics(_FRANKA_STATE_STATS, 8, 15),
            right_gripper=_slice_statistics(_FRANKA_STATE_STATS, 15, 16),
        ),
        action=dict(
            left_eef_pose=_slice_statistics(_FRANKA_ACTION_STATS, 0, 7),
            left_gripper=_slice_statistics(_FRANKA_ACTION_STATS, 7, 8),
            right_eef_pose=_slice_statistics(_FRANKA_ACTION_STATS, 8, 15),
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
    use_relative_action=False,
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
    use_relative_action=False,
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
        type='DistributedRepeatingDataset',
        name_mappings={
            '_delete_': True,
            'observation.eepose': ['states', 'actions'],
        },
        statistic_keys=['observation.eepose', 'timestamp'],
        statistic_name=_STATISTIC_NAME,
        dataset_statistics=dict(
            _delete_=True,
            **_DATASET_STATISTICS,
        ),
        shuffle=True,
        reshuffle_each_epoch=True,
        seed=42,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=_DATASET_ROOT,
                statistic_name=_STATISTIC_NAME,
                action_key='observation.eepose',
                use_delta=False,
                window_start_idx=0,
                train_episode_fraction=1.0,
                repeat_to_full_length=False,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=_EMBODIMENT_ID,
                        parquet_keys=[
                            'observation.eepose',
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
                            'observation.eepose': ['states'],
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
    type='FSDPTrainRunner',
    max_steps=3625,
    optimizer=dict(
        lr=1e-4,
        type='AdamW',
        weight_decay=1e-5,
    ),
    max_grad_norm=1.0,
    grad_accumulation_steps=1,
    sampler=None,
    save_iter_interval=450,
    save_epoch_interval=1,
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
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler=dict(
        _delete_=True,
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='shard-grad-op',
    change_key_name=False,
    keep_params_fp32=True,
)

inference = dict(
    type='FrankaInferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    task_descriptions={'1': _TASK_DESCRIPTION},
    seed=7,
    action_mode='cartesian',
    active_arms=('left', 'right'),
    async_execution=False,
    execute_horizon=20,
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
        _delete_=True,
        type='DenormalizePrivateAction',
        statistic_name=_STATISTIC_NAME,
        norm_type='quantile',
        action_dim=16,
    ),
    action_chunk=40,
    operator=dict(
        _delete_=True,
        type='FrankaDualOperator',
        image_encoding='rgb8',
        command_mode='cartesian',
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
    ),
)
