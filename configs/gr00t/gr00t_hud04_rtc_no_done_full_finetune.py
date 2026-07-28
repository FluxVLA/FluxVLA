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

# WBT action layout (42 dims, padded to 64 for the model):
#   [0:31]   joint_cmd (31)            - continuous
#   [31:34]  base_pos x/y/z (3)        - continuous
#   [34:40]  base_rot6d (6)            - continuous
#   [40:42]  hand_closed left/right    - discrete {0, 1}
# This dataset has no done/progress action dimension.
WBT_DISCRETE_ACTION_DIMS = [40, 41]
WBT_DISCRETE_STATE_DIMS = [31, 32]
WBT_DISCRETE_NORM_TYPE = 'min_max'
WBT_CONTINUOUS_NORM_TYPE = 'quantile'
WBT_NORM_KW = dict(
    norm_type=WBT_CONTINUOUS_NORM_TYPE,
    discrete_action_dims=WBT_DISCRETE_ACTION_DIMS,
    discrete_state_dims=WBT_DISCRETE_STATE_DIMS,
    discrete_norm_type=WBT_DISCRETE_NORM_TYPE,
)
WBT_DENORM_KW = dict(
    norm_type=WBT_CONTINUOUS_NORM_TYPE,
    discrete_action_dims=WBT_DISCRETE_ACTION_DIMS,
    discrete_norm_type=WBT_DISCRETE_NORM_TYPE,
)

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_inference_timesteps=4,
        num_steps=32,
        action_dim=64,  # 42 raw dims padded to 64
        ori_action_dim=42,
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',  # 'exponential'（推荐）或 'uniform'
        )),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=580)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_inference_timesteps=4,
        num_steps=32,
        action_dim=64,  # 42 raw dims padded to 64
        ori_action_dim=42),
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            '/data/jace/VLA_dataset/Fixed-Feet-Mani/0518_21_22_25_26_27_0601_02_03_04_05_09_10_11_12_15_16_basket_full_task_prompt_v4_filtered_base_delta',  # noqa: E501
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=0,
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.head',
                        'observation.images.left_wrist',
                    ],
                    name_mappings={'observation.state': ['states']}),
                dict(type='ParquetPrompter'),
                dict(
                    type='ProcessPromptsWithImage',
                    max_len=900,
                    num_images=2,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        'fluxvla/models/third_party_models/eagle2_hg_model',
                    )),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375],
                           [123.515625, 116.04492188, 103.59375]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [58.27148438, 57.02636719, 57.27539062]],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    state_dim=64,
                    action_dim=64,
                    state_key='proprio',
                    action_key='action',
                    **WBT_NORM_KW)
            ],
            action_window_size=32,
            action_key='action')))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=30,
    save_epoch_interval=2,
    max_keep_ckpts=15,
    optimizer=dict(lr=2e-5, type='AdamW', weight_decay=0.0),
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        # special_tokens={'pad_token': '<PAD>'}
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks', 'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(type='constant'),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

inference = dict(
    type='Teleop02WbtRTCInferenceRunner',
    seed=7,
    use_done_state_machine=False,
    async_execution=True,
    async_remaining_actions_threshold=6,
    execute_horizon=16,
    target_hz=50,
    interpolation_method='cubic',
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=4,
    ),
    task_descriptions={
        '0':
        'Lift up the red basket with right arm, put all the objects on the white table into the red basket with left arm, place the red basket on the table.',  # noqa: E501
    },
    interactive=False,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=0,
        img_keys=['head', 'left_wrist'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=580,
                num_images=2,
                tokenizer=dict(type='PretrainedTokenizer')),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                **WBT_NORM_KW)
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction', action_dim=42, **WBT_DENORM_KW),
    operator=dict(
        type='Teleop02WbtOperator',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic='/left_wrist_camera/color/image_raw/compressed',
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        teleop_wbt_topic='/teleop_cmd_WBT',
        cmd_vel_topic='/sdk_cmd_vel_vla',
    ))
