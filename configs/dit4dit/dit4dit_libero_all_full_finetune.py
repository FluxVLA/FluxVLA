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

_ckpt_root = './checkpoints'
_cosmos_base_model = _ckpt_root + '/Cosmos-Predict2.5-2B'
_official_dit4dit_ckpt = (
    _ckpt_root + '/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt')

_action_dim = 8
_ori_action_dim = 7
_state_dim = 16
_action_horizon = 8
_frame_window_size = 9
_image_frame_stride = 2
_image_size = 224

_action_norm_mask = [True, True, True, True, True, True, False]
_libero_data_roots = [
    './datasets/libero_10_no_noops_lerobotv2.1',
    './datasets/libero_goal_no_noops_lerobotv2.1',
    './datasets/libero_spatial_no_noops_lerobotv2.1',
    './datasets/libero_object_no_noops_lerobotv2.1',
]

# The official release evaluates these suites independently. FluxVLA's eval
# runner takes one suite per launch; override eval.task_suite_name for each.
official_eval_task_suites = [
    'libero_spatial',
    'libero_object',
    'libero_goal',
    'libero_10',
]

model = dict(
    type='DiT4DiTVLA',
    # Keep None for official train-from-Cosmos behavior. Set this to
    # _official_dit4dit_ckpt to initialize from the released DiT4DiT weights.
    pretrained_name_or_path=None,
    name_mapping={
        'vlm_backbone.text_encoder':
        'backbone_interface.extractor.text_encoder',  # noqa: E501
        'vlm_backbone.transformer': 'backbone_interface.extractor.transformer',
        'vlm_backbone.vae': 'backbone_interface.extractor.vae',
        'vla_head': 'action_model',
    },
    strict_mapping=False,
    repeated_diffusion_steps=4,
    image_layout='btchw',
    multiview_strategy='tile',
    vlm_backbone=dict(
        type='Cosmos25Backbone',
        base_model=_cosmos_base_model,
        revision='diffusers/base/post-trained',
        torch_dtype='bf16',
        local_files_only=True,
        extract_layer=17,
        max_sequence_length=512,
        trainable=True,
        frozen_submodules=['text_encoder', 'vae'],
        split_future_frames=True,
        num_frames_out=5,
        fixed_seed=None,
        num_inference_steps=1,
        conditional_frame_timestep=0.0001,
        future_loss_type='flow_matching',
        detach_hidden_states=True,
        flow_matching_time_distribution='logit_normal',
        flow_matching_high_sigma_ratio=0.05,
        flow_matching_high_sigma_min=0.98,
        fsdp_min_num_params=10_000_000,
    ),
    vla_head=dict(
        type='DiT4DiTActionHead',
        action_model_type='DiT-B',
        hidden_size=2560,
        add_pos_embed=True,
        max_seq_len=1024,
        action_dim=_action_dim,
        ori_action_dim=_ori_action_dim,
        state_dim=_state_dim,
        future_action_window_size=7,
        action_horizon=_action_horizon,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        num_inference_timesteps=4,
        diffusion_model_cfg=dict(
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_layers=16,
            output_dim=2560,
            positional_embeddings=None,
        ),
    ),
    freeze_vlm_backbone=False,
)

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='franka',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_libero_data_roots,
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
                dict(type='ParquetPrompter', use_conversation=False),
                dict(
                    type='ResizeImages', height=_image_size,
                    width=_image_size),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='ConcatImagesHorizontally',
                    key='images',
                    num_views=2,
                    frame_stride=_image_frame_stride,
                    views_first=True,
                    keep_time_dim=True,
                ),
                dict(
                    type='SinCosKeys',
                    keys=['states'],
                    target_dims={'states': _state_dim},
                    interleave=True,
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    state_key='proprio',
                    action_key='action',
                    action_dim=None,
                    state_dim=None,
                    state_norm_type='none',
                    action_norm_type='min_max',
                    norm_type='min_max',
                    action_norm_mask=_action_norm_mask,
                    clip_norm=False,
                ),
                dict(
                    type='PadActionsAndActionMasks',
                    action_dim=_action_dim,
                    valid_action_dim=_ori_action_dim,
                ),
            ],
            action_window_size=_action_horizon,
            action_key='action',
            use_delta=False,
            statistic_name='franka',
            window_start_idx=0,
            frame_window_size=_frame_window_size,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=100000,
    optimizer=dict(
        lr=1e-5,
        type='AdamW',
        weight_decay=1e-8,
        eps=1e-8,
        betas=(0.9, 0.95),
        paramwise_learning_rate={
            'vlm_backbone.transformer': 1e-5,
            'vla_head': 1e-4,
        },
    ),
    max_grad_norm=1.0,
    save_iter_interval=5000,
    max_keep_ckpts=2,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'frame_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='cosine_with_min_lr',
        warmup_steps=5000,
        min_lr=5e-7,
    ),
    sharding_strategy='full-shard',
    pre_fsdp_param_dtype='bf16',
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name=official_eval_task_suites,
    model_family='dit4dit',
    norm_stats_key='franka',
    eval_chunk_size=_action_horizon,
    resize_size=_image_size,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    enable_mixed_precision_training=False,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        img_buffer_len=1,
        include_task_description=True,
        require_lang_tokens=False,
        transforms=[
            dict(
                type='ProcessDiT4DiTLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                image_size=_image_size,
            ),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='none',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                state_dim=None,
                out_key='states',
            ),
            dict(
                type='SinCosKeys',
                keys=['states'],
                target_dims={'states': _state_dim},
                interleave=True,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='min_max',
        action_dim=_ori_action_dim,
        action_norm_mask=_action_norm_mask,
        clip_normalized_action=True,
        normalize_gripper_action=True,
        invert_gripper_action=True,
    ),
)
