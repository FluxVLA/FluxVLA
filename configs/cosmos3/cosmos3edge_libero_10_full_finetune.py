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
#
# Cosmos3-Edge post-training on LIBERO-10 benchmark.
#
# Usage (8 GPU):
#   torchrun --nproc-per-node=8 scripts/train.py \
#     --config configs/cosmos3/cosmos3edge_libero_10_full_finetune.py
#
# LIBERO-specific settings:
# * LIBERO: action_dim=10 (eef pos + rot6d + gripper),
#   max_state_dim=64, 2 views.
# * Video keys: observation.images.image + observation.images.wrist_image.
# * Single dataset group (no multi-embodiment split).
# * Cosmos3 uses LIBERO embodiment_id=5 for the action projector.
# * 192×320 (H×W) model canvas: two 256×256 views → 512×256 concat →
#   aspect-preserving resize + bottom reflection pad; third-person left,
#   wrist right.
from copy import deepcopy

# Cosmos3's 256/16:9 video bucket is (320, 192) (W, H); see
# third_party_models/cosmos3/data/vfm/utils.py::VIDEO_RES_SIZE_INFO.
# Keep this explicit: resize/padding and prompt video metadata must be
# changed together.
_video_height = 192
_video_width = 320

model = dict(
    type='Cosmos3FlowMatching',
    action_horizon=16,
    action_in_proj=dict(
        type='DomainAwareLinear',
        input_size=64,
        num_domains=32,
        output_size=2048,
    ),
    action_out_proj=dict(
        type='DomainAwareLinear',
        input_size=2048,
        num_domains=32,
        output_size=64,
    ),
    base_fps=24.0,
    enable_fps_modulation=True,
    enable_vision_loss=True,
    freeze_non_moe_vlm_backbone=True,
    freeze_vlm_backbone=False,
    latent_patch_size=2,
    max_action_dim=64,
    name_mapping=dict({
        '.self_attn.k_norm_moe_gen.':
        '.self_attn.norm_added_k.',
        '.self_attn.k_proj.':
        '.self_attn.to_k.',
        '.self_attn.k_proj_moe_gen.':
        '.self_attn.add_k_proj.',
        '.self_attn.o_proj.':
        '.self_attn.to_out.',
        '.self_attn.o_proj_moe_gen.':
        '.self_attn.to_add_out.',
        '.self_attn.q_norm_moe_gen.':
        '.self_attn.norm_added_q.',
        '.self_attn.q_proj.':
        '.self_attn.to_q.',
        '.self_attn.q_proj_moe_gen.':
        '.self_attn.add_q_proj.',
        '.self_attn.v_proj.':
        '.self_attn.to_v.',
        '.self_attn.v_proj_moe_gen.':
        '.self_attn.add_v_proj.',
        'action_in_proj.':
        'action_proj_in.',
        'action_modality_embed':
        'action_modality_embed',
        'action_out_proj.':
        'action_proj_out.',
        'time_embedder.mlp.0.':
        'time_embedder.linear_1.',
        'time_embedder.mlp.2.':
        'time_embedder.linear_2.',
        'vision_in_proj.projector.':
        'proj_in.',
        'vision_out_proj.projector.':
        'proj_out.',
        'vlm_backbone.lm_head.weight':
        'lm_head.weight',
        'vlm_backbone.model.language_model.embed_tokens.weight':
        'embed_tokens.weight',
        'vlm_backbone.model.language_model.layers.':
        'layers.',
        'vlm_backbone.model.language_model.norm.weight':
        'norm.weight',
        'vlm_backbone.model.language_model.norm_moe_gen.weight':
        'norm_moe_gen.weight'
    }),
    num_embodiment_domains=32,
    ori_action_dim=10,
    packed_attention_backend='flash2',
    position_embedding_type='unified_3d_mrope',
    pretrained_name_or_path='./checkpoints/Cosmos3-Edge/transformer',
    reinitialize_action_policy=True,
    rectified_flow_inference_config=dict(
        num_steps=30,
        num_train_timesteps=1000,
        scheduler_type='unipc',
        shift=10.0,
        use_dynamic_shifting=False,
        use_karras_sigmas=False),
    rectified_flow_training_config=dict(
        action_loss_weight=10.0,
        independent_action_schedule=False,
        normalize_loss_by_active=False,
        shift=dict({
            '256': 3,
            '480': 5,
            '720': 10
        }),
        shift_action=None,
        train_time_action_distribution='logitnormal',
        train_time_image_distribution='logitnormal',
        train_time_video_distribution='waver',
        train_time_weight='uniform',
        use_discrete_rf=False,
        use_dynamic_shift=False,
        use_high_sigma_strategy=False,
        use_high_sigma_strategy_action=False,
        vision_loss_weight=10.0),
    special_tokens=dict(
        end_of_generation=21, eos_token_id=11, start_of_generation=20),
    strict_mapping=True,
    timestep_scale=0.001,
    unified_3d_mrope_reset_spatial_ids=True,
    unified_3d_mrope_temporal_modality_margin=15000,
    vision_in_proj=dict(type='LinearProjector', in_dim=192, out_dim=2048),
    vision_latent_dim=48,
    vision_out_proj=dict(type='LinearProjector', in_dim=2048, out_dim=192),
    vision_vae=dict(
        type='Cosmos3Wan22VAE',
        encode_exact_durations=[
            17,
        ],
        pretrained_name_or_path='./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',
    ),
    vlm_backbone=dict(
        type='Cosmos3MoTBackbone',
        include_visual=False,
        skip_init_weights=True,
        vlm_config=dict(
            attention_bias=False,
            bos_token_id=1,
            enable_mrope=True,
            eos_token_id=11,
            head_dim=128,
            hidden_size=2048,
            intermediate_size=9216,
            layer_norm_epsilon=1e-05,
            max_position_embeddings=131072,
            mlp_bias=False,
            mlp_hidden_act='relu2',
            model_type='nemotron_3_dense_vl_text',
            mrope_section=[
                24,
                20,
                20,
            ],
            num_attention_heads=16,
            num_hidden_layers=28,
            num_key_value_heads=8,
            pad_token_id=11,
            rope_theta=100000000.0,
            tie_word_embeddings=False,
            use_und_k_norm_for_gen=True,
            vocab_size=131072)))
inference_model = deepcopy(model)
# Eval builds the VAE without the external Wan2.2 file; the finetuned
# checkpoint already carries the frozen VAE weights.
inference_model['vision_vae']['pretrained_name_or_path'] = None

runner = dict(
    type='FSDPTrainRunner',
    change_key_name=False,
    collator=dict(
        type='DictCollator',
        meta_keys=[
            'text_token_ids',
            'sequence_plan',
            'task_description',
            'stats',
            'info',
            'timestamp',
        ],
        keys=[
            'images',
            'actions',
            'embodiment_ids',
            'raw_action_dim',
            'conditioning_fps',
            'action_fps',
        ],
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    grad_accumulation_steps=32,
    lr_scheduler=dict(
        type='linear-warmup+linear-decay',
        cycle_length=16000,
        warmup_steps=500),
    max_epochs=None,
    max_grad_norm=1.0,
    max_keep_ckpts=2,
    max_steps=2000,
    metric=dict(
        type='VLAMetric',
        active_trackers=(
            'jsonl',
            'wandb',
        ),
        grad_accumulation_steps=32,
        run_dir='work_dirs',
        window_size=1),
    mixed_precision_dtype='bf16',
    optimizer=dict(
        type='AdamW',
        betas=(
            0.9,
            0.99,
        ),
        eps=1e-08,
        exclude_1d_from_weight_decay=False,
        fused=True,
        lr=5e-05,
        paramwise_learning_rate=dict({
            'action_in_proj.': 0.00025,
            'action_modality_embed': 0.00025,
            'action_out_proj.': 0.00025
        }),
        weight_decay=0.05),
    sampler=None,
    save_iter_interval=500,
    sharding_strategy='full-shard',
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_max_length=4096,
        model_path='./checkpoints/Cosmos3-Edge',
        padding_side='right',
        trust_remote_code=True,
    ),
)
seed = 7
train_dataloader = dict(
    dataset=dict(
        type='DistributedRepeatingDataset',
        datasets=dict(
            type='ParquetDataset',
            action_key='action',
            action_window_size=16,
            data_root_path='./datasets/libero_10_no_noops_lerobotv2.1',
            frame_window_size=17,
            require_full_window=True,
            statistic_name='libero_10_no_noops',
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=5,
                    name_mappings=dict({
                        'actions': [
                            'actions',
                        ],
                        'observation.state': [
                            'states',
                        ]
                    }),
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.wrist_image',
                    ]),
                dict(type='LiberoFramewiseActionToRot6D'),
                dict(
                    type='ProcessCosmos3Prompt',
                    action_metadata=dict(
                        append_viewpoint=False,
                        conditioning_fps=20.0,
                        frame_window_size=17,
                        video_height=_video_height,
                        video_width=_video_width),
                    cfg_dropout_rate=0.1,
                    format_prompt_as_json=True,
                    max_len=512,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_max_length=4096,
                        model_path='./checkpoints/Cosmos3-Edge',
                        padding_side='right',
                        trust_remote_code=True,
                    ),
                ),
                dict(type='ResizeImages', height=256, width=256),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=64,
                    action_key='action',
                    action_norm_type='quantile',
                    state_norm_type='none',
                    state_dim=64,
                    state_key='proprio',
                ),
                dict(
                    type='BuildCosmos3Sequence',
                    conditioning_fps=20.0,
                    frame_window_size=17,
                    mode='wam',
                    prepend_state_to_action=False,
                    raw_action_dim=10,
                ),
                dict(
                    type='PrepareVideo',
                    frame_window_size=17,
                    num_views=2,
                    tile_direction='horizontal',
                ),
                dict(
                    type='ResizeAndReflectPad',
                    height=_video_height,
                    width=_video_width),
            ],
            use_delta=False,
            window_start_idx=0),
        name_mappings=dict({
            'action': [
                'action',
            ],
            'observation.state': [
                'proprio',
            ]
        }),
        statistic_keys=[
            'observation.state',
            'timestamp',
            'action',
        ],
        statistic_name='libero_10_no_noops',
        statistics_overrides={
            'libero_10_no_noops':
            dict(
                action=dict(
                    mean=[
                        0.050704, 0.097407, -0.094833, 0.994873, -0.004579,
                        -0.004288, 0.004389, 0.996104, 0.001109, 0.476725
                    ],
                    std=[
                        0.333621, 0.387175, 0.45714, 0.010807, 0.077802,
                        0.063386, 0.078571, 0.009994, 0.038504, 0.49946
                    ],
                    min=[
                        -0.9375, -0.9375, -0.9375, 0.902028, -0.356085,
                        -0.367416, -0.370434, 0.921907, -0.255, 0.0
                    ],
                    max=[
                        0.9375, 0.9375, 0.9375, 1.0, 0.368853, 0.341214,
                        0.356395, 1.0, 0.348251, 1.0
                    ],
                    q01=[
                        -0.723214, -0.808929, -0.9375, 0.934955, -0.223431,
                        -0.189878, -0.334735, 0.938516, -0.107736, 0.0
                    ],
                    q99=[
                        0.9375, 0.870536, 0.9375, 1.0, 0.331, 0.163153,
                        0.226216, 1.0, 0.127158, 1.0
                    ]))
        },
    ),
    per_device_batch_size=8,
    per_device_num_workers=4)

eval = dict(
    type='LiberoEvalRunner',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        extra_tensor_keys=[
            'conditioning_fps',
            'prepend_state_to_action',
        ],
        img_buffer_len=1,
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                embodiment_id=5,
                img_keys=[
                    'agentview_image',
                    'robot0_eye_in_hand_image',
                ],
            ),
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=20.0,
                prepend_state_to_action=False,
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-crop',
                input_sizes=[
                    [
                        3,
                        256,
                        256,
                    ],
                    [
                        3,
                        256,
                        256,
                    ],
                ],
                means=[
                    [
                        127.5,
                        127.5,
                        127.5,
                    ],
                    [
                        127.5,
                        127.5,
                        127.5,
                    ],
                ],
                stds=[
                    [
                        127.5,
                        127.5,
                        127.5,
                    ],
                    [
                        127.5,
                        127.5,
                        127.5,
                    ],
                ],
            ),
            dict(
                type='ProcessCosmos3Prompt',
                action_metadata=dict(
                    append_viewpoint=False,
                    conditioning_fps=20.0,
                    frame_window_size=17,
                    video_height=_video_height,
                    video_width=_video_width),
                cfg_dropout_rate=0.0,
                format_prompt_as_json=True,
                max_len=512,
                output_attention_mask_key='lang_masks',
                output_key='lang_tokens',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_max_length=4096,
                    model_path='./checkpoints/Cosmos3-Edge',
                    padding_side='right',
                    trust_remote_code=True,
                ),
            ),
            dict(
                type='LiberoProprioFromInputs',
                gripper_key='robot0_gripper_qpos',
                norm_type='mean_std',
                out_key='states',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                state_dim=64,
            ),
            dict(
                type='PrepareVideo',
                frame_window_size=1,
                num_views=2,
                tile_direction='horizontal',
            ),
            dict(
                type='ResizeAndReflectPad',
                height=_video_height,
                width=_video_width),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoFramewiseRot6DAction',
        action_dim=10,
        norm_type='quantile_rot',
    ),
    enable_mixed_precision_training=True,
    eval_chunk_size=16,
    inference_seed=7,
    mixed_precision_dtype='bf16',
    model_family='cosmos3',
    norm_stats_key='libero_10_no_noops',
    num_inference_steps=30,
    num_steps_wait=10,
    num_trials_per_task=50,
    resize_size=256,
    seed=7,
    task_ids=None,
    task_suite_name='libero_10',
)
