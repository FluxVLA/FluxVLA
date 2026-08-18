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
"""Cosmos3-Edge policy post-training on the ALOHA example dataset.

The Edge settings below are the real-robot validated recipe.  The data root
and task prompts intentionally remain the repository's ALOHA example so this
config is runnable without access to the internal towel dataset.
"""

from copy import deepcopy

_example_data_root = (
    './datasets/RealRobot_AgileX_aloha_lerobot_v2/aloha_example')
_example_statistic_name = 'private'

_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path='./checkpoints/Cosmos3-Edge',
    model_max_length=4096,
    padding_side='right',
    trust_remote_code=True)

_action_prompt_metadata = dict(
    append_viewpoint=False,
    conditioning_fps=30.0,
    frame_window_size=33,
    video_height=384,
    video_width=256)

_transforms = [
    dict(
        type='ProcessParquetInputs',
        embodiment_id=21,
        parquet_keys=[
            'observation.state',
            'observation.eepose',
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
        }),
    dict(type='ResizeImages', height=256, width=256),
    dict(
        type='AugVideo',
        rotation_range=0.0,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.6, 1.4),
        crop_scale=(0.95, 0.95),
        crop_ratio=(1.0, 1.0),
        prob=1.0,
        saturation_range=(0.5, 1.5),
        hue_delta=0.08),
    dict(
        type='ProcessCosmos3Prompt',
        tokenizer=_tokenizer,
        max_len=512,
        cfg_dropout_rate=0.1,
        action_metadata=_action_prompt_metadata),
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=64,
        state_dim=64,
        state_key='proprio',
        action_key='action',
        norm_type='mean_std'),
    dict(
        type='BuildCosmos3Sequence',
        raw_action_dim=14,
        mode='wam',
        frame_window_size=33,
        prepend_state_to_action=True,
        conditioning_fps=30.0),
    dict(
        type='PrepareVideo',
        num_views=3,
        frame_window_size=33,
        tile_direction='top_bottom_pair',
        top_view=0,
        bottom_views=(1, 2),
        bottom_height_ratio=0.5),
]

model = dict(
    type='Cosmos3FlowMatching',
    action_horizon=32,
    action_in_proj=dict(
        type='DomainAwareLinear',
        input_size=64,
        num_domains=32,
        output_size=2048),
    action_out_proj=dict(
        type='DomainAwareLinear',
        input_size=2048,
        num_domains=32,
        output_size=64),
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
    ori_action_dim=14,
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
        vision_loss_weight=1.0),
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
            33,
        ],
        pretrained_name_or_path='./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'),
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
# Eval builds the VAE without the external Wan2.2 file; the fine-tuned
# checkpoint already carries the frozen VAE weights.
inference_model['vision_vae']['pretrained_name_or_path'] = None
eval = None

train_dataloader = dict(
    per_device_batch_size=16,
    per_device_num_workers=4,
    prefetch_factor=1,
    dataset=dict(
        type='DistributedRepeatingDataset',
        statistic_name=_example_statistic_name,
        name_mappings={
            'observation.state': ['proprio', 'action'],
        },
        statistic_keys=[
            'observation.state',
            'observation.eepose',
            'timestamp',
        ],
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_example_data_root,
            transforms=_transforms,
            action_window_size=32,
            action_key='observation.state',
            use_delta=False,
            window_start_idx=1,
            frame_window_size=33,
            require_full_window=True,
            statistic_name=_example_statistic_name)))

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
        ]),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    grad_accumulation_steps=2,
    lr_scheduler=dict(type='linear-warmup+linear-decay', warmup_steps=500),
    max_epochs=12,
    max_grad_norm=1.0,
    max_keep_ckpts=4,
    max_steps=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=(
            'jsonl',
            'wandb',
        ),
        grad_accumulation_steps=2,
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
    save_epoch_interval=1,
    save_iter_interval=500,
    sharding_strategy='full-shard',
    tokenizer=_tokenizer)

seed = 7

# Keep the existing public ALOHA task prompts as inference examples.
inference = dict(
    type='AlohaInferenceRunner',
    seed=7,
    task_descriptions={
        '1': 'pick up the brown bird toy with left arm',
        '2': 'pick up the brown bird toy with right arm',
        '3': 'pick up the pruple knitted teddy bear toy with left arm',
        '4': 'pick up the purple knitted teddy bear toy with right arm',
        '5': 'pick up the white racing car toy with left arm',
        '6': 'pick up the white racing car toy with right arm',
        '7': 'pick up the pruple caterpillar toy with left arm',
        '8': 'pick up the pruple caterpillar toy with right arm',
        '9': 'place it in the brown flat cardboard box with left arm',
        '10': 'place it in the brown flat cardboard box with right arm',
    },
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=21,
        extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=30.0,
                prepend_state_to_action=True),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer=_tokenizer,
                max_len=512,
                cfg_dropout_rate=0.0,
                action_metadata=_action_prompt_metadata,
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks'),
            dict(type='ResizeImages', height=256, width=256),
            dict(type='SimpleNormalizeImages'),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=64,
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std'),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='top_bottom_pair',
                top_view=0,
                bottom_views=(1, 2),
                bottom_height_ratio=0.5),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction', norm_type='mean_std', action_dim=14))
