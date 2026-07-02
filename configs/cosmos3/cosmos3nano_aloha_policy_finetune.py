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
"""Cosmos3-Nano ALOHA action-policy finetune config.

This is the production-oriented counterpart of
``cosmos3nano_aloha_debug_single_gpu.py``.  It keeps the FluxVLA ALOHA data
contract but aligns the Cosmos3 robot-policy pieces that transfer cleanly from
the native DROID recipe:

* current state + 32 future actions with 33 observation frames;
* ``prepend_state_to_action=True`` so action row 0 is clean conditioning;
* native Cosmos3 ``mode='wam'`` policy-only sampling (DROID recipe);
* Cosmos3-style action prompt metadata for viewpoint, duration/FPS, and
  resolution;
* CFG text dropout at 0.1;
* vision flow-matching loss weighted 10x (DROID ``loss_scale=10.0``);
* DROID-style shared image augmentation across all views/frames;
* a DROID-like three-view canvas: front view on top, two wrist views below.

Intentional diffs from native DROID:

* ALOHA actions are 14D dual-arm joint states normalized with FluxVLA stats;
  DROID uses raw 8D single-arm joint commands.
* ALOHA data is treated as 30 FPS; DROID policy uses 15 FPS.
"""

from copy import deepcopy

_ckpt_root = './checkpoints'
_cosmos3_nano_ckpt = _ckpt_root + '/Cosmos3-Nano'
_cosmos3_nano_transformer = _cosmos3_nano_ckpt + '/transformer'
_cosmos3_nano_vision_encoder = _cosmos3_nano_ckpt + '/vision_encoder'
_cosmos3_nano_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path=_cosmos3_nano_ckpt + '/text_tokenizer',
    model_max_length=4096,
    padding_side='right',
    trust_remote_code=True,
)
_wan22_vae_path = _ckpt_root + '/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'

_action_dim = 14
_max_action_dim = 64
_max_state_dim = 64
_embodiment_id = 21  # Config-local fine-tuning slot for action projector.
_action_horizon = 32
_frame_window_size = _action_horizon + 1
_prepend_state_to_action = True
_image_height = 224
_image_width = 224
_video_height = 336
_video_width = 224
_conditioning_fps = 30.0
_cfg_dropout_rate = 0.1
_base_lr = 8e-5  # DROID: 2e-4 x LambdaLinear f_max=0.4
_action_lr = _base_lr * 5.0  # DROID: 5x action-head LR
_viewpoint_description = (
    'The top row shows the high camera view looking at the dual-arm ALOHA '
    'workspace. The bottom row contains two horizontally concatenated '
    'wrist-mounted camera views: the left wrist camera on the left and the '
    'right wrist camera on the right.')
_vision_vae = dict(
    type='Cosmos3Wan22VAE',
    pretrained_name_or_path=_wan22_vae_path,
    encode_exact_durations=[_frame_window_size],
)

_cosmos3_nano_special_tokens = dict(
    eos_token_id=151645,
    start_of_generation=151652,
    end_of_generation=151653,
)

_cosmos3_nano_vlm_config = dict(
    model_type='qwen3_vl',
    vocab_size=151936,
    tie_word_embeddings=False,
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    vision_end_token_id=151653,
    text_config=dict(
        model_type='qwen3_vl_text',
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act='silu',
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151645,
        pad_token_id=0,
        tie_word_embeddings=False,
        rope_scaling=dict(
            rope_type='default',
            mrope_interleaved=True,
            mrope_section=[24, 20, 20],
        ),
        layer_types=['full_attention'] * 36,
    ),
    vision_config=dict(
        model_type='qwen3_vl',
        hidden_size=1152,
        hidden_act='gelu_pytorch_tanh',
        intermediate_size=4304,
        depth=27,
        num_heads=16,
        in_channels=3,
        initializer_range=0.02,
        out_hidden_size=4096,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
    ),
)

_cosmos3_nano_name_mapping = {
    # Official Cosmos3 checkpoint keeps this parameter name unchanged.
    'action_modality_embed': 'action_modality_embed',
    'vlm_backbone.model.language_model.embed_tokens.weight':
    'embed_tokens.weight',
    'vlm_backbone.lm_head.weight': 'lm_head.weight',
    'vlm_backbone.model.language_model.norm.weight': 'norm.weight',
    'vlm_backbone.model.language_model.norm_moe_gen.weight':
    'norm_moe_gen.weight',
    'vlm_backbone.model.language_model.layers.': 'layers.',
    'vision_in_proj.projector.': 'proj_in.',
    'vision_out_proj.projector.': 'proj_out.',
    'time_embedder.mlp.0.': 'time_embedder.linear_1.',
    'time_embedder.mlp.2.': 'time_embedder.linear_2.',
    'action_in_proj.': 'action_proj_in.',
    'action_out_proj.': 'action_proj_out.',
    '.self_attn.q_proj.': '.self_attn.to_q.',
    '.self_attn.k_proj.': '.self_attn.to_k.',
    '.self_attn.v_proj.': '.self_attn.to_v.',
    '.self_attn.o_proj.': '.self_attn.to_out.',
    '.self_attn.q_proj_moe_gen.': '.self_attn.add_q_proj.',
    '.self_attn.k_proj_moe_gen.': '.self_attn.add_k_proj.',
    '.self_attn.v_proj_moe_gen.': '.self_attn.add_v_proj.',
    '.self_attn.o_proj_moe_gen.': '.self_attn.to_add_out.',
    '.self_attn.q_norm.': '.self_attn.norm_q.',
    '.self_attn.k_norm.': '.self_attn.norm_k.',
    '.self_attn.q_norm_moe_gen.': '.self_attn.norm_added_q.',
    '.self_attn.k_norm_moe_gen.': '.self_attn.norm_added_k.',
    'vlm_backbone.model.visual.patch_embed.': 'patch_embed.',
    'vlm_backbone.model.visual.blocks.': 'blocks.',
    'vlm_backbone.model.visual.pos_embed.': 'pos_embed.',
    'vlm_backbone.model.visual.merger.': 'merger.',
    'vlm_backbone.model.visual.deepstack_merger_list.':
    'deepstack_merger_list.',
}

_rectified_flow_inference_config = dict(
    num_train_timesteps=1000,
    scheduler_type='unipc',
    num_steps=30,
    shift=10.0,
    use_dynamic_shifting=False,
    use_karras_sigmas=False,
)

model = dict(
    type='Cosmos3FlowMatching',
    vlm_backbone=dict(
        type='Cosmos3MoTBackbone',
        vlm_config=_cosmos3_nano_vlm_config,
        include_visual=False,
        vision_encoder_path=_cosmos3_nano_vision_encoder,
        skip_init_weights=True,
    ),
    vision_latent_dim=48,
    latent_patch_size=2,
    max_action_dim=64,
    num_embodiment_domains=32,
    vision_in_proj=dict(
        type='LinearProjector',
        in_dim=48 * 2 * 2,
        out_dim=4096,
    ),
    vision_out_proj=dict(
        type='LinearProjector',
        in_dim=4096,
        out_dim=48 * 2 * 2,
    ),
    action_in_proj=dict(
        type='DomainAwareLinear',
        input_size=64,
        output_size=4096,
        num_domains=32,
    ),
    action_out_proj=dict(
        type='DomainAwareLinear',
        input_size=4096,
        output_size=64,
        num_domains=32,
    ),
    rectified_flow_training_config=dict(
        shift={
            '256': 3,
            '480': 5,
            '720': 10,
        },
        use_dynamic_shift=False,
        train_time_image_distribution='logitnormal',
        train_time_video_distribution='waver',
        train_time_action_distribution='logitnormal',
        train_time_weight='uniform',
        independent_action_schedule=False,
        shift_action=None,
        use_high_sigma_strategy=False,
        high_sigma_ratio=0.05,
        high_sigma_timesteps_min=995,
        high_sigma_timesteps_max=1000,
        use_high_sigma_strategy_action=False,
        use_discrete_rf=False,
        normalize_loss_by_active=False,
        action_loss_weight=10.0,
        vision_loss_weight=10.0,  # DROID loss_scale=10.0
    ),
    rectified_flow_inference_config=_rectified_flow_inference_config,
    timestep_scale=0.001,
    packed_attention_backend='flash2',
    position_embedding_type='unified_3d_mrope',
    unified_3d_mrope_reset_spatial_ids=True,
    unified_3d_mrope_temporal_modality_margin=15000,
    enable_fps_modulation=True,
    base_fps=24.0,
    special_tokens=_cosmos3_nano_special_tokens,
    pretrained_name_or_path=_cosmos3_nano_transformer,
    name_mapping=_cosmos3_nano_name_mapping,
    vision_vae=_vision_vae,
    ori_action_dim=_action_dim,
    action_horizon=_action_horizon,
    # DROID: moe_gen + heads only.
    freeze_vlm_backbone=False,
    freeze_non_moe_vlm_backbone=True,
    enable_vision_loss=True,
)
inference_model = deepcopy(model)
# Eval builds the VAE without the external Wan2.2 file; the finetuned
# checkpoint already carries the frozen VAE weights.
inference_model['vision_vae']['pretrained_name_or_path'] = None

_transforms = [
    dict(
        type='ProcessParquetInputs',
        embodiment_id=_embodiment_id,
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
        name_mappings={'observation.state': ['states']},
    ),
    dict(
        type='AugVideo',
        rotation_range=0.0,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.6, 1.4),
        crop_scale=(0.95, 0.95),
        crop_ratio=(1.0, 1.0),
        prob=1.0,
        saturation_range=(0.5, 1.5),
        hue_delta=0.08,
    ),
    dict(
        type='ProcessCosmos3Prompt',
        tokenizer=_cosmos3_nano_tokenizer,
        max_len=256,
        cfg_dropout_rate=_cfg_dropout_rate,
        action_metadata=dict(
            viewpoint='concat_view',
            viewpoint_description=_viewpoint_description,
            frame_window_size=_frame_window_size,
            conditioning_fps=_conditioning_fps,
            video_height=_video_height,
            video_width=_video_width,
        ),
    ),
    dict(type='ResizeImages', height=_image_height, width=_image_width),
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_max_action_dim,
        state_dim=_max_state_dim,
        state_key='proprio',
        action_key='action',
        norm_type='mean_std',
    ),
    dict(
        type='BuildCosmos3Sequence',
        raw_action_dim=_action_dim,
        mode='wam',  # DROID: policy-only
        frame_window_size=_frame_window_size,
        prepend_state_to_action=_prepend_state_to_action,
        conditioning_fps=_conditioning_fps,
    ),
    dict(
        type='PrepareVideo',
        num_views=3,
        frame_window_size=_frame_window_size,
        tile_direction='top_bottom_pair',
        top_view=0,
        bottom_views=(1, 2),
        bottom_height_ratio=0.5,
    ),
]

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={'observation.state': ['proprio', 'action']},
        statistic_keys=[
            'observation.state',
            'observation.eepose',
            'timestamp',
        ],
        datasets=dict(
            type='ParquetDataset',
            data_root_path=(
                './datasets/RealRobot_AgileX_aloha_lerobot_v2/aloha_example'),
            transforms=_transforms,
            action_window_size=_action_horizon,
            action_key='observation.state',
            use_delta=False,
            statistic_name='private',
            window_start_idx=1,
            frame_window_size=_frame_window_size,
            require_full_window=True,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=None,
    max_epochs=6,
    optimizer=dict(
        type='AdamW',
        lr=_base_lr,
        weight_decay=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        fused=True,
        exclude_1d_from_weight_decay=False,
        paramwise_learning_rate={
            'action_in_proj.': _action_lr,
            'action_out_proj.': _action_lr,
            'action_modality_embed': _action_lr,
        }),
    max_grad_norm=1.0,
    save_iter_interval=1000,
    max_keep_ckpts=3,
    tokenizer=_cosmos3_nano_tokenizer,
    collator=dict(
        type='DictCollator',
        keys=[
            'images',
            'actions',
            'embodiment_ids',
            'raw_action_dim',
            'conditioning_fps',
            'action_fps',
        ],
        meta_keys=[
            'text_token_ids',
            'sequence_plan',
            'task_description',
            'stats',
            'info',
            'timestamp',
        ],
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
        # Linear decay over the epoch-based run.
        type='linear-warmup+linear-decay',
        warmup_steps=0,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

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
        embodiment_id=_embodiment_id,
        extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=_conditioning_fps,
                prepend_state_to_action=_prepend_state_to_action,
            ),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer=_cosmos3_nano_tokenizer,
                max_len=256,
                cfg_dropout_rate=0.0,
                action_metadata=dict(
                    viewpoint='concat_view',
                    viewpoint_description=_viewpoint_description,
                    frame_window_size=_frame_window_size,
                    conditioning_fps=_conditioning_fps,
                    video_height=_video_height,
                    video_width=_video_width,
                ),
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks',
            ),
            dict(
                type='ResizeImages',
                height=_image_height,
                width=_image_width,
            ),
            dict(type='SimpleNormalizeImages'),
            dict(
                type='NormalizeStatesAndActions',
                action_dim=_max_action_dim,
                state_dim=_max_state_dim,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std',
            ),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='top_bottom_pair',
                top_view=0,
                bottom_views=(1, 2),
                bottom_height_ratio=0.5,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=_action_dim,
    ),
)
