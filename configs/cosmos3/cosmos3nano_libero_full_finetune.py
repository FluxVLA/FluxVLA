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
# Cosmos3-Nano full finetune on all four LIBERO no-noops suites.
#
# Usage (8 GPU):
#   torchrun --nproc-per-node=8 scripts/train.py \
#     --config configs/cosmos3/cosmos3nano_libero_full_finetune.py
#
# LIBERO-specific settings:
# * LIBERO: action_dim=10 (eef pos + rot6d + gripper),
#   max_state_dim=64, 2 views.
# * Video keys: observation.images.image + observation.images.wrist_image.
# * Mixed dataset group: spatial + object + goal + 10 no-noops suites.
# * Cosmos3 uses LIBERO embodiment_id=5 for the action projector.
# * 192×320 (H×W) model canvas: two 256×256 views → 512×256 concat →
#   aspect-preserving resize + bottom reflection pad; third-person left,
#   wrist right.
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
_vision_vae = dict(
    type='Cosmos3Wan22VAE',
    pretrained_name_or_path=_wan22_vae_path,
)

# LIBERO robot spec
_action_dim = 10  # eef_pos(3) + eef_rot6d(6) + gripper(1)
_max_action_dim = 64  # Normalize target action width / Cosmos3 projector width
_max_state_dim = 64  # Normalize target state width
_action_horizon = 16
_frame_window_size = _action_horizon + 1
_prepend_state_to_action = False
# Official pipeline: 256×256 views → 512×256 concat → aspect-preserving
# resize to 320×160 + bottom reflection pad (32px) → 192×320 (H×W) canvas.
_image_height = 256
_image_width = 256  # per view; official image_size=256
# Cosmos3's 256/16:9 video bucket is (320, 192) (W, H); see
# third_party_models/cosmos3/data/vfm/utils.py::VIDEO_RES_SIZE_INFO.
# Keep this explicit: resize/padding and prompt video metadata must be
# changed together.
_video_height = 192
_video_width = 320
_conditioning_fps = 20.0  # Official LIBERO action-policy stats use 20 FPS
_libero_action_stats = dict(
    mean=[
        0.050704, 0.097407, -0.094833, 0.994873, -0.004579, -0.004288,
        0.004389, 0.996104, 0.001109, 0.476725
    ],
    std=[
        0.333621, 0.387175, 0.45714, 0.010807, 0.077802, 0.063386, 0.078571,
        0.009994, 0.038504, 0.49946
    ],
    min=[
        -0.9375, -0.9375, -0.9375, 0.902028, -0.356085, -0.367416, -0.370434,
        0.921907, -0.255, 0.0
    ],
    max=[
        0.9375, 0.9375, 0.9375, 1.0, 0.368853, 0.341214, 0.356395, 1.0,
        0.348251, 1.0
    ],
    q01=[
        -0.723214, -0.808929, -0.9375, 0.934955, -0.223431, -0.189878,
        -0.334735, 0.938516, -0.107736, 0.0
    ],
    q99=[
        0.9375, 0.870536, 0.9375, 1.0, 0.331, 0.163153, 0.226216, 1.0,
        0.127158, 1.0
    ],
)
_cfg_dropout_rate = 0.1
_base_lr = 5e-5  # Official Cosmos3 LIBERO action-policy base LR
_action_lr = _base_lr * 5.0
# Official LIBERO-all trains for 5000 optimizer steps at global batch 2048.
# 192×320 (H×W) canvas ≈ 1.9x the old lowered tokens → per-device 16;
# 16 GPUs × 16 × 8 accum = 2048 (8 GPUs: use grad_accumulation_steps=16).
_per_device_batch_size = 16
_grad_accumulation_steps = 8
_max_steps = 5000
_save_iter_interval = 500
_vision_vae['encode_exact_durations'] = [_frame_window_size]
_data_root_path = [
    './datasets/libero_spatial_no_noops_lerobotv2.1',
    './datasets/libero_object_no_noops_lerobotv2.1',
    './datasets/libero_goal_no_noops_lerobotv2.1',
    './datasets/libero_10_no_noops_lerobotv2.1',
]
# Approximate a 1:1:1:1 suite mix by repeating dataset configs. Current
# sample counts are spatial=53k, object=67k, goal=53k, libero_10=104k, so
# [2, 2, 2, 1] keeps the small suites close to libero_10 without adding
# custom sampler code.
_data_root_path = ([_data_root_path[0]] * 2 + [_data_root_path[1]] * 2 +
                   [_data_root_path[2]] * 2 + [_data_root_path[3]])
_statistic_name = 'all_libero_no_noops'

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
        vision_loss_weight=10.0,  # Official LIBERO recipe: loss_scale=10.0
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
    ),
    rectified_flow_inference_config=dict(
        num_train_timesteps=1000,
        scheduler_type='unipc',
        num_steps=30,
        shift=10.0,
        use_dynamic_shifting=False,
        use_karras_sigmas=False,
    ),
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
        embodiment_id=5,
    ),
    dict(type='LiberoFramewiseActionToRot6D'),
    dict(
        type='ProcessCosmos3Prompt',
        tokenizer=_cosmos3_nano_tokenizer,
        max_len=512,
        cfg_dropout_rate=_cfg_dropout_rate,
        format_prompt_as_json=True,
        action_metadata=dict(
            append_viewpoint=False,
            frame_window_size=_frame_window_size,
            conditioning_fps=_conditioning_fps,
            video_height=_video_height,
            video_width=_video_width,
        ),
    ),
    dict(type='ResizeImages', height=_image_height, width=_image_width),
    # SimpleNormalizeImages: scales [0,255] uint8 → [-1,1] float32
    # (required by Wan2.2 VAE)
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_max_action_dim,
        state_dim=_max_state_dim,
        state_key='proprio',
        action_key='action',
        state_norm_type='none',
        action_norm_type='quantile',
    ),
    dict(
        type='BuildCosmos3Sequence',
        raw_action_dim=_action_dim,
        mode='wam',
        frame_window_size=_frame_window_size,
        prepend_state_to_action=_prepend_state_to_action,
        conditioning_fps=_conditioning_fps,
    ),
    dict(
        type='PrepareVideo',
        num_views=2,
        frame_window_size=_frame_window_size,
        tile_direction='horizontal',
    ),
    dict(
        type='ResizeAndReflectPad',
        height=_video_height,
        width=_video_width,
    ),
]

train_dataloader = dict(
    per_device_batch_size=_per_device_batch_size,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        statistics_overrides={
            _statistic_name: dict(action=_libero_action_stats)
        },
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=data_root_path,
                transforms=_transforms,
                action_window_size=_action_horizon,
                action_key='action',
                use_delta=False,
                statistic_name=_statistic_name,
                window_start_idx=0,
                frame_window_size=_frame_window_size,
                require_full_window=True,
            ) for data_root_path in _data_root_path
        ],
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=_max_steps,
    save_iter_interval=_save_iter_interval,
    max_keep_ckpts=2,
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
    tokenizer=_cosmos3_nano_tokenizer,
    collator=dict(
        type='DictCollator',
        keys=[
            'images',
            'states',
            'actions',
            'action_masks',
            'img_masks',
            'frame_masks',
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
            'viewpoint',
        ],
    ),
    sampler=None,
    grad_accumulation_steps=_grad_accumulation_steps,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=_grad_accumulation_steps,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+linear-decay',
        warmup_steps=500,
        # Matches official Cosmos3 LIBERO cycle_lengths=[16000].
        cycle_length=16000,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_10',
    model_family='cosmos3',
    norm_stats_key=_statistic_name,
    eval_chunk_size=_action_horizon,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        img_buffer_len=1,
        extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                resize_size=_image_height,
                embodiment_id=5,
            ),
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=_conditioning_fps,
                prepend_state_to_action=_prepend_state_to_action,
            ),
            dict(
                type='TransformImage',
                # input_sizes use (C, W, H) PIL ordering; per-view 256×256.
                image_resize_strategy='resize-crop',
                input_sizes=[[3, _image_width, _image_height],
                             [3, _image_width, _image_height]],
                means=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
            ),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer=_cosmos3_nano_tokenizer,
                max_len=512,
                cfg_dropout_rate=0.0,
                format_prompt_as_json=True,
                action_metadata=dict(
                    append_viewpoint=False,
                    frame_window_size=_frame_window_size,
                    conditioning_fps=_conditioning_fps,
                    video_height=_video_height,
                    video_width=_video_width,
                ),
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks',
            ),
            dict(
                type='PrepareVideo',
                num_views=2,
                frame_window_size=1,
                tile_direction='horizontal',
            ),
            dict(
                type='ResizeAndReflectPad',
                height=_video_height,
                width=_video_width,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoFramewiseRot6DAction',
        norm_type='quantile_rot',
        action_dim=_action_dim,
    ),
)
