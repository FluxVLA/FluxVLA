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
"""Cosmos3-Nano training and evaluation on RoboDojo ARX-X5.

The policy consumes all three RoboDojo cameras and absolute 14D joint
states/actions.  Its action-policy settings follow the official Cosmos3 DROID
recipe in the sibling ``cosmos-framework`` checkout: 32-step chunks (33 video
frames), structured JSON prompts, CFG dropout, 10k updates, and a 5x
learning-rate multiplier for the action projections. State conditioning is
disabled because it regressed RoboDojo closed-loop simulation performance.

Training (16 GPUs; global batch 2048):
    torchrun --nproc-per-node=8 --nnodes=2 scripts/train.py \
        --config \
        configs/cosmos3/cosmos3nano_robodojo_full_finetune.py \
        --work-dir work_dirs/cosmos3nano_robodojo_full_finetune

Evaluation is served through the RoboDojo/FluxThemis manager using the
``eval`` and ``themis`` sections below.
"""

from copy import deepcopy

_CKPT_ROOT = './checkpoints'
_COSMOS3_NANO_CKPT = _CKPT_ROOT + '/Cosmos3-Nano'
_COSMOS3_NANO_TRANSFORMER = _COSMOS3_NANO_CKPT + '/transformer'
_COSMOS3_NANO_VISION_ENCODER = _COSMOS3_NANO_CKPT + '/vision_encoder'
_COSMOS3_NANO_TOKENIZER = dict(
    type='PretrainedTokenizer',
    model_path=_COSMOS3_NANO_CKPT + '/text_tokenizer',
    model_max_length=4096,
    padding_side='right',
    trust_remote_code=True,
)
_WAN22_VAE_PATH = _CKPT_ROOT + '/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'

_COSMOS3_NANO_SPECIAL_TOKENS = dict(
    eos_token_id=151645,
    start_of_generation=151652,
    end_of_generation=151653,
)

_COSMOS3_NANO_VLM_CONFIG = dict(
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

_COSMOS3_NANO_NAME_MAPPING = {
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

_ROBODOJO_DATA_ROOT = './datasets/RoboDojo_lerobot_v21_video'
_STATISTIC_NAME = 'robodojo_arx_x5'
_ACTION_DIM = 14
_MAX_ACTION_DIM = 64
_MAX_STATE_DIM = 64
_ACTION_HORIZON = 32
_FRAME_WINDOW_SIZE = _ACTION_HORIZON + 1
_CONDITIONING_FPS = 25.0
_EMBODIMENT_ID = 31
_PREPEND_STATE_TO_ACTION = False

# Exact frame-level absolute-action quantiles computed from all 1,859,602
# RoboDojo samples. The dataset's automatically aggregated statistics do not
# contain quantiles, so provide them explicitly for Cosmos3 normalization.
_ROBODOJO_ACTION_QUANTILES = dict(
    q01=[
        -1.051365569829941, -3.5348887174584814e-14, 1.741881830823392e-16,
        -1.5984010362625123, -0.6003574305772781, -1.6678147149085998, 0.0,
        -0.4509398394823074, -2.5859282299029243e-14, 1.234041714549172e-16,
        -1.64055180311203, -1.2633746123313903, -1.7645285475254058, 0.0
    ],
    q99=[
        0.5431358617544174, 2.495765209197998, 2.492974226474762,
        1.3241519677639007, 1.2496635341644287, 1.7392305016517635, 1.0,
        1.0814965963363647, 2.4167031812667843, 2.3470158338546754,
        1.141731116771698, 0.5208872479200363, 1.489715996980667, 1.0
    ])

# Preserve RoboDojo's native 4:3 camera aspect ratio while downsampling each
# 640x480 view. The three views are arranged as one overhead view above two
# half-height wrist views, then padded to Cosmos3's official 256-tier 3:4
# video bucket: VIDEO_RES_SIZE_INFO['256']['3,4'] == (W=256, H=320).
_IMAGE_HEIGHT = 192
_IMAGE_WIDTH = 256
_VIDEO_HEIGHT = 320
_VIDEO_WIDTH = 256
# Encode the 32 frames after the causal key frame as two 16-frame chunks.
# This preserves all 33 input frames and the resulting 9 latent frames while
# reducing the frozen Wan VAE's peak activation memory.
_VAE_ENCODE_CHUNK_FRAMES = {'256': 16, '480': 24, '720': 12}

_ROBODOJO_GENERALIZATION_BASE_TASKS = (
    'stack_bowls',
    'push_T',
    'pack_objects_into_box',
    'fold_clothes',
    'hang_mugs',
    'sweep_blocks',
    'pour_liquid_into_cup',
    'make_toast',
    'arrange_largest_number',
    'sort_nesting_dolls_by_size',
    'store_laptop_and_headphones',
    'stack_blocks',
)
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

_ROBODOJO_TRANSFORMS = [
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
            'observation.images.cam_high',
            'observation.images.cam_left_wrist',
            'observation.images.cam_right_wrist',
        ],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        }),
    dict(type='ResizeImages', height=_IMAGE_HEIGHT, width=_IMAGE_WIDTH),
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
        tokenizer=_COSMOS3_NANO_TOKENIZER,
        max_len=512,
        cfg_dropout_rate=0.1,
        format_prompt_as_json=True,
        action_metadata=dict(
            append_viewpoint=False,
            viewpoint='concat_view',
            frame_window_size=_FRAME_WINDOW_SIZE,
            conditioning_fps=_CONDITIONING_FPS,
            video_height=_VIDEO_HEIGHT,
            video_width=_VIDEO_WIDTH)),
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_MAX_ACTION_DIM,
        state_dim=_MAX_STATE_DIM,
        state_key='proprio',
        action_key='action',
        state_norm_type='none',
        action_norm_type='quantile'),
    dict(
        type='BuildCosmos3Sequence',
        raw_action_dim=_ACTION_DIM,
        mode='wam',
        frame_window_size=_FRAME_WINDOW_SIZE,
        prepend_state_to_action=_PREPEND_STATE_TO_ACTION,
        conditioning_fps=_CONDITIONING_FPS),
    dict(
        type='PrepareVideo',
        num_views=3,
        frame_window_size=_FRAME_WINDOW_SIZE,
        tile_direction='top_bottom_pair',
        top_view=0,
        bottom_views=(1, 2),
        bottom_height_ratio=0.5),
    dict(type='ResizeAndReflectPad', height=_VIDEO_HEIGHT, width=_VIDEO_WIDTH),
]

model = dict(
    type='Cosmos3FlowMatching',
    vlm_backbone=dict(
        type='Cosmos3MoTBackbone',
        vlm_config=_COSMOS3_NANO_VLM_CONFIG,
        include_visual=False,
        vision_encoder_path=_COSMOS3_NANO_VISION_ENCODER,
        skip_init_weights=True,
    ),
    vision_latent_dim=48,
    latent_patch_size=2,
    max_action_dim=_MAX_ACTION_DIM,
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
        input_size=_MAX_ACTION_DIM,
        output_size=4096,
        num_domains=32,
    ),
    action_out_proj=dict(
        type='DomainAwareLinear',
        input_size=4096,
        output_size=_MAX_ACTION_DIM,
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
        vision_loss_weight=10.0,
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
    special_tokens=_COSMOS3_NANO_SPECIAL_TOKENS,
    pretrained_name_or_path=_COSMOS3_NANO_TRANSFORMER,
    name_mapping=_COSMOS3_NANO_NAME_MAPPING,
    vision_vae=dict(
        type='Cosmos3Wan22VAE',
        pretrained_name_or_path=_WAN22_VAE_PATH,
        encode_chunk_frames=_VAE_ENCODE_CHUNK_FRAMES,
        encode_exact_durations=[_FRAME_WINDOW_SIZE],
    ),
    ori_action_dim=_ACTION_DIM,
    action_horizon=_ACTION_HORIZON,
    freeze_vlm_backbone=False,
    freeze_non_moe_vlm_backbone=True,
    enable_vision_loss=True,
    reinitialize_action_policy=True,
)

inference_model = deepcopy(model)
inference_model['vision_vae'].pop('encode_chunk_frames')
inference_model['vision_vae']['pretrained_name_or_path'] = None

train_dataloader = dict(
    # Batch 16 exhausts an 80 GB GPU while the frozen Wan VAE encodes the
    # causal key frame. Keep the same global batch via extra accumulation.
    per_device_batch_size=8,
    per_device_num_workers=4,
    prefetch_factor=1,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=42,
        reshuffle_each_epoch=True,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action', 'timestamp'],
        statistic_name=_STATISTIC_NAME,
        statistics_overrides=dict(
            robodojo_arx_x5=dict(action=_ROBODOJO_ACTION_QUANTILES)),
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[_ROBODOJO_DATA_ROOT],
            transforms=_ROBODOJO_TRANSFORMS,
            action_window_size=_ACTION_HORIZON,
            action_key='action',
            use_delta=False,
            statistic_name=_STATISTIC_NAME,
            window_start_idx=0,
            frame_window_size=_FRAME_WINDOW_SIZE,
            require_full_window=True)),
)

# Match the official Cosmos3 LIBERO optimization scale: global batch 2048,
# base LR 5e-5, and 5x LR for the action projections. With 16 GPUs this is
# 8 samples/GPU * 16 GPUs * 16 accumulation steps = 2048.
runner = dict(
    type='FSDPTrainRunner',
    max_steps=10000,
    save_iter_interval=1000,
    max_keep_ckpts=2,
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        fused=True,
        exclude_1d_from_weight_decay=False,
        paramwise_learning_rate={
            'action_in_proj.': 2.5e-4,
            'action_out_proj.': 2.5e-4,
            'action_modality_embed': 2.5e-4,
        }),
    max_grad_norm=1.0,
    tokenizer=_COSMOS3_NANO_TOKENIZER,
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
    grad_accumulation_steps=16,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=16,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+linear-decay', warmup_steps=0,
        cycle_length=100000),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='cosmos3',
    eval_chunk_size=_ACTION_HORIZON,
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='RoboDojoEvalDataset',
        unnorm_key=_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                embodiment_id=_EMBODIMENT_ID),
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=_CONDITIONING_FPS,
                prepend_state_to_action=_PREPEND_STATE_TO_ACTION),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer=_COSMOS3_NANO_TOKENIZER,
                max_len=512,
                cfg_dropout_rate=0.0,
                format_prompt_as_json=True,
                action_metadata=dict(
                    append_viewpoint=False,
                    viewpoint='concat_view',
                    frame_window_size=_FRAME_WINDOW_SIZE,
                    conditioning_fps=_CONDITIONING_FPS,
                    video_height=_VIDEO_HEIGHT,
                    video_width=_VIDEO_WIDTH),
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks'),
            dict(
                type='ResizeImages',
                height=_IMAGE_HEIGHT,
                width=_IMAGE_WIDTH,
                key='pixel_values'),
            dict(
                type='SimpleNormalizeImages',
                key='pixel_values',
                output_type='torch'),
            dict(
                type='PrepareVideo',
                num_views=3,
                frame_window_size=1,
                tile_direction='top_bottom_pair',
                top_view=0,
                bottom_views=(1, 2),
                bottom_height_ratio=0.5),
            dict(
                type='ResizeAndReflectPad',
                height=_VIDEO_HEIGHT,
                width=_VIDEO_WIDTH),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='quantile',
        action_dim=_ACTION_DIM,
        statistic_name=_STATISTIC_NAME),
)

themis = dict(
    transport=dict(
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key=_STATISTIC_NAME,
    ),
    evaluation=dict(
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root='/root/projects/RoboDojo',
            device_id=1,
            action_mode='joint',
            headless=True,
            save_videos=False,
        ),
        policy_client=dict(type='FluxVLAPolicyClient'),
        reporter=dict(type='FluxVLAEvaluationReporter'),
        evaluator=dict(type='SuccessRateEvaluator'),
        base_seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        execute_horizon=_ACTION_HORIZON,
        stop_on_success=True,
    ),
    server=dict(
        runner=dict(type='FluxVLAServingRunner'),
        evaluation_reporting=dict(enabled=True),
        denormalize_context=dict(task_suite_name=_STATISTIC_NAME),
    ),
)
