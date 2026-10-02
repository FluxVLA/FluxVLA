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
"""Cosmos3-Nano full-data fine-tuning and evaluation on RoboTwin.

The data combines clean and randomized LeRobot v2.1 datasets for all 50
RoboTwin tasks (27,500 episodes), with three 480x640 (H x W) RGB cameras.
The policy uses 14-D absolute joint/gripper targets and mean/std statistics
shared by training and closed-loop evaluation.

Usage (16 GPUs):
    # Run on both nodes with NODE_RANK=0/1 and the same MASTER_ADDR.
    torchrun --nproc-per-node=8 --nnodes=2 \
        --node-rank=${NODE_RANK} --master-addr=${MASTER_ADDR} \
        --master-port=29500 scripts/train.py \
        --config \
        configs/cosmos3/cosmos3nano_robotwin_all_data_full_finetune.py \
        --work-dir work_dirs/cosmos3nano_robotwin_all_data_full_finetune

Evaluation:
    torchrun --nproc-per-node=1 scripts/eval.py \
        --config \
        configs/cosmos3/cosmos3nano_robotwin_all_data_full_finetune.py \
        --ckpt-path <checkpoint.safetensors>

Evaluation defaults to all 50 tasks in both clean and random suites.
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

_ACTION_DIM = 14
_MAX_ACTION_DIM = 64
_MAX_STATE_DIM = 64
_ACTION_HORIZON = 32
_FRAME_WINDOW_SIZE = _ACTION_HORIZON + 1
_PREPEND_STATE_TO_ACTION = True
# Reuse the ALOHA embodiment domain for 14-D joint-position control.
_ROBOTWIN_EMBODIMENT_ID = 21
_CONDITIONING_FPS = 15.0
_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 256
_VIDEO_HEIGHT = 384
_VIDEO_WIDTH = 256
_CFG_DROPOUT_RATE = 0.1

_BASE_LR = 5e-5
_ACTION_LR = _BASE_LR * 5.0
# Global batch = GPU count * per-device batch * accumulation steps.
_PER_DEVICE_BATCH_SIZE = 16
_GRAD_ACCUMULATION_STEPS = 1
_MAX_STEPS = None
_MAX_EPOCHS = 5
_SAVE_ITER_INTERVAL = 500
_SAVE_EPOCH_INTERVAL = 1

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
        rms_norm_eps=1e-06,
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
    '.self_attn.k_norm.': '.self_attn.norm_k.',
    '.self_attn.k_norm_moe_gen.': '.self_attn.norm_added_k.',
    '.self_attn.k_proj.': '.self_attn.to_k.',
    '.self_attn.k_proj_moe_gen.': '.self_attn.add_k_proj.',
    '.self_attn.o_proj.': '.self_attn.to_out.',
    '.self_attn.o_proj_moe_gen.': '.self_attn.to_add_out.',
    '.self_attn.q_norm.': '.self_attn.norm_q.',
    '.self_attn.q_norm_moe_gen.': '.self_attn.norm_added_q.',
    '.self_attn.q_proj.': '.self_attn.to_q.',
    '.self_attn.q_proj_moe_gen.': '.self_attn.add_q_proj.',
    '.self_attn.v_proj.': '.self_attn.to_v.',
    '.self_attn.v_proj_moe_gen.': '.self_attn.add_v_proj.',
    'action_in_proj.': 'action_proj_in.',
    'action_modality_embed': 'action_modality_embed',
    'action_out_proj.': 'action_proj_out.',
    'time_embedder.mlp.0.': 'time_embedder.linear_1.',
    'time_embedder.mlp.2.': 'time_embedder.linear_2.',
    'vision_in_proj.projector.': 'proj_in.',
    'vision_out_proj.projector.': 'proj_out.',
    'vlm_backbone.lm_head.weight': 'lm_head.weight',
    'vlm_backbone.model.language_model.embed_tokens.weight':
    'embed_tokens.weight',
    'vlm_backbone.model.language_model.layers.': 'layers.',
    'vlm_backbone.model.language_model.norm.weight': 'norm.weight',
    'vlm_backbone.model.language_model.norm_moe_gen.weight':
    'norm_moe_gen.weight',
    'vlm_backbone.model.visual.blocks.': 'blocks.',
    'vlm_backbone.model.visual.deepstack_merger_list.':
    'deepstack_merger_list.',
    'vlm_backbone.model.visual.merger.': 'merger.',
    'vlm_backbone.model.visual.patch_embed.': 'patch_embed.',
    'vlm_backbone.model.visual.pos_embed.': 'pos_embed.',
}

_STATISTIC_NAME = 'robotwin_all'
_ROBOTWIN_DATA_ROOTS = [
    './datasets/robotwin_clean_lerobotv2.1',
    './datasets/robotwin_randomized_lerobotv2.1',
]

# All 50 RoboTwin tasks, matching the upstream RoboTwin benchmark.
_ROBOTWIN_TASK_LIST = [
    'adjust_bottle',
    'beat_block_hammer',
    'blocks_ranking_rgb',
    'blocks_ranking_size',
    'click_alarmclock',
    'click_bell',
    'dump_bin_bigbin',
    'grab_roller',
    'handover_block',
    'handover_mic',
    'hanging_mug',
    'lift_pot',
    'move_can_pot',
    'move_pillbottle_pad',
    'move_playingcard_away',
    'move_stapler_pad',
    'open_laptop',
    'open_microwave',
    'pick_diverse_bottles',
    'pick_dual_bottles',
    'place_a2b_left',
    'place_a2b_right',
    'place_bread_basket',
    'place_bread_skillet',
    'place_burger_fries',
    'place_can_basket',
    'place_cans_plasticbox',
    'place_container_plate',
    'place_dual_shoes',
    'place_empty_cup',
    'place_fan',
    'place_mouse_pad',
    'place_object_basket',
    'place_object_scale',
    'place_object_stand',
    'place_phone_stand',
    'place_shoe',
    'press_stapler',
    'put_bottles_dustbin',
    'put_object_cabinet',
    'rotate_qrcode',
    'scan_object',
    'shake_bottle',
    'shake_bottle_horizontally',
    'stack_blocks_three',
    'stack_blocks_two',
    'stack_bowls_three',
    'stack_bowls_two',
    'stamp_seal',
    'turn_switch',
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
        in_dim=192,
        out_dim=4096,
    ),
    vision_out_proj=dict(
        type='LinearProjector',
        in_dim=4096,
        out_dim=192,
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
        vision_loss_weight=1.0,
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
        encode_exact_durations=[_FRAME_WINDOW_SIZE],
    ),
    ori_action_dim=_ACTION_DIM,
    action_horizon=_ACTION_HORIZON,
    freeze_vlm_backbone=False,
    freeze_non_moe_vlm_backbone=True,
    reinitialize_action_policy=True,
    enable_vision_loss=True,
)

# Eval builds the VAE without the external Wan2.2 file; the fine-tuned
# checkpoint already carries the frozen VAE weights.
inference_model = deepcopy(model)
inference_model['vision_vae']['pretrained_name_or_path'] = None

_ACTION_PROMPT_METADATA = dict(
    append_viewpoint=False,
    conditioning_fps=_CONDITIONING_FPS,
    frame_window_size=_FRAME_WINDOW_SIZE,
    video_height=_VIDEO_HEIGHT,
    video_width=_VIDEO_WIDTH,
)

_TRANSFORMS = [
    dict(
        type='ProcessParquetInputs',
        embodiment_id=_ROBOTWIN_EMBODIMENT_ID,
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
        },
    ),
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
        hue_delta=0.08,
    ),
    dict(
        type='ProcessCosmos3Prompt',
        tokenizer=_COSMOS3_NANO_TOKENIZER,
        max_len=512,
        cfg_dropout_rate=_CFG_DROPOUT_RATE,
        action_metadata=_ACTION_PROMPT_METADATA,
    ),
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_MAX_ACTION_DIM,
        state_dim=_MAX_STATE_DIM,
        state_key='proprio',
        action_key='action',
        norm_type='mean_std',
    ),
    dict(
        type='BuildCosmos3Sequence',
        raw_action_dim=_ACTION_DIM,
        mode='wam',
        frame_window_size=_FRAME_WINDOW_SIZE,
        prepend_state_to_action=_PREPEND_STATE_TO_ACTION,
        conditioning_fps=_CONDITIONING_FPS,
    ),
    dict(
        type='PrepareVideo',
        num_views=3,
        frame_window_size=_FRAME_WINDOW_SIZE,
        tile_direction='top_bottom_pair',
        top_view=0,
        bottom_views=(1, 2),
        bottom_height_ratio=0.5,
    ),
]

train_dataloader = dict(
    per_device_batch_size=_PER_DEVICE_BATCH_SIZE,
    per_device_num_workers=4,
    prefetch_factor=1,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_ROBOTWIN_DATA_ROOTS,
            transforms=_TRANSFORMS,
            action_window_size=_ACTION_HORIZON,
            action_key='action',
            use_delta=False,
            statistic_name=_STATISTIC_NAME,
            # action[t] is the control target for observation[t].
            window_start_idx=0,
            frame_window_size=_FRAME_WINDOW_SIZE,
            require_full_window=True,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=_MAX_STEPS,
    max_epochs=_MAX_EPOCHS,
    save_iter_interval=_SAVE_ITER_INTERVAL,
    save_epoch_interval=_SAVE_EPOCH_INTERVAL,
    max_keep_ckpts=3,
    optimizer=dict(
        type='AdamW',
        lr=_BASE_LR,
        weight_decay=0.05,
        betas=(0.9, 0.99),
        eps=1e-08,
        fused=True,
        exclude_1d_from_weight_decay=False,
        paramwise_learning_rate={
            'action_in_proj.': _ACTION_LR,
            'action_modality_embed': _ACTION_LR,
            'action_out_proj.': _ACTION_LR,
        },
    ),
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
    grad_accumulation_steps=_GRAD_ACCUMULATION_STEPS,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=_GRAD_ACCUMULATION_STEPS,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+linear-decay',
        warmup_steps=500,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

seed = 7

# Both clean and random suites are evaluated by default.
# scripts/eval_robotwin_manager.sh also reads this default from the config.
eval = dict(
    runner=dict(
        type='RobotwinEvalRunner',
        task_suite_name=['clean', 'random'],
        model_family='cosmos3',
        task_list=_ROBOTWIN_TASK_LIST,
        instruction_type='unseen',
        eval_chunk_size=_ACTION_HORIZON,
        num_trials_per_task=100,
        seed=7,
        mixed_precision_dtype='bf16',
        save_video=False,
        unnorm_key=_STATISTIC_NAME,
        dataset=dict(
            type='PrivateInferenceDataset',
            extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
            transforms=[
                dict(
                    type='SetCosmos3ActionMetadata',
                    conditioning_fps=_CONDITIONING_FPS,
                    prepend_state_to_action=_PREPEND_STATE_TO_ACTION,
                ),
                dict(
                    type='ProcessCosmos3Prompt',
                    tokenizer=_COSMOS3_NANO_TOKENIZER,
                    max_len=512,
                    cfg_dropout_rate=0.0,
                    action_metadata=_ACTION_PROMPT_METADATA,
                    output_key='lang_tokens',
                    output_attention_mask_key='lang_masks',
                ),
                dict(
                    type='ResizeImages',
                    height=_IMAGE_HEIGHT,
                    width=_IMAGE_WIDTH),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=_MAX_ACTION_DIM,
                    state_dim=_MAX_STATE_DIM,
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
            embodiment_id=_ROBOTWIN_EMBODIMENT_ID,
            inject_model_path=False,
            img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        ),
        denormalize_action=dict(
            type='DenormalizePrivateAction',
            norm_type='mean_std',
            action_dim=_ACTION_DIM,
        ),
    ),
    manager=dict(
        num_gpus=1,
        max_tasks_per_gpu=1,
        master_port_base=29690,
        monitor_interval=5,
        status_interval=30,
        launch_delay=0.5,
    ),
)
