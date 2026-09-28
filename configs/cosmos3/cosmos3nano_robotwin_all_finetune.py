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
"""Cosmos3-Nano training on all 50 RoboTwin tasks (clean + randomized).

Run from the FluxVLA repository root:
    torchrun --standalone --nproc-per-node=1 scripts/train.py \
        --config configs/cosmos3/cosmos3nano_robotwin_all_finetune.py

The model architecture follows the Nano LIBERO recipe; data processing and
loss weights follow the Edge RoboTwin config. Microbatch 16 is used after
batch 8 left about 30% GPU memory unused.
"""

from copy import deepcopy

_data_root_paths = [
    './datasets/robotwin_clean_lerobotv2.1',
    './datasets/robotwin_randomized_lerobotv2.1',
]
_statistic_name = 'robotwin_all'

_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path='./checkpoints/Cosmos3-Nano/text_tokenizer',
    model_max_length=4096,
    padding_side='right',
    trust_remote_code=True)

_action_prompt_metadata = dict(
    append_viewpoint=False,
    conditioning_fps=15.0,
    frame_window_size=33,
    video_height=384,
    video_width=256)

_transforms = [
    dict(
        type='ProcessParquetInputs',
        # Reuse the ALOHA embodiment domain for 14-D joint-position control.
        embodiment_id=21,
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
        conditioning_fps=15.0),
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
    action_horizon=32,
    action_in_proj=dict(
        input_size=64,
        num_domains=32,
        output_size=4096,
        type='DomainAwareLinear'),
    action_out_proj=dict(
        input_size=4096,
        num_domains=32,
        output_size=64,
        type='DomainAwareLinear'),
    base_fps=24.0,
    enable_fps_modulation=True,
    enable_vision_loss=True,
    freeze_non_moe_vlm_backbone=True,
    freeze_vlm_backbone=False,
    latent_patch_size=2,
    max_action_dim=64,
    name_mapping=dict({
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
        'vlm_backbone.model.visual.pos_embed.': 'pos_embed.'
    }),
    num_embodiment_domains=32,
    ori_action_dim=14,
    packed_attention_backend='flash2',
    position_embedding_type='unified_3d_mrope',
    pretrained_name_or_path='./checkpoints/Cosmos3-Nano/transformer',
    rectified_flow_inference_config=dict(
        num_steps=30,
        num_train_timesteps=1000,
        scheduler_type='unipc',
        shift=10.0,
        use_dynamic_shifting=False,
        use_karras_sigmas=False),
    rectified_flow_training_config=dict(
        action_loss_weight=10.0,
        high_sigma_ratio=0.05,
        high_sigma_timesteps_max=1000,
        high_sigma_timesteps_min=995,
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
    reinitialize_action_policy=True,
    special_tokens=dict(
        end_of_generation=151653,
        eos_token_id=151645,
        start_of_generation=151652),
    timestep_scale=0.001,
    type='Cosmos3FlowMatching',
    unified_3d_mrope_reset_spatial_ids=True,
    unified_3d_mrope_temporal_modality_margin=15000,
    vision_in_proj=dict(in_dim=192, out_dim=4096, type='LinearProjector'),
    vision_latent_dim=48,
    vision_out_proj=dict(in_dim=4096, out_dim=192, type='LinearProjector'),
    vision_vae=dict(
        encode_exact_durations=[
            33,
        ],
        pretrained_name_or_path='./checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',
        type='Cosmos3Wan22VAE'),
    vlm_backbone=dict(
        include_visual=False,
        skip_init_weights=True,
        type='Cosmos3MoTBackbone',
        vision_encoder_path='./checkpoints/Cosmos3-Nano/vision_encoder',
        vlm_config=dict(
            image_token_id=151655,
            model_type='qwen3_vl',
            text_config=dict(
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                head_dim=128,
                hidden_act='silu',
                hidden_size=4096,
                intermediate_size=12288,
                layer_types=[
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                    'full_attention',
                ],
                max_position_embeddings=262144,
                model_type='qwen3_vl_text',
                num_attention_heads=32,
                num_hidden_layers=36,
                num_key_value_heads=8,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_scaling=dict(
                    mrope_interleaved=True,
                    mrope_section=[
                        24,
                        20,
                        20,
                    ],
                    rope_type='default'),
                rope_theta=5000000,
                tie_word_embeddings=False,
                vocab_size=151936),
            tie_word_embeddings=False,
            video_token_id=151656,
            vision_config=dict(
                deepstack_visual_indexes=[
                    8,
                    16,
                    24,
                ],
                depth=27,
                hidden_act='gelu_pytorch_tanh',
                hidden_size=1152,
                in_channels=3,
                initializer_range=0.02,
                intermediate_size=4304,
                model_type='qwen3_vl',
                num_heads=16,
                num_position_embeddings=2304,
                out_hidden_size=4096,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2),
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vocab_size=151936)))

inference_model = deepcopy(model)
# Eval builds the VAE without the external Wan2.2 file; the fine-tuned
# checkpoint already carries the frozen VAE weights.
inference_model['vision_vae']['pretrained_name_or_path'] = None

train_dataloader = dict(
    # Global batch = GPU count * 16 * grad_accumulation_steps.
    per_device_batch_size=16,
    per_device_num_workers=4,
    prefetch_factor=1,
    dataset=dict(
        type='DistributedRepeatingDataset',
        statistic_name=_statistic_name,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=[
            'observation.state',
            'timestamp',
            'action',
        ],
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_data_root_paths,
            transforms=_transforms,
            action_window_size=32,
            action_key='action',
            use_delta=False,
            # action[t] is the control target for observation[t].
            window_start_idx=0,
            frame_window_size=33,
            require_full_window=True,
            statistic_name=_statistic_name)))

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
            'viewpoint',
        ],
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
        ]),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    grad_accumulation_steps=1,
    lr_scheduler=dict(type='linear-warmup+linear-decay', warmup_steps=500),
    max_epochs=5,
    max_grad_norm=1.0,
    max_keep_ckpts=3,
    max_steps=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=(
            'jsonl',
            'wandb',
        ),
        grad_accumulation_steps=1,
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

# Evaluate both conditions with TASK_SUITE_NAME="clean random" when using
# scripts/eval_robotwin_manager.sh. The default evaluates randomized scenes.
eval = dict(
    runner=dict(
        type='RobotwinEvalRunner',
        model_family='cosmos3',
        task_list=_ROBOTWIN_TASK_LIST,
        task_suite_name='random',
        instruction_type='unseen',
        eval_chunk_size=32,
        num_trials_per_task=100,
        seed=7,
        unnorm_key=_statistic_name,
        mixed_precision_dtype='bf16',
        save_video=False,
        dataset=dict(
            type='PrivateInferenceDataset',
            embodiment_id=21,
            inject_model_path=False,
            extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
            img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            transforms=[
                dict(
                    type='SetCosmos3ActionMetadata',
                    conditioning_fps=15.0,
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
            type='DenormalizePrivateAction',
            norm_type='mean_std',
            action_dim=14)),
    manager=dict(
        num_gpus=1,
        max_tasks_per_gpu=1,
        master_port_base=29690,
        monitor_interval=5,
        status_interval=30,
        launch_delay=0.5))
