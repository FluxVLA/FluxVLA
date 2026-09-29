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
"""FastWAM (uncond) full-data fine-tuning and evaluation on RoboTwin.

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
        configs/fastwam/fastwam_robotwin_all_data_full_finetune.py \
        --work-dir work_dirs/fastwam_robotwin_all_data_full_finetune

Evaluation:
    torchrun --nproc-per-node=1 scripts/eval.py \
        --config \
        configs/fastwam/fastwam_robotwin_all_data_full_finetune.py \
        --ckpt-path <checkpoint.safetensors>

Evaluation defaults to all 50 tasks in both clean and random suites.
"""

seed = 42

_CKPT_ROOT = './checkpoints'
_FASTWAM_ROOT = _CKPT_ROOT + '/fastwam_base_full'
_FASTWAM_CHECKPOINT = _FASTWAM_ROOT + '/fastwam_base_full.safetensors'
_FASTWAM_TOKENIZER = _FASTWAM_ROOT + '/tokenizer'

_FRAME_WINDOW_SIZE = 9
_ACTION_WINDOW_SIZE = 32
_FRAME_SAMPLE_STRIDE = 4
_ACTION_DIM = 14
_STATE_DIM = 14
_ACTION_NORM_TYPE = 'mean_std'

# Train jointly on the merged clean + randomized RoboTwin LeRobot sets.
_ROBOTWIN_DATA_ROOTS = [
    'datasets/robotwin_clean_lerobotv2.1',
    'datasets/robotwin_randomized_lerobotv2.1',
]
_STATISTIC_NAME = 'robotwin_all'
_TEXT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')

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
    type='FastWAMVLA',
    pretrained_name_or_path=_FASTWAM_CHECKPOINT,
    pretrained_skip_prefixes=[
        'vla_head.mot.mixtures.action.action_encoder.',
        'vla_head.mot.mixtures.action.head.',
        'vla_head.proprio_encoder.',
    ],
    torch_dtype='bf16',
    num_views=3,
    frame_window_size=_FRAME_WINDOW_SIZE,
    proprio_dim=_STATE_DIM,
    action_horizon=_ACTION_WINDOW_SIZE,
    mot_checkpoint_mixed_attn=True,
    vlm_backbone=dict(
        type='Wan22Backbone',
        text_embed_cache_context_len=128,
        text_embed_cache_size=256,
        text_embed_cache_device='cpu',
    ),
    vla_head=dict(
        type='FastWAMHead',
        video_dit_config=dict(
            has_image_input=False,
            patch_size=[1, 2, 2],
            in_dim=48,
            hidden_dim=3072,
            ffn_dim=14336,
            freq_dim=256,
            text_dim=4096,
            out_dim=48,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            eps=1.0e-06,
            seperated_timestep=True,
            require_clip_embedding=False,
            require_vae_embedding=False,
            fuse_vae_embedding_in_latents=True,
            video_attention_mask_mode='first_frame_causal',
            action_conditioned=False,
            action_dim=_ACTION_DIM,
            action_group_causal_mask_mode='group_diagonal',
            use_gradient_checkpointing=True,
        ),
        action_dit_config=dict(
            action_dim=_ACTION_DIM,
            hidden_dim=1024,
            ffn_dim=4096,
            num_heads=24,
            attn_head_dim=128,
            num_layers=30,
            text_dim=4096,
            freq_dim=256,
            eps=1.0e-06,
            use_gradient_checkpointing=True,
        ),
        video_scheduler=dict(
            train_shift=5.0, infer_shift=5.0, num_train_timesteps=1000),
        # Upstream FastWAM uses shift 1.0 for the action scheduler on
        # every task, including RoboTwin.
        action_scheduler=dict(
            train_shift=1.0, infer_shift=1.0, num_train_timesteps=1000),
        loss=dict(lambda_video=1.0, lambda_action=1.0),
    ),
)

# Training and evaluation encode unseen prompts online, then reuse a
# per-process CPU-memory LRU cache without reading or writing cache files.
inference_model = model.copy()

train_dataloader = dict(
    # Upstream RoboTwin recipe trains with global batch 16; on 4x72GB GPUs
    # this is 2 microbatch x 4 GPUs x 2 accumulation (microbatch 4 OOMs on
    # the 384x320x9 video branch).
    per_device_batch_size=2,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        reshuffle_each_epoch=True,
        seed=42,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_ROBOTWIN_DATA_ROOTS,
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
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                    video_backend='torchcodec',
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=0,
                ),
                dict(
                    type='ResizeImages',
                    height=256,
                    width=320,
                    backend='torchvision',
                    scale_to_unit_interval=True,
                ),
                dict(
                    type='NormalizeImages',
                    means=[0.5, 0.5, 0.5],
                    stds=[0.5, 0.5, 0.5],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=_ACTION_DIM,
                    state_dim=_STATE_DIM,
                    state_key='proprio',
                    action_key='action',
                    norm_type=_ACTION_NORM_TYPE,
                ),
                # cam_high on top (256x320), wrist cameras side by side
                # below (128x160 each) -> 384x320, matching the upstream
                # ``concat_multi_camera="robotwin"`` layout.
                dict(
                    type='PrepareVideo',
                    num_views=3,
                    frame_window_size=_FRAME_WINDOW_SIZE,
                    tile_direction='top_bottom_pair',
                    top_view=0,
                    bottom_views=(1, 2),
                    bottom_height_ratio=0.5,
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_FASTWAM_TOKENIZER),
                    max_len=128,
                    use_conversation=False,
                    prompt_template=_TEXT_PROMPT_TEMPLATE,
                ),
            ],
            action_window_size=_ACTION_WINDOW_SIZE,
            action_key='action',
            use_delta=False,
            statistic_name=_STATISTIC_NAME,
            window_start_idx=0,
            frame_window_size=_FRAME_WINDOW_SIZE,
            frame_sample_stride=_FRAME_SAMPLE_STRIDE,
        ),
    ),
)

val_dataloader = None
eval_dataset = None

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=5,
    max_steps=None,
    save_epoch_interval=1,
    # RoboTwin epochs are ~47k steps at global batch 128; disable the
    # step-based trigger so checkpoints are epoch-only, matching the
    # effective LIBERO behavior (whose short epochs rarely hit 10000).
    save_iter_interval=10000000,
    max_keep_ckpts=10,
    optimizer=dict(lr=1e-4, type='AdamW', weight_decay=1e-2),
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['task_description', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    tokenizer=dict(type='PretrainedTokenizer', model_path=_FASTWAM_TOKENIZER),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay-min-lr',
        warmup_ratio=0.05,
        min_lr_ratio=0.01,
        betas=(0.9, 0.95),
        weight_decay_style='uniform',
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    # 4x72GB memory budget: the whole 6B FastWAM head is one FSDP flat
    # unit, so fp32 masters + fp32 grad reduction do not fit. Keep managed
    # params, AdamW states, and gradient reduction in bf16 (matching the
    # upstream pure-bf16 execution).
    reduce_in_full_precision=False,
    pre_fsdp_param_dtype='bf16',
    grad_accumulation_steps=2,
    mixed_precision_dtype='bf16',
    dataset_sharding_strategy='blockwise',
    # full-shard: on 4x72GB GPUs the 12.4B model's AdamW states alone are
    # ~25GB per rank under shard-grad-op, which OOMs; FULL_SHARD also
    # shards parameters.
    sharding_strategy='full-shard',
    evaluator=dict(
        type='training-eval',
        eval_every=1000,
        num_inference_steps=10,
        seed=42,
        save_video=False,
        video_fps=8,
    ),
)

eval = dict(
    runner=dict(
        type='RobotwinEvalRunner',
        # Evaluate clean and random suites with unseen instructions,
        # 100 episodes per task per suite, replanning every 24 of 32 actions.
        task_suite_name=['clean', 'random'],
        model_family='fastwam',
        task_list=_ROBOTWIN_TASK_LIST,
        instruction_type='unseen',
        eval_chunk_size=24,
        num_trials_per_task=100,
        seed=42,
        unnorm_key=_STATISTIC_NAME,
        mixed_precision_dtype='bf16',
        save_video=False,
        dataset=dict(
            type='PrivateInferenceDataset',
            img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            transforms=[
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=_ACTION_DIM,
                    state_dim=_STATE_DIM,
                    state_key='proprio',
                    action_key='action',
                    norm_type=_ACTION_NORM_TYPE,
                ),
                dict(
                    type='LiberoPromptFromInputs',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_FASTWAM_TOKENIZER),
                    max_len=128,
                    use_conversation=False,
                    prompt_template=_TEXT_PROMPT_TEMPLATE,
                ),
                dict(type='ResizeImages', height=256, width=320),
                dict(type='SimpleNormalizeImages'),
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
            norm_type=_ACTION_NORM_TYPE,
            action_dim=_ACTION_DIM,
        ),
    ),
)
