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
"""Cosmos3-Nano full-data fine-tuning and evaluation on RoboCasa GR1.

The data is produced by ``scripts/convert_robocasa_for_fluxvla.py``.  It has
one 256x256 ego camera, 29-D absolute joint-position actions in FluxVLA GR1
order, and q01/q99 statistics shared by training and closed-loop evaluation.

Usage (16 GPUs):
    torchrun --nproc-per-node=8 --nnodes=2 scripts/train.py \
        --config \
        configs/cosmos3/cosmos3nano_robocasa_full_data_full_finetune.py \
        --work-dir work_dirs/cosmos3nano_robocasa_full_data_full_finetune

Evaluation:
    MUJOCO_GL=egl torchrun --nproc-per-node=1 scripts/eval.py \
        --config \
        configs/cosmos3/cosmos3nano_robocasa_full_data_full_finetune.py \
        --ckpt-path <checkpoint.safetensors>
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

# RoboCasa GR1 uses left_arm + left_hand + right_arm + right_hand + waist.
# This 29-D action contract must stay in the native FluxVLA order because the
# simulator action split and the official statistics use the same order.
_ACTION_DIM = 29
_MAX_ACTION_DIM = 64
_MAX_STATE_DIM = 64
_ACTION_HORIZON = 16
_FRAME_WINDOW_SIZE = _ACTION_HORIZON + 1  # Wan2.2 temporal 4n + 1 convention.
_PREPEND_STATE_TO_ACTION = True
_ROBOCASA_EMBODIMENT_ID = 24  # Dedicated, unused DomainAwareLinear row.
_CONDITIONING_FPS = 20.0
_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 256
_VIDEO_HEIGHT = 256
_VIDEO_WIDTH = 256
_CFG_DROPOUT_RATE = 0.1

# This follows the DROID action-policy recipe's action loss scale, CFG prompt
# dropout, structured JSON prompt, and long LR horizon.  The per-device batch
# and base LR match the comparable 16-frame Cosmos3 LIBERO recipe: 16 GPUs x
# 16 samples x 8 accumulation = global batch 2048.
_BASE_LR = 5e-5
_ACTION_LR = _BASE_LR * 5.0
_PER_DEVICE_BATCH_SIZE = 16
_GRAD_ACCUMULATION_STEPS = 8
_MAX_STEPS = 10000
_SAVE_ITER_INTERVAL = 1000

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

_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_OFFICIAL_GR1_STATS_PATH = ('./datasets/robocasa_gr1_24tasks_first30ep/'
                            'official_groot_gr1_dataset_statistics.json')
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'

_ROBOCASA_TASK_DIRS = [
    'PnPBottleToCabinetClose',
    'PnPCanToDrawerClose',
    'PnPCupToDrawerClose',
    'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose',
    'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA',
    'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA',
    'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
    'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA',
    'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
    'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA',
    'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA',
    'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA',
    'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA',
    'PosttrainPnPNovelFromTrayToTieredshelfSplitA',
]


def _robocasa_data_path(task_name):
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name):
    return f'{_ROBOCASA_TASK_PREFIX}/{task_name}{_ROBOCASA_ENV_SUFFIX}'


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
        # DROID's action-policy recipe balances both generation targets at 10.
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
        encode_exact_durations=[_FRAME_WINDOW_SIZE],
    ),
    ori_action_dim=_ACTION_DIM,
    action_horizon=_ACTION_HORIZON,
    freeze_vlm_backbone=False,
    freeze_non_moe_vlm_backbone=True,
    enable_vision_loss=True,
)

# Evaluation checkpoints already contain the frozen Wan VAE weights.
inference_model = deepcopy(model)
inference_model['vision_vae']['pretrained_name_or_path'] = None

_TRANSFORMS = [
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
        video_keys=['observation.images.ego_view'],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        },
        embodiment_id=_ROBOCASA_EMBODIMENT_ID,
    ),
    # The DROID recipe uses a structured action prompt.  RoboCasa has one
    # first-person camera, rather than DROID's concatenated multi-view input.
    dict(
        type='ProcessCosmos3Prompt',
        tokenizer=_COSMOS3_NANO_TOKENIZER,
        max_len=512,
        cfg_dropout_rate=_CFG_DROPOUT_RATE,
        format_prompt_as_json=True,
        action_metadata=dict(
            append_viewpoint=False,
            viewpoint='ego_view',
            frame_window_size=_FRAME_WINDOW_SIZE,
            conditioning_fps=_CONDITIONING_FPS,
            video_height=_VIDEO_HEIGHT,
            video_width=_VIDEO_WIDTH,
        ),
    ),
    # Match PI0.5's RoboCasa augmentation, but retain Cosmos3's 256x256
    # canvas and [-1, 1] Wan2.2 VAE input range.
    dict(type='RandomCropImages', scale=0.95),
    dict(type='ResizeImages', height=_IMAGE_HEIGHT, width=_IMAGE_WIDTH),
    dict(
        type='ColorJitterImages',
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08,
    ),
    dict(type='SimpleNormalizeImages'),
    dict(
        type='NormalizeStatesAndActions',
        action_dim=_MAX_ACTION_DIM,
        state_dim=_MAX_STATE_DIM,
        # State and action are the same absolute 29-D GR1 joint-position
        # representation.  Use the action q01/q99 parameters for both so the
        # prepended state token shares the action-token value space.
        state_key='action',
        action_key='action',
        state_norm_type='quantile',
        action_norm_type='quantile',
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
        num_views=1,
        frame_window_size=_FRAME_WINDOW_SIZE,
        tile_direction='horizontal',
    ),
    dict(
        type='ResizeAndReflectPad',
        height=_VIDEO_HEIGHT,
        width=_VIDEO_WIDTH,
    ),
]

train_dataloader = dict(
    per_device_batch_size=_PER_DEVICE_BATCH_SIZE,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STATISTIC_NAME,
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[
                _robocasa_data_path(task_dir)
                for task_dir in _ROBOCASA_TASK_DIRS
            ],
            transforms=_TRANSFORMS,
            action_window_size=_ACTION_HORIZON,
            action_key='action',
            use_delta=False,
            statistic_name=_STATISTIC_NAME,
            window_start_idx=0,
            frame_window_size=_FRAME_WINDOW_SIZE,
            require_full_window=True,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=_MAX_STEPS,
    save_iter_interval=_SAVE_ITER_INTERVAL,
    max_keep_ckpts=2,
    optimizer=dict(
        type='AdamW',
        lr=_BASE_LR,
        weight_decay=0.05,
        betas=(0.9, 0.99),
        eps=1e-8,
        fused=True,
        exclude_1d_from_weight_decay=False,
        paramwise_learning_rate={
            'action_in_proj.': _ACTION_LR,
            'action_out_proj.': _ACTION_LR,
            'action_modality_embed': _ACTION_LR,
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
        # Cosmos3 DROID keeps a 100k-step cycle for a 10k-step policy run.
        cycle_length=100000,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

# Evaluate all 24 GR1 tabletop tasks.  Evaluation uses the same structured
# first-person prompt, 20 Hz metadata, q01/q99 action statistics, and native
# FluxVLA joint order as training.
eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='cosmos3',
    task_list=[
        _robocasa_task_env(task_dir) for task_dir in _ROBOCASA_TASK_DIRS
    ],
    total_tasks=len(_ROBOCASA_TASK_DIRS),
    eval_chunk_size=8,
    max_episode_steps=720,
    num_trials_per_task=50,
    seed=7,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    unnorm_key=_STATISTIC_NAME,
    action_order='fluxvla',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_STATISTIC_NAME,
        extra_tensor_keys=['conditioning_fps', 'prepend_state_to_action'],
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=_IMAGE_HEIGHT,
                center_crop_scale=0.95,
                normalize=True,
                value_range='tanh',
                embodiment_id=_ROBOCASA_EMBODIMENT_ID,
            ),
            dict(
                type='SetCosmos3ActionMetadata',
                conditioning_fps=_CONDITIONING_FPS,
                prepend_state_to_action=_PREPEND_STATE_TO_ACTION,
            ),
            # Apply the action q01/q99 statistics to the current 29-D joint
            # state and pad it to Cosmos3's 64-D action width.  ``state_key``
            # selects the statistics entry; the values being normalized are
            # still data['states'].
            dict(
                type='NormalizeStatesAndActions',
                state_dim=_MAX_STATE_DIM,
                state_key='action',
                action_key=None,
                state_norm_type='quantile',
            ),
            dict(
                type='ProcessCosmos3Prompt',
                tokenizer=_COSMOS3_NANO_TOKENIZER,
                max_len=512,
                cfg_dropout_rate=0.0,
                format_prompt_as_json=True,
                action_metadata=dict(
                    append_viewpoint=False,
                    viewpoint='ego_view',
                    frame_window_size=_FRAME_WINDOW_SIZE,
                    conditioning_fps=_CONDITIONING_FPS,
                    video_height=_VIDEO_HEIGHT,
                    video_width=_VIDEO_WIDTH,
                ),
                output_key='lang_tokens',
                output_attention_mask_key='lang_masks',
            ),
            dict(
                type='PrepareVideo',
                num_views=1,
                frame_window_size=1,
                tile_direction='horizontal',
            ),
            dict(
                type='ResizeAndReflectPad',
                height=_VIDEO_HEIGHT,
                width=_VIDEO_WIDTH,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='quantile',
        action_dim=_ACTION_DIM,
        clip_actions=False,
        stats_order='native',
    ),
)

# The manager-based RoboCasa evaluator uses the same ROS bridge as the PI0.5
# and GR00T configs.  It reads the ``eval`` section above for model-specific
# preprocessing and action denormalization.
themis = dict(
    transport=dict(
        service_name='/fluxvla/predict_action',
        report_service_name='/fluxvla/report_evaluation',
        timeout_s=30.0,
        image_keys=['video.ego_view_bg_crop_pad_res256_freq20'],
        state_keys=[
            'state.left_arm',
            'state.left_hand',
            'state.right_arm',
            'state.right_hand',
            'state.waist',
        ],
        unnorm_key=_STATISTIC_NAME,
        image_encoding='rgb8',
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='RoboCasaEnvironment',
            task_list=eval['task_list'],
            action_order=eval['action_order'],
            deterministic_env=True,
            prompt_key='annotation.human.coarse_action',
            render_key='video.ego_view_pad_res256_freq20',
        ),
        model_client=dict(type='FluxVLAROSModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=eval['seed'],
        episodes_per_task=eval['num_trials_per_task'],
        max_episode_steps=eval['max_episode_steps'],
        execute_horizon=eval['eval_chunk_size'],
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/fluxthemis',
    ),
    ros_server=dict(
        ros_version=1,
        dataset_section='eval',
        evaluation_reporting=dict(
            result_output_dir='work_dirs/fluxthemis',
            report_kind='robocasa',
        ),
        device='cuda:0',
        workers=dict(
            startup_timeout_s=900.0,
            request_timeout_s=120.0,
            lease_timeout_s=900.0,
        ),
        mixed_precision_dtype='bf16',
        enable_mixed_precision=True,
        model_outputs_environment_actions=False,
        forward_seed=False,
        denormalize_context={},
        denormalize_per_action=True,
    ),
)
