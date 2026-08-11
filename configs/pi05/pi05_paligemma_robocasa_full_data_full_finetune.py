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
"""Low-change, BF16 PI0.5 RoboCasa score-target recipe.

This is the single production recipe selected after auditing the experiment
sheet, RLinf, OpenPI, and StarVLA. It deliberately starts from the official
PI0.5 base checkpoint instead of continuing the 31.58% RoboCasa checkpoint.

The public StarVLA 43.9% result is from QwenPI_v2, not OpenPI PI0.5, so 40% is
a target rather than a reproduced guarantee. Only its uniform 24-task mixture
and larger sample budget are transferred. The optimizer schedule follows the
RLinf/OpenPI values. Global SHARD_GRAD_OP uses BF16 forward/backward compute
with globally sharded FP32 master parameters, reductions, and buffers.

The converted dataset uses a single ego-view camera, 29-dimensional joint
states and absolute joint-position actions, q01/q99 quantile normalization,
and a 16-step action horizon. Set ``ROBOCASA_DATA_ROOT`` when the converted
LeRobot dataset is not in one of the checked default locations.

Expected topology: 32 RTX PRO 5000 72GB GPUs, for example 4 nodes x 8 GPUs.
Per-device batch 8 without accumulation gives an effective global batch 256.

Example for four 8-GPU nodes sharing MASTER_ADDR and MASTER_PORT:
    torchrun --nnodes=4 --nproc_per_node=8 \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} scripts/train.py \
        --config \
        configs/pi05/\
pi05_paligemma_robocasa_full_data_full_finetune.py \
        --work-dir \
        work_dirs/pi05_paligemma_robocasa_full_data_full_finetune
"""

import os

train_seed = 42
eval_seed = 7

_LOCAL_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_SHARED_ROBOCASA_DATA_ROOT = (
    '/mnt/data/cpfs/mnt/data/yiming/fluxvla/upload_staging/'
    'robocasa_lerobot_V2.1')
_LOCAL_ROBOCASA_DATA_READY = os.path.isdir(
    f'{_LOCAL_ROBOCASA_DATA_ROOT}/PnPBottleToCabinetClose/videos')
_DEFAULT_ROBOCASA_DATA_ROOT = (
    _LOCAL_ROBOCASA_DATA_ROOT
    if _LOCAL_ROBOCASA_DATA_READY else _SHARED_ROBOCASA_DATA_ROOT)
_PI05_CHECKPOINT = os.environ.get('PI05_CHECKPOINT',
                                  './checkpoints/pi05_base/model.safetensors')
_PI05_TOKENIZER = os.environ.get('PI05_TOKENIZER', './checkpoints/pi05_base')

# The PI0.5 architecture matches the LIBERO and ALOHA variants. Its internal
# action dimension is 32; the 29 RoboCasa joints are padded with three zeros.
model = dict(
    type='PI05FlowMatching',
    # Match OpenPI's flow-matching objective and supervise all padded action
    # dimensions.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    loss_action_dim=32,
    openpi_fp32_flow=True,
    # PaliGemma backbone for image and language tokens.
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    # SigLIP vision encoder.
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        openpi_stem_fp32=True,
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    # Vision-to-LLM projection.
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    # A 16-step chunk covers roughly 0.8 seconds at 20 Hz.
    proj_width=1024,
    n_action_steps=16,
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    # Gemma expert conditioned on state, action, and diffusion time through
    # adaptive RMS normalization.
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    # PI0.5 injects the normalized 29D proprio state through discrete prompt
    # tokens, so the language backbone remains trainable during adaptation.
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    # Initialize from the general PI0.5 base model rather than LIBERO weights.
    pretrained_name_or_path=_PI05_CHECKPOINT,
    # Map upstream PI0.5 checkpoint keys to FluxVLA parameter names.
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
        'llm_expert.embed_tokens':
        'paligemma_with_expert.gemma_expert.lm_head',
    },
    strict_mapping=True,
    # Convert the large transformer modules to bf16 to reduce memory use.
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=29,
)

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = os.environ.get('ROBOCASA_DATA_ROOT',
                                     _DEFAULT_ROBOCASA_DATA_ROOT)
_OFFICIAL_GR1_STATS_PATH = os.environ.get(
    'ROBOCASA_STATS_PATH', './datasets/robocasa_gr1_24tasks_first30ep/'
    'official_groot_gr1_dataset_statistics.json')
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'


def _robocasa_data_path(task_name):
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name):
    return f'{_ROBOCASA_TASK_PREFIX}/{task_name}{_ROBOCASA_ENV_SUFFIX}'


# The full dataset contains about 1,000 episodes for each of 24 tasks (6 seen
# and 18 novel), one 256x256 ego-view camera, 29-dimensional joint states and
# absolute actions, and fixed q01/q99 quantile statistics shared with eval.
train_dataloader = dict(
    # 8 samples/GPU x 32 GPUs x 1 accumulation step = global batch 256.
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        # Sample all 24 tasks uniformly, independently of episode count.
        type='DistributedBalancedRepeatingDataset',
        seed=train_seed,
        reshuffle_each_epoch=True,
        # Keep state and action statistics separate. Action statistics must
        # come from the action column rather than observation.state.
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        # PI0.5 upstream uses q01/q99 quantile statistics. Reuse one fixed
        # robot/task statistics asset for both full-data training and eval.
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        datasets=dict(
            type='ParquetDataset',
            supervise_terminal_padding=True,
            # Converted task directories produced by
            # convert_robocasa_for_fluxvla.py.
            data_root_path=[
                _robocasa_data_path('PnPBottleToCabinetClose'),
                _robocasa_data_path('PnPCanToDrawerClose'),
                _robocasa_data_path('PnPCupToDrawerClose'),
                _robocasa_data_path('PnPMilkToMicrowaveClose'),
                _robocasa_data_path('PnPPotatoToMicrowaveClose'),
                _robocasa_data_path('PnPWineToCabinetClose'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToBasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToPanSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToPotSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromCuttingboard'
                                    'ToTieredbasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToBasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToBowlSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlacemat'
                                    'ToTieredshelfSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToBowlSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlate'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToPanSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromPlateToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToCardboardboxSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTrayToPlateSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTrayToPotSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToTieredbasketSplitA'),
                _robocasa_data_path('PosttrainPnPNovelFromTray'
                                    'ToTieredshelfSplitA'),
            ],
            transforms=[
                # Decode the requested Parquet columns and video frames.
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',  # 29D joint angles
                        'timestamp',  # Seconds
                        'actions',  # 29D target joint positions
                        'info',  # Dataset metadata
                        'stats',  # Normalization statistics
                        'action_masks',  # Valid-action masks
                    ],
                    # RoboCasa uses a single ego-view camera.
                    video_keys=[
                        'observation.images.ego_view',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    }),
                # Preserve native state ordering and tokenize the normalized
                # 29D state, matching OpenPI.
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,  # Zero-pad to the model action dimension.
                    state_dim=29,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile',
                    output_dtype='float32'),
                # Build the OpenPI-compatible state-conditioned prompt.
                dict(
                    type='PreparePromptWithState',
                    max_state_dim=29,
                    lowercase_task_description=False,
                    add_action_prefix=True),
                # Tokenize the prompt.
                dict(
                    type='ProcessPrompts',
                    max_len=200,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_PI05_TOKENIZER,
                    )),
                # Resize to 224 and apply the crop/color augmentations used by
                # the RoboCasa training recipe.
                dict(type='RandomCropImages', scale=0.95),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='ColorJitterImages',
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08),
                # Match OpenPI PI0.5 image normalization: pixel / 255 * 2 - 1.
                dict(type='SimpleNormalizeImages'),
            ],
            action_window_size=16,
            action_key='action',
            use_delta=False,
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
        )))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    # 100k global-256 updates expose 25.6M samples.
    max_steps=100000,
    grad_accumulation_steps=1,
    ema_decay=0.99,
    seed=train_seed,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        # Avoid the model-sized peak allocation from AdamW foreach state.
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    # Keep enough periodic checkpoints for closed-loop model selection.
    save_epoch_interval=1,
    save_iter_interval=10000,
    max_keep_ckpts=10,
    # 72GB cards fit SHARD_GRAD_OP at batch 8 and avoid FULL_SHARD's extra
    # backward all-gathers across nodes.
    sharding_strategy='global-shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',  # (B, 29) quantile-normalized joint state
            'observation.eepose',  # Optional; DictCollator skips missing keys.
            'timestamp',  # (B,)
            'images',  # (B, N_views, C, H, W)
            'img_masks',  # (B, N_views)
            'lang_tokens',  # (B, max_len)
            'lang_masks',  # (B, max_len)
            'actions',  # (B, chunk_size, 32), normalized and padded
            'action_masks',  # (B, chunk_size)
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=_PI05_TOKENIZER,
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1),
    lr_scheduler=dict(
        type='openpi-warmup+cosine-decay',
        warmup_steps=5000,
        decay_steps=100000,
        min_lr=2.5e-6,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)

# Evaluate all 24 RoboCasa tasks.
# Example:
#   conda activate fluxvla && cd /root/projects/FluxVLA
#   CONFIG_DIR=configs/pi05
#   CONFIG=$CONFIG_DIR/pi05_paligemma_robocasa_full_data_full_finetune.py
#   CKPT=work_dirs/pi05_paligemma_robocasa_full_data_full_finetune/\
# checkpoints/latest-checkpoint.safetensors
#   NUM_GPUS=8 bash scripts/eval_robocasa_manager.sh "$CONFIG" "$CKPT"
#
# Optional override:
#   --cfg-options eval.num_trials_per_task=50 eval.seed=7
#
# unnorm_key must match the training statistic_name.
eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='pi0',
    task_list=[
        _robocasa_task_env('PnPBottleToCabinetClose'),
        _robocasa_task_env('PnPCanToDrawerClose'),
        _robocasa_task_env('PnPCupToDrawerClose'),
        _robocasa_task_env('PnPMilkToMicrowaveClose'),
        _robocasa_task_env('PnPPotatoToMicrowaveClose'),
        _robocasa_task_env('PnPWineToCabinetClose'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToBasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToPanSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToPotSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromCuttingboard'
                           'ToTieredbasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToBasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToBowlSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlacemat'
                           'ToTieredshelfSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToBowlSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlate'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToPanSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromPlateToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToCardboardboxSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTrayToPlateSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTrayToPotSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToTieredbasketSplitA'),
        _robocasa_task_env('PosttrainPnPNovelFromTray'
                           'ToTieredshelfSplitA'),
    ],
    total_tasks=24,
    # Keep the 16-step prediction horizon, but replan halfway through it.
    # At 20 Hz this reduces open-loop execution from 0.8 s to 0.4 s without
    # changing the positive 100k-step training recipe.
    eval_chunk_size=8,
    max_episode_steps=720,
    num_trials_per_task=50,  # 1,200 episodes across 24 tasks.
    episode_seed_stride=50,
    seed=eval_seed,  # Match the GR00T RoboCasa evaluation initial states.
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    action_order='fluxvla',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            # Evaluation preprocessing must match training: the 0.95 center
            # crop mirrors RandomCropImages, tanh maps pixels to [-1, 1], and
            # the bg_crop ego-view key matches the converted training camera.
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop_scale=0.95,
                normalize=True,
                value_range='tanh'),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=29,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(
                type='PreparePromptWithState',
                max_state_dim=29,
                lowercase_task_description=False,
                add_action_prefix=True),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer', model_path=_PI05_TOKENIZER)),
        ]),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='quantile',
        action_dim=29,
        clip_actions=False,
        stats_order='native',
    ),
)

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
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
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
        # Preserve the old inherited config's base-time value. The formal
        # RobocasaEvalRunner protocol above still evaluates 50 trials/task.
        episodes_per_task=20,
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
