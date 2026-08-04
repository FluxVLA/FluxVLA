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
"""Fine-tune the official PI0.5 base model on all RoboCasa demonstrations.

This StarVLA-inspired recipe starts from the converted official PI0.5 base
checkpoint, uses all 1,000 episodes from each of the 24 task roots, samples
task roots uniformly, and applies a lower learning rate to the pretrained
vision-language backbone than to the action expert. The historical filename
is retained so existing launch automation does not need to change.

Example for a 16-rank job:
    torchrun --nproc_per_node=16 scripts/train.py \
        --config \
        configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py \
        --work-dir work_dirs/pi05_robocasa_official_starvla_full24k
"""

# Make training ablations reproducible across model setup, flow noise, image
# augmentation, dataloader workers, and distributed ranks.
seed = 7

# The PI0.5 architecture matches the LIBERO and ALOHA variants. Its internal
# action dimension is 32; the 29 RoboCasa joints are padded with three zeros.
model = dict(
    type='PI05FlowMatching',
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
    # Match OpenPI exactly: t ~ Beta(1.5, 1.0), then t = 0.999*t + 0.001.
    # The legacy FluxVLA power-ratio sampler substantially undersampled the
    # high-noise region and can still be selected for old-run reproduction.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
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
    # Keep this recipe a genuine full fine-tune. The frozen-backbone variant
    # reduced the 24-task closed-loop score despite reaching a similar flow
    # loss, because the loss does not measure visual-language adaptation or
    # compounding rollout error.
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    freeze_projector=False,
    # Initialize from the converted official PI0.5 base model. This is a new
    # RoboCasa adaptation run, not a continuation of a previous task policy.
    pretrained_name_or_path='./checkpoints/pi05_base/model.safetensors',
    # Map the upstream OpenPI checkpoint names to FluxVLA module names.
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
    # OpenPI pads RoboCasa actions to 32D and supervises all model outputs.
    # Evaluation still discards the final three padding dimensions.
    loss_action_dim=32,
)

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_lerobot_V2.1'
_ROBOCASA_STATS_ROOT = './datasets/robocasa_gr1_24tasks_first30ep'
_OFFICIAL_GR1_STATS_PATH = (
    f'{_ROBOCASA_STATS_ROOT}/official_groot_gr1_dataset_statistics.json')
_ROBOCASA_TASK_PREFIX = 'gr1_unified'
_ROBOCASA_ENV_SUFFIX = '_GR1ArmsAndWaistFourierHands_Env'
# Keep the sheet's established 24x50 benchmark comparable. OpenPI itself
# evaluates an uncropped resized view; test that source-style variant with
# ``eval.dataset.transforms.0.center_crop_scale=None`` as a separate A/B.
_ROBOCASA_EVAL_CENTER_CROP_SCALE = 0.95

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


# The full dataset contains 24 tasks x 1,000 episodes. Reuse the fixed official
# q01/q99 statistics for both training normalization and eval denormalization.
train_dataloader = dict(
    # Expected production topology: 16 A800 ranks x 4 samples = global 64,
    # matching the public StarVLA RoboCasa launcher. At 100k updates this is
    # a 6.4M-sample raw-frame-equivalent budget; task balancing intentionally
    # repeats frames from shorter roots.
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedBalancedRepeatingDataset',
        seed=seed,
        # Generate a fresh distributed permutation after every pass. The
        # virtual task-balanced stream is deterministic for a given epoch.
        reshuffle_each_epoch=True,
        # Keep state and action statistics separate. Action statistics must
        # come from the action column rather than observation.state.
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        datasets=dict(
            type='ParquetDataset',
            # One multi-root ParquetDataset shares a single tokenizer and
            # transform pipeline. The outer wrapper recognizes these roots
            # as 24 sources and balances them without copying the datasets.
            data_root_path=[
                _robocasa_data_path(task_dir)
                for task_dir in _ROBOCASA_TASK_DIRS
            ],
            transforms=[
                # Decode the requested Parquet columns and video frames.
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=24,
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
                # Normalize the native 29D robot state before tokenization,
                # without GR00T reordering or sine/cosine expansion.
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,  # Zero-pad to the model action dimension.
                    state_dim=29,
                    state_key='proprio',
                    action_key='action',
                    norm_type='quantile'),
                # Match OpenPI's exact "Task: ..., State: ...;\nAction: "
                # prompt format.
                dict(
                    type='PreparePromptWithState',
                    max_state_dim=29,
                    lowercase_task_description=False,
                    add_action_prefix=True),
                # Tokenize the state-conditioned prompt.
                dict(
                    type='ProcessPrompts',
                    max_len=200,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path='checkpoints/pi05_base',
                    )),
                # Match the official OpenPI image policy: resize/crop,
                # rotation with black borders, and probabilistic HSV jitter.
                dict(
                    type='OpenPIImageAugment',
                    height=224,
                    width=224,
                    crop_scale=0.95,
                    rotation_degrees=5.0,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.1,
                    jitter_probability=0.5),
            ],
            action_window_size=16,
            action_key='action',
            use_delta=False,
            # Match LeRobot/OpenPI: repeated terminal actions remain valid
            # targets when a 16-step query crosses the episode boundary.
            supervise_terminal_padding=True,
            statistic_name=_ROBOCASA_STATISTIC_NAME,
            window_start_idx=0,
        )))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    # Override this to 2 on 8 GPUs or 8 on 2 GPUs to retain global batch 64.
    grad_accumulation_steps=1,
    # With global batch 64 this exposes about 6.4M task-balanced samples.
    # Select checkpoints by closed-loop score, not flow loss.
    max_steps=100000,
    # StarVLA protects the pretrained VLM with a lower LR while adapting its
    # action model more quickly. Use a conservative 5x ratio for the already
    # trained PI0.5 expert rather than StarVLA's from-scratch 10x ratio.
    optimizer=dict(
        lr=1e-5,
        type='AdamW',
        betas=(0.9, 0.95),
        weight_decay=0.0,
        paramwise_learning_rate={
            'llm_expert.': 5e-5,
            'action_in_proj.': 5e-5,
            'action_out_proj.': 5e-5,
            'time_mlp_in.': 5e-5,
            'time_mlp_out.': 5e-5,
        },
    ),
    max_grad_norm=1.0,
    # Keep enough periodic checkpoints for closed-loop model selection.
    save_epoch_interval=1,
    save_iter_interval=10000,
    max_keep_ckpts=10,
    # Use DDP-style replicated parameters with bf16 master weights to avoid
    # wrapping hundreds of small FSDP submodules.
    sharding_strategy='no-shard',
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
            'embodiment_ids',  # (B,)
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path='checkpoints/pi05_base',
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    # Follow StarVLA's five-percent warmup and cosine decay. Checkpoints at
    # 10k intervals cover the rising, peak, and decaying parts of the run.
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

# Evaluate all 24 RoboCasa tasks.
# Example:
#   conda activate fluxvla && cd /root/projects/fluxvla
#   bash scripts/eval_robocasa.sh \
#       --config \
#       configs/pi05/pi05_paligemma_robocasa_30_eps_full_finetune.py \
#       --ckpt-path checkpoints/pi05_paligemma_robocasa_30_eps/checkpoints/\
#       latest-checkpoint.safetensors
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
    # changing the positive 100k-step training recipe. Blend the overlapping
    # half with the previous chunk to avoid fresh flow noise causing contact-
    # breaking jumps at every replan boundary.
    eval_chunk_size=8,
    action_chunk_ensemble_weight=0.5,
    max_episode_steps=720,
    num_trials_per_task=50,  # 1,200 episodes across 24 tasks.
    # Reduced gates must reuse the formal 50-trial protocol's exact scenes:
    # task t, trial i always uses seed + 50 * t + i.
    episode_seed_stride=50,
    seed=7,  # Match the GR00T RoboCasa evaluation initial states.
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    action_order='fluxvla',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            # Retain the historical sheet protocol here. OpenPI's no-crop
            # evaluation must be reported as a separate preprocessing A/B.
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop_scale=_ROBOCASA_EVAL_CENTER_CROP_SCALE,
                normalize=True,
                value_range='tanh'),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=29,
                state_key='proprio',
                action_key='action',
                norm_type='quantile'),
            dict(
                type='PreparePromptWithState',
                max_state_dim=29,
                lowercase_task_description=False,
                add_action_prefix=True),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path='checkpoints/pi05_base')),
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
