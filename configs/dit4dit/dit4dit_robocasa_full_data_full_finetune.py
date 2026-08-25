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
"""Full-data DiT4DiT fine-tuning on the 24 RoboCasa GR1 tasks.

The model and optimization settings follow DiT4DiT's RoboCasa recipe. The
dataset/evaluation layout follows the PI0.5 full-data RoboCasa config and uses
the converted LeRobot V2.1 task directories available in FluxVLA.

The converted dataset stores the 29 GR1 joints in FluxVLA order. DiT4DiT's
source recipe uses the GR00T N1.5 order, sin/cos-encodes the state to 58
features and pads it to 64, pads actions to 32, and predicts 16 action steps.
``RobocasaGR1N15Bridge`` performs that conversion for both train and eval.
"""

import os

_ckpt_root = './checkpoints'
_cosmos_base_model = _ckpt_root + '/Cosmos-Predict2.5-2B'
_cosmos_revision = 'diffusers/base/post-trained'
_cosmos_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path=_cosmos_base_model + '/tokenizer',
    model_max_length=512,
)

_action_dim = 32
_ori_action_dim = 29
_state_dim = 64
_action_horizon = 16
_frame_window_size = 9
_image_frame_stride = 2
_image_size = 224

_ROBOCASA_STATISTIC_NAME = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = os.environ.get('ROBOCASA_DATA_ROOT',
                                     './datasets/robocasa_lerobot_V2.1')
_OFFICIAL_GR1_STATS_PATH = os.environ.get(
    'ROBOCASA_STATS_PATH', './datasets/robocasa_gr1_24tasks_first30ep/'
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

seed = 42
eval_seed = 7


def _robocasa_data_path(task_name):
    return f'{_ROBOCASA_DATA_ROOT}/{task_name}'


def _robocasa_task_env(task_name):
    return f'{_ROBOCASA_TASK_PREFIX}/{task_name}{_ROBOCASA_ENV_SUFFIX}'


model = dict(
    type='DiT4DiTVLA',
    # Train from the Cosmos base model rather than a released DiT4DiT policy
    # checkpoint. Training resume is handled by the runner.
    # The source trainer repeats each sample four times with independently
    # sampled action diffusion noise.
    repeated_diffusion_steps=4,
    vlm_backbone=dict(
        type='Cosmos25Backbone',
        base_model=_cosmos_base_model,
        revision=_cosmos_revision,
        torch_dtype='bf16',
        local_files_only=True,
        extract_layer=17,
        trainable=True,
        frozen_submodules=['text_encoder', 'vae'],
        split_future_frames=True,
        # video_delta_indices=0..16 with action_video_freq_ratio=2.
        num_frames_out=_frame_window_size,
        fixed_seed=42,
        num_inference_steps=1,
        conditional_frame_timestep=0.0001,
        future_loss_type='flow_matching',
        detach_hidden_states=True,
        flow_matching_time_distribution='logit_normal',
        flow_matching_high_sigma_ratio=0.05,
        flow_matching_high_sigma_min=0.98,
        fsdp_min_num_params=0,
    ),
    vla_head=dict(
        type='DiT4DiTActionHead',
        action_model_type='DiT-B',
        hidden_size=2560,
        add_pos_embed=True,
        max_seq_len=1024,
        action_dim=_action_dim,
        ori_action_dim=_ori_action_dim,
        state_dim=_state_dim,
        future_action_window_size=_action_horizon - 1,
        action_horizon=_action_horizon,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        num_inference_timesteps=4,
        diffusion_model_cfg=dict(
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_layers=16,
            output_dim=2560,
            positional_embeddings=None,
        ),
    ),
    freeze_vlm_backbone=False,
)

# The saved FluxVLA checkpoint contains the full model. Construct inference on
# meta tensors without reading pretrained model weights, then assign the eval
# checkpoint in the runner.
inference_model = dict(
    model,
    init_empty_weights=True,
    vlm_backbone=dict(
        model['vlm_backbone'],
        load_pretrained_weights=False,
    ),
)

_dit4dit_train_transforms = [
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
    ),
    # Converted data is left_arm,left_hand,right_arm,right_hand,waist. The
    # source DiT4DiT recipe is left_arm,right_arm,left_hand,right_hand,waist.
    # The bridge also applies per-group sin/cos state encoding and reorders
    # action statistics before min-max normalization.
    dict(type='RobocasaGR1N15Bridge', expand_state_axis=0),
    dict(
        type='ProcessCosmos25Prompt',
        tokenizer=_cosmos_tokenizer,
        input_key='task_description',
        remove_input_key=True,
    ),
    # DiT4DiT's RoboCasa loader converts to float CHW / 255 and then performs
    # a torch bilinear resize. It does not apply the PI0.5 crop/color jitter.
    dict(
        type='ResizeImages',
        height=_image_size,
        width=_image_size,
        backend='torch',
        scale_divisor=255.0,
        output_layout='nchw',
    ),
    dict(
        type='PrepareVideo',
        input_layout='tchw',
    ),
    dict(
        type='NormalizeStatesAndActions',
        state_key='proprio',
        action_key='action',
        state_dim=_state_dim,
        action_dim=_action_dim,
        state_norm_type='none',
        action_norm_type='min_max',
        norm_type='min_max',
        normalize_states=False,
        clip_norm=False,
        normalization_epsilon=0.0,
        preserve_input_dtype=True,
        valid_action_dim=_ori_action_dim,
        mark_all_action_steps_valid=True,
        output_dtype='float16',
    ),
]

_dit4dit_parquet_dataset = dict(
    type='ParquetDataset',
    transforms=_dit4dit_train_transforms,
    action_window_size=_action_horizon,
    action_key='action',
    use_delta=False,
    statistic_name=_ROBOCASA_STATISTIC_NAME,
    window_start_idx=0,
    frame_window_size=_frame_window_size,
    frame_sample_stride=_image_frame_stride,
)

train_dataloader = dict(
    # 32 GPUs * 8 samples/GPU = the source global batch of 256.
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedBalancedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_ROBOCASA_STATISTIC_NAME,
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        # Keep the full-data recipe's 24 tasks as equal-probability sources.
        datasets=[
            dict(
                _dit4dit_parquet_dataset,
                data_root_path=_robocasa_data_path(task_dir),
            ) for task_dir in _ROBOCASA_TASK_DIRS
        ],
        sampling_weights=[1.0] * len(_ROBOCASA_TASK_DIRS),
        shuffle=False,
        reshuffle_each_epoch=True,
        seed=seed,
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    max_steps=100000,
    grad_accumulation_steps=1,
    optimizer=dict(
        type='AdamW',
        lr=1e-5,
        weight_decay=1e-8,
        eps=1e-8,
        betas=(0.9, 0.95),
        paramwise_learning_rate={
            'vlm_backbone.transformer': 1e-5,
            'vla_head': 1e-4,
        },
    ),
    max_grad_norm=1.0,
    save_iter_interval=5000,
    max_keep_ckpts=10,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['info', 'stats'],
    ),
    tokenizer=_cosmos_tokenizer,
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_steps=5000,
        min_lr=5e-7,
    ),
    # Preserve the pre-rebase global ZeRO-2 / SHARD_GRAD_OP behavior. On the
    # latest main, ``shard-grad-op`` names the private hybrid variant.
    sharding_strategy='global-shard-grad-op',
    pre_fsdp_param_dtype='fp32',
    # The source flag is not wired into the released training path. Enabling
    # it here would add recomputation absent from the source recipe.
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    reduce_in_full_precision=False,
    change_key_name=False,
)

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='dit4dit',
    task_list=[_robocasa_task_env(task) for task in _ROBOCASA_TASK_DIRS],
    total_tasks=len(_ROBOCASA_TASK_DIRS),
    # The released DiT4DiT RoboCasa batch-eval script executes 12 of the 16
    # predicted actions before replanning.
    eval_chunk_size=12,
    max_episode_steps=720,
    num_trials_per_task=50,
    seed=eval_seed,
    unnorm_key=_ROBOCASA_STATISTIC_NAME,
    norm_stats_path=_OFFICIAL_GR1_STATS_PATH,
    action_order='n15',
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_ROBOCASA_STATISTIC_NAME,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                # Convert the source 256x256 frame to float CHW / 255, but
                # preserve its spatial size until the torch resize below.
                resize_size=256,
                normalize=True,
                value_range='unit',
            ),
            dict(type='RobocasaGR1N15Bridge', expand_state_axis=0),
            dict(
                type='ProcessCosmos25Prompt',
                tokenizer=_cosmos_tokenizer,
                input_key='task_description',
                remove_input_key=True,
            ),
            dict(
                type='ResizeImages',
                key='pixel_values',
                height=_image_size,
                width=_image_size,
                backend='torch',
                output_layout='nchw',
            ),
            dict(
                type='PrepareVideo',
                input_layout='tchw',
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_key='proprio',
                action_key=None,
                state_dim=_state_dim,
                state_norm_type='none',
                norm_type='min_max',
                normalize_states=False,
                output_dtype='float16',
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='min_max',
        action_dim=_ori_action_dim,
        clip_actions=False,
        # Reorder the stored FluxVLA statistics to the N1.5/DiT4DiT output
        # order before denormalization.
        stats_order='fluxvla',
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
