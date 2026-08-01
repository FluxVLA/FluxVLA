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
"""GR00T-N1.5 Eagle 3B RoboCasa GR1 finetuning config."""

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        dtype='bf16',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900),
        tune_llm=False,
        tune_visual=True),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_inference_timesteps=4,
        num_steps=16,
        noise_beta_alpha=1.0,
        noise_beta_beta=1.5,
        action_dim=32,
        ori_action_dim=29),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

_STAT = 'robocasa_gr1_24tasks_30ep'
_ROBOCASA_DATA_ROOT = './datasets/robocasa_gr1_24tasks_first30ep'
_OFFICIAL_GR1_STATS_PATH = (
    f'{_ROBOCASA_DATA_ROOT}/official_groot_gr1_dataset_statistics.json')

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

train_dataloader = dict(
    per_device_batch_size=16,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STAT,
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=[
                    f'{_ROBOCASA_DATA_ROOT}/{task_dir}'
                    for task_dir in _ROBOCASA_TASK_DIRS
                ],
                statistic_name=_STAT,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=24,
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=['observation.images.ego_view'],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        }),
                    dict(type='RobocasaGR1N15Bridge'),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=1,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=('fluxvla/models/third_party_models/'
                                        'eagle2_hg_model'))),
                    dict(type='RandomCropImages', scale=0.95),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='ColorJitterImages',
                        brightness=0.3,
                        contrast=0.4,
                        saturation=0.5,
                        hue=0.08),
                    dict(
                        type='NormalizeImages',
                        means=[[127.5, 127.5, 127.5]],
                        stds=[[127.5, 127.5, 127.5]]),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='min_max',
                        normalize_states=False)
                ],
                action_window_size=16)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    max_steps=30000,
    optimizer=dict(lr=6e-5, type='AdamW', weight_decay=1e-5),
    max_grad_norm=1.0,
    sampler=None,
    save_iter_interval=5000,
    save_epoch_interval=1,
    max_keep_ckpts=1,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path='fluxvla/models/third_party_models/eagle2_hg_model'),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks',
            'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.05,
    ),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='no-shard',
    change_key_name=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_steps=16,
        num_inference_timesteps=4,
        ori_action_dim=29,
        action_dim=32,
        max_input_seq_len=900,
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None)))

eval = dict(
    type='RobocasaEvalRunner',
    benchmark='robocasa',
    task_suite_name='robocasa',
    model_family='groot',
    task_list=[
        f'gr1_unified/{task_dir}_GR1ArmsAndWaistFourierHands_Env'
        for task_dir in _ROBOCASA_TASK_DIRS
    ],
    total_tasks=24,
    eval_chunk_size=16,
    max_episode_steps=720,
    num_trials_per_task=50,
    seed=7,
    unnorm_key=_STAT,
    action_order='n15',
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_STAT,
        transforms=[
            dict(
                type='ProcessRobocasaEvalInputs',
                img_key='video.ego_view_bg_crop_pad_res256_freq20',
                resize_size=224,
                center_crop_scale=0.95,
                normalize=False,
                embodiment_id=24),
            dict(type='RobocasaGR1N15Bridge'),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='min_max',
                normalize_states=False),
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=1,
                return_text=True,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=('fluxvla/models/third_party_models/'
                                'eagle2_hg_model'))),
            dict(
                type='NormalizeImages',
                means=[[127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5]]),
        ]),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='min_max',
        action_dim=29,
        clip_actions=False,
        stats_order='fluxvla'),
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
        unnorm_key=_STAT,
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
