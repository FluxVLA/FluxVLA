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
"""GR00T-N1.5 Eagle 3B RoboDojo ARX-X5 finetuning config.

The dataset is the LeRobot v2.1 RoboDojo release
(``./datasets/RoboDojo_lerobot_v21_video``): 3,500 episodes across 35
tasks, three 480x640 cameras (``cam_high``, ``cam_left_wrist``,
``cam_right_wrist``), and 14-dimensional joint states/actions (7 per arm)
recorded at 25 Hz.

The data and optimization contract follows StarVLA's released QwenGR00T
RoboDojo recipe while retaining the GR00T-N1.5 Eagle backbone and action-head
parameterization: absolute 14D actions, a 50-step prediction horizon,
flat q01/q99 statistics, and full-model finetuning with separate backbone and
action-head learning rates.

Example:
    torchrun --nproc_per_node=8 scripts/train.py \
        --config \
        configs/gr00t/gr00t_eagle_3b_robodojo_full_finetune.py \
        --work-dir work_dirs/gr00t_eagle_3b_robodojo_full_finetune
"""

_ROBODOJO_DATA_ROOT = './datasets/RoboDojo_lerobot_v21_video'
# Exact frame-level statistics computed from all 1,859,602 RoboDojo samples,
# matching StarVLA's absolute-action q01/q99 normalization contract.
_GR00T_ROBODOJO_STATS = {'robodojo_arx_x5': {'proprio': {'mean': [-0.20028550922870636,
                                          0.9250829815864563,
                                          0.6878653764724731,
                                          -0.3399182856082916,
                                          0.06579338014125824,
                                          0.004018673673272133,
                                          0.7717168927192688,
                                          0.17282654345035553,
                                          0.8213191032409668,
                                          0.6190997958183289,
                                          -0.3601549565792084,
                                          -0.06264610588550568,
                                          0.0037324042059481144,
                                          0.7856565713882446],
                                 'std': [0.34361061453819275,
                                         0.8903892636299133,
                                         0.7319481372833252,
                                         0.6505074501037598,
                                         0.29847919940948486,
                                         0.5842824578285217,
                                         0.34185686707496643,
                                         0.3206428587436676,
                                         0.8718962669372559,
                                         0.7120621204376221,
                                         0.6014657616615295,
                                         0.2840825021266937,
                                         0.5366978049278259,
                                         0.3415590524673462],
                                 'max': [1.7537026405334473,
                                         3.0546703338623047,
                                         3.8525187969207764,
                                         2.0974574089050293,
                                         1.8103265762329102,
                                         3.1366536617279053,
                                         1.0,
                                         1.6732014417648315,
                                         3.3797459602355957,
                                         4.252756595611572,
                                         2.5287232398986816,
                                         3.1440579891204834,
                                         3.0381109714508057,
                                         1.0],
                                 'min': [-1.4863648414611816,
                                         -0.2586335241794586,
                                         -0.06850092858076096,
                                         -2.2642147541046143,
                                         -1.4876768589019775,
                                         -3.0168392658233643,
                                         -3.212450869184004e-17,
                                         -1.4924354553222656,
                                         -0.27809593081474304,
                                         -0.18774886429309845,
                                         -2.5354628562927246,
                                         -3.0663280487060547,
                                         -3.0486762523651123,
                                         -3.212450869184004e-17],
                                 'q01': [-1.051365569829941,
                                         -3.52302449637288e-14,
                                         1.6996893991849812e-16,
                                         -1.5984010362625123,
                                         -0.6003412520885467,
                                         -1.6678147149085998,
                                         0.0,
                                         -0.4500492787361145,
                                         -2.5859282299029243e-14,
                                         1.194697284288627e-16,
                                         -1.6398817586898804,
                                         -1.263367258310318,
                                         -1.760098042488098,
                                         0.0],
                                 'q99': [0.5431358617544174,
                                         2.495765209197998,
                                         2.492974226474762,
                                         1.323737324476242,
                                         1.2496635341644287,
                                         1.7391636216640465,
                                         1.0,
                                         1.0814965963363647,
                                         2.416655488014221,
                                         2.346989154815674,
                                         1.1415718960762022,
                                         0.5208872479200363,
                                         1.4889542925357817,
                                         1.0],
                                 'count': 1859602},
                     'action': {'mean': [-0.20032191276550293,
                                         0.925260603427887,
                                         0.6879700422286987,
                                         -0.33993399143218994,
                                         0.06580745428800583,
                                         0.003984309732913971,
                                         0.771653950214386,
                                         0.17281471192836761,
                                         0.8216421008110046,
                                         0.6193687319755554,
                                         -0.3603578805923462,
                                         -0.06270533800125122,
                                         0.0036061492282897234,
                                         0.7855709195137024],
                                'std': [0.34360724687576294,
                                        0.8902974724769592,
                                        0.7318998575210571,
                                        0.6505846977233887,
                                        0.29849204421043396,
                                        0.5843351483345032,
                                        0.34186822175979614,
                                        0.3206902742385864,
                                        0.8718543648719788,
                                        0.7120647430419922,
                                        0.6016045212745667,
                                        0.28415030241012573,
                                        0.5370385050773621,
                                        0.34119123220443726],
                                'max': [1.7537026405334473,
                                        3.0546703338623047,
                                        3.8525187969207764,
                                        2.0974574089050293,
                                        1.8103265762329102,
                                        3.1366536617279053,
                                        1.0,
                                        1.6732014417648315,
                                        3.3797459602355957,
                                        4.252756595611572,
                                        2.5287232398986816,
                                        3.1440579891204834,
                                        3.0381109714508057,
                                        1.0],
                                'min': [-1.4863648414611816,
                                        -0.2586335241794586,
                                        -0.06850092858076096,
                                        -2.2642147541046143,
                                        -1.4876768589019775,
                                        -3.0168392658233643,
                                        -3.212450869184004e-17,
                                        -1.4924354553222656,
                                        -0.27809593081474304,
                                        -0.18774886429309845,
                                        -2.5354628562927246,
                                        -3.0663280487060547,
                                        -3.0486762523651123,
                                        -3.212450869184004e-17],
                                'q01': [-1.051365569829941,
                                        -3.5348887174584814e-14,
                                        1.741881830823392e-16,
                                        -1.5984010362625123,
                                        -0.6003574305772781,
                                        -1.6678147149085998,
                                        0.0,
                                        -0.4509398394823074,
                                        -2.5859282299029243e-14,
                                        1.234041714549172e-16,
                                        -1.64055180311203,
                                        -1.2633746123313903,
                                        -1.7645285475254058,
                                        0.0],
                                'q99': [0.5431358617544174,
                                        2.495765209197998,
                                        2.492974226474762,
                                        1.3241519677639007,
                                        1.2496635341644287,
                                        1.7392305016517635,
                                        1.0,
                                        1.0814965963363647,
                                        2.4167031812667843,
                                        2.3470158338546754,
                                        1.141731116771698,
                                        0.5208872479200363,
                                        1.489715996980667,
                                        1.0],
                                'count': 1859602}}}
_ROBODOJO_GENERALIZATION_BASE_TASKS = (
    'stack_bowls', 'push_T', 'pack_objects_into_box', 'fold_clothes',
    'hang_mugs', 'sweep_blocks', 'pour_liquid_into_cup', 'make_toast',
    'arrange_largest_number', 'sort_nesting_dolls_by_size',
    'store_laptop_and_headphones', 'stack_blocks')
_ROBODOJO_EPISODE_OVERRIDES = {
    task_name: 25
    for base_task in _ROBODOJO_GENERALIZATION_BASE_TASKS
    for task_name in (base_task, f'{base_task}_random')
}

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        dtype='bf16',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=1100),
        tune_llm=True,
        tune_visual=True),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_inference_timesteps=4,
        num_steps=50,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        action_dim=32,
        ori_action_dim=14),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=1100)),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_steps=50,
        num_inference_timesteps=4,
        ori_action_dim=14,
        action_dim=32,
        max_input_seq_len=1100,
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

train_dataloader = dict(
    per_device_batch_size=16,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action', 'timestamp'],
        statistic_name='robodojo_arx_x5',
        dataset_statistics=_GR00T_ROBODOJO_STATS,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    _ROBODOJO_DATA_ROOT,
                ],
                action_key='action',
                use_delta=False,
                statistic_name='robodojo_arx_x5',
                window_start_idx=0,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        # CategorySpecificMLP needs a valid embodiment id
                        # (0-31). RoboDojo ARX-X5 uses 31, which is unused by
                        # the other configs; evaluation must use the same id.
                        embodiment_id=31,
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
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
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=1100,
                        num_images=3,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'fluxvla/models/third_party_models/eagle2_hg_model',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(type='SimpleNormalizeImages'),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='quantile',
                        normalize_states=True)
                ],
                action_window_size=50,
                supervise_terminal_padding=True)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    max_steps=130000,
    grad_accumulation_steps=1,
    optimizer=dict(
        lr=1e-5,
        type='AdamW',
        weight_decay=1e-8,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
        exclude_1d_from_weight_decay=False,
        paramwise_learning_rate={'vla_head': 1e-4}),
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path='fluxvla/models/third_party_models/eagle2_hg_model'),
    collator=dict(
        type='DictCollator',
        keys=[
            'states',  # (B, 64) raw 14D state, zero-padded
            'timestamp',  # (B,)
            'images',  # (B, N_views, C, H, W)
            'img_masks',  # (B, N_views)
            'lang_tokens',  # (B, max_len)
            'lang_masks',  # (B, max_len)
            'actions',  # (B, chunk_size, 32), normalized and padded
            'action_masks',  # (B, chunk_size)
            'embodiment_ids',  # (B,) RoboDojo ARX-X5 = 31
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
        warmup_steps=5000,
        # The released 130k checkpoint was taken from a 1M-step cosine run.
        decay_steps=1000000,
        min_lr=5e-7,
    ),
    save_epoch_interval=1,
    save_iter_interval=10000,
    max_keep_ckpts=5,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='global-shard-grad-op',
    change_key_name=False)

eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='groot',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    dataset=dict(
        type='RoboDojoEvalDataset',
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=[
                    'cam_high', 'cam_left_wrist', 'cam_right_wrist'
                ],
                embodiment_id=31),
            dict(
                type='StateFromInputs',
                stat_key='proprio',
                norm_type='quantile',
                state_dim=64),
            dict(
                type='ProcessPromptsWithImage',
                max_len=1100,
                num_images=3,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path='fluxvla/models/third_party_models/'
                    'eagle2_hg_model')),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224], [3, 224, 224]],
                means=[[127.5, 127.5, 127.5]] * 3,
                stds=[[127.5, 127.5, 127.5]] * 3,
            ),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='quantile',
        action_dim=14,
        statistic_name='robodojo_arx_x5',
    ),
)

themis = dict(
    transport=dict(
        host='127.0.0.1',
        port=5555,
        timeout_s=30.0,
        image_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        state_keys=['states'],
        unnorm_key='robodojo_arx_x5',
        image_encoding='rgb8',
        report_service_name='/fluxvla/report_evaluation',
    ),
    runner=dict(
        type='EvalRunner',
        environment=dict(
            type='RoboDojoEnvironment',
            task_name='all',
            env_cfg_type='arx_x5',
            robodojo_root='/root/projects/RoboDojo',
            device_id=1,
            action_mode='joint',
            headless=True,
            save_videos=False),
        model_client=dict(type='FluxVLAZMQModelClient'),
        evaluator=dict(type='SuccessRateEvaluator'),
        seed=0,
        episodes_per_task=50,
        episodes_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
        max_episode_steps=2000,
        execute_horizon=16,
        stop_on_success=True,
        parallel_workers=1,
        simulator_gpu_ids=None,
        work_dir='work_dirs/fluxthemis',
    ),
    ros_server=dict(
        dataset_section='eval',
        evaluation_reporting=dict(result_output_dir='work_dirs/fluxthemis'),
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
        denormalize_context=dict(task_suite_name='robodojo_arx_x5'),
    ),
)
