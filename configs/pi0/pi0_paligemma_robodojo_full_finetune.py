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
"""Fine-tune PI0 on the full RoboDojo ARX-X5 benchmark suite.

The dataset is the LeRobot v2.1 RoboDojo release
(``./datasets/RoboDojo_lerobot_v21_video``): 3,500 episodes across 35
tasks, three 480x640 cameras (``cam_high``, ``cam_left_wrist``,
``cam_right_wrist``), and 14-dimensional joint states/actions (7 per arm)
recorded at 25 Hz.

Training parameters mirror the RoboDojo OpenPI recipe
(``pi0_base_aloha_full_sim_arx-x5_seed_0`` in
``XPolicyLab/policy/Pi_0/openpi/src/openpi/training/config.py``): a 50-step
action horizon, global batch 256, 60k optimizer updates, a 1k-step warmup
followed by cosine decay from 2.5e-5, and checkpoints kept every 5k steps.
On 16 GPUs use ``grad_accumulation_steps=2`` so that 8 x 16 x 2 = 256
matches the reference global batch.

The first six joints of each arm are trained as state-relative actions while
the two gripper dimensions remain absolute, matching OpenPI's ALOHA contract.
States remain absolute. States and transformed actions use mean/std statistics
computed offline over the exact 50-step training chunks.

Example:
    torchrun --nproc_per_node=8 scripts/train.py \
        --config configs/pi0/pi0_paligemma_robodojo_full_finetune.py \
        --work-dir work_dirs/pi0_paligemma_robodojo_full_finetune
"""

_PI0_ROBODOJO_DELTA_MASK = [True] * 6 + [False] + [True] * 6 + [False]
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

# Generated from all 3,500 RoboDojo episodes with
# tools/compute_pi05_norm_stats.py after applying
# _PI0_ROBODOJO_DELTA_MASK and repeating terminal actions to fill the
# supervised 50-step horizon. PI0 consumes the mean/std entries.
_PI0_ROBODOJO_STATS = {'robodojo_arx_x5': {'proprio': {'mean': [-0.20040483418845204,
                                          0.9260308650671458,
                                          0.6881893789087474,
                                          -0.3400484544283471,
                                          0.06579394032953159,
                                          0.004003331131509558,
                                          0.771446548553906,
                                          0.17299834959584776,
                                          0.8224941837716736,
                                          0.6196513047400859,
                                          -0.3604178625374315,
                                          -0.06267071413702419,
                                          0.0037534824507240434,
                                          0.7860054714056368],
                                 'std': [0.34305506136703257,
                                         0.8880398716756615,
                                         0.7302413867018247,
                                         0.6487680858145891,
                                         0.2989095916390101,
                                         0.584300843989697,
                                         0.34316664624786114,
                                         0.31989806135013626,
                                         0.8736479432260295,
                                         0.7135350385777451,
                                         0.6027223159771133,
                                         0.2836475966439172,
                                         0.5368540820848867,
                                         0.3406560080480477],
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
                                 'q01': [-1.051365613937378,
                                         -3.52302449637288e-14,
                                         1.6996893991849812e-16,
                                         -1.5984010696411133,
                                         -0.600341260433197,
                                         -1.6678147315979004,
                                         0.0,
                                         -0.4500492811203003,
                                         -2.5859282299029243e-14,
                                         1.194697223408134e-16,
                                         -1.6398817300796509,
                                         -1.2633672952651978,
                                         -1.7600980997085571,
                                         0.0],
                                 'q99': [0.5431358814239502,
                                         2.495765209197998,
                                         2.492974281311035,
                                         1.323737382888794,
                                         1.2496635913848877,
                                         1.7391636371612549,
                                         1.0,
                                         1.0814965963363647,
                                         2.4166555404663086,
                                         2.346989154815674,
                                         1.1415718793869019,
                                         0.5208872556686401,
                                         1.4889543056488037,
                                         1.0],
                                 'count': 1859602},
                     'action': {'mean': [0.002944688230520478,
                                         -0.02273978544478616,
                                         -0.01595988345759602,
                                         0.012653045052335679,
                                         -0.00036925004286021907,
                                         -0.002810816886079648,
                                         0.7733198894402199,
                                         -0.001665844293170471,
                                         -0.005889869578631844,
                                         -0.003142197456721388,
                                         0.0012835924178307308,
                                         -0.0006098368572494885,
                                         -0.0025841846669483358,
                                         0.7851101517491746],
                                'std': [0.3085495891486599,
                                        0.6060294868883956,
                                        0.5104044521775194,
                                        0.43551522745980215,
                                        0.21062307871176156,
                                        0.45044876330011924,
                                        0.34265666903134123,
                                        0.27541140464717384,
                                        0.5886515952314446,
                                        0.5021945001704038,
                                        0.4273689086823123,
                                        0.19945708096405648,
                                        0.43223737747826557,
                                        0.34143679451875986],
                                'min': [-2.3431708812713623,
                                        -2.8565127849578857,
                                        -2.9247982501983643,
                                        -3.2954680919647217,
                                        -1.9588444232940674,
                                        -4.495245933532715,
                                        -3.212450869184004e-17,
                                        -1.9679970741271973,
                                        -2.856095790863037,
                                        -2.9085304737091064,
                                        -3.380599021911621,
                                        -3.140778064727783,
                                        -4.215540885925293,
                                        -3.212450869184004e-17],
                                'max': [2.190984010696411,
                                        2.860210657119751,
                                        2.9098896980285645,
                                        3.081826686859131,
                                        2.3564412593841553,
                                        4.020153999328613,
                                        1.0,
                                        2.032360553741455,
                                        3.3797459602355957,
                                        4.252756595611572,
                                        3.3967480659484863,
                                        3.1407182216644287,
                                        3.4700708389282227,
                                        1.0],
                                'q01': [-0.9816640615463257,
                                        -2.056368589401245,
                                        -1.8049654960632324,
                                        -1.309919834136963,
                                        -0.7569671869277954,
                                        -1.5582654476165771,
                                        0.0,
                                        -0.8919917345046997,
                                        -2.026737689971924,
                                        -1.7403379678726196,
                                        -1.3211758136749268,
                                        -0.8450621366500854,
                                        -1.6245449781417847,
                                        0.0],
                                'q99': [0.9855517148971558,
                                        1.8684489727020264,
                                        1.543192982673645,
                                        1.3830254077911377,
                                        0.7971591353416443,
                                        1.651127576828003,
                                        1.0,
                                        0.9411922097206116,
                                        1.8169732093811035,
                                        1.4770119190216064,
                                        1.3545492887496948,
                                        0.704921305179596,
                                        1.5059595108032227,
                                        1.0],
                                'count': 92980100}}}

model = dict(
    type='PI0FlowMatching',
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    loss_action_dim=32,
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
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=50,
    state_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    action_time_mlp_in=dict(type='LinearProjector', in_dim=2048, out_dim=1024),
    action_time_mlp_out=dict(
        type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=None,
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
        use_adarms=False,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/pi0_base/model.safetensors',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'llm_expert.embed_tokens':
        'paligemma_with_expert.gemma_expert.lm_head',
        'action_time_mlp_in.projector': 'action_time_mlp_in',
        'action_time_mlp_out.projector': 'action_time_mlp_out',
        'state_proj.projector': 'state_proj',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    strict_mapping=True,
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=14,
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=0,
        reshuffle_each_epoch=True,
        # Keep state and action statistics separate. Action statistics are
        # computed after converting arm joints to relative actions.
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'action', 'timestamp'],
        statistic_name='robodojo_arx_x5',
        dataset_statistics=_PI0_ROBODOJO_STATS,
        datasets=[
            dict(
                type='ParquetDataset',
                supervise_terminal_padding=True,
                data_root_path=  # noqa: E251
                [
                    './datasets/RoboDojo_lerobot_v21_video',  # noqa: E501
                ],
                action_key='action',
                statistic_name='robodojo_arx_x5',
                window_start_idx=0,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
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
                        },
                        video_backend='pyav'),
                    # Match OpenPI: arm joints are relative to the current
                    # state while each gripper remains an absolute command.
                    dict(type='DeltaActions', mask=_PI0_ROBODOJO_DELTA_MASK),
                    # Mean/std-normalize native 14D states and relative
                    # actions, then pad both to the model's 32D widths.
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=32,
                        state_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std'),
                    dict(
                        type='ParquetPrompter',
                        use_conversation=False,
                        add_new_line=True),
                    dict(
                        type='ProcessPrompts',
                        max_len=48,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'checkpoints/pi05_base',
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    # OpenPI first performs PIL resize-with-pad, normalizes to
                    # [-1, 1], then augments only the base camera geometrically.
                    dict(
                        type='ResizeImagesWithPad',
                        height=224,
                        width=224,
                        backend='pil'),
                    dict(type='SimpleNormalizeImages'),
                    dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
                ],
                # 50-step action chunk, matching OpenPI's action_horizon=50.
                action_window_size=50,
                # Delta conversion is handled explicitly after decoding so it
                # uses the same raw current state as OpenPI.
                use_delta=False)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=None,
    # OpenPI reference: 60k optimizer updates with a global batch of 256.
    # On 16 GPUs, 8 x 16 x grad_accumulation_steps(2) = 256.
    max_steps=60000,
    grad_accumulation_steps=2,
    deterministic_algorithms=True,
    ema_decay=0.99,
    seed=0,
    optimizer=dict(
        type='AdamW',
        lr=2.5e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        weight_decay_all_params=True,
        foreach=False,
        fused=True,
    ),
    max_grad_norm=1.0,
    sharding_strategy='global-shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'lang_tokens',
            'lang_masks',
            'actions',
            'action_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'checkpoints/pi05_base',
        # special_tokens={'pad_token': '<PAD>'}
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1),
    lr_scheduler=dict(
        type='openpi-warmup+cosine-decay',
        warmup_steps=1000,
        decay_steps=30000,
        min_lr=2.5e-6,
    ),
    save_epoch_interval=1,
    save_iter_interval=5000,
    max_keep_ckpts=5,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)

eval = dict(
    report_kind='robodojo',
    task_suite_name='robodojo',
    model_family='pi0',
    num_trials_per_task=50,
    num_trials_per_task_overrides=_ROBODOJO_EPISODE_OVERRIDES,
    dataset=dict(
        type='RoboDojoEvalDataset',
        transforms=[
            dict(
                type='ProcessEvalInputs',
                img_keys=[
                    'cam_high', 'cam_left_wrist', 'cam_right_wrist'
                ]),
            dict(
                type='StateFromInputs',
                stat_key='proprio',
                norm_type='mean_std',
                state_dim=32),
            dict(
                type='ParquetPrompter',
                use_conversation=False,
                add_new_line=True),
            dict(
                type='ProcessPrompts',
                max_len=48,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path='checkpoints/pi05_base')),
            dict(
                type='TransformImage',
                image_resize_strategy='letterbox',
                input_sizes=[[3, 224, 224], [3, 224, 224], [3, 224, 224]],
                means=[[127.5, 127.5, 127.5]] * 3,
                stds=[[127.5, 127.5, 127.5]] * 3,
                letterbox_fill=[0, 0, 0],
                letterbox_pad_position='center',
            ),
        ]),
    denormalize_action=dict(
        type='DenormalizeDeltaAction',
        norm_type='mean_std',
        action_dim=14,
        delta_action_mask=_PI0_ROBODOJO_DELTA_MASK,
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
        execute_horizon=50,
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
