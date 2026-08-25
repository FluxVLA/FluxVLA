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

# Generated from the exact public dual-Franka training root with
# tools/compute_pi05_norm_stats.py --profile franka-eepose
# --action-horizon 50. Cartesian poses remain absolute.
_PI05_FRANKA_EEPOSE_STATS = {
    'private': {
        'proprio': {
            'mean': [
                0.42964787813401917, -0.09692170756619255, 0.3592090779752679,
                0.9005328079680991, 0.29016532458420324, 0.027143752858663273,
                -0.008862519601147052, 0.07367047184567013,
                0.41213064710322195, 0.10349228315432209, 0.3754232669699068,
                0.9988647455708408, 0.0007219255557120206,
                0.019584024154903835, 0.005790611560116755, 0.0744173229819657
            ],
            'std': [
                0.12605131723752344, 0.14224903852227466, 0.16046737097724478,
                0.1214594014326893, 0.2978196575799411, 0.016954992575343213,
                0.017117274491925315, 0.013909948982729898,
                0.12901287136562492, 0.1555301047618608, 0.15375201065404484,
                0.0020035154770163812, 0.026407173839674673,
                0.029480606255667396, 0.01676800868149931, 0.012897158068437983
            ],
            'min': [
                0.22812223434448242, -0.4910210371017456, 0.06128664314746857,
                0.6709647178649902, -0.09293060004711151, -0.0720209851861,
                -0.11751241236925125, 1.0506666512810625e-05, 0.25687175989151,
                -0.04022833704948425, 0.06256761401891708, 0.9833524823188782,
                -0.14233872294425964, -0.06986254453659058,
                -0.08542066067457199, 0.00012739333033096045
            ],
            'max': [
                0.6602121591567993, 0.1194400042295456, 0.5307644605636597,
                0.9999995827674866, 0.739533007144928, 0.1105477586388588,
                0.07910763472318649, 0.08088426291942596, 0.6568440794944763,
                0.5664681792259216, 0.5469597578048706, 0.9999999403953552,
                0.12353609502315521, 0.17121100425720215, 0.09826808422803879,
                0.08085142821073532
            ],
            'q01': [
                0.2760472798347473, -0.4548420667648315, 0.06611331924796104,
                0.7022522699832916, -0.00041697174368891864,
                -0.00985528726130724, -0.054041027203202245,
                0.04453447833657265, 0.3040494853258133, -0.00285750106908381,
                0.06809134155511856, 0.9905693280696869, -0.08659736528992652,
                -0.01195803001523018, -0.02154774736613035,
                0.046659450978040695
            ],
            'q99': [
                0.6307594072818755, 0.03464343227446078, 0.5131553411483765,
                0.9999475014209748, 0.7107134449481964, 0.07139649197459214,
                0.03959224335849284, 0.08088228851556778, 0.6320570492744446,
                0.49852439939975685, 0.4946299970149992, 0.9999958276748657,
                0.07661584362387648, 0.11076131671667092, 0.06110300414264201,
                0.08085011690855026
            ],
            'count':
            635179
        },
        'action': {
            'mean': [
                0.4295745346176471, -0.09727397332784957, 0.3590095384311181,
                0.9007244253525049, 0.2893689395969941, 0.027502622773746627,
                -0.009192194361722495, 0.07364908731321883,
                0.41282852807503806, 0.10424888319754266, 0.37529599553344906,
                0.9988290843676396, 0.001118609966739622, 0.020707953342020247,
                0.00598703275498847, 0.0744180326312667
            ],
            'std': [
                0.12610867663431177, 0.14205357269812138, 0.1603293530243628,
                0.12155532249153071, 0.29795137385577636, 0.016711188697899088,
                0.017036522524531708, 0.013918604004554075,
                0.12860156575101137, 0.15516098260182726, 0.1536992772489054,
                0.001999132883245944, 0.026661125882603033,
                0.029642553413954803, 0.016764349811848954,
                0.012892374814383234
            ],
            'min': [
                0.22812223434448242, -0.4910210371017456, 0.06128664314746857,
                0.6709647178649902, -0.09293060004711151, -0.0720209851861,
                -0.11751241236925125, 1.0506666512810625e-05, 0.25687175989151,
                -0.04022833704948425, 0.06256761401891708, 0.9833524823188782,
                -0.14233872294425964, -0.06986254453659058,
                -0.08542066067457199, 0.00012739333033096045
            ],
            'max': [
                0.6602121591567993, 0.1194400042295456, 0.5307644605636597,
                0.9999995827674866, 0.739533007144928, 0.1105477586388588,
                0.07910763472318649, 0.08088426291942596, 0.6568440794944763,
                0.5664681792259216, 0.5469597578048706, 0.9999999403953552,
                0.12353609502315521, 0.17121100425720215, 0.09826808422803879,
                0.08085142821073532
            ],
            'q01': [
                0.2757352292537689, -0.4548434615135193, 0.0661131739616394,
                0.7022518515586853, -0.0005095336236990988,
                -0.00979709718376398, -0.054043058305978775,
                0.04453447833657265, 0.30379921197891235,
                -0.0029757677111774683, 0.06809121370315552,
                0.9905688166618347, -0.08659835904836655,
                -0.011958152055740356, -0.021547963842749596,
                0.046659450978040695
            ],
            'q99': [
                0.6307608485221863, 0.03455594927072525, 0.513458788394928,
                0.9998650550842285, 0.7107138633728027, 0.07134172320365906,
                0.039514388889074326, 0.08088228851556778, 0.6320576071739197,
                0.4985400438308716, 0.5015515685081482, 0.999995768070221,
                0.07661841809749603, 0.11076349020004272, 0.06110329180955887,
                0.08085011690855026
            ],
            'count':
            31758950
        }
    }
}

model = dict(
    type='PI05FlowMatching',
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
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    # Match the OpenPI-aligned RoboCasa flow-matching objective.
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    max_action_dim=32,
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
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/pi05_base/model.safetensors',  # noqa: E501
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
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=16,
    # Supervise all padded model dimensions, as in OpenPI.
    loss_action_dim=32,
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        dataset_statistics=_PI05_FRANKA_EEPOSE_STATS,
        name_mappings={'observation.eepose': ['proprio', 'action']},
        statistic_keys=['observation.eepose', 'timestamp'],
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    './datasets/RealRobot_Franka_dual_lerobot_v2/franka_dual_example'  # noqa: E501
                ],
                action_key='observation.eepose',
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.eepose', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_front',
                            'observation.images.cam_wrist_left',
                            'observation.images.cam_wrist_right'
                        ],
                        name_mappings={
                            'observation.eepose': ['states'],
                            'actions': ['actions']
                        }),
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=32,
                        state_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='quantile',
                        output_dtype='float32'),
                    dict(type='PreparePromptWithState'),
                    dict[str, str | dict[str, str]](
                        type='ProcessPrompts',
                        max_len=200,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'checkpoints/pi05_base',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(
                        type='ResizeImagesWithPad',
                        height=224,
                        width=224,
                        backend='pil'),
                    dict(type='SimpleNormalizeImages'),
                    dict(type='OpenPIImageAugment', base_camera_indices=(0, )),
                ],
                action_window_size=50,
                window_start_idx=0,
                supervise_terminal_padding=True)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_steps=20_000,
    # 8 samples/GPU x 4 GPUs x 2 accumulation steps = global batch 64.
    grad_accumulation_steps=2,
    ema_decay=0.99,
    seed=42,
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
    # BF16 compute with FP32 sharded master parameters and reductions.
    sharding_strategy='global-shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'checkpoints/pi05_base',  # noqa: E501
        # special_tokens={'pad_token': '<PAD>'}
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        schedule_style='openpi',
        warmup_steps=1000,
        decay_steps=30000,
        min_lr=2.5e-6),
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    keep_params_fp32=True,
    change_key_name=False)

inference = dict(
    type='FrankaInferenceRunner',
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    task_descriptions={
        '1':
        'The right arm picks up the shuttlecock bucket, hands it to the left arm, and places it on the plate.'  # noqa: E501
    },
    seed=7,
    action_mode='cartesian',
    active_arms=('left', 'right'),
    async_execution=False,
    execute_horizon=20,
    # Prepare eepose: [left_arm_eepose, right_arm_eepose]
    # Each arm: [x, y, z, qx, qy, qz, qw, gripper_width]
    prepare_pose=None,  # None uses operator default prepare eepose
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_front', 'cam_wrist_left', 'cam_wrist_right'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                state_dim=32,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                output_dtype='float32'),
            dict(type='PreparePromptWithState'),
            dict[str, str | dict[str, str]](
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    'checkpoints/pi05_base',
                    # special_tokens={'pad_token': '<PAD>'}
                )),
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages'),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='quantile',
        action_dim=16,
    ),
    action_chunk=50,
    operator=dict(
        type='FrankaDualOperator',
        image_encoding='rgb8',
        command_mode='cartesian',
        img_left_topic='/camera_left_wrist/color/image_raw',
        img_right_topic='/camera_right_wrist/color/image_raw',
        img_front_topic='/camera_front/color/image_raw',
        puppet_arm_left_topic='/left_arm/joint_states',
        puppet_arm_right_topic='/right_arm/joint_states',
        puppet_franka_state_left_topic=(
            '/left_arm/franka_state_controller/franka_states'),
        puppet_franka_state_right_topic=(
            '/right_arm/franka_state_controller/franka_states'),
        sync_warning_enabled=True,
        cartesian_cmd_left_topic=(
            '/left_arm/cartesian_impedance_controller/equilibrium_pose'),
        cartesian_cmd_right_topic=(
            '/right_arm/cartesian_impedance_controller/equilibrium_pose'),
        gripper_left_topic='/left_arm/franka_gripper/move/goal',
        gripper_right_topic='/right_arm/franka_gripper/move/goal',
    ))
