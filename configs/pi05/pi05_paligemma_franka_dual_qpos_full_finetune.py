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
# tools/compute_pi05_norm_stats.py --profile franka-qpos
# --action-horizon 50.
_PI05_FRANKA_QPOS_STATS = {
    'private': {
        'proprio': {
            'mean': [
                -0.03520631383961793, -0.24870669605728632,
                -0.1563442682015423, -2.136823639268365, 0.12521307080809424,
                1.899299916936001, -0.07185138369152991, 0.07367047184567013,
                0.04401634108912645, -0.29158947073606484, 0.15016879733862376,
                -2.136833263751491, -0.13353510830419327, 1.8429483002675644,
                1.0243280952004241, 0.0744173229819657
            ],
            'std': [
                0.07012418117399274, 0.6558160987386015, 0.20405740748883702,
                0.37501532824439365, 0.20127993393544147, 0.3374275150555436,
                0.9045100961215209, 0.013909948982729898, 0.07561737207089334,
                0.6553570939651002, 0.22834811014206682, 0.3751477686819759,
                0.23410997684691207, 0.3616363026751854, 0.3518435101316706,
                0.012897158068437983
            ],
            'min': [
                -0.33701425790786743, -1.1533950567245483, -0.7735654711723328,
                -2.578178882598877, -0.14115752279758453, 1.4155923128128052,
                -1.7631734609603882, 1.0506666512810625e-05,
                -0.030184879899024963, -1.0676981210708618,
                -0.10015326738357544, -2.5690605640411377, -0.8455336689949036,
                1.4720702171325684, 0.6783227920532227, 0.00012739333033096045
            ],
            'max': [
                0.08990167081356049, 1.2466726303100586, 0.18151268362998962,
                -0.8355942964553833, 0.8450835943222046, 2.8272337913513184,
                0.8956989049911499, 0.08088426291942596, 0.45842117071151733,
                1.2710249423980713, 0.84023517370224, -0.7715049982070923,
                0.1736655980348587, 2.963266372680664, 2.0796995162963867,
                0.08085142821073532
            ],
            'q01': [
                -0.28118944466114043, -0.9277725219726562, -0.6488878655433655,
                -2.4912457990646364, -0.06577091827988625, 1.5402279305458069,
                -1.6284139657020569, 0.04453447833657265,
                -0.0016815223777666688, -0.7893159985542297,
                -0.005849852226674557, -2.3886012172698976,
                -0.7559263372421264, 1.55976567029953, 0.7769999504089355,
                0.046659450978040695
            ],
            'q99': [
                0.040276070758700364, 1.1085592007637022, 0.0710255146026611,
                -1.0707644772529603, 0.6830312812328337, 2.7049813795089723,
                0.7739931607246399, 0.08088228851556778, 0.31405098259449,
                1.0929124736785887, 0.7159089958667749, -1.0355463767051698,
                0.08248163789510712, 2.7948518228530883, 1.9065979099273678,
                0.08085011690855026
            ],
            'count':
            635179
        },
        'action': {
            'mean': [
                -0.00011703465610524009, -0.0007745629211515547,
                -0.0009341182356114215, -0.001087550877694867,
                8.805371750111061e-06, 0.0010006687917139695,
                0.0008390552214850067, 0.07364908731321883,
                0.00043671659496946375, 0.0007287837938472708,
                0.0014506766411778909, -0.0003298808593848223,
                0.00046642300527724613, 0.003327422351308836,
                0.0007102200138259378, 0.0744180326312667
            ],
            'std': [
                0.03255101194959409, 0.20559184856489374, 0.08011703250808425,
                0.1345610373027473, 0.08605788879214331, 0.1217142182481323,
                0.28868217749988284, 0.013918604004554075, 0.03418588815134726,
                0.2327346625672978, 0.0851124334623602, 0.1512043254137981,
                0.11404712017164544, 0.13001783068862002, 0.1314278599805504,
                0.012892374814383234
            ],
            'min': [
                -0.24526117742061615, -1.4327025413513184, -0.5362956523895264,
                -1.2103910446166992, -0.7781462669372559, -0.7345454692840576,
                -1.4309709072113037, 1.0506666512810625e-05,
                -0.44921788573265076, -1.6099255084991455, -0.7155240774154663,
                -1.2777348756790161, -0.6213087439537048, -0.7701249122619629,
                -1.0929076671600342, 0.00012739333033096045
            ],
            'max': [
                0.3339442014694214, 0.957870602607727, 0.6903188824653625,
                0.8791354894638062, 0.5073481798171997, 0.8374570608139038,
                2.1733717918395996, 0.08088426291942596, 0.32327771186828613,
                1.1706854104995728, 0.5174639225006104, 0.9989891052246094,
                0.9230328798294067, 0.9811059236526489, 0.7197147607803345,
                0.08085142821073532
            ],
            'q01': [
                -0.11112188711762429, -0.8241139090061188, -0.2952014285326004,
                -0.5497994422912598, -0.39895492762327195,
                -0.40284574270248413, -0.8554342651367188, 0.04453447833657265,
                -0.15827589705586434, -0.9610398423671722,
                -0.38868317782878875, -0.6567311942577362,
                -0.33178246378898624, -0.3821221613883972, -0.6200929588079452,
                0.046659450978040695
            ],
            'q99': [
                0.1491532182693449, 0.5166074937582015, 0.331217319369316,
                0.4491676092147827, 0.24778359606862022, 0.4195587682723989,
                1.3099922156333879, 0.08088228851556778, 0.10056318432092648,
                0.6596419543027872, 0.2679590004682537, 0.5044398963451373,
                0.5484890997409808, 0.5441479146480548, 0.35347610831260656,
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
        dataset_statistics=_PI05_FRANKA_QPOS_STATS,
        name_mappings={'observation.state': ['proprio', 'action']},
        statistic_keys=['observation.state', 'timestamp'],
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    './datasets/RealRobot_Franka_dual_lerobot_v2/franka_dual_example'  # noqa: E501
                ],
                action_key='observation.state',
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_front',
                            'observation.images.cam_wrist_left',
                            'observation.images.cam_wrist_right'
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        }),
                    dict(
                        type='DeltaActions',
                        mask=[True] * 7 + [False] + [True] * 7 + [False]),
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
        type='openpi-warmup+cosine-decay',
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
        # 'The right arm picks up the shuttlecock bucket, hands it to the left arm, and places it on the plate.'  # noqa: E501
        '1':
        'Left arm picks up the green block and stacks it on the yellow block in the plate; right arm picks up the red block and stacks it on the green block.'  # noqa: E501
    },
    seed=7,
    action_mode='joint',
    active_arms=('left', 'right'),
    async_execution=False,
    execute_horizon=50,
    # Prepare joints: [left_arm_joints, right_arm_joints]
    # Each arm: [joint1..joint7, gripper_width]
    prepare_pose=None,  # None uses operator default prepare joints
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
        type='DenormalizeDeltaAction',
        norm_type='quantile',
        action_dim=16,
        delta_action_mask=[True] * 7 + [False] + [True] * 7 + [False],
    ),
    action_chunk=50,
    operator=dict(
        type='FrankaDualOperator',
        image_encoding='rgb8',
        command_mode='joint',
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
        joint_cmd_left_topic=(
            '/left_arm/ruckig_joint_impedance_controller/target_joint_state'
        ),  # noqa: E501
        joint_cmd_right_topic=(
            '/right_arm/ruckig_joint_impedance_controller/target_joint_state'
        ),  # noqa: E501
        gripper_left_topic='/left_arm/franka_gripper/move/goal',
        gripper_right_topic='/right_arm/franka_gripper/move/goal',
        gripper_control_mode='grasp'))
