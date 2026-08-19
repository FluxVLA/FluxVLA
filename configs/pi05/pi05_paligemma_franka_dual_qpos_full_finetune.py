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

# Generated from the exact training data with
# tools/compute_pi05_norm_stats.py --profile franka-qpos.
_PI05_FRANKA_QPOS_STATS = {
    'private': {
        'proprio': {
            'mean': [
                -0.006821800625934213, -0.40521847319122656,
                -0.05450514653308278, -2.1000402468863704,
                -0.44536325218481776, 1.7549236937415997, 0.850297009123846,
                0.05766711124051247, 0.032901275758265194, -0.3167395128284768,
                0.0532142311894464, -2.170835554290666, 0.483674350605563,
                1.9052721929363325, 0.6324705814186767, 0.05111560404453615
            ],
            'std': [
                0.016472626110189223, 0.4654394821384937, 0.1104501833451347,
                0.2855271774997365, 0.6943504984308186, 0.21541108245739898,
                0.20093335211578156, 0.03565488327787719, 0.020967230478024748,
                0.4000962585512831, 0.10134939782804978, 0.2021468115159534,
                0.7584693033417772, 0.28597755346694503, 0.28282574397347454,
                0.03763030112896345
            ],
            'min': [
                -0.0661570280790329, -0.8396243453025818, -0.3471769094467163,
                -2.4722084999084473, -1.7153528928756714, 1.526877999305725,
                0.39955440163612366, 2.6266666282026563e-06,
                -0.009997420944273472, -0.8233413100242615,
                -0.4480891823768616, -2.5085887908935547, -0.3984763026237488,
                1.5354458093643188, 0.030898187309503555, 0.0009101399919018149
            ],
            'max': [
                0.04373755306005478, 0.89531010389328, 0.37325969338417053,
                -1.129136562347412, 0.18208235502243042, 2.551578998565674,
                1.3214914798736572, 0.08086193352937698, 0.1031181737780571,
                0.9369364976882935, 0.34958934783935547, -1.1585062742233276,
                1.8249077796936035, 2.884308099746704, 1.2611782550811768,
                0.08082253485918045
            ],
            'q01': [
                -0.03914828851819038, -0.8123818564414979,
                -0.29229462444782256, -2.357604742050171, -1.6529088115692139,
                1.5588371849060059, 0.48917633056640625,
                0.00039071665378287435, 0.0001337525379494764,
                -0.7974990296363831, -0.2607133948802948, -2.4808950853347778,
                -0.23938447266817092, 1.5709584951400757, 0.08988710731267929,
                0.0015195267042145133
            ],
            'q99': [
                0.030357560664415358, 0.7884451413154607, 0.2536210554838183,
                -1.3048278450965878, 0.09813202053308487, 2.3545701169967654,
                1.2594540476799012, 0.0808599665760994, 0.07816524744033813,
                0.6832878422737122, 0.24506871283054354, -1.538973252773283,
                1.7701099276542667, 2.732522406578064, 1.081895608901978,
                0.08082187920808792
            ],
            'count':
            77327
        },
        'action': {
            'mean': [
                -0.0006126436530497384, 0.007257787345300351,
                -0.001291458275893944, 0.001307211707237011,
                -0.0003656603845959942, 0.006381494531233484,
                -0.003518696118210785, 0.05757749757156749,
                0.0007075968957615133, 0.004839441939297425,
                0.0007632812221833914, 0.0006795987849545278,
                -0.0006156190544475198, 0.006068304156348138,
                0.0009274401459332497, 0.051115605347820725
            ],
            'std': [
                0.010079786926998076, 0.21789657466878706, 0.07238494008014622,
                0.14969861949803245, 0.3085470604260948, 0.08421220329164891,
                0.09297307515003443, 0.03568904300006961, 0.008352045355818865,
                0.21249002125265667, 0.08491044124999082, 0.10659255240196093,
                0.33875751805625764, 0.14579253814657564, 0.1406414967376247,
                0.0376303021571727
            ],
            'min': [
                -0.08760914206504822, -1.4489758014678955, -0.588826596736908,
                -1.1771193742752075, -1.6043111085891724, -0.6667472124099731,
                -0.6760205626487732, 2.6266666282026563e-06,
                -0.05786112695932388, -1.321929931640625, -0.5144758224487305,
                -1.0040768384933472, -1.883864402770996, -0.9771838188171387,
                -1.0237770080566406, 0.0009101399919018149
            ],
            'max': [
                0.06690680980682373, 1.1033998727798462, 0.3738029897212982,
                0.8432270288467407, 1.7118467092514038, 0.5905137062072754,
                0.5358557105064392, 0.08086193352937698, 0.05790921300649643,
                1.286797046661377, 0.6433027982711792, 0.8383045196533203,
                1.8404945135116577, 0.8577947616577148, 0.7956088185310364,
                0.08082253485918045
            ],
            'q01': [
                -0.03508321955800056, -0.927380074262619, -0.3236503484845162,
                -0.706730649471283, -1.1979312765598298, -0.3072397756576538,
                -0.4018883168697357, 0.0003920299932360649,
                -0.02795792240649462, -0.7779699826240539, -0.2855138486623764,
                -0.3657342553138733, -1.4835465264320373, -0.47738827466964723,
                -0.638928741812706, 0.0015195267042145133
            ],
            'q99': [
                0.02903852557763444, 0.6866954135894772, 0.2221186743676661,
                0.3931978082656826, 1.3720208597183223, 0.2697093939781179,
                0.3257311290502548, 0.0808599665760994, 0.026975513361394386,
                0.7459093928337084, 0.35851511627435645, 0.3525113666057582,
                1.343698980808258, 0.5126444828510281, 0.5077816033363334,
                0.08082187920808792
            ],
            'count':
            3866350
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
                    './datasets/RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop'  # noqa: E501
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
                        norm_type='quantile'),
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
                norm_type='quantile'),
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
