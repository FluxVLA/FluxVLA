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
# tools/compute_pi05_norm_stats.py --profile franka-eepose.
_PI05_FRANKA_EEPOSE_STATS = {
    'private': {
        'proprio': {
            'mean': [
                0.4297850236341168, -0.08895618923284448, 0.49928558973936177,
                0.9243202075382054, 0.0002207550289174728, 0.04329197650393954,
                0.19563486537977712, 0.05766711124051247, 0.46347494510356396,
                0.10086746857406358, 0.45572189996687806, 0.49130005489590206,
                0.000237820739117546, 0.012168133312696582,
                0.19352553373602857, 0.05111560404453615
            ],
            'std': [
                0.13811076639849035, 0.1321748444670963, 0.12581609868796692,
                0.11651828740368826, 0.04251399752000619, 0.03975932429383864,
                0.29752335367323357, 0.03565488327787719, 0.11463055318312505,
                0.12599910879882698, 0.11811582179057525, 0.7811647102295556,
                0.035866162707209756, 0.08459945577790502, 0.3199409963718324,
                0.03763030112896345
            ],
            'min': [
                0.3027248680591583, -0.38332295417785645, 0.13262200355529785,
                0.6982001066207886, -0.1375977247953415, -0.057386089116334915,
                -0.06531272828578949, 2.6266666282026563e-06,
                0.3064102530479431, -0.07807203382253647, 0.13519157469272614,
                -0.8597608804702759, -0.13035796582698822,
                -0.24709585309028625, -0.4998726546764374,
                0.0009101399919018149
            ],
            'max': [
                0.7639314532279968, 0.03266870230436325, 0.6907702684402466,
                1.0, 0.13553950190544128, 0.20811569690704346,
                0.7079764604568481, 0.08086193352937698, 0.8194141387939453,
                0.37946924567222595, 0.6170250177383423, 1.0,
                0.11493460088968277, 0.25644761323928833, 0.7295733690261841,
                0.08082253485918045
            ],
            'q01': [
                0.3037597620487213, -0.35402622520923616, 0.14794699043035509,
                0.7140382289886474, -0.09163575455546379, -0.0202855958789587,
                -0.029231580384075643, 0.00039071665378287435,
                0.3069322407245636, -0.05707312546670437, 0.14529650568962096,
                -0.7987435495853424, -0.0965876829624176, -0.1802465319633484,
                -0.3971003895998001, 0.0015195267042145133
            ],
            'q99': [
                0.7443822014331818, 0.008142047487199306, 0.6667602038383484,
                0.9999997615814209, 0.10866434335708625, 0.14994624316692357,
                0.6941786932945252, 0.0808599665760994, 0.7438275456428529,
                0.3428739887475968, 0.5934124124050141, 0.9999999403953552,
                0.07118354380130769, 0.177644230425358, 0.702909963130951,
                0.08082187920808792
            ],
            'count':
            77327
        },
        'action': {
            'mean': [
                0.43227380235410495, -0.0897681347007823, 0.4979206161179475,
                0.9242949640233978, 0.0011547293972972068, 0.04351417430936695,
                0.19550013691269977, 0.05757749757156749, 0.4655108574435572,
                0.1013003139451169, 0.454940246666036, 0.4912709892509422,
                0.0004927232978202265, 0.013134920977366548,
                0.19402383388275093, 0.051115605347820725
            ],
            'std': [
                0.13663790857896632, 0.13172570276395615, 0.12621534512707736,
                0.11650205322926567, 0.0429610310887676, 0.03963025778361833,
                0.29761511216988584, 0.03568904300006961, 0.1125203746672725,
                0.12569847444048504, 0.11802711565462688, 0.7811458073013892,
                0.03601455464321274, 0.08467403681679588, 0.3196549334250355,
                0.0376303021571727
            ],
            'min': [
                0.3027248680591583, -0.38332295417785645, 0.13262200355529785,
                0.6982001066207886, -0.1375977247953415, -0.057386089116334915,
                -0.06531272828578949, 2.6266666282026563e-06,
                0.3064102530479431, -0.07807203382253647, 0.13519157469272614,
                -0.8597608804702759, -0.13035796582698822,
                -0.24709585309028625, -0.4998726546764374,
                0.0009101399919018149
            ],
            'max': [
                0.7639314532279968, 0.03266870230436325, 0.6907702684402466,
                1.0, 0.13553950190544128, 0.20811569690704346,
                0.7079764604568481, 0.08086193352937698, 0.8194141387939453,
                0.37946924567222595, 0.6170250177383423, 1.0,
                0.11493460088968277, 0.25644761323928833, 0.7295733690261841,
                0.08082253485918045
            ],
            'q01': [
                0.3037613332271576, -0.35402998328208923, 0.1479463279247284,
                0.714038074016571, -0.09164456278085709, -0.0215742364525795,
                -0.02923443913459778, 0.0003920299932360649,
                0.3069392442703247, -0.0570814348757267, 0.1452961415052414,
                -0.7987663149833679, -0.0965888723731041, -0.1802748143672943,
                -0.39711296558380127, 0.0015195267042145133
            ],
            'q99': [
                0.7443833947181702, 0.008147988468408585, 0.6667618155479431,
                0.9999997019767761, 0.10866738855838776, 0.14994816482067108,
                0.6941827535629272, 0.0808599665760994, 0.7438315749168396,
                0.34287577867507935, 0.5934155583381653, 0.9999998807907104,
                0.07118412107229233, 0.1776532381772995, 0.7029126286506653,
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
        dataset_statistics=_PI05_FRANKA_EEPOSE_STATS,
        name_mappings={'observation.eepose': ['proprio', 'action']},
        statistic_keys=['observation.eepose', 'timestamp'],
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    './datasets/RealRobot_franka_dual_lerobotv2.1/20260519_dual_franka_teleop'  # noqa: E501
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
