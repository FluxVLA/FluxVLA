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

# Generated from the exact public ALOHA training root with
# tools/compute_pi05_norm_stats.py --profile aloha
# --action-key observation.state --action-horizon 50.
_PI05_ALOHA_STATS = {
    'private': {
        'proprio': {
            'mean': [
                -0.2999958246772495, -1.7743772715889088, 0.7636950516458562,
                -0.2763671858455072, 0.06718097732100804, 1.0491808900940889,
                0.7470487813658261, 0.0896551348805329, -1.6812935632999924,
                0.6822415206173743, 0.36018804572351004, 0.2558468433346743,
                -1.212118090579343, 0.7625112997101525
            ],
            'std': [
                0.21205855575064714, 0.3049492539603072, 0.3148689639937349,
                1.2736942554221051, 0.7191890629774356, 1.2528553622388192,
                0.40373586600413786, 0.23411756658630614, 0.31351718069351414,
                0.27169390258543386, 1.1808965241973037, 0.6868840674219273,
                1.2047939687529736, 0.40123040385301756
            ],
            'min': [
                -0.9729216694831848, -2.5685243606567383, 0.22106781601905823,
                -1.7402657270431519, -1.1505190134048462, -1.4535212516784668,
                -0.8099625706672668, -0.45234036445617676, -2.4218204021453857,
                0.16739262640476227, -1.7486040592193604, -1.047128438949585,
                -2.1289007663726807, -0.8886476159095764
            ],
            'max': [
                0.2721612751483917, -0.9491803646087646, 2.156130790710449,
                1.7563666105270386, 1.1646660566329956, 2.133819818496704,
                0.9456529021263123, 0.9330098032951355, -0.7430969476699829,
                1.877776861190796, 1.746266484260559, 1.206304907798767,
                1.2431989908218384, 0.9456529021263123
            ],
            'q01': [
                -0.7785780429840088, -2.355672597885132, 0.2917160093784332,
                -1.6561682224273682, -1.0551700592041016, -0.9225311696529388,
                -0.34393277764320374, -0.29764696955680847,
                -2.2614402770996094, 0.24705937504768372, -1.6390557289123535,
                -0.8972495794296265, -2.1159048080444336, -0.40792906284332275
            ],
            'q99': [
                0.08437662571668625, -1.1562057733535767, 1.6183304429054255,
                1.7416439056396484, 0.9748753905296326, 2.1333489418029785,
                0.9456529021263123, 0.7384393811225891, -1.0099029541015625,
                1.4218482732772826, 1.5936803102493273, 1.0136935472488402,
                0.8564480543136597, 0.9456529021263123
            ],
            'count':
            315371
        },
        'action': {
            'mean': [
                -0.0021457228739425356, 0.009690166810823563,
                -0.0073363921569867905, -0.00414552019666049,
                -0.0014674119144848876, -0.0011147737170408498,
                0.9673767470481657, 0.007045450043772695, 0.014621964501084786,
                -0.009558105881873308, 0.004359588663568153,
                0.000383585472160792, 0.0019294249150465575, 0.9541503737291883
            ],
            'std': [
                0.11579498350417643, 0.18073317959926316, 0.17561051541668257,
                0.08748633738900026, 0.07644222030488367, 0.08076098749026431,
                0.48595533535824886, 0.12109834768325645, 0.1642728723741622,
                0.1507214979089093, 0.07238599432926658, 0.0773407225098651,
                0.06346165345143974, 0.4732598384800975
            ],
            'min': [
                -0.9224038124084473, -1.0240674018859863, -1.5682157278060913,
                -1.0850342512130737, -0.7343051433563232, -1.3961305618286133,
                -0.9647109508514404, -1.0880869626998901, -1.0157641172409058,
                -1.3611727952957153, -0.8863296508789062, -0.7219199538230896,
                -1.0825920104980469, -1.0327739715576172
            ],
            'max': [
                1.0139847993850708, 1.173422932624817, 1.0701196193695068,
                1.0682182312011719, 0.7441436052322388, 1.0262478590011597,
                1.2672858238220215, 0.9847661256790161, 1.1337553262710571,
                0.9932962656021118, 1.0170201063156128, 0.8107622265815735,
                0.6591390371322632, 1.283714771270752
            ],
            'q01': [
                -0.38715213537216187, -0.5396825075149536, -0.6396889090538025,
                -0.3228186368942261, -0.25464755296707153, -0.27629554271698,
                -0.41316598653793335, -0.45000284910202026,
                -0.46824920177459717, -0.5741518139839172, -0.2353370189666748,
                -0.2648696753382683, -0.22328317165374756, -0.5023519992828369
            ],
            'q99': [
                0.422057569026947, 0.6254023313522339, 0.535478413105011,
                0.27291131019592285, 0.26560235023498535, 0.2848081588745117,
                1.2578978538513184, 0.43442538380622864, 0.60923171043396,
                0.490028316974636, 0.2873201370239258, 0.279540091753006,
                0.22473108768463135, 1.283714771270752
            ],
            'count':
            15768550
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
    ori_action_dim=14,
    loss_action_dim=32,
)

inference_model = dict(
    type='PI05FlowMatchingRTCInference',
    num_view=3,
    triton_max_prompt_len=48,
    num_steps=10,
    llm_backbone=dict(
        type='ConditionGemmaInferenceModel',
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
        type='SigLIPViTBackboneInference',
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
    projector=dict(
        type='LinearProjectorInference',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=50,
    action_in_proj=dict(
        type='LinearProjectorInference', in_dim=32, out_dim=1024),
    action_out_proj=dict(
        type='LinearProjectorInference', in_dim=1024, out_dim=32),
    time_mlp_in=dict(
        type='LinearProjectorInference', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(
        type='LinearProjectorInference', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaInferenceModel',
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
    ori_action_dim=14,
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        dataset_statistics=_PI05_ALOHA_STATS,
        name_mappings={'observation.state': ['proprio', 'action']},
        statistic_keys=[
            'observation.state', 'observation.eepose', 'timestamp'
        ],
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                [
                    './datasets/RealRobot_AgileX_aloha_lerobot_v2/aloha_example',  # noqa: E501
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
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist'
                        ],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        }),
                    dict(
                        type='JointSignTransform',
                        signs=[1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]),
                    dict(
                        type='OpenPIAlohaGripperCoordinates',
                        gripper_input_range=(-0.01, 0.08)),
                    dict(
                        type='RelativeActions',
                        mask=([True] * 6 + [False] + [True] * 6 + [False])),
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
        fused=True),
    max_grad_norm=1.0,
    sharding_strategy='global-shard-grad-op',
    fsdp_wrap_policy='execution-block',
    reduce_in_full_precision=True,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks'
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
    type='AlohaRTCInferenceRunner',
    async_execution=True,
    execute_horizon=0,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=5,  # based on deployment inference frequency
    ),
    publish_rate=150,
    task_descriptions={
        '1': 'pick up the brown bird toy with left arm',
        '2': 'pick up the brown bird toy with right arm',
        '3': 'pick up the pruple knitted teddy bear toy with left arm',
        '4': 'pick up the purple knitted teddy bear toy with right arm',
        '5': 'pick up the white racing car toy with left arm',
        '6': 'pick up the white racing car toy with right arm',
        '7': 'pick up the pruple caterpillar toy with left arm',
        '8': 'pick up the pruple caterpillar toy with right arm',
        '9': 'place it in the brown flat cardboard box with left arm',
        '10': 'place it in the brown flat cardboard box with right arm',
    },
    seed=7,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='JointSignTransform',
                signs=[1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]),
            dict(
                type='OpenPIAlohaGripperCoordinates',
                gripper_input_range=(-0.01, 0.08)),
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
        type='OpenPIAlohaActionPostprocess',
        norm_stats=_PI05_ALOHA_STATS,
        action_dim=14,
        gripper_input_range=(-0.01, 0.08),
        gripper_output_range=(-0.01, 0.08),
    ),
    # Equivalent to threshold=0.05 in the standardized [0, 1] space.
    gripper_threshold=-0.0055,
    gripper_closed_value=-0.01,
    action_chunk=50,
    operator=dict(
        type='AlohaOperator',
        image_encoding='rgb8',
        img_front_topic='/camera_h/color/image_raw',
        img_left_topic='/camera_l/color/image_raw',
        img_right_topic='/camera_r/color/image_raw',
        img_front_depth_topic='/camera_h/depth/image_raw',
        img_left_depth_topic='/camera_l/depth/image_raw',
        img_right_depth_topic='/camera_r/depth/image_raw',
        puppet_arm_left_cmd_topic='/master/joint_left',
        puppet_arm_right_cmd_topic='/master/joint_right',
        puppet_arm_left_topic='/puppet/joint_left',
        puppet_arm_right_topic='/puppet/joint_right',
        robot_base_topic='/odom_raw',
        robot_base_cmd_topic='/cmd_vel',
    ))
