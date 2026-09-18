# Copyright 2026 Limx Dynamics
"""Standalone asynchronous RTC inference config for Boris Candy PI0.5.

Run this config with a fine-tuned checkpoint and its sibling
``dataset_statistics.json`` through ``scripts/inference_real_robot.py``.
The RTC design follows Boris's Candy deployment, with its timing window sized
for 150-200 ms inference latency. Oli publishes the selected model actions
directly at 30 Hz without Teleop02's 50 Hz interpolation.
"""

_state_dim = 43
_action_dim = 52
_model_action_dim = 64
_action_horizon = 32
_statistic_name = 'private'

_task_prompts = {
    '1': ('pick up the white candy and place it in the left section of the '
          'snack tray with left arm'),
    '2': ('pick up the purple candy and place it in the right section of the '
          'snack tray with left arm'),
    '3': ('pick up the red candy and place it in the middle section of the '
          'snack tray with left arm'),
}

# Captured WBT standing pose in canonical 31-joint order, followed by twelve
# open-finger targets. The base anchor is intentionally held at its current
# value by OliOperator.gohome().
_prepare_pose = [
    -0.0376718,
    0.0743988,
    0.0287065,
    -0.00910288,
    -0.0216001,
    -0.0656817,
    -0.0658258,
    -0.101149,
    -0.178495,
    0.0896175,
    -0.0522503,
    0.112336,
    -0.0160254,
    -0.00427001,
    0.00303777,
    -0.067359,
    0.433261,
    0.0380359,
    0.355374,
    -0.471247,
    -1.27736,
    0.191936,
    -0.771603,
    0.118538,
    0.336419,
    -0.266369,
    0.264269,
    -1.5451,
    -0.0991593,
    -0.592179,
    -0.182942,
] + [0.0, 0.0, 100.0, 100.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

inference_model = dict(
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
        vocab_size=257152),
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
            vision_use_head=False)),
    projector=dict(type='LinearProjector', in_dim=1152, out_dim=2048),
    proj_width=1024,
    n_action_steps=_action_horizon,
    action_in_proj=dict(
        type='LinearProjector', in_dim=_model_action_dim, out_dim=1024),
    action_out_proj=dict(
        type='LinearProjector', in_dim=1024, out_dim=_model_action_dim),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_sampler='beta',
    time_beta_alpha=1.5,
    time_beta_beta=1.0,
    openpi_fp32_flow=True,
    max_action_dim=_model_action_dim,
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
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=_action_dim,
    loss_action_dim=_action_dim,
    zero_padded_action_dims=True,
    trim_action_prediction=True)

inference = dict(
    type='OliRTCInferenceRunner',
    seed=7,
    state_dim=_state_dim,
    action_chunk=_action_horizon,
    publish_rate=30,
    max_publish_step=10000,
    # At 30 Hz, seven prefix steps cover about 233 ms for the measured
    # 150-200 ms PI0.5 latency. Keep nine executable steps after activation
    # so the next request can start before the active chunk is exhausted.
    execute_horizon=16,
    async_remaining_actions_threshold=9,
    rtc_config=dict(enabled=True, method='prefix', prefix_len=7),
    interactive=True,
    default_prompt_id='1',
    # Keep one interactive selection alive long enough for overlapping RTC.
    default_execution_count=1000,
    prepare_pose=_prepare_pose,
    prepare_pose_duration_sec=4.0,
    prepare_pose_prompt_id='0',
    apply_jpeg_compression=True,
    keep_params_fp32=True,
    mixed_precision_dtype='bf16',
    camera_names=['head', 'left_wrist'],
    task_descriptions=_task_prompts,
    dataset=dict(
        type='PrivateInferenceDataset',
        statistic_name=_statistic_name,
        img_keys=['head', 'left_wrist'],
        transforms=[
            dict(
                type='NormalizeStatesAndActions',
                action_dim=None,
                state_dim=None,
                state_key='proprio',
                action_key='action',
                norm_type='quantile',
                discrete_state_dims=list(range(31, 43)),
                discrete_action_dims=list(range(40, 52)),
                discrete_norm_type='min_max',
                output_dtype='float32'),
            dict(type='PreparePromptWithState'),
            dict(
                type='ProcessPrompts',
                max_len=200,
                tokenizer=dict(type='PretrainedTokenizer')),
            dict(
                type='PadStatesAndActions',
                model_action_dim=_model_action_dim),
            dict(
                type='ResizeImagesWithPad',
                height=224,
                width=224,
                backend='pil'),
            dict(type='SimpleNormalizeImages')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        statistic_name=_statistic_name,
        action_dim=_action_dim,
        norm_type='quantile',
        discrete_action_dims=list(range(40, 52)),
        discrete_norm_type='min_max'),
    operator=dict(
        type='OliOperator',
        control_backend='mros',
        hand_mode='finger',
        head_rgb_topic='/head/color/image_raw/compressed',
        left_wrist_rgb_topic=('/left_wrist_camera/color/image_raw/compressed'),
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        finger_force_levels=(2.0, 2.0),
        teleop_wbt_topic='/teleop_cmd_WBT'))
