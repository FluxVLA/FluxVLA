# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# FluxVLA training config that mirrors SonicStar's 78-dim SONIC action space
# (motion_token[64] + left_hand[7] + right_hand[7]) on top of the FluxVLA
# LlavaVLA (Eagle 3B backbone + FlowMatching DiT head) stack.
#
# IMPORTANT — run the data prep first:
#     python tools/build_sonic78_dataset.py \
#         --src ~/Erwin/Datasets/merged_dataset_001 \
#         --dst datasets/merged_dataset_001_sonic78
# This adds the single concatenated action column `action.sonic78` (78,) AND the
# concatenated proprio column `observation.sonic_state46` (46,), each with its
# per-episode stats, which FluxVLA's single-column action window and the
# SONIC-aligned proprio branch require.
#
# Action layout (must match inference-time split):
#     [ 0:64]  action.motion_token        (FSQ-quantized, normalized min/max)
#     [64:71]  teleop.left_hand_joints
#     [71:78]  teleop.right_hand_joints
#
# State layout (must match the SONIC deploy observation):
#     [ 0:43]  observation.state            (whole-body q)
#     [43:46]  observation.projected_gravity
#
# Padded internal action_dim = 96 (next multiple of 32 >= 78); the real action
# dim is ori_action_dim = 78 and the loss is computed only on the first 78 dims.
# Normalization is min/max for BOTH state and action, matching the SONIC deploy
# unnormalize and SonicStar-BlackOtters training.

# ----------------------------------------------------------------------------
# Action / state dimensions
# ----------------------------------------------------------------------------
ORI_ACTION_DIM = 78   # real SONIC action width (64 + 7 + 7)
ACTION_DIM = 128       # padded internal width fed to the DiT (>= 78)
STATE_DIM = 64        # observation.sonic_state46 (46) padded to 64
TRAJ_LEN = 10         # action horizon; MUST equal action_window_size below

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=STATE_DIM,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=TRAJ_LEN,
        action_dim=ACTION_DIM,
        ori_action_dim=ORI_ACTION_DIM),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model'),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=STATE_DIM,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=TRAJ_LEN,
        action_dim=ACTION_DIM,
        ori_action_dim=ORI_ACTION_DIM,
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
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        # Map raw stat columns -> the names NormalizeStatesAndActions expects.
        name_mappings={
            'observation.sonic_state46': ['proprio'],
            'action.sonic78': ['action'],
        },
        statistic_keys=[
            'observation.sonic_state46', 'timestamp', 'action.sonic78'
        ],
        statistic_name='sonic_motion78',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            'datasets/merged_dataset_001_sonic78',  # noqa: E501
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=0,
                    # 'actions'/'action_masks'/'info'/'stats' are the derived
                    # keys built by ParquetDataset.__getitem__ and must be
                    # listed here so they survive into the model inputs.
                    parquet_keys=[
                        'observation.sonic_state46', 'timestamp', 'actions',
                        'info', 'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.ego_view',
                    ],
                    name_mappings={'observation.sonic_state46': ['states']}),
                dict(type='ParquetPrompter'),
                dict(
                    type='ProcessPromptsWithImage',
                    max_len=600,
                    num_images=1,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        'fluxvla/models/third_party_models/eagle2_hg_model',
                    )),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375]],
                    stds=[[58.27148438, 57.02636719, 57.27539062]],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=ACTION_DIM,
                    state_dim=STATE_DIM,
                    state_key='proprio',
                    action_key='action',
                    norm_type='min_max')
            ],
            # The 78-dim action window is built from this single column.
            action_window_size=TRAJ_LEN,
            action_key='action.sonic78',
            use_delta=False,
            statistic_name='sonic_motion78',
            window_start_idx=0)))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=5,
    learning_rate=1.5e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks',
            'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=['jsonl'],
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.03,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)
