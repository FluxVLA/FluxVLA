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

_ckpt_root = './checkpoints'
_cosmos_base_model = _ckpt_root + '/Cosmos-Predict2.5-2B'
_cosmos_revision = 'diffusers/base/post-trained'
_cosmos_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path=_cosmos_base_model + '/tokenizer',
    model_max_length=512,
)
_official_dataset_statistics = dict(
    franka=dict(
        action=dict(
            mean=[
                0.07237596483901143,
                0.08987006871029735,
                -0.10144743137061596,
                -0.00045383188989944756,
                0.006273590726777911,
                -0.003878799732774496,
                0.524486355483532,
            ],
            std=[
                0.3498823308902479,
                0.37794140366375184,
                0.460084266976933,
                0.0403885784928603,
                0.06616144248501059,
                0.07763074391911857,
                0.4994683356809767,
            ],
            max=[
                0.9375,
                0.9375,
                0.9375,
                0.3557142913341522,
                0.375,
                0.375,
                1.0,
            ],
            min=[
                -0.9375,
                -0.9375,
                -0.9375,
                -0.2582142949104309,
                -0.375,
                -0.3675000071525574,
                0.0,
            ],
            q01=[
                -0.8785714507102966,
                -0.8758928775787354,
                -0.9375,
                -0.1510714292526245,
                -0.20678570866584778,
                -0.2742857038974762,
                0.0,
            ],
            q99=[
                0.9375,
                0.9107142686843872,
                0.9375,
                0.20357142388820648,
                0.26357144117355347,
                0.375,
                1.0,
            ],
            mask=[True, True, True, True, True, True, False],
        ),
        state=dict(
            mean=[
                -0.04889854742214084,
                0.03689368185587227,
                0.7890402488410473,
                2.9771945476531982,
                -0.1417286954820156,
                -0.11769362539052963,
                0.026436020154505968,
                -0.02665513101965189,
            ],
            std=[
                0.10639013941746686,
                0.15115733130675715,
                0.38406895599530033,
                0.3530238395244304,
                0.8227341427331599,
                0.32357567121520087,
                0.014583991652936385,
                0.014467005007200339,
            ],
            max=[
                0.21031762659549713,
                0.39128610491752625,
                1.3660105466842651,
                3.6714255809783936,
                3.560650587081909,
                1.386339545249939,
                0.04233968257904053,
                0.0013633022317662835,
            ],
            min=[
                -0.4828203022480011,
                -0.3255046010017395,
                0.008128180168569088,
                0.35277295112609863,
                -3.641430377960205,
                -1.842738389968872,
                -0.0013586411951109767,
                -0.042040832340717316,
            ],
            q01=[
                -0.42401049643754957,
                -0.2838300323486328,
                0.009925739830359817,
                1.3085840785503386,
                -2.886677579879761,
                -1.1599004411697387,
                0.001503719249740243,
                -0.040336399003863335,
            ],
            q99=[
                0.1530261474847791,
                0.3629165390133857,
                1.2910678112506866,
                3.303542451858519,
                2.7496529006957933,
                0.6893712210655194,
                0.040610933862626555,
                -0.0015016929572448147,
            ],
        ),
        num_transitions=273465,
        num_trajectories=1693,
    ), )

_action_dim = 8
_ori_action_dim = 7
_state_dim = 16
_action_horizon = 8
_frame_window_size = 5
_image_frame_stride = 2
_image_size = 224

_action_norm_mask = [True, True, True, True, True, True, False]
# This local conversion is the closest available match to the released
# 1.0.0 mixture (273233/1692 vs. source 273465/1693 transitions/trajectories)
# and reproduces all released action min/max values exactly.
_libero_data_roots = [
    './datasets/libero_object_no_noops_lerobotv2.1',
    './datasets/libero_goal_no_noops_lerobotv2.1',
    './datasets/libero_spatial_no_noops_lerobotv2.1',
    './datasets/libero_10_no_noops_lerobotv2.1',
]
# Match the source training RNG and its equal-weight mixture sampler.
seed = 42

# The official release evaluates these suites independently. FluxVLA's eval
# runner takes one suite per launch; override eval.task_suite_name for each.
official_eval_task_suites = [
    'libero_spatial',
    'libero_object',
    'libero_goal',
    'libero_10',
]

model = dict(
    type='DiT4DiTVLA',
    # Match the released training recipe: initialize the backbone from Cosmos,
    # but do not initialize from a released DiT4DiT policy checkpoint.
    repeated_diffusion_steps=4,
    vlm_backbone=dict(
        type='Cosmos25Backbone',
        base_model=_cosmos_base_model,
        revision=_cosmos_revision,
        torch_dtype='bf16',
        local_files_only=True,
        extract_layer=17,
        trainable=True,
        frozen_submodules=['text_encoder', 'vae'],
        split_future_frames=True,
        num_frames_out=_frame_window_size,
        fixed_seed=None,
        num_inference_steps=1,
        conditional_frame_timestep=0.0001,
        future_loss_type='flow_matching',
        detach_hidden_states=True,
        flow_matching_time_distribution='uniform',
        flow_matching_high_sigma_ratio=None,
        flow_matching_high_sigma_min=None,
        # Wrap transformer blocks only. Combining the block policy with the
        # 10M size policy recursively wrapped large linears inside each block
        # and created hundreds of tiny FSDP collectives.
        fsdp_min_num_params=0,
    ),
    vla_head=dict(
        type='DiT4DiTActionHead',
        action_model_type='DiT-B',
        hidden_size=2560,
        add_pos_embed=True,
        max_seq_len=1024,
        action_dim=_action_dim,
        ori_action_dim=_ori_action_dim,
        state_dim=_state_dim,
        future_action_window_size=7,
        action_horizon=_action_horizon,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        num_inference_timesteps=4,
        diffusion_model_cfg=dict(
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_layers=16,
            output_dim=2560,
            positional_embeddings=None,
        ),
    ),
    freeze_vlm_backbone=False,
)

# Evaluation checkpoints contain the complete Cosmos + action-head state.
# Build only the architecture on meta tensors, then let the eval runner assign
# the requested FluxVLA checkpoint. This avoids loading any pretrained model
# weights while constructing inference_model.
inference_model = dict(
    model,
    init_empty_weights=True,
    vlm_backbone=dict(
        model['vlm_backbone'],
        load_pretrained_weights=False,
    ),
)

_dit4dit_train_transforms = [
    dict(
        type='ProcessParquetInputs',
        parquet_keys=[
            'observation.state',
            'timestamp',
            'actions',
            'info',
            'stats',
            'action_masks',
        ],
        video_keys=[
            'observation.images.image',
            'observation.images.wrist_image',
        ],
        name_mappings={
            'observation.state': ['states'],
            'actions': ['actions'],
        },
    ),
    dict(
        type='ProcessCosmos25Prompt',
        tokenizer=_cosmos_tokenizer,
        input_key='task_description',
        remove_input_key=True,
    ),
    # Match the source loader: float CHW / 255 first, then torch bilinear
    # resize. OpenCV uint8 resize followed by normalization quantizes the
    # interpolation result and measurably drifts the Cosmos inputs.
    dict(
        type='ResizeImages',
        height=_image_size,
        width=_image_size,
        backend='torch',
        scale_divisor=255.0,
        output_layout='flattened_chw',
    ),
    dict(
        type='PrepareVideo',
        num_views=2,
        frame_window_size=_frame_window_size,
        tile_direction='horizontal',
        combine_view_masks=True,
    ),
    dict(
        type='SinCosKeys',
        keys=['states'],
        target_dims={'states': _state_dim},
        interleave=True,
        expand_axis=0,
        # The source StateActionSinCosTransform uses torch.float32 kernels.
        backend='torch',
    ),
    dict(
        type='NormalizeStatesAndActions',
        state_key='proprio',
        action_key='action',
        action_dim=_action_dim,
        state_dim=None,
        state_norm_type='none',
        action_norm_type='min_max',
        norm_type='min_max',
        action_norm_mask=_action_norm_mask,
        clip_norm=False,
        # The source torch normalizer divides by (max - min) exactly.
        normalization_epsilon=0.0,
        preserve_input_dtype=True,
        valid_action_dim=_ori_action_dim,
        mark_all_action_steps_valid=True,
        # The source casts normalized actions and sin/cos states to float16
        # before padding and collation.
        output_dtype='float16',
    ),
]

_dit4dit_parquet_dataset = dict(
    type='ParquetDataset',
    transforms=_dit4dit_train_transforms,
    action_window_size=_action_horizon,
    action_key='action',
    use_delta=False,
    statistic_name='franka',
    window_start_idx=0,
    frame_window_size=_frame_window_size,
    frame_sample_stride=_image_frame_stride,
)

train_dataloader = dict(
    # Four nodes * eight H100s preserving the source effective batch:
    # 32 GPUs * 8 samples/GPU * 1 accumulation step = batch 256.
    # This avoids a second forward/backward micro-step per optimizer update.
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedBalancedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='franka',
        # Source DiT4DiT samples all four suites uniformly with replacement.
        datasets=[
            dict(_dit4dit_parquet_dataset, data_root_path=data_root)
            for data_root in _libero_data_roots
        ],
        sampling_weights=[1.0, 1.0, 1.0, 1.0],
        shuffle=False,
        reshuffle_each_epoch=True,
        seed=seed,
        # Pin normalization to the released model instead of transition-count
        # weighted statistics from a locally converted dataset revision.
        dataset_statistics=_official_dataset_statistics,
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    # The released 98.6% checkpoint was trained for 160k optimizer updates.
    # The public run script's older 80k setting does not describe that model.
    max_steps=160000,
    # One B=8 micro-batch per rank retains the source global batch of 256.
    grad_accumulation_steps=1,
    optimizer=dict(
        # Match dit4dit_libero/config.yaml shipped with the official model.
        lr=3e-5,
        type='AdamW',
        weight_decay=1e-8,
        eps=1e-8,
        betas=(0.9, 0.95),
        paramwise_learning_rate={
            'vlm_backbone.transformer': 1e-4,
            'vla_head': 1e-4,
        },
    ),
    max_grad_norm=1.0,
    # The public source launch saves every 40k steps. Full FSDP checkpoints
    # gather the complete model and optimizer, so a 5k interval creates long
    # periodic stalls that look like a training hang.
    save_iter_interval=40000,
    max_keep_ckpts=2,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['info', 'stats'],
    ),
    tokenizer=_cosmos_tokenizer,
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler=dict(
        type='cosine_with_min_lr',
        warmup_steps=10000,
        min_lr=5e-7,
    ),
    # DiT4DiT is released with DeepSpeed ZeRO-2: parameters remain resident
    # while gradients and optimizer state are sharded. FULL_SHARD repeatedly
    # all-gathers the Cosmos parameters across its two forwards per step.
    sharding_strategy='shard-grad-op',
    # Keep FP32 master parameters/Adam moments like DeepSpeed BF16_Optimizer;
    # FSDP still casts forward parameters to BF16 below.
    pre_fsdp_param_dtype='fp32',
    # The upstream flag is not wired into the model and therefore does not
    # actually enable checkpointing. Enabling it here recomputes Cosmos during
    # backward and compounds FSDP communication.
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    # Match the source BF16 communication path instead of doubling gradient
    # traffic with FP32 reductions.
    reduce_in_full_precision=False,
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name=official_eval_task_suites,
    model_family='dit4dit',
    norm_stats_key='franka',
    eval_chunk_size=_action_horizon,
    resize_size=_image_size,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    enable_mixed_precision_training=False,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        img_buffer_len=1,
        transforms=[
            dict(
                type='ProcessCosmos25Prompt',
                tokenizer=_cosmos_tokenizer,
                input_key='task_description',
                remove_input_key=True,
            ),
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                use_pil=False,
            ),
            dict(
                type='ResizeImages',
                key='pixel_values',
                height=_image_size,
                width=_image_size,
                backend='cv2',
                interpolation='area',
                output_layout='flattened_chw',
            ),
            dict(
                type='SimpleNormalizeImages',
                key='pixel_values',
                preserve_leading_dims=True,
                output_type='torch',
            ),
            dict(
                type='PrepareVideo',
                num_views=2,
                frame_window_size=1,
                tile_direction='horizontal',
                combine_view_masks=True,
            ),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='none',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                state_dim=None,
                out_key='states',
            ),
            dict(
                type='SinCosKeys',
                keys=['states'],
                target_dims={'states': _state_dim},
                interleave=True,
                expand_axis=0,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='min_max',
        action_dim=_ori_action_dim,
        action_norm_mask=_action_norm_mask,
        clip_normalized_action=True,
        normalize_gripper_action=True,
        invert_gripper_action=True,
    ),
)
