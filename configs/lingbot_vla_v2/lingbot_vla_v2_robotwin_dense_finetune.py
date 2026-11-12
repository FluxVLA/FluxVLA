"""LingBot-VLA-v2 DP4 FULL_SHARD training on an offline teacher cache."""

_ROOT = "/ceph/teamGolden/andyzeng/VLA"
_SHARED = "/ceph/teamGolden/sjkang/rl/rlinf_adapt"
_CACHE = f"{_ROOT}/FluxVLA/work_dirs/lingbot_backend_alignment_20260930"

model = dict(
    type="LingBotVLAV2",
    source_root=f"{_ROOT}/lingbot-vla-v2",
    tokenizer_path=f"{_SHARED}/weights/Qwen3-VL-4B-Instruct",
    pretrained_name_or_path=(
        f"{_SHARED}/weights/lingbot-vla-v2-6b-robotwin/"
        "checkpoints/global_step_50000/hf_ckpt"
    ),
    native_config_path=f"{_ROOT}/lingbot-vla-v2/golden_artifacts/robotwin_reduced.yaml",
    # Native train.enable_fp32 -> model.action_fp32 contract.
    action_fp32=True,
)

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=0,
    dataset=dict(
        type="LingBotTeacherCacheDataset",
        manifest_path=f"{_CACHE}/lingbot_teacher_cache_manifest.json",
        cache_in_memory=True,
        validate_targets=True,
        fixed_time=0.5,
    ),
)

runner = dict(
    type="FSDPTrainRunner",
    max_steps=2,
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        # Native use_moe_expert_lr scales routed experts by sqrt(32 / 4).
        paramwise_learning_rate_regex={
            r"(?:^|\.)layers\.\d+\.mlp\.experts\.": 2.8284271247461903e-4,
        },
    ),
    max_grad_norm=1.0,
    # Native FSDP2 uses reshard_after_forward=false (ZeRO-2 semantics).
    sharding_strategy="global-shard-grad-op",
    fsdp_api_version=2,
    # Native FSDP2 can shard one root unit within the A100 memory budget. FSDP1
    # needs decoder-sized handles to avoid a 6.376B-parameter flat-grad peak;
    # every unit still uses the same native BF16/FP32-reduce precision policy.
    fsdp_wrap_policy="model",
    # Native enable_fp32 keeps trainable parameters in FP32 while autocast
    # controls transformer compute precision.
    pre_fsdp_param_dtype=None,
    reduce_in_full_precision=True,
    collator=dict(
        type="DictCollator",
        keys=[
            "images",
            "img_masks",
            "image_grid_thw",
            "lang_tokens",
            "lang_masks",
            "state",
            "state_joint_mask",
            "actions",
            "joint_mask",
            "action_is_pad",
            "noise",
            "time",
            "depth_targets",
            "future_depth_targets",
            "future_video_targets",
            "future_video_current_patch",
        ],
    ),
    sampler="distributed",
    metric=dict(
        type="VLAMetric",
        active_trackers=("jsonl",),
        run_dir="work_dirs",
        window_size=1,
    ),
    lr_scheduler=dict(
        type="linear-warmup+cosine-decay",
        warmup_ratio=0.0,
        min_lr=5e-5,
    ),
    save_iter_interval=10000,
    max_keep_ckpts=1,
    # Match the native RobotWin resolved training contract.
    seed=42,
    enable_gradient_checkpointing=True,
    # Match the native root FSDP2 BF16 parameter-compute policy.
    enable_mixed_precision_training=True,
    enable_forward_autocast=False,
    mixed_precision_dtype="bf16",
    keep_params_fp32=False,
    report_dp_mean_metrics=True,
    deterministic_algorithms=False,
    change_key_name=False,
)
