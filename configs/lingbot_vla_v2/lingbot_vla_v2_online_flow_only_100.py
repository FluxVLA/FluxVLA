"""100-step Native FSDP run on the FluxVLA-owned online LeRobot pipeline.

This is the fair first-stage baseline: flow loss only, with auxiliary teacher
targets disabled in all three runners until the production teacher path is
migrated into FluxVLA.
"""

_base_ = ["./lingbot_vla_v2_robotwin_dense_finetune.py"]

_ROOT = "/ceph/teamGolden/andyzeng/VLA"
_DATA = "/ceph/teamGolden/datasets/huggingface/lerobot/lingbot_beat_block_hammer_reduced"

model = dict(
    native_config_path=f"{_ROOT}/lingbot_runs/reduced_beat_block_hammer/lingbotvla_cli.yaml",
    pretrained_name_or_path=f"{_ROOT}/models/lingbot-vla-v2-6b",
    tokenizer_path=f"{_ROOT}/models/Qwen3-VL-4B-Instruct",
    action_fp32=False,
    attention_implementation="eager",
    vit_attn_implementation="eager",
    moe_implementation="fused",
    align_params={},
    attention=dict(implementation="eager", vit_implementation="eager"),
)

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=0,
    dataset=dict(
        _delete_=True,
        type="LingBotOnlineDataset",
        data_path=_DATA,
        norm_stats_path=(
            f"{_ROOT}/FluxVLA/fluxvla/datasets/lingbot_vla_v2/"
            "assets/robotwin_norm_stats.json"
        ),
        tokenizer_path=f"{_ROOT}/models/Qwen3-VL-4B-Instruct",
        action_horizon=50,
        image_size=256,
        tokenizer_max_length=72,
        use_future_images=True,
        return_teacher_images=False,
        flow_rng_seed=42,
    ),
)

runner = dict(
    max_steps=100,
    seed=42,
    sharding_strategy="global-shard-grad-op",
    fsdp_api_version=2,
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        foreach=True,
        fused=False,
    ),
    max_grad_norm=1.0,
    enable_mixed_precision_training=True,
    # MCore executes the policy under BF16 autocast. Keep the Native parity
    # baseline on the same compute path instead of inheriting the production
    # base config's explicit autocast disable.
    enable_forward_autocast=True,
    mixed_precision_dtype="bf16",
    keep_params_fp32=False,
    reduce_in_full_precision=True,
    enable_gradient_checkpointing=True,
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
            "action_joint_mask",
            "joint_mask",
            "action_is_pad",
            "sample_index",
            "noise",
            "time",
        ],
    ),
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
    report_dp_mean_metrics=True,
    deterministic_algorithms=False,
    change_key_name=False,
)
