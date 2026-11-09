"""FluxVLA replay of the official LingBot DP2 sampler trace."""

_base_ = ["./lingbot_vla_v2_robotwin_dense_finetune.py"]

model = dict(
    # Use the same local production checkpoint and eager backend as the
    # official compatibility run.  The nested form is consumed by the
    # FluxVLA adapter after the native YAML is merged.
    pretrained_name_or_path="/ceph/teamGolden/andyzeng/VLA/models/lingbot-vla-v2-6b",
    tokenizer_path="/ceph/teamGolden/andyzeng/VLA/models/Qwen3-VL-4B-Instruct",
    # Native production config has train.enable_fp32=true. Keep action expert
    # parameters and its loss in FP32 for an apples-to-apples replay.
    action_fp32=True,
    attention_implementation="eager",
    vit_attn_implementation="eager",
    # Official production config uses the fused token-MoE path.
    moe_implementation="fused",
    attention=dict(implementation="eager", vit_implementation="eager"),
)

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=0,
    dataset=dict(
        type="LingBotTeacherCacheDataset",
        manifest_path=(
            "/ceph/teamGolden/andyzeng/VLA/FluxVLA/configs/lingbot_vla_v2/"
            "official_trace_manifest_captured_20261016.json"
        ),
        cache_in_memory=True,
        validate_targets=True,
        fixed_time=0.5,
        flow_from_manifest=True,
        require_flow=True,
    ),
)

runner = dict(
    max_steps=1,
    seed=42,
    sharding_strategy="global-shard-grad-op",
    fsdp_api_version=2,
    enable_mixed_precision_training=True,
    mixed_precision_dtype="bf16",
    # The official trainer applies FP32 only to action-expert/basic modules.
    # Its VLM decoder/root FSDP2 units still use BF16 parameters with FP32
    # gradient reduction, so keeping every parameter FP32 would be a different
    # numerical contract.
    keep_params_fp32=False,
    reduce_in_full_precision=True,
)
