"""Four-GPU, GB64, 100-step LingBot Native FSDP parity baseline."""

_base_ = ["./lingbot_vla_v2_official_production_100_20261016.py"]

_REPLAY_ROOT = (
    "/ceph/teamGolden/andyzeng/VLA/FluxVLA/"
    "work_dirs/lingbot_gb64_4gpu_100step_20261026"
)

train_dataloader = dict(
    per_device_batch_size=1,
    per_device_num_workers=0,
    dataset=dict(
        manifest_path=f"{_REPLAY_ROOT}/gb64_100step_manifest.json",
        cache_in_memory=False,
        replay_order="official_step_rank",
        flow_from_manifest=True,
        require_flow=True,
    ),
)

runner = dict(
    max_steps=100,
    grad_accumulation_steps=16,
    seed=42,
    enable_gradient_checkpointing=False,
    save_iter_interval=10000,
    metric=dict(
        type="VLAMetric",
        active_trackers=("jsonl",),
        run_dir=f"{_REPLAY_ROOT}/native_fsdp4_mb1_norecompute",
        window_size=1,
    ),
)
