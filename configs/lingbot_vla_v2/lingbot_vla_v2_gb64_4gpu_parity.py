"""Four-GPU, GB64, 100-step diagnostic replay, not an online recipe.

Set LINGBOT_MANIFEST to an existing captured-input manifest. WORK_DIR only
controls output; no input artifacts are assumed to exist under that directory.
"""

_base_ = ["./lingbot_vla_v2_official_production_100_20261016.py"]
import os  # noqa: E402

train_dataloader = dict(
    per_device_batch_size=1,
    # Keep the diagnostic conservative; online worker counts are configurable.
    per_device_num_workers=0,
    dataset=dict(
        manifest_path=os.getenv("LINGBOT_MANIFEST", ""),
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
        run_dir=os.getenv("WORK_DIR", "work_dirs/lingbot_gb64_native"),
        window_size=1,
    ),
)
