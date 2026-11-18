import json

import pytest
import torch

from fluxvla.datasets.lingbot_teacher_cache import LingBotTeacherCacheDataset


def _write_sample(path, cache_key, state_value):
    sample = {
        "images": torch.ones(3, 192, 1536),
        "image_grid_thw": torch.tensor([[1, 12, 16]] * 3),
        "img_masks": torch.ones(3, dtype=torch.bool),
        "lang_tokens": torch.ones(1, 72, dtype=torch.long),
        "lang_masks": torch.ones(1, 72, dtype=torch.bool),
        "state": torch.full((1, 55), state_value),
        "state_joint_mask": torch.ones(55, dtype=torch.bool),
        "actions": torch.ones(1, 50, 55),
        "joint_mask": torch.ones(50, 55, dtype=torch.bool),
        "action_is_pad": torch.zeros(50, dtype=torch.bool),
        "depth_targets": torch.ones(1, 252, 1024, dtype=torch.bfloat16),
        "future_depth_targets": torch.ones(1, 252, 1024, dtype=torch.bfloat16),
        "future_video_targets": torch.ones(1, 256, 1024, dtype=torch.bfloat16),
        "future_video_current_patch": torch.ones(1, 256, 1024, dtype=torch.bfloat16),
        "source_metadata": {
            "auxiliary_targets": "native_moge_morgbd_dino",
            "teacher_cache_key": cache_key,
        },
    }
    torch.save(sample, path)


def _write_manifest(path, entries):
    path.write_text(
        json.dumps(
            {
                "format": "fluxvla.lingbot_teacher_cache.v1",
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )


def _write_captured_batch(path):
    sample = {
        "images": torch.arange(3 * 256 * 16, dtype=torch.float32).reshape(1, 3, 256, 16),
        "image_grid_thw": torch.tensor([[[1, 16, 16]] * 3]),
        "img_masks": torch.tensor([[True, False, True]]),
        "lang_tokens": torch.arange(72).reshape(1, 72),
        "lang_masks": torch.ones(1, 72, dtype=torch.bool),
        "state": torch.arange(55, dtype=torch.bfloat16).reshape(1, 55),
        "state_joint_mask": torch.ones(1, 55, dtype=torch.bool),
        "actions": torch.arange(50 * 55, dtype=torch.bfloat16).reshape(1, 50, 55),
        "joint_mask": torch.ones(1, 50, 55, dtype=torch.bool),
        "action_is_pad": torch.zeros(1, 50, dtype=torch.bool),
    }
    for offset, key in enumerate(
        (
            "depth_targets",
            "future_depth_targets",
            "future_video_targets",
            "future_video_current_patch",
        ),
        start=1,
    ):
        sample[key] = torch.full((1, 256, 1024), offset, dtype=torch.bfloat16)
    torch.save(sample, path)
    return sample


def test_lingbot_teacher_cache_builds_unbatched_model_contract(tmp_path):
    _write_sample(tmp_path / "sample.pt", "sample-0", 3.0)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [{"path": "sample.pt", "cache_key": "sample-0"}])

    dataset = LingBotTeacherCacheDataset(str(manifest), cache_in_memory=True)
    sample = dataset[0]

    assert sample["images"].shape == (3, 192, 1536)
    assert sample["state"].shape == (55,)
    assert sample["actions"].shape == (50, 55)
    assert sample["depth_targets"].shape == (256, 1024)
    assert sample["future_video_targets"].shape == (256, 1024)
    assert torch.count_nonzero(sample["depth_targets"]).item() == 252 * 1024
    assert torch.equal(sample["noise"], torch.zeros_like(sample["actions"]))
    assert sample["time"].item() == pytest.approx(0.5)
    assert dataset.identity(0)["cache_key"] == "sample-0"


def test_lingbot_teacher_cache_preserves_manifest_order_for_rollover(tmp_path):
    entries = []
    for index in range(2):
        cache_key = f"sample-{index}"
        sample_path = tmp_path / f"sample{index}.pt"
        _write_sample(sample_path, cache_key, float(index))
        entries.append({"path": sample_path.name, "cache_key": cache_key})
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, entries)

    dataset = LingBotTeacherCacheDataset(str(manifest))
    sequence = [dataset[index % len(dataset)]["state"][0].item() for index in range(5)]

    assert sequence == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_lingbot_teacher_cache_rejects_key_mismatch(tmp_path):
    _write_sample(tmp_path / "sample.pt", "actual", 0.0)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [{"path": "sample.pt", "cache_key": "expected"}])

    dataset = LingBotTeacherCacheDataset(str(manifest))
    with pytest.raises(ValueError, match="cache key mismatch"):
        dataset[0]


def test_lingbot_teacher_cache_rejects_duplicate_flow_for_strict_replay(tmp_path):
    for index in range(2):
        _write_sample(tmp_path / f"sample{index}.pt", f"sample-{index}", float(index))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [
        {"path": "sample0.pt", "cache_key": "sample-0", "flow_path": "flow.pt"},
        {"path": "sample1.pt", "cache_key": "sample-1", "flow_path": "flow.pt"},
    ])
    with pytest.raises(ValueError, match="unique flow_path"):
        LingBotTeacherCacheDataset(str(manifest), require_flow=True)


def test_lingbot_teacher_cache_can_restore_official_step_rank_order(tmp_path):
    entries = []
    for step, rank, value in ((2, 1, 21.0), (1, 1, 11.0), (2, 0, 20.0), (1, 0, 10.0)):
        cache_key = f"sample-r{rank}-s{step}"
        sample_path = tmp_path / f"{cache_key}.pt"
        _write_sample(sample_path, cache_key, value)
        entries.append({
            "path": sample_path.name,
            "cache_key": cache_key,
            "official_step": step,
            "official_rank": rank,
        })
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, entries)

    dataset = LingBotTeacherCacheDataset(
        str(manifest), replay_order="official_step_rank")

    assert dataset.deterministic_order is True
    assert [dataset.identity(index)["cache_key"] for index in range(4)] == [
        "sample-r0-s1", "sample-r1-s1", "sample-r0-s2", "sample-r1-s2"
    ]


def test_lingbot_teacher_cache_replays_captured_batch_and_flow_exactly(tmp_path):
    captured = _write_captured_batch(tmp_path / "batch.pt")
    flow = {
        "noise": torch.arange(50 * 55, dtype=torch.bfloat16).reshape(1, 50, 55),
        "time": torch.tensor([0.625], dtype=torch.bfloat16),
    }
    torch.save(flow, tmp_path / "flow.pt")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "path": "batch.pt",
                "flow_path": "flow.pt",
                "cache_key": "official-r0-s1",
                "captured_batch": True,
            }
        ],
    )

    sample = LingBotTeacherCacheDataset(str(manifest), require_flow=True)[0]

    for key, expected in captured.items():
        assert torch.equal(sample[key], expected.squeeze(0)), key
    assert torch.equal(sample["noise"], flow["noise"].squeeze(0))
    assert torch.equal(sample["time"], flow["time"].squeeze(0))
    assert sample["time"].dtype == torch.bfloat16


def test_lingbot_teacher_cache_captured_batch_requires_singleton_batch(tmp_path):
    captured = _write_captured_batch(tmp_path / "batch.pt")
    captured["images"] = captured["images"].expand(2, -1, -1, -1)
    torch.save(captured, tmp_path / "batch.pt")
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        [{"path": "batch.pt", "cache_key": "official-r0-s1", "captured_batch": True}],
    )

    with pytest.raises(ValueError, match="images must have a leading singleton"):
        LingBotTeacherCacheDataset(str(manifest))[0]


def test_lingbot_teacher_cache_traces_rank_identity_and_tensor_hashes(
    tmp_path, monkeypatch
):
    entries = []
    for index, state_value in enumerate((3.0, 4.0)):
        cache_key = f"sample-{index}"
        sample_path = tmp_path / f"sample{index}.pt"
        _write_sample(sample_path, cache_key, state_value)
        entries.append(
            {
                "path": sample_path.name,
                "cache_key": cache_key,
                "official_step": 7,
                "official_rank": index,
            }
        )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, entries)
    trace_dir = tmp_path / "trace"
    monkeypatch.setenv("FLUXVLA_LINGBOT_SAMPLE_TRACE_DIR", str(trace_dir))
    monkeypatch.setenv("RANK", "2")

    dataset = LingBotTeacherCacheDataset(str(manifest), cache_in_memory=True)
    dataset[0]
    dataset[0]  # Exercise the cache-hit path.
    dataset[1]

    rows = [
        json.loads(line)
        for line in (trace_dir / "rank2.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["call_index"] for row in rows] == [1, 2, 3]
    assert [row["dataset_index"] for row in rows] == [0, 0, 1]
    assert rows[0]["identity"] == {
        "cache_key": "sample-0",
        "official_rank": 0,
        "official_step": 7,
    }
    assert rows[0]["rank"] == 2
    assert rows[0]["tensors"]["state"]["dtype"] == "torch.float32"
    assert rows[0]["tensors"]["state"]["shape"] == [55]
    assert rows[0]["tensors"]["state"]["sha256"] == rows[1]["tensors"]["state"]["sha256"]
    assert rows[0]["tensors"]["state"]["sha256"] != rows[2]["tensors"]["state"]["sha256"]
    assert len(rows[0]["tensors"]["time"]["sha256"]) == 64
