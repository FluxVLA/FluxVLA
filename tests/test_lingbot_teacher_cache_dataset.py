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
