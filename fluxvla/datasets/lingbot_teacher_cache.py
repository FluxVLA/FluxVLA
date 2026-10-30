"""Dataset for model-ready LingBot replays with offline teacher targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fluxvla.engines import DATASETS


_CACHE_FORMAT = "fluxvla.lingbot_teacher_cache.v1"
_MODEL_KEYS = (
    "images",
    "image_grid_thw",
    "img_masks",
    "lang_tokens",
    "lang_masks",
    "state",
    "state_joint_mask",
    "actions",
    "joint_mask",
    "action_is_pad",
    "depth_targets",
    "future_depth_targets",
    "future_video_targets",
    "future_video_current_patch",
)
_BATCHED_CACHE_KEYS = (
    "lang_tokens",
    "lang_masks",
    "state",
    "actions",
    "depth_targets",
    "future_depth_targets",
    "future_video_targets",
    "future_video_current_patch",
)


@DATASETS.register_module()
class LingBotTeacherCacheDataset(Dataset):
    """Read deterministic, model-ready LingBot samples from a cache manifest."""

    def __init__(
        self,
        manifest_path: str,
        cache_in_memory: bool = False,
        validate_targets: bool = True,
        fixed_time: float = 0.5,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != _CACHE_FORMAT:
            raise ValueError(
                f"Unsupported LingBot teacher cache format: {manifest.get('format')!r}"
            )
        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("LingBot teacher cache manifest must contain entries")
        self.entries = [self._normalize_entry(entry) for entry in entries]
        cache_keys = [entry["cache_key"] for entry in self.entries]
        if len(cache_keys) != len(set(cache_keys)):
            raise ValueError("LingBot teacher cache keys must be unique")
        self.cache_in_memory = bool(cache_in_memory)
        self.validate_targets = bool(validate_targets)
        self.fixed_time = float(fixed_time)
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def _normalize_entry(self, entry: Any) -> dict[str, Any]:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("cache_key"):
            raise ValueError("Each LingBot cache entry requires path and cache_key")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        normalized = dict(entry)
        normalized["path"] = path.resolve()
        return normalized

    def __len__(self) -> int:
        return len(self.entries)

    def identity(self, index: int) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.entries[index].items()
            if key != "path"
        }

    def _load(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        sample = torch.load(entry["path"], map_location="cpu", weights_only=False)
        metadata = sample.get("source_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Cached sample {entry['path']} has no source_metadata")
        if metadata.get("teacher_cache_key") != entry["cache_key"]:
            raise ValueError(f"Teacher cache key mismatch for {entry['path']}")
        if metadata.get("auxiliary_targets") != "native_moge_morgbd_dino":
            raise ValueError(f"Cached sample {entry['path']} has no native teacher targets")

        missing = [key for key in _MODEL_KEYS if key not in sample]
        if missing:
            raise KeyError(f"Cached sample {entry['path']} is missing keys: {missing}")
        output = {key: torch.as_tensor(sample[key]) for key in _MODEL_KEYS}
        for key in _BATCHED_CACHE_KEYS:
            if output[key].shape[0] != 1:
                raise ValueError(f"Cached {key} must have a leading singleton batch dimension")
            output[key] = output[key].squeeze(0)
        for key in ("depth_targets", "future_depth_targets"):
            if output[key].shape[-2] == 252:
                output[key] = F.pad(output[key], (0, 0, 0, 4))
        if self.validate_targets:
            for key in (
                "depth_targets",
                "future_depth_targets",
                "future_video_targets",
                "future_video_current_patch",
            ):
                target = output[key]
                if target.shape != (256, 1024):
                    raise ValueError(f"Cached {key} has invalid shape {tuple(target.shape)}")
                if not torch.isfinite(target.float()).all() or torch.count_nonzero(target).item() == 0:
                    raise ValueError(f"Cached {key} must be finite and nonzero")
        output["noise"] = torch.zeros_like(output["actions"])
        output["time"] = torch.tensor(self.fixed_time, dtype=torch.float32)
        return output

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        if self.cache_in_memory and index in self._cache:
            return dict(self._cache[index])
        sample = self._load(index)
        if self.cache_in_memory:
            self._cache[index] = sample
        return dict(sample)
