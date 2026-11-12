"""Dataset for model-ready LingBot replays with offline teacher targets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

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
        flow_from_manifest: bool = True,
        require_flow: bool = False,
        replay_order: str = "manifest",
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
        self.flow_from_manifest = bool(flow_from_manifest)
        self.require_flow = bool(require_flow)
        if replay_order not in ("manifest", "official_step_rank"):
            raise ValueError(
                "replay_order must be 'manifest' or 'official_step_rank'"
            )
        if replay_order == "official_step_rank":
            if any(
                "official_step" not in entry or "official_rank" not in entry
                for entry in self.entries
            ):
                raise ValueError(
                    "official_step_rank replay requires official_step and official_rank"
                )
            self.entries.sort(
                key=lambda entry: (
                    int(entry["official_step"]),
                    int(entry["official_rank"]),
                )
            )
        self.deterministic_order = replay_order == "official_step_rank"
        if self.flow_from_manifest:
            flow_paths = [entry.get("flow_path") for entry in self.entries if entry.get("flow_path")]
            if len(flow_paths) != len(set(flow_paths)):
                raise ValueError(
                    "Each replay entry must have a unique flow_path; duplicated flow inputs "
                    "would invalidate optimizer/trajectory parity"
                )
            if self.require_flow and any(not entry.get("flow_path") for entry in self.entries):
                raise ValueError("Replay manifest is incomplete: every entry requires a captured flow_path")
        self._cache: dict[int, dict[str, torch.Tensor]] = {}
        self._trace_call_index = 0

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
        captured_batch = bool(entry.get("captured_batch", False))
        if not captured_batch:
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
        batched_keys = _MODEL_KEYS if captured_batch else _BATCHED_CACHE_KEYS
        for key in batched_keys:
            if output[key].shape[0] != 1:
                raise ValueError(f"Cached {key} must have a leading singleton batch dimension")
            output[key] = output[key].squeeze(0)
        if not captured_batch:
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
        flow_path = entry.get("flow_path") if self.flow_from_manifest else None
        if flow_path:
            flow_path = Path(flow_path)
            if not flow_path.is_absolute():
                flow_path = self.manifest_path.parent / flow_path
            flow = torch.load(flow_path.resolve(), map_location="cpu", weights_only=False)
            if not isinstance(flow, dict) or "noise" not in flow or "time" not in flow:
                raise ValueError(f"Flow cache {flow_path} must contain noise and time")
            output["noise"] = torch.as_tensor(flow["noise"]).squeeze(0)
            output["time"] = torch.as_tensor(flow["time"]).reshape(-1)[0]
            if output["noise"].shape != output["actions"].shape:
                raise ValueError(
                    f"Flow noise shape {tuple(output['noise'].shape)} does not match "
                    f"actions {tuple(output['actions'].shape)} for {flow_path}"
                )
        else:
            output["noise"] = torch.zeros_like(output["actions"])
            output["time"] = torch.tensor(self.fixed_time, dtype=torch.float32)
        return output

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        if self.cache_in_memory and index in self._cache:
            sample = self._cache[index]
        else:
            sample = self._load(index)
            if self.cache_in_memory:
                self._cache[index] = sample
        self._write_sample_trace(index, sample)
        return dict(sample)

    def _write_sample_trace(
        self, index: int, sample: Mapping[str, torch.Tensor]
    ) -> None:
        root_value = os.environ.get("FLUXVLA_LINGBOT_SAMPLE_TRACE_DIR")
        if not root_value:
            return
        self._trace_call_index += 1
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        tensors = {}
        for name, value in sample.items():
            if not isinstance(value, torch.Tensor):
                continue
            contiguous = value.detach().cpu().contiguous()
            payload = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
            tensors[name] = {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        root = Path(root_value)
        root.mkdir(parents=True, exist_ok=True)
        row = {
            "call_index": self._trace_call_index,
            "dataset_index": index,
            "identity": self.identity(index),
            "rank": rank,
            "tensors": tensors,
        }
        with (root / f"rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
