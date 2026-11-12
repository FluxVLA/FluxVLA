"""Temporary launcher for official LingBot parity checks."""

import runpy
import sys
import json
import os
import builtins

import torch
import transformers
import transformers.modeling_utils as _modeling_utils
import transformers.utils.import_utils as _import_utils
import transformers.cache_utils as _cache_utils
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen25vl
import transformers.models.qwen3_vl.modeling_qwen3_vl as _qwen3vl
from contextlib import contextmanager

# Transformers 5.x exposes this symbol through a lazy module that can report
# a false negative to ``hasattr`` while importing the official LingBot package.
# Install an explicit top-level alias before any official module is imported.
if not hasattr(transformers, "AutoModelForVision2Seq"):
    setattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)

if not hasattr(_modeling_utils, "no_init_weights"):
    @contextmanager
    def no_init_weights(_enable=True):
        yield

    _modeling_utils.no_init_weights = no_init_weights

if not getattr(_modeling_utils.PreTrainedModel, "_official_lingbot_tie_compat", False):
    _original_tied_keys = _modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys

    def _get_expanded_tied_weights_keys(self, all_submodels=False):
        if isinstance(getattr(self, "_tied_weights_keys", None), list):
            return {}
        return _original_tied_keys(self, all_submodels=all_submodels)

    _modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = _get_expanded_tied_weights_keys
    _modeling_utils.PreTrainedModel._official_lingbot_tie_compat = True

if not hasattr(_import_utils, "is_safetensors_available"):
    _import_utils.is_safetensors_available = lambda: True

if not hasattr(_cache_utils, "SlidingWindowCache"):
    _cache_utils.SlidingWindowCache = _cache_utils.DynamicCache

if not hasattr(_qwen25vl, "Qwen2RMSNorm"):
    _qwen25vl.Qwen2RMSNorm = _qwen25vl.Qwen2_5_VLRMSNorm

if not hasattr(_qwen3vl.Qwen3VLForConditionalGeneration, "visual"):
    _qwen3vl.Qwen3VLForConditionalGeneration.visual = property(
        lambda self: self.model.visual
    )

if not getattr(_qwen3vl.Qwen3VLModel, "_official_lingbot_rope_compat", False):
    _original_rope_index = _qwen3vl.Qwen3VLModel.get_rope_index

    def _get_rope_index_compat(self, input_ids, mm_token_type_ids=None, **kwargs):
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids = mm_token_type_ids.masked_fill(
                input_ids == self.config.image_token_id, 1
            )
            if hasattr(self.config, "video_token_id"):
                mm_token_type_ids = mm_token_type_ids.masked_fill(
                    input_ids == self.config.video_token_id, 2
                )
        return _original_rope_index(
            self,
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

    _qwen3vl.Qwen3VLModel.get_rope_index = _get_rope_index_compat
    _qwen3vl.Qwen3VLModel._official_lingbot_rope_compat = True
from lerobot.datasets import utils as _lerobot_utils
from lerobot.datasets import io_utils as _lerobot_io_utils
from lerobot.datasets import lerobot_dataset as _lerobot_dataset

if not hasattr(_lerobot_utils, "hf_transform_to_torch"):
    _lerobot_utils.hf_transform_to_torch = _lerobot_io_utils.hf_transform_to_torch

if not getattr(_lerobot_dataset.LeRobotDataset, "_official_lingbot_api_compat", False):
    _original_dataset_init = _lerobot_dataset.LeRobotDataset.__init__

    def _dataset_init(self, *args, **kwargs):
        _original_dataset_init(self, *args, **kwargs)
        # LingBot's v2 wrapper checks this legacy name before querying the
        # v3 delta-timestamp implementation.
        self.delta_indices = {
            key: [round(float(delta) * float(self.meta.fps)) for delta in deltas]
            for key, deltas in (self.delta_timestamps or {}).items()
        }
        self.episode_data_index = {
            "from": torch.as_tensor(self.meta.episodes["dataset_from_index"]),
            "to": torch.as_tensor(self.meta.episodes["dataset_to_index"]),
        }

    def _get_query_indices(self, idx, ep_idx):
        ep_start = int(self.episode_data_index["from"][ep_idx])
        ep_end = int(self.episode_data_index["to"][ep_idx])
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in deltas]
            for key, deltas in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) or (idx + delta >= ep_end) for delta in deltas]
            )
            for key, deltas in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(self, current_ts, query_indices=None):
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = [
                    float(self.hf_dataset.select(query_indices[key])["timestamp"][i])
                    for i in range(len(query_indices[key]))
                ]
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    _lerobot_dataset.LeRobotDataset.__init__ = _dataset_init
    _lerobot_dataset.LeRobotDataset._get_query_indices = _get_query_indices
    _lerobot_dataset.LeRobotDataset._get_query_timestamps = _get_query_timestamps
    _lerobot_dataset.LeRobotDataset._official_lingbot_api_compat = True

# Optional, read-only sampler trace for parity replay.  This is deliberately
# installed only by the official compatibility launcher and is disabled by
# default, so production training has no extra I/O or changed sampling path.
if os.environ.get("LINGBOT_OFFICIAL_SAMPLE_TRACE"):
    _trace_path = os.environ["LINGBOT_OFFICIAL_SAMPLE_TRACE"]
    _trace_rank = int(os.environ.get("RANK", "0"))
    _trace_count = 0
    _trace_file = open(_trace_path, "a", encoding="utf-8")
    _trace_original_getitem = _lerobot_dataset.LeRobotDataset.__getitem__

    def _trace_getitem(self, idx):
        global _trace_count
        item = _trace_original_getitem(self, idx)
        row = {
            "rank": _trace_rank,
            "call": _trace_count,
            "dataset_index": int(idx),
        }
        _trace_count += 1
        for key in ("episode_index", "frame_index", "timestamp", "index"):
            value = item.get(key) if isinstance(item, dict) else None
            if hasattr(value, "item"):
                value = value.item()
            if value is not None:
                row[key] = value
        # The v2 wrapper may hide metadata in a nested observation mapping.
        metadata = item.get("metadata") if isinstance(item, dict) else None
        if isinstance(metadata, dict):
            for key in ("episode_index", "frame_index", "timestamp"):
                value = metadata.get(key)
                if hasattr(value, "item"):
                    value = value.item()
                if value is not None:
                    row[key] = value
        _trace_file.write(json.dumps(row, sort_keys=True) + "\n")
        _trace_file.flush()
        return item

    _lerobot_dataset.LeRobotDataset.__getitem__ = _trace_getitem

    # LingBot wraps LeRobotDataset in VLADataset and overrides __getitem__;
    # trace at that boundary so the actual sampler index is retained.
    from lingbotvla.data.vla_data.base_dataset import VLADataset as _OfficialVLADataset

    _vla_original_getitem = _OfficialVLADataset.getitem

    def _trace_vla_getitem(self, idx):
        global _trace_count
        try:
            raw = self.dataset.hf_dataset[idx]
            row = {
                "rank": _trace_rank,
                "call": _trace_count,
                "dataset_index": int(idx),
            }
            for key in ("episode_index", "frame_index", "timestamp", "index"):
                value = raw.get(key) if isinstance(raw, dict) else None
                if hasattr(value, "item"):
                    value = value.item()
                if value is not None:
                    row[key] = value
            _trace_file.write(json.dumps(row, sort_keys=True) + "\n")
            _trace_file.flush()
        except Exception as exc:
            # Tracing must never alter official training behavior.
            if os.environ.get("LINGBOT_COMPAT_DEBUG") == "1":
                print(f"OFFICIAL_SAMPLE_TRACE_ERROR {exc!r}", flush=True)
        _trace_count += 1
        return _vla_original_getitem(self, idx)

    _OfficialVLADataset.getitem = _trace_vla_getitem

# Some Transformers lazy-module imports reinitialize the public namespace;
# repeat the alias immediately before executing the official entry point.
if not hasattr(transformers, "AutoModelForVision2Seq"):
    setattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
_transformers_getattr = getattr(transformers, "__getattr__", None)

def _official_transformers_getattr(name):
    if name == "AutoModelForVision2Seq":
        return transformers.AutoModel
    if _transformers_getattr is None:
        raise AttributeError(name)
    return _transformers_getattr(name)

transformers.__getattr__ = _official_transformers_getattr

# Python's import-from fast path in Transformers 5.x bypasses the module
# dictionary for a missing lazy export.  Intercept only this legacy symbol so
# the official (read-only) LingBot loader can import without modifying it.
_official_import = builtins.__import__

def _official_import_compat(name, globals=None, locals=None, fromlist=(), level=0):
    module = _official_import(name, globals, locals, fromlist, level)
    if name == "transformers" and fromlist and "AutoModelForVision2Seq" in fromlist:
        setattr(module, "AutoModelForVision2Seq", module.AutoModel)
    return module

builtins.__import__ = _official_import_compat
if os.environ.get("LINGBOT_COMPAT_DEBUG") == "1":
    print("OFFICIAL_COMPAT_VISION2SEQ", "AutoModelForVision2Seq" in transformers.__dict__, flush=True)

sys.argv[0] = "/ceph/teamGolden/andyzeng/VLA/lingbot-vla-v2/tasks/vla/train_lingbotvla.py"
runpy.run_path(sys.argv[0], run_name="__main__")
