import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from safetensors import safe_open


@dataclass
class ModelConfig:
    path: Union[str, list[str], None] = None
    model_id: Optional[str] = None
    origin_file_pattern: Union[str, list[str], None] = None
    local_model_path: Optional[str] = None
    state_dict: Optional[Dict[str, torch.Tensor]] = None

    def check_input(self):
        if self.path is None and self.model_id is None:
            raise ValueError("ModelConfig requires either `path` or (`model_id`, `origin_file_pattern`).")

    def parse_original_file_pattern(self):
        if self.origin_file_pattern in [None, "", "./"]:
            return "*"
        if isinstance(self.origin_file_pattern, list):
            return self.origin_file_pattern
        if self.origin_file_pattern.endswith("/"):
            return self.origin_file_pattern + "*"
        return self.origin_file_pattern

    def resolve_local_model_path(self):
        if os.environ.get("DIFFSYNTH_MODEL_BASE_PATH") is not None:
            self.local_model_path = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
        elif self.local_model_path is None:
            self.local_model_path = "./checkpoints"

    def find_local_matches(self):
        origin_file_pattern = self.parse_original_file_pattern()
        local_root = os.path.join(self.local_model_path, self.model_id)
        if isinstance(origin_file_pattern, list):
            matches = []
            missing_patterns = []
            for pattern in origin_file_pattern:
                pattern_matches = glob.glob(os.path.join(local_root, pattern))
                pattern_matches.sort()
                if len(pattern_matches) == 0:
                    missing_patterns.append(pattern)
                matches.extend(pattern_matches)
            return matches, missing_patterns

        matches = glob.glob(os.path.join(local_root, origin_file_pattern))
        matches.sort()
        missing_patterns = [] if len(matches) > 0 else [origin_file_pattern]
        return matches, missing_patterns

    def require_local_files(self):
        matches, missing_patterns = self.find_local_matches()
        if len(missing_patterns) == 0:
            return matches

        local_root = os.path.join(self.local_model_path, self.model_id)
        raise FileNotFoundError(
            "Missing required local model weights. "
            f"model_id={self.model_id!r}, base_path={self.local_model_path!r}, "
            f"expected_root={local_root!r}, "
            f"missing_patterns={missing_patterns!r}. "
            "FastWAM does not auto-download model weights; place the files under "
            "./checkpoints or set DIFFSYNTH_MODEL_BASE_PATH to the local "
            "checkpoint root."
        )

    def resolve_local_path(self):
        self.check_input()
        self.resolve_local_model_path()
        if self.path is None:
            if self.origin_file_pattern in [None, "", "./"]:
                self.require_local_files()
                self.path = os.path.join(self.local_model_path, self.model_id)
            else:
                matches = self.require_local_files()
                self.path = matches
        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype=torch_dtype, device=device))
        return state_dict
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            value = f.get_tensor(key)
            if torch_dtype is not None:
                value = value.to(torch_dtype)
            state_dict[key] = value
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(torch_dtype)
    return state_dict


def _load_keys_dict_from_safetensors(file_path):
    keys_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            keys_dict[key] = f.get_slice(key).get_shape()
    return keys_dict


def _convert_state_dict_to_keys_dict(state_dict):
    keys_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            keys_dict[key] = list(value.shape)
        else:
            keys_dict[key] = _convert_state_dict_to_keys_dict(value)
    return keys_dict


def _load_keys_dict_from_bin(file_path):
    state_dict = load_state_dict_from_bin(file_path)
    return _convert_state_dict_to_keys_dict(state_dict)


def _load_keys_dict(file_path):
    if isinstance(file_path, list):
        merged = {}
        for path in file_path:
            merged.update(_load_keys_dict(path))
        return merged
    if file_path.endswith(".safetensors"):
        return _load_keys_dict_from_safetensors(file_path)
    return _load_keys_dict_from_bin(file_path)


def _convert_keys_dict_to_single_str(keys_dict, with_shape=True):
    keys = []
    for key, value in keys_dict.items():
        if isinstance(key, str):
            if isinstance(value, dict):
                keys.append(key + "|" + _convert_keys_dict_to_single_str(value, with_shape=with_shape))
            else:
                if with_shape:
                    shape = "_".join(map(str, list(value)))
                    keys.append(key + ":" + shape)
                keys.append(key)
    keys.sort()
    return ",".join(keys)


def hash_model_file(path, with_shape=True):
    keys_dict = _load_keys_dict(path)
    keys_str = _convert_keys_dict_to_single_str(keys_dict, with_shape=with_shape).encode("UTF-8")
    return hashlib.md5(keys_str).hexdigest()
