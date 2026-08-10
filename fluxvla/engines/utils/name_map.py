# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .heterogeneous_runtime import is_heterogeneous_torch_distributed_error

_DTYPE_MAP = {
    'fp32': torch.float32,
    'float32': torch.float32,
    'fp16': torch.float16,
    'float16': torch.float16,
    'bf16': torch.bfloat16,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'uint8': torch.uint8,
    'bool': torch.bool,
}

_STATE_DICT_TYPE_ATTRS = {
    'full_state_dict': 'FULL_STATE_DICT',
    'local_state_dict': 'LOCAL_STATE_DICT',
    'sharded_state_dict': 'SHARDED_STATE_DICT',
}


def _get_state_dict_type_cls():
    try:
        from torch.distributed.fsdp import StateDictType
    except ImportError as exc:
        if not is_heterogeneous_torch_distributed_error(exc):
            raise
        raise RuntimeError(
            'FSDP state-dict types are unavailable in this torch build'
        ) from exc
    return StateDictType


def str_to_dtype(s: str):
    try:
        return _DTYPE_MAP[s.lower()]
    except KeyError as exc:
        raise ValueError(f'Unsupported dtype string: {s}') from exc


def state_dict_type_map(s: str):
    state_dict_type_cls = _get_state_dict_type_cls()
    try:
        return getattr(state_dict_type_cls, _STATE_DICT_TYPE_ATTRS[s])
    except KeyError as exc:
        raise ValueError(f'Unsupported state dict type: {s}') from exc
