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

from typing import Dict, MutableMapping, Optional, Tuple

import torch
import torch.distributed as dist

CFG_PARALLEL_PREDICT = 1
CFG_PARALLEL_STOP = 2

# Tensor order is part of the communication protocol. Both ranks must execute
# collectives in exactly this order.
CFG_PARALLEL_TENSOR_KEYS = (
    'images',
    'lang_tokens',
    'lang_masks',
    'states',
    'embodiment_ids',
)
CFG_PARALLEL_REQUIRED_TENSOR_KEYS = CFG_PARALLEL_TENSOR_KEYS[:4]

_MAX_TENSOR_NDIM = 5
_HEADER_CONTROL_VALUES = 3
_HEADER_VALUES_PER_TENSOR = 3 + _MAX_TENSOR_NDIM
CFG_PARALLEL_HEADER_SIZE = (
    _HEADER_CONTROL_VALUES +
    len(CFG_PARALLEL_TENSOR_KEYS) * _HEADER_VALUES_PER_TENSOR)

_DTYPE_TO_CODE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
    torch.float64: 4,
    torch.int8: 5,
    torch.uint8: 6,
    torch.int16: 7,
    torch.int32: 8,
    torch.int64: 9,
    torch.bool: 10,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _select_predict_tensors(
        predict_kwargs: Dict,
        require_cuda: bool = True) -> Dict[str, torch.Tensor]:
    missing = [
        key for key in CFG_PARALLEL_REQUIRED_TENSOR_KEYS
        if key not in predict_kwargs
    ]
    if missing:
        raise KeyError('Missing required DreamZero CFG tensor input(s): '
                       f'{missing}.')

    tensors = {}
    expected_device = None
    for key in CFG_PARALLEL_TENSOR_KEYS:
        if key not in predict_kwargs:
            continue
        tensor = predict_kwargs[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f'DreamZero CFG input {key!r} must be a tensor, '
                            f'got {type(tensor).__name__}.')
        if tensor.layout != torch.strided:
            raise ValueError(f'DreamZero CFG input {key!r} must use strided '
                             f'layout, got {tensor.layout}.')
        if tensor.ndim > _MAX_TENSOR_NDIM:
            raise ValueError(
                f'DreamZero CFG input {key!r} has {tensor.ndim} dimensions; '
                f'the protocol supports at most {_MAX_TENSOR_NDIM}.')
        if tensor.dtype not in _DTYPE_TO_CODE:
            raise TypeError(f'DreamZero CFG input {key!r} has unsupported '
                            f'dtype {tensor.dtype}.')
        if require_cuda and tensor.device.type != 'cuda':
            raise ValueError(f'DreamZero CFG input {key!r} must already be '
                             f'on CUDA, got {tensor.device}.')
        if expected_device is None:
            expected_device = tensor.device
        elif tensor.device != expected_device:
            raise ValueError('All DreamZero CFG tensor inputs must be on the '
                             f'same device, got {expected_device} and '
                             f'{tensor.device}.')
        tensors[key] = tensor.contiguous()
    return tensors


def build_dreamzero_cfg_header(
    command: int,
    predict_kwargs: Optional[Dict] = None,
    device: torch.device | str = 'cpu',
) -> torch.Tensor:
    """Build the fixed-size control and tensor-metadata header."""
    if command not in (CFG_PARALLEL_PREDICT, CFG_PARALLEL_STOP):
        raise ValueError(f'Unknown DreamZero CFG command: {command}.')
    if command == CFG_PARALLEL_STOP:
        if predict_kwargs:
            raise ValueError('The DreamZero CFG stop command has no payload.')
        values = [CFG_PARALLEL_STOP, 0, -1]
        values.extend([0] * (CFG_PARALLEL_HEADER_SIZE - len(values)))
        return torch.tensor(values, dtype=torch.int64, device=device)

    if predict_kwargs is None:
        raise ValueError('The DreamZero CFG predict command requires inputs.')
    tensors = _select_predict_tensors(predict_kwargs, require_cuda=False)
    reset_history = int(bool(predict_kwargs.get('reset_history', False)))
    num_inference_steps = predict_kwargs.get('num_inference_steps', -1)
    if num_inference_steps is None:
        num_inference_steps = -1
    if not isinstance(num_inference_steps, int):
        raise TypeError('num_inference_steps must be an int or None, got '
                        f'{type(num_inference_steps).__name__}.')

    values = [CFG_PARALLEL_PREDICT, reset_history, num_inference_steps]
    for key in CFG_PARALLEL_TENSOR_KEYS:
        tensor = tensors.get(key)
        if tensor is None:
            values.extend([0] * _HEADER_VALUES_PER_TENSOR)
            continue
        shape = list(tensor.shape)
        values.extend([1, _DTYPE_TO_CODE[tensor.dtype], tensor.ndim])
        values.extend(shape)
        values.extend([0] * (_MAX_TENSOR_NDIM - tensor.ndim))
    assert len(values) == CFG_PARALLEL_HEADER_SIZE
    return torch.tensor(values, dtype=torch.int64, device=device)


def decode_dreamzero_cfg_header(
    header: torch.Tensor,
) -> Tuple[int, bool, Optional[int], Dict[str, Tuple[torch.dtype,
                                                     torch.Size]]]:
    """Decode a header after its small CUDA-to-host synchronization."""
    if header.dtype != torch.int64 or header.numel(
    ) != CFG_PARALLEL_HEADER_SIZE:
        raise ValueError('Invalid DreamZero CFG header: expected '
                         f'{CFG_PARALLEL_HEADER_SIZE} int64 values, got '
                         f'dtype={header.dtype}, numel={header.numel()}.')
    values = header.tolist()
    command = int(values[0])
    if command not in (CFG_PARALLEL_PREDICT, CFG_PARALLEL_STOP):
        raise ValueError(f'Unknown DreamZero CFG command: {command}.')
    reset_history = bool(values[1])
    raw_num_inference_steps = int(values[2])
    num_inference_steps = (None if raw_num_inference_steps == -1 else
                           raw_num_inference_steps)

    specs = {}
    offset = _HEADER_CONTROL_VALUES
    for key in CFG_PARALLEL_TENSOR_KEYS:
        present = int(values[offset])
        dtype_code = int(values[offset + 1])
        ndim = int(values[offset + 2])
        raw_shape = values[offset + 3:offset + 3 + _MAX_TENSOR_NDIM]
        offset += _HEADER_VALUES_PER_TENSOR
        if not present:
            continue
        if dtype_code not in _CODE_TO_DTYPE:
            raise ValueError(f'Unsupported dtype code {dtype_code} for '
                             f'DreamZero CFG input {key!r}.')
        if ndim < 0 or ndim > _MAX_TENSOR_NDIM:
            raise ValueError(f'Invalid ndim {ndim} for DreamZero CFG input '
                             f'{key!r}.')
        shape = torch.Size(int(dim) for dim in raw_shape[:ndim])
        if any(dim < 0 for dim in shape):
            raise ValueError(f'Invalid shape {tuple(shape)} for DreamZero CFG '
                             f'input {key!r}.')
        specs[key] = (_CODE_TO_DTYPE[dtype_code], shape)

    if command == CFG_PARALLEL_PREDICT:
        missing = [
            key for key in CFG_PARALLEL_REQUIRED_TENSOR_KEYS
            if key not in specs
        ]
        if missing:
            raise ValueError(
                'DreamZero CFG predict header is missing required '
                f'tensor input(s): {missing}.')
    elif specs:
        raise ValueError('DreamZero CFG stop header must not contain tensors.')
    return command, reset_history, num_inference_steps, specs


def send_dreamzero_cfg_predict(predict_kwargs: Dict, src: int = 0) -> None:
    """Broadcast required predict inputs directly from source CUDA tensors."""
    if dist.get_rank() != src:
        raise RuntimeError('send_dreamzero_cfg_predict must run on the source '
                           f'rank {src}.')
    tensors = _select_predict_tensors(predict_kwargs, require_cuda=True)
    device = tensors['images'].device
    header = build_dreamzero_cfg_header(
        CFG_PARALLEL_PREDICT, predict_kwargs, device=device)
    dist.broadcast(header, src=src)
    for key in CFG_PARALLEL_TENSOR_KEYS:
        if key in tensors:
            dist.broadcast(tensors[key], src=src)


def send_dreamzero_cfg_stop(src: int = 0) -> None:
    """Tell the CFG worker to leave its receive loop."""
    if dist.get_rank() != src:
        raise RuntimeError('send_dreamzero_cfg_stop must run on the source '
                           f'rank {src}.')
    device = torch.device('cuda', torch.cuda.current_device())
    header = build_dreamzero_cfg_header(CFG_PARALLEL_STOP, device=device)
    dist.broadcast(header, src=src)


def receive_dreamzero_cfg_command(
    receive_buffers: MutableMapping[str, torch.Tensor],
    src: int = 0,
) -> Tuple[int, Optional[Dict]]:
    """Receive one direct-CUDA predict payload or a stop command."""
    if dist.get_rank() == src:
        raise RuntimeError('receive_dreamzero_cfg_command must run on a '
                           'non-source rank.')
    device = torch.device('cuda', torch.cuda.current_device())
    header = torch.empty(
        CFG_PARALLEL_HEADER_SIZE, dtype=torch.int64, device=device)
    dist.broadcast(header, src=src)
    command, reset_history, num_inference_steps, specs = (
        decode_dreamzero_cfg_header(header))
    if command == CFG_PARALLEL_STOP:
        return command, None

    predict_kwargs = {}
    for key in CFG_PARALLEL_TENSOR_KEYS:
        if key not in specs:
            continue
        dtype, shape = specs[key]
        buffer = receive_buffers.get(key)
        if (buffer is None or buffer.device != device or buffer.dtype != dtype
                or buffer.shape != shape):
            buffer = torch.empty(shape, dtype=dtype, device=device)
            receive_buffers[key] = buffer
        dist.broadcast(buffer, src=src)
        predict_kwargs[key] = buffer
    predict_kwargs['reset_history'] = reset_history
    if num_inference_steps is not None:
        predict_kwargs['num_inference_steps'] = num_inference_steps
    return command, predict_kwargs


def set_model_cfg_parallel(vla, enabled: bool) -> None:
    """Propagate CFG parallel mode to VLA heads that support it."""
    vla_head = getattr(vla, 'vla_head', None)
    if vla_head is not None:
        setattr(vla_head, 'cfg_parallel', bool(enabled))


def validate_dreamzero_cfg_parallel(enabled: bool, model_family: str,
                                    world_size: int) -> None:
    """Validate the currently supported DreamZero CFG parallel eval mode."""
    if not enabled:
        return
    if world_size != 2:
        raise ValueError('eval.cfg_parallel=True requires exactly 2 ranks, '
                         f'got world_size={world_size}.')
    if model_family != 'dreamzero':
        raise ValueError('eval.cfg_parallel=True is currently only '
                         'implemented for DreamZero.')
