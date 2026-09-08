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

"""Minimal DreamZero TensorRT runtime wrapper.

This module intentionally imports TensorRT only when an engine is loaded. Normal
training and eval should not require the optional TensorRT Python package.
"""

import atexit
import os
from typing import Any

import torch
import torch.nn as nn


def _import_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            'TensorRT runtime is required to load a DreamZero TRT engine. '
            'Install tensorrt or unset LOAD_TRT_ENGINE / trt_engine_path.'
        ) from exc
    return trt


def _torch_dtype(trt_dtype):
    trt = _import_tensorrt()
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if hasattr(trt, 'bfloat16'):
        mapping[trt.bfloat16] = torch.bfloat16
    if trt_dtype not in mapping:
        raise TypeError(f'Unsupported TensorRT dtype: {trt_dtype}')
    return mapping[trt_dtype]


class TensorRTEngine:
    """Small wrapper around TensorRT execute_async_v3."""

    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f'TensorRT engine not found: {engine_path}')

        trt = _import_tensorrt()
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine: {engine_path}')
        self.context = self.engine.create_execution_context()
        self.input_meta: list[tuple[str, torch.dtype]] = []
        self.output_meta: list[tuple[str, torch.dtype]] = []
        for tensor_name in self.engine:
            dtype = _torch_dtype(self.engine.get_tensor_dtype(tensor_name))
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_meta.append((tensor_name, dtype))
            else:
                self.output_meta.append((tensor_name, dtype))

        def destroy():
            self.context = None
            self.engine = None

        atexit.register(destroy)

    def set_runtime_tensor_shape(self, name: str, shape: torch.Size) -> None:
        self.context.set_input_shape(name, tuple(shape))

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        stream = torch.cuda.current_stream()
        references: list[torch.Tensor] = []

        for index, tensor in enumerate(args):
            name, dtype = self.input_meta[index]
            self._bind_input(name, dtype, tensor, references)

        for name, dtype in self.input_meta:
            if name in kwargs:
                self._bind_input(name, dtype, kwargs[name], references)

        outputs = {}
        for name, dtype in self.output_meta:
            runtime_shape = tuple(self.context.get_tensor_shape(name))
            output = torch.empty(runtime_shape, dtype=dtype, device=references[0].device)
            self.context.set_tensor_address(name, output.data_ptr())
            references.append(output)
            outputs[name] = output

        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs

    def _bind_input(
        self,
        name: str,
        dtype: torch.dtype,
        tensor: torch.Tensor,
        references: list[torch.Tensor],
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'TensorRT input {name} must be a tensor, got {type(tensor)}')
        if not tensor.is_cuda:
            raise ValueError(f'TensorRT input {name} must be CUDA tensor')
        if tensor.dtype != dtype:
            raise TypeError(
                f'TensorRT input {name} expects {dtype}, got {tensor.dtype}')
        runtime_shape = tuple(self.context.get_tensor_shape(name))
        if runtime_shape != tuple(tensor.shape):
            raise ValueError(
                f'TensorRT input {name} shape mismatch: engine {runtime_shape}, '
                f'input {tuple(tensor.shape)}')
        tensor = tensor.contiguous()
        self.context.set_tensor_address(name, tensor.data_ptr())
        references.append(tensor)

    __call__ = forward


class WanTrtModelAr14B(nn.Module):
    """Runtime wrapper for DreamZero AR 14B TensorRT engines."""

    def __init__(self, engine_path: str):
        super().__init__()
        self.engine = TensorRTEngine(engine_path)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        kv_cache: list[torch.Tensor],
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_cache_packed = torch.stack(kv_cache, dim=0)

        self.engine.set_runtime_tensor_shape('x', x.shape)
        self.engine.set_runtime_tensor_shape('timestep', timestep.shape)
        self.engine.set_runtime_tensor_shape('context', context.shape)
        self.engine.set_runtime_tensor_shape('kv_cache_packed', kv_cache_packed.shape)
        self.engine.set_runtime_tensor_shape('y', y.shape)
        self.engine.set_runtime_tensor_shape('clip_feature', clip_feature.shape)
        self.engine.set_runtime_tensor_shape('action', action.shape)
        self.engine.set_runtime_tensor_shape('timestep_action', timestep_action.shape)
        self.engine.set_runtime_tensor_shape('state', state.shape)

        output = self.engine(
            x.to(torch.float16),
            timestep.to(torch.float16),
            context.to(torch.float16),
            kv_cache_packed.to(torch.float16),
            y.to(torch.float16),
            clip_feature.to(torch.float16),
            action.to(torch.float16),
            timestep_action.to(torch.float16),
            state.to(torch.float16),
        )
        if 'out.0' in output:
            video_pred = output['out.0']
            action_pred = output['out.1']
        else:
            video_pred = output['video_noise_pred']
            action_pred = output['action_noise_pred']
        return video_pred.to(torch.bfloat16).contiguous(), action_pred.to(torch.bfloat16).contiguous()


def load_tensorrt_engine(engine_path: str, model_type: str = 'ar_14B') -> nn.Module:
    if model_type not in ('ar_14B', 'ar_14B_droid'):
        raise ValueError(f'Unsupported DreamZero TensorRT model type: {model_type}')
    return WanTrtModelAr14B(engine_path)


def describe_tensorrt_engine(engine: Any) -> str:
    return engine.__class__.__name__
