"""Optional Megatron-Core runtime used by the FluxVLA compatibility backend.

Megatron-LM is intentionally not vendored or modified.  The package is
resolved from ``PYTHONPATH`` at runtime and is imported lazily so existing
DDP/FSDP installations remain unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist


@dataclass
class MegatronParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    @property
    def model_parallel_size(self) -> int:
        return (self.tensor_model_parallel_size *
                self.pipeline_model_parallel_size *
                self.context_parallel_size * self.expert_model_parallel_size)


class MegatronRuntime:
    """Own Megatron-Core process-group lifecycle without changing its source."""

    def __init__(self, config: MegatronParallelConfig) -> None:
        self.config = config
        self.parallel_state = None
        self._initialized_here = False

    def initialize(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                'MegatronRuntime requires an initialized torch.distributed '
                'process group. Launch FluxVLA with torchrun.')
        if self.config.model_parallel_size < 1:
            raise ValueError('Megatron model-parallel sizes must be positive.')
        world_size = dist.get_world_size()
        if world_size % self.config.model_parallel_size != 0:
            raise ValueError(
                'world_size must be divisible by tensor * pipeline * context '
                '* expert parallel sizes: '
                f'{world_size} % {self.config.model_parallel_size} != 0')

        try:
            from megatron.core import parallel_state
        except ImportError as exc:
            raise ImportError(
                'Megatron backend requires Megatron-LM on PYTHONPATH, for '
                'example: export PYTHONPATH=/ceph/teamGolden/andyzeng/VLA/'
                'Megatron-LM:$PYTHONPATH') from exc

        self.parallel_state = parallel_state
        already_initialized = bool(parallel_state.is_initialized())
        if already_initialized:
            return
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.config.tensor_model_parallel_size,
            pipeline_model_parallel_size=
            self.config.pipeline_model_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
            expert_model_parallel_size=self.config.expert_model_parallel_size)
        self._initialized_here = True

    def data_parallel_group(self):
        if self.parallel_state is None:
            raise RuntimeError('MegatronRuntime.initialize() was not called.')
        return self.parallel_state.get_data_parallel_group()

    def tensor_model_parallel_group(self):
        """Return MCore's TP process group for backend model wrappers."""
        if self.parallel_state is None:
            raise RuntimeError('MegatronRuntime.initialize() was not called.')
        return self.parallel_state.get_tensor_model_parallel_group()

    def summary(self) -> str:
        if self.parallel_state is None:
            return 'uninitialized'
        ps = self.parallel_state
        return (
            f'DP={ps.get_data_parallel_world_size()} '
            f'TP={ps.get_tensor_model_parallel_world_size()} '
            f'PP={ps.get_pipeline_model_parallel_world_size()} '
            f'CP={ps.get_context_parallel_world_size()}')

    def destroy(self) -> None:
        if self._initialized_here and self.parallel_state is not None:
            self.parallel_state.destroy_model_parallel()
            self._initialized_here = False


def get_optional_parallel_config(cfg) -> MegatronParallelConfig:
    """Read runner fields while accepting plain dict/ConfigDict objects."""
    def get(name: str, default: int) -> int:
        value = cfg.get(name, default) if hasattr(cfg, 'get') else getattr(
            cfg, name, default)
        return int(value)

    return MegatronParallelConfig(
        tensor_model_parallel_size=get('tensor_model_parallel_size', 1),
        pipeline_model_parallel_size=get('pipeline_model_parallel_size', 1),
        context_parallel_size=get('context_parallel_size', 1),
        expert_model_parallel_size=get('expert_model_parallel_size', 1))
