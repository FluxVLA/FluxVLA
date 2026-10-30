"""Compatibility Megatron-Core training backend for FluxVLA.

The first implementation keeps FluxVLA's proven model and training loop and
uses Megatron-Core for process-group topology plus the data-parallel group.
Tensor/pipeline sharding is deliberately rejected until a model-specific
PI0.5 bridge is enabled; silently running replicated parameters under a TP
configuration would produce incorrect training semantics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from ..utils.overwatch import initialize_overwatch
from ..utils.root import RUNNERS
from .ddp_train_runner import DDPTrainRunner
from .megatron_runtime import MegatronRuntime, get_optional_parallel_config
from .megatron_parallel import ShardedOptimizer, apply_tensor_parallel

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class MegatronTrainRunner(DDPTrainRunner):
    """FluxVLA runner using Megatron-Core's process topology.

    ``tensor_model_parallel_size`` and ``pipeline_model_parallel_size`` are
    TP partitions eligible linear layers after checkpoint loading and routes
    gradient synchronization through Megatron's DP group.  The optional
    ``distributed_optimizer`` flag enables ZeRO-1 style optimizer-state
    sharding over that same DP group.
    """

    def __init__(self, *args, **kwargs) -> None:
        runner_cfg = kwargs.copy()
        self.parallel_config = get_optional_parallel_config(runner_cfg)
        if self.parallel_config.pipeline_model_parallel_size != 1:
            raise NotImplementedError(
                'PI0.5 pipeline parallel bridge is not enabled yet; use '
                'pipeline_model_parallel_size=1 for the compatibility backend.')
        if self.parallel_config.context_parallel_size != 1:
            raise NotImplementedError(
                'PI0.5 context parallel bridge is not enabled yet; use '
                'context_parallel_size=1 for the compatibility backend.')
        if self.parallel_config.expert_model_parallel_size != 1:
            raise NotImplementedError(
                'PI0.5 expert parallel bridge is not enabled yet; use '
                'expert_model_parallel_size=1 for the compatibility backend.')
        self.distributed_optimizer = bool(
            runner_cfg.get('distributed_optimizer', False))
        self.tensor_parallel_min_out_features = int(
            runner_cfg.get('tensor_parallel_min_out_features', 128))
        self.megatron_runtime: Optional[MegatronRuntime] = None
        # PI0.5 recipes are shared with FSDP and carry FSDP-only fields.
        # Consume those and the MCore topology fields before delegating to
        # the compatible DDP lifecycle and checkpoint implementation.
        for key in ('tensor_model_parallel_size',
                    'pipeline_model_parallel_size', 'context_parallel_size',
                    'expert_model_parallel_size', 'sharding_strategy',
                    'change_key_name', 'distributed_optimizer',
                    'tensor_parallel_min_out_features'):
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)

    def run_setup(self, n_train_examples: int) -> None:
        self.megatron_runtime = MegatronRuntime(self.parallel_config)
        self.megatron_runtime.initialize()
        self.ddp_process_group = self.megatron_runtime.data_parallel_group()
        try:
            super().run_setup(n_train_examples)
        except Exception:
            self.megatron_runtime.destroy()
            self.megatron_runtime = None
            raise

        if overwatch.is_rank_zero():
            overwatch.info(
                'Megatron-Core compatibility backend initialized: '
                f'{self.megatron_runtime.summary()} (replicated PI0.5 model)')

    def _prepare_backend_model(self) -> None:
        if self.parallel_config.tensor_model_parallel_size <= 1:
            return
        count = apply_tensor_parallel(
            self.vla, self.megatron_runtime.tensor_model_parallel_group(),
            minimum_out_features=self.tensor_parallel_min_out_features)
        if overwatch.is_rank_zero():
            overwatch.info('Megatron TP enabled: partitioned %d Linear modules', count)

    def _setup_optimizer_and_scheduler(self, n_train_examples: int) -> None:
        super()._setup_optimizer_and_scheduler(n_train_examples)
        if self.distributed_optimizer:
            self.optimizer = ShardedOptimizer(self.optimizer,
                                               self.ddp_process_group)
            self.lr_scheduler.bind_optimizer(self.optimizer)
            if overwatch.is_rank_zero():
                overwatch.info('Megatron distributed optimizer enabled: ZeRO-1 DP sharding')

    def _reinit_optimizer(self):
        """Rebuild optimizer after a state mismatch, preserving ZeRO-1."""
        super()._reinit_optimizer()
        if self.distributed_optimizer:
            self.optimizer = ShardedOptimizer(self.optimizer,
                                               self.ddp_process_group)
            self.lr_scheduler.bind_optimizer(self.optimizer)

    def cleanup(self) -> None:
        try:
            super().cleanup()
        finally:
            if self.megatron_runtime is not None:
                self.megatron_runtime.destroy()
                self.megatron_runtime = None


@RUNNERS.register_module()
class MegatronNativeTrainRunner(MegatronTrainRunner):
    """Native MCore provider runner used by phase-one recipes.

    The provider owns MCore-native linear modules while FluxVLA still owns the
    multimodal loss and dataloader contract.  This is the migration boundary
    for adding MCore transformer blocks and pipeline stages in phase two.
    """

    def __init__(self, *args, **kwargs):
        runner_cfg = kwargs.copy()
        self.sequence_parallel = bool(runner_cfg.get('sequence_parallel', False))
        kwargs.pop('sequence_parallel', None)
        super().__init__(*args, **kwargs)

    def _prepare_backend_model(self) -> None:
        from fluxvla.mcore import build_pi05_mcore_provider

        self.vla = build_pi05_mcore_provider(
            self.vla,
            self.megatron_runtime,
            sequence_parallel=self.sequence_parallel,
            minimum_out_features=self.tensor_parallel_min_out_features,
        )
        if overwatch.is_rank_zero():
            overwatch.info(
                'Native MCore PI0.5 provider enabled: '
                f'{self.vla.converted_linear_count} TP linear modules, '
                f'SP={self.sequence_parallel}')
