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


class _MegatronNativeLifecycleRunner(DDPTrainRunner):
    """Internal native-MCore lifecycle shared by the registered runner.

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
        self.optimizer_bucket_cap_mb = float(
            runner_cfg.get('optimizer_bucket_cap_mb', 25.0))
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
                    'tensor_parallel_min_out_features',
                    'optimizer_bucket_cap_mb'):
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)

    def run_setup(self, n_train_examples: int) -> None:
        self.megatron_runtime = MegatronRuntime(self.parallel_config)
        self.megatron_runtime.initialize()
        self.ddp_process_group = self.megatron_runtime.data_parallel_group()
        self.global_batch_size = (self.per_device_batch_size *
                                  self.megatron_runtime.parallel_state
                                  .get_data_parallel_world_size() *
                                  self.grad_accumulation_steps)
        try:
            super().run_setup(n_train_examples)
        except Exception:
            self.megatron_runtime.destroy()
            self.megatron_runtime = None
            raise

        if overwatch.is_rank_zero():
            backend_name = (
                'native MCore'
                if isinstance(self, MegatronNativeTrainRunner)
                else 'Megatron-Core compatibility')
            overwatch.info(
                f'{backend_name} backend initialized: '
                f'{self.megatron_runtime.summary()}')

    def _data_parallel_world_size(self) -> int:
        if self.megatron_runtime is not None:
            return self.megatron_runtime.parallel_state.get_data_parallel_world_size()
        return super()._data_parallel_world_size()

    def _data_parallel_rank(self) -> int:
        if self.megatron_runtime is not None:
            return self.megatron_runtime.parallel_state.get_data_parallel_rank()
        return super()._data_parallel_rank()

    def _reduce_training_loss_for_metrics(self, loss: torch.Tensor) -> torch.Tensor:
        """Log the DP-global mean instead of rank-local microbatch loss."""
        if self.megatron_runtime is None:
            return loss
        group = self.megatron_runtime.data_parallel_group()
        if dist.get_world_size(group) <= 1:
            return loss
        reduced = loss.detach().float().clone()
        dist.all_reduce(reduced, group=group)
        reduced /= dist.get_world_size(group)
        return reduced.to(dtype=loss.dtype)

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
                                               self.ddp_process_group,
                                               bucket_cap_mb=self.optimizer_bucket_cap_mb)
            self.lr_scheduler.bind_optimizer(self.optimizer)
            if overwatch.is_rank_zero():
                overwatch.info(
                    'Megatron distributed optimizer enabled: ZeRO-1 DP '
                    f'sharding, bucket_count={self.optimizer.bucket_count}, '
                    f'bucket_cap_mb={self.optimizer_bucket_cap_mb:g}')

    def _reinit_optimizer(self):
        """Rebuild optimizer after a state mismatch, preserving ZeRO-1."""
        super()._reinit_optimizer()
        if self.distributed_optimizer:
            self.optimizer = ShardedOptimizer(self.optimizer,
                                               self.ddp_process_group,
                                               bucket_cap_mb=self.optimizer_bucket_cap_mb)
            self.lr_scheduler.bind_optimizer(self.optimizer)

    def clip_grad_norm(self):
        """Clip against the norm of the logical, unsharded PI0.5 model."""
        if self.max_grad_norm is None:
            return None

        tp_group = self.megatron_runtime.tensor_model_parallel_group()
        tp_rank = dist.get_rank(tp_group)
        total_sq = torch.zeros((), device=self.device_id, dtype=torch.float32)
        parameters = []
        for parameter in self.vla.parameters():
            if parameter.grad is None:
                continue
            parameters.append(parameter)
            is_tp_sharded = bool(
                getattr(parameter, '_tp_sharded', False)
                or getattr(parameter, 'tensor_model_parallel', False))
            if is_tp_sharded or tp_rank == 0:
                grad = parameter.grad.detach().float()
                total_sq += torch.sum(grad * grad)

        dist.all_reduce(total_sq, group=tp_group)
        total_norm = total_sq.sqrt()
        clip_coefficient = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coefficient < 1:
            for parameter in parameters:
                parameter.grad.mul_(clip_coefficient.to(parameter.grad.dtype))
        return total_norm

    def _log_gradient_norm_breakdown(self, top_k: int = 20) -> None:
        """Log logical parameter norms after combining TP weight shards."""
        tp_group = self.megatron_runtime.tensor_model_parallel_group()
        tp_rank = dist.get_rank(tp_group)
        norms = []
        for name, parameter in self.vla.named_parameters():
            if parameter.grad is None:
                local_sq = torch.zeros(
                    (), device=self.device_id, dtype=torch.float32)
            else:
                is_tp_sharded = bool(
                    getattr(parameter, '_tp_sharded', False)
                    or getattr(parameter, 'tensor_model_parallel', False))
                if is_tp_sharded or tp_rank == 0:
                    grad = parameter.grad.detach().float()
                    local_sq = torch.sum(grad * grad)
                else:
                    local_sq = torch.zeros(
                        (), device=parameter.device, dtype=torch.float32)
            dist.all_reduce(local_sq, group=tp_group)
            if tp_rank == 0:
                canonical_name = name.replace('.layer.weight', '.weight') \
                    .replace('.layer.bias', '.bias')
                norms.append((canonical_name, local_sq.sqrt().item()))
        if tp_rank == 0:
            norms.sort(key=lambda item: item[1], reverse=True)
            overwatch.info('TP gradient norm breakdown (top %d): %s', top_k,
                           norms[:top_k])

    def cleanup(self) -> None:
        try:
            super().cleanup()
        finally:
            if self.megatron_runtime is not None:
                self.megatron_runtime.destroy()
                self.megatron_runtime = None


@RUNNERS.register_module()
class MegatronNativeTrainRunner(_MegatronNativeLifecycleRunner):
    """Native MCore provider runner used by phase-one recipes.

    The provider owns MCore-native linear modules while FluxVLA still owns the
    multimodal loss and dataloader contract.  This is the migration boundary
    for adding MCore transformer blocks and pipeline stages in phase two.
    """

    def __init__(self, *args, **kwargs):
        runner_cfg = kwargs.copy()
        # TP conversion is safe on CPU and must happen before the device move
        # to avoid a full-model plus shard peak allocation.
        self.prepare_backend_on_cpu = True
        self.sequence_parallel = bool(runner_cfg.get('sequence_parallel', False))
        self.tensor_parallel_strategy = str(
            runner_cfg.get('tensor_parallel_strategy', 'gemma_mlp'))
        kwargs.pop('sequence_parallel', None)
        kwargs.pop('tensor_parallel_strategy', None)
        super().__init__(*args, **kwargs)

    def _prepare_backend_model(self) -> None:
        raise RuntimeError(
            'The legacy post-hoc Megatron runner was removed. Use the single '
            'native entrypoint scripts/train_mcore_pi05.py, which owns the '
            'MCore PI0.5 model and Megatron pretrain lifecycle.')
