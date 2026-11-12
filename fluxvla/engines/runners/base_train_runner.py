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

import contextlib
import gc
import inspect
import json
import math
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines.utils import check_bf16_supported
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import (
    configure_deterministic_training, worker_init_function)
from ..utils import (build_evaluator_from_cfg, build_lr_scheduler_from_cfg,
                     build_tokenizer_from_cfg, initialize_overwatch)

overwatch = initialize_overwatch(__name__)


class BaseTrainRunner(ABC):
    """Basic class for training VLA models.
    This class is designed to be subclassed and should not be used directly.

    Args:
        cfg (dict): Configuration dictionary containing model and training
            settings.
        stage (str): Stage of training (e.g., 'vla-train', 'vla-train').
        device_id (int): Device ID for training.
        epochs (int): Number of epochs to train.
        max_steps (int): Maximum number of training steps.
        collator (Dict): Collator configuration.
        save_iter_interval (int, optional): Interval for saving checkpoints
            based on iterations. Defaults to 10000.
        save_epoch_interval (int, optional): Interval for saving checkpoints
            based on epochs. Defaults to 10000.
        max_keep_ckpts (int, optional): Maximum number of checkpoints to keep.
            Defaults to 2.
        optimizer (Dict): Optimizer configuration.
        lr_scheduler (Dict): Learning rate scheduler policy configuration.
        enable_gradient_checkpointing (bool, optional): Enable gradient
            checkpointing. Defaults to True.
        enable_mixed_precision_training (bool, optional): Enable mixed
            precision training. Defaults to True.
        enable_forward_autocast (bool, optional): Control forward autocast
            independently from distributed mixed-precision policies. When
            omitted it follows ``enable_mixed_precision_training``.
        reduce_in_full_precision (bool, optional): Reduce in full precision.
            Defaults to True.
        mixed_precision_dtype (str, optional): Data type for mixed
            precision training. Defaults to 'bf16'.
        batch_dtype (str, optional): Explicit dtype for floating model inputs.
            When omitted, the legacy parameter-policy-derived dtype is used.
        keep_params_fp32 (bool, optional): Keep model parameters and floating
            batch inputs in FP32 by default while autocast controls compute
            precision. ``batch_dtype`` can override the input side.
        deterministic_algorithms (bool, optional): Require deterministic
            PyTorch/CUDA kernels, including SDPA backward. Defaults to False.
        dataset_sharding_strategy (str, optional): Dataset sharding strategy.
            Defaults to 'round_robin'.
        sharding_strategy (str, optional): Sharding strategy for
            distributed training. Defaults to 'full-shard'.
    """

    def __init__(self,
                 cfg: dict,
                 device_id: int,
                 collator: Dict,
                 sampler: str,
                 metric: Dict,
                 dataset_sharding_strategy: str = 'round_robin',
                 optimizer: Optional[Dict] = None,
                 max_epochs: int = None,
                 max_steps: Optional[int] = None,
                 save_epoch_interval: int = 1,
                 save_iter_interval: int = 10000,
                 max_keep_ckpts: int = 2,
                 lr_scheduler: Dict = None,
                 enable_gradient_checkpointing: bool = True,
                 enable_mixed_precision_training: bool = True,
                 enable_forward_autocast: Optional[bool] = None,
                 reduce_in_full_precision: bool = True,
                 mixed_precision_dtype: str = 'bf16',
                 batch_dtype: Optional[str] = None,
                 keep_params_fp32: bool = False,
                 grad_accumulation_steps: int = 1,
                 deterministic_algorithms: bool = False,
                 ema_decay: Optional[float] = None,
                 seed: Optional[int] = None,
                 evaluator: Optional[Dict] = None,
                 tokenizer: Optional[Dict] = None,
                 resume_from: Optional[str] = None,
                 train_task: str = 'vla',
                 mixed_training: Optional[Dict] = None):
        from ..utils.builder import (build_collator_from_cfg,
                                     build_metric_from_cfg, build_vla_from_cfg)

        configure_deterministic_training(deterministic_algorithms)
        if train_task not in ('vla', 'vlm', 'mixed'):
            raise ValueError('runner.train_task must be vla, vlm or mixed.')
        self.train_task = train_task
        self._vlm_step_totals = None
        self._mixed_training = None
        if train_task == 'mixed':
            from .mixed_training import MixedTraining
            if max_steps is None or max_epochs is not None:
                raise ValueError('Mixed training requires max_steps and '
                                 'max_epochs=None.')
            self._mixed_training = MixedTraining(cfg, **(mixed_training or {}))
        elif mixed_training is not None:
            raise ValueError('mixed_training requires train_task="mixed".')

        per_device_batch_size = cfg.train_dataloader.per_device_batch_size
        grad_accumulation_steps = int(grad_accumulation_steps)
        assert grad_accumulation_steps >= 1, \
            'Gradient accumulation steps must be >= 1!'

        metric = metric.copy()
        metric['hparams'] = cfg
        metric['grad_accumulation_steps'] = grad_accumulation_steps
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        metric['run_id'] = (
            f"{os.path.basename(cfg.filename).replace('.py', '')}_{timestamp}")
        self.metric = build_metric_from_cfg(metric)

        # Ensure only one training mode is set
        assert max_steps is None or max_epochs is None, \
            'Only one of `max_steps` or `max_epochs` can be set!'
        assert max_steps is not None or max_epochs is not None, \
            'One of `max_steps` or `max_epochs` must be set!'

        # Determine training mode
        self.training_mode = 'step_based' if max_steps is not None else 'epoch_based'  # noqa: E501

        self.vla = build_vla_from_cfg(cfg.model)
        if train_task in ('vlm', 'mixed') and keep_params_fp32:
            # Promote before from_pretrained: Pi05 initializes RMSNorm in
            # BF16, which would otherwise round the source FP32 weights.
            self.vla.float()
        self.vla.validate_training_task(train_task)
        self.all_module_keys = self.vla.all_module_keys
        if self.vla.llm_backbone is not None:
            self.llm_transformer_layer_cls = self.vla.llm_backbone.transformer_layer_cls  # noqa: E501
        elif (self.vla.vlm_backbone is not None
              and hasattr(self.vla.vlm_backbone, 'transformer_layer_cls')):
            self.llm_transformer_layer_cls = self.vla.vlm_backbone.transformer_layer_cls  # noqa: E501
        else:
            self.llm_transformer_layer_cls = None

        optimizer_cfg = self._normalize_optimizer_cfg(optimizer)

        self.device_id = device_id
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.optimizer_cfg = optimizer_cfg
        self.collator = build_collator_from_cfg(collator)
        self.sampler = sampler
        self.save_iter_interval = save_iter_interval
        self.save_epoch_interval = save_epoch_interval
        self.max_keep_ckpts = max_keep_ckpts
        self.lr_scheduler_cfg = lr_scheduler
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.enable_forward_autocast = (
            bool(enable_mixed_precision_training)
            if enable_forward_autocast is None else bool(enable_forward_autocast))
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.batch_dtype = (
            None if batch_dtype is None else str_to_dtype(batch_dtype))
        self.keep_params_fp32 = bool(keep_params_fp32)
        self.deterministic_algorithms = bool(deterministic_algorithms)
        self.per_device_batch_size = per_device_batch_size
        self.dataset_sharding_strategy = dataset_sharding_strategy
        self.grad_accumulation_steps = grad_accumulation_steps
        if ema_decay is not None and not 0.0 < float(ema_decay) < 1.0:
            raise ValueError('ema_decay must be in (0, 1) when provided.')
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        self.seed = None if seed is None else int(seed)
        self._ema_params = None
        self.evaluator = (
            build_evaluator_from_cfg(evaluator)
            if evaluator is not None else None)
        self.global_batch_size = self.per_device_batch_size * \
            overwatch.world_size() * self.grad_accumulation_steps
        if hasattr(cfg.train_dataloader, 'per_device_num_workers'):
            self.per_device_num_workers = cfg.train_dataloader.per_device_num_workers  # noqa: E501
        else:
            self.per_device_num_workers = 0
        if tokenizer is not None:
            self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        else:
            self.tokenizer = None

        # Initialize training state
        self.current_epoch = 0
        # Number of micro-batches already consumed in ``current_epoch``.
        # Persisting this cursor is required for sampler-exact resume; epoch
        # alone is insufficient when a checkpoint is taken mid-epoch.
        self._sampler_batch_offset = 0
        self._dataloader_generator = None
        self.steps_per_epoch = None  # Determined at runtime
        # Accumulate losses for checkpoint interval averaging
        self._loss_accumulator = []

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None
        self.wandb_mode = os.environ.get('WANDB_MODE', 'online')
        self.resume_from = resume_from
        # Track if optimizer state was successfully loaded
        self.optimizer_state_loaded = False
        self.num_training_steps = None
        self._active_dataloader = None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), 'Per-device batch size must evenly divide global batch size!'

        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, \
                'Only BF16 mixed precision training is supported!'
            assert check_bf16_supported(), \
                'BFloat16 is not supported on this hardware; unset `mixed_precision`'  # noqa: E501

    def _unwrap_vla(self) -> torch.nn.Module:
        """Return the model underneath a distributed wrapper."""
        return self.vla.module if hasattr(self.vla, 'module') else self.vla

    def _prepare_training_task(self):
        self.vla.prepare_training_task(self.train_task)
        if not any(p.requires_grad for p in self.vla.parameters()):
            raise ValueError(f'No trainable parameters for {self.train_task}; '
                             'check the backbone freezing configuration.')

    def _finish_vlm_accumulation(self, output):
        """Normalize averaged distributed gradients by global target count.

        Each backward uses a token *sum*. DDP/FSDP average gradients across
        ranks; undo that average and divide once by the total target count
        across the accumulation window, before clipping or optimizer.step.
        """
        totals = self._vlm_step_totals
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
            world_size = dist.get_world_size()
        self._vlm_step_totals = None
        if totals[1].item() == 0:
            self.optimizer.zero_grad()
            raise ValueError('VLM optimizer step has no supervised tokens.')
        scale = world_size / totals[1]
        for parameter in self.vla.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(parameter.grad.dtype))
        loss = (totals[0] / totals[1]).to(output['loss'].dtype)
        self.metric.commit(
            loss=loss,
            vlm_loss=loss,
            token_accuracy=(totals[2] / totals[1]).float(),
            supervised_tokens=totals[1].float())
        return loss

    def _save_vlm_assets(self, dataset):
        if self.train_task not in ('vlm',
                                   'mixed') or not overwatch.is_rank_zero():
            return
        asset_dir = Path(self.metric.run_dir) / 'vlm_assets'
        asset_dir.mkdir(parents=True, exist_ok=True)
        model = self._unwrap_vla()
        backbone = model.vlm_backbone
        native = (
            backbone.vlm if backbone is not None else getattr(
                model.llm_backbone, 'llm', model.llm_backbone))
        native.config.save_pretrained(asset_dir)
        if getattr(native, 'generation_config', None) is not None:
            native.generation_config.save_pretrained(asset_dir)
        for transform in getattr(dataset, 'transforms', []):
            save_assets = getattr(transform, 'save_pretrained', None)
            if callable(save_assets):
                save_assets(asset_dir)
        if self.tokenizer is not None:
            tokenizer = getattr(self.tokenizer, 'tokenizer', self.tokenizer)
            if callable(getattr(tokenizer, 'save_pretrained', None)):
                tokenizer.save_pretrained(asset_dir)

    def can_run_training_eval(self) -> bool:
        """Return whether the underlying VLA exposes a training-eval hook."""
        unwrapped_vla = self._unwrap_vla()
        training_eval_fn = getattr(unwrapped_vla, 'compute_training_eval',
                                   None)
        return callable(training_eval_fn)

    def run_training_eval(
        self,
        batch: Dict[str, Any],
        num_inference_steps: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Execute a training-eval hook for an unsharded model.

        Distributed strategies that require a specialized execution path
        override this method. DDP keeps complete parameters on every rank, so
        calling the underlying model hook directly is safe.

        Args:
            batch: Collated evaluation batch.
            num_inference_steps: Number of diffusion inference steps.
            seed: Evaluation random seed.

        Returns:
            Training-evaluation output dictionary.
        """
        unwrapped_vla = self._unwrap_vla()
        training_eval_fn = getattr(unwrapped_vla, 'compute_training_eval',
                                   None)
        if not callable(training_eval_fn):
            raise RuntimeError(
                f'{unwrapped_vla.__class__.__name__} does not implement '
                '`compute_training_eval`.')
        return training_eval_fn(
            batch,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

    @staticmethod
    def _normalize_optimizer_cfg(optimizer: Optional[Dict]) -> Dict:
        if optimizer is None:
            raise ValueError('runner.optimizer must be provided.')
        optimizer_cfg = dict(optimizer)
        optimizer_type = optimizer_cfg.get('type', 'AdamW')
        if 'lr' not in optimizer_cfg:
            raise ValueError('optimizer.lr must be provided.')

        normalized_cfg = dict(optimizer_cfg)
        normalized_cfg['type'] = optimizer_type
        normalized_cfg['lr'] = float(normalized_cfg['lr'])

        if 'betas' in normalized_cfg:
            normalized_cfg['betas'] = tuple(
                float(beta) for beta in normalized_cfg['betas'])
            if len(normalized_cfg['betas']) != 2:
                raise ValueError(
                    'optimizer.betas must contain two values when provided.')
        if 'eps' in normalized_cfg:
            normalized_cfg['eps'] = float(normalized_cfg['eps'])
        if (normalized_cfg.get('weight_decay') is not None
                and 'weight_decay' in normalized_cfg):
            normalized_cfg['weight_decay'] = float(
                normalized_cfg['weight_decay'])
        normalized_cfg['paramwise_learning_rate'] = dict(
            normalized_cfg.get('paramwise_learning_rate', {}) or {})
        normalized_cfg['paramwise_learning_rate_regex'] = dict(
            normalized_cfg.get('paramwise_learning_rate_regex', {}) or {})
        return normalized_cfg

    def _prepare_batch(self,
                       batch: Dict,
                       device: torch.device | int,
                       dtype: Optional[torch.dtype] = None) -> Dict:
        """Move tensor batch values to device and optionally cast floats.

        Floating point tensors are cast to ``dtype`` when provided. Integer
        and bool tensors keep their dtype.

        Args:
            batch (Dict): Input batch dictionary.
            device: Target device.
            dtype (torch.dtype): Optional floating point target dtype.

        Returns:
            Dict: Batch with tensors on the target device.
        """
        converted_batch = {}
        target_device = (
            torch.device('cuda', device)
            if isinstance(device, int) else device)

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if dtype is not None and value.dtype.is_floating_point:
                    converted_batch[key] = value.to(
                        device=target_device, dtype=dtype, non_blocking=True)
                else:
                    converted_batch[key] = value.to(
                        device=target_device, non_blocking=True)
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                converted_batch[key] = self._prepare_batch(
                    value, device, dtype)
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples that may contain tensors
                converted_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        if dtype is not None and item.dtype.is_floating_point:
                            converted_list.append(
                                item.to(
                                    device=target_device,
                                    dtype=dtype,
                                    non_blocking=True))
                        else:
                            converted_list.append(
                                item.to(
                                    device=target_device, non_blocking=True))
                    elif isinstance(item, dict):
                        converted_list.append(
                            self._prepare_batch(item, device, dtype))
                    else:
                        converted_list.append(item)
                converted_batch[key] = (
                    tuple(converted_list)
                    if isinstance(value, tuple) else converted_list)
            else:
                # Keep non-tensor values as is
                converted_batch[key] = value

        return converted_batch

    def _convert_batch_to_dtype(self, batch: Dict, dtype: torch.dtype) -> Dict:
        """Convert floating point tensors in batch to specified dtype."""
        return self._prepare_batch(batch, self.device_id, dtype)

    def _prepare_model_training_batch(self, batch: Dict) -> Dict:
        """Apply an optional model-owned input contract after device casting."""
        module = self.vla.module if hasattr(self.vla, 'module') else self.vla
        prepare = getattr(module, 'prepare_training_batch', None)
        return prepare(batch) if callable(prepare) else batch

    @staticmethod
    def _shutdown_dataloader(dataloader: Optional[DataLoader]) -> None:
        """Stop DataLoader workers so they do not overlap with evaluation."""
        if dataloader is None:
            return

        iterator = getattr(dataloader, '_iterator', None)
        shutdown_fn = getattr(iterator, '_shutdown_workers', None)
        if callable(shutdown_fn):
            shutdown_fn()

    def cleanup(self) -> None:
        """Release training resources before launching evaluation."""
        if self._mixed_training is not None:
            self._mixed_training.close(self)
        self._shutdown_dataloader(self._active_dataloader)
        self._active_dataloader = None

        if self.optimizer is not None:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.optimizer.zero_grad()

        if self.vla is not None:
            try:
                self.vla.zero_grad(set_to_none=True)
            except TypeError:
                self.vla.zero_grad()

        self.optimizer = None
        self.lr_scheduler = None
        self._ema_params = None
        self.collator = None
        self.tokenizer = None
        self.vla = None
        self._loss_accumulator.clear()

        if hasattr(self, 'recent_losses'):
            self.recent_losses.clear()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, 'ipc_collect', None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except RuntimeError:
                    pass

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        """Save checkpoint including model, optimizer, and scheduler states.

        Subclasses should save:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - Global step and epoch information
        """
        ...

    @abstractmethod
    def clip_grad_norm(self):
        """Clip gradient norm. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def _load_model_state(self, checkpoint_model_state: dict) -> None:
        """Load model state from checkpoint.

        Args:
            checkpoint_model_state (dict): Model state dict from checkpoint.
        """
        ...

    @abstractmethod
    def _load_optimizer_state(self, checkpoint_optimizer_state: dict) -> bool:
        """Load optimizer state from checkpoint.

        Args:
            checkpoint_optimizer_state (dict): Optimizer state dict from
                checkpoint.

        Returns:
            bool: True if optimizer state was successfully loaded,
                False otherwise.
        """
        ...

    def resume(self) -> None:
        """Resume training from checkpoint if specified.

        This method handles:
        - Loading training state (global_step, epoch, etc.)
        - Loading optimizer state (delegated to subclasses)
        - Loading scheduler state
        - Synchronizing all ranks after resume
        """
        if self.resume_from is None:
            return

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Resuming training from checkpoint: {self.resume_from}')
        checkpoint_info = torch.load(
            self.resume_from, map_location='cpu', weights_only=True)
        mixed = getattr(self, '_mixed_training', None)
        if mixed is not None:
            required = {
                'model', 'optimizer_state_dict', 'scheduler_state_dict'
            }
            if not required.issubset(checkpoint_info):
                raise ValueError('Mixed resume needs a complete training '
                                 'checkpoint, including optimizer/scheduler.')
            mixed.load_state_dict(
                checkpoint_info.get('mixed_training'),
                checkpoint_info.get('global_step', 0))

        # Restore model state (delegated to subclasses for FSDP/DDP-specific
        # handling)
        if 'model' in checkpoint_info:
            if (getattr(self, 'ema_decay', None) is not None
                    and 'train_model' in checkpoint_info):
                self._load_model_state(checkpoint_info['model'])
                self._initialize_ema_parameters()
                self._load_model_state(checkpoint_info['train_model'])
            else:
                self._load_model_state(checkpoint_info['model'])

        # Restore training state
        if 'global_step' in checkpoint_info:
            self.metric.global_step = checkpoint_info['global_step']
        if 'epoch' in checkpoint_info:
            self.current_epoch = checkpoint_info['epoch']
        data_state = checkpoint_info.get('data_state')
        if isinstance(data_state, dict):
            self._sampler_batch_offset = int(data_state.get('batch_offset', 0))
            generator_state = data_state.get('generator_state')
            if (generator_state is not None
                    and self._dataloader_generator is not None):
                self._dataloader_generator.set_state(generator_state)
            self._restore_rng_states(data_state.get('rng_states'))

        # Restore optimizer state (delegated to subclasses)
        # Store checkpoint_info as instance variable for subclasses to access
        # additional information (e.g., parameter mappings)
        if ('optimizer_state_dict' in checkpoint_info
                and self.optimizer is not None):
            checkpoint_optimizer_state = checkpoint_info[
                'optimizer_state_dict']
            # Store checkpoint_info temporarily for _load_optimizer_state
            # to access
            self._current_checkpoint_info = checkpoint_info
            try:
                success = self._load_optimizer_state(
                    checkpoint_optimizer_state)
                if not success:
                    if mixed is not None:
                        raise RuntimeError('Cannot restore mixed optimizer.')
                    if overwatch.is_rank_zero():
                        overwatch.warning(
                            'Failed to load optimizer state. '
                            'Training will continue with fresh optimizer '
                            'state.')
            except Exception as e:
                if mixed is not None:
                    raise RuntimeError(
                        'Cannot restore mixed optimizer.') from e
                if overwatch.is_rank_zero():
                    overwatch.warning(
                        f'Error loading optimizer state: {e}. '
                        f'Training will continue with fresh optimizer state.')
            finally:
                # Clean up temporary instance variable
                self._current_checkpoint_info = None
                # Ensure all ranks synchronize even if loading failed
                # This prevents deadlock if some ranks succeed and others fail
                if dist.is_initialized():
                    dist.barrier()

        # Restore scheduler state
        if ('scheduler_state_dict' in checkpoint_info
                and self.lr_scheduler is not None):
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint_info['scheduler_state_dict'])
                if overwatch.is_rank_zero():
                    overwatch.info('Scheduler state restored from checkpoint')
            except Exception as e:
                if mixed is not None:
                    raise RuntimeError(
                        'Cannot restore mixed scheduler.') from e
                overwatch.warning(f'Failed to load scheduler state: {e}')

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Resumed training from step {self.metric.global_step}, '
                f'epoch {self.current_epoch}')
        if dist.is_initialized():
            dist.barrier()

    def _initialize_ema_parameters(self) -> None:
        """Create FP32/local-shard EMA buffers from current parameters."""
        if (getattr(self, 'ema_decay', None) is None
                or getattr(self, '_ema_params', None) is not None):
            return
        self._ema_params = {
            name: parameter.detach().clone()
            for name, parameter in self.vla.named_parameters()
        }

    @torch.no_grad()
    def _update_ema_parameters(self) -> None:
        if getattr(self, 'ema_decay', None) is None:
            return
        self._initialize_ema_parameters()
        one_minus_decay = 1.0 - self.ema_decay
        for name, parameter in self.vla.named_parameters():
            shadow = self._ema_params[name]
            if shadow.shape != parameter.shape:
                raise RuntimeError(f'EMA parameter shape changed for {name}: '
                                   f'{shadow.shape} != {parameter.shape}.')
            shadow.lerp_(parameter.detach(), one_minus_decay)

    @contextlib.contextmanager
    def _use_ema_parameters(self):
        """Temporarily expose EMA weights through the wrapped model."""
        if getattr(self, '_ema_params', None) is None:
            yield
            return
        current = {}
        with torch.no_grad():
            for name, parameter in self.vla.named_parameters():
                current[name] = parameter.detach().clone()
                parameter.copy_(self._ema_params[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.vla.named_parameters():
                    parameter.copy_(current[name])

    def _should_save_step_checkpoint(self) -> bool:
        """Check if checkpoint should be saved (step-based)."""
        return (self.metric.global_step % self.save_iter_interval) == 0

    def _should_save_epoch_checkpoint(self) -> bool:
        """Check if checkpoint should be saved (epoch-based)."""
        return (self.current_epoch % self.save_epoch_interval) == 0

    def _get_effective_dataset_size(self, dataset, sampler):
        """Get effective dataset size for finite and sampler-backed datasets.

        Args:
            dataset: The dataset object.
            sampler: The sampler used in DataLoader.
        """
        if sampler is not None:
            # Effective size after DistributedSampler processing
            return len(sampler)
        else:
            try:
                dataset_len = len(dataset)
                # If dataset has a finite length, use it
                return dataset_len // self.per_device_batch_size
            except (TypeError, AttributeError):
                return None

    def _estimate_steps_per_epoch(self, dataset, sampler):
        """Estimate steps per epoch for finite and sampler-backed datasets."""
        if sampler is not None:
            # Effective size after DistributedSampler processing
            return len(sampler)
        else:
            dataset_len = len(dataset)
            # If dataset has a finite length, use it
            return math.ceil(dataset_len / self.global_batch_size)

    def _build_dataloader_generator(self) -> Optional[torch.Generator]:
        seed = os.environ.get('EXPERIMENT_GLOBAL_SEED')
        if seed is None:
            return None

        generator = torch.Generator()
        generator.manual_seed(int(seed) + self._data_parallel_rank())
        self._dataloader_generator = generator
        return generator

    def _data_state_dict(self) -> dict:
        """Return sampler cursor and loader RNG state for checkpointing."""
        state = {'batch_offset': int(self._sampler_batch_offset)}
        if self._dataloader_generator is not None:
            state['generator_state'] = self._dataloader_generator.get_state()
        return state

    def _capture_rng_states(self):
        """Gather per-rank torch RNG states for exact checkpoint resume.

        Model code may sample dropout/noise independently on every rank.  A
        resumed process has re-created the model and therefore cannot rely on
        its freshly seeded RNG stream matching the stream after the previous
        optimizer step.
        """
        local = {
            'torch': torch.get_rng_state(),
            'cuda': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        }
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
            dist.gather_object(local, gathered, dst=0)
            return gathered
        return [local]

    def _restore_rng_states(self, states):
        """Restore the current rank's torch RNG stream from a checkpoint."""
        if not isinstance(states, (list, tuple)):
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank >= len(states) or not isinstance(states[rank], dict):
            return
        state = states[rank]
        if isinstance(state.get('torch'), torch.Tensor):
            torch.set_rng_state(state['torch'])
        cuda_state = state.get('cuda')
        if torch.cuda.is_available() and isinstance(cuda_state, (list, tuple)):
            torch.cuda.set_rng_state_all(cuda_state)

    @staticmethod
    def _tensor_for_safetensors(tensor):
        """Return a contiguous tensor for safetensors export."""
        if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
            return tensor.contiguous()
        return tensor

    @staticmethod
    def _save_model_safetensors(model_state_dicts, safetensors_path):
        """Save model weights as safetensors alongside the .pt checkpoint.

        Handles both flat state dicts and nested dicts (FSDP with
        change_key_name) by flattening to a single-level {str: tensor} dict.
        """
        flat_dict = {}
        for key, value in model_state_dicts.items():
            if isinstance(value, dict):
                for sub_key, tensor in value.items():
                    flat_dict[f'{key}.{sub_key}'] = (
                        BaseTrainRunner._tensor_for_safetensors(tensor))
            elif isinstance(value, torch.Tensor):
                flat_dict[key] = BaseTrainRunner._tensor_for_safetensors(value)
        if flat_dict:
            save_file(flat_dict, safetensors_path)

    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Clean up old checkpoint files, keeping only the most recent ones."""
        ckpt_files = sorted(
            [
                f for f in os.listdir(checkpoint_dir)
                if f.endswith('.pt') and f != 'latest-checkpoint.pt'
            ],
            key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        if len(ckpt_files) > self.max_keep_ckpts:
            for old_ckpt in ckpt_files[:-self.max_keep_ckpts]:
                try:
                    os.remove(os.path.join(checkpoint_dir, old_ckpt))
                    overwatch.info(f'Removed old checkpoint: {old_ckpt}')
                    sf_file = old_ckpt.replace('.pt', '.safetensors')
                    sf_path = os.path.join(checkpoint_dir, sf_file)
                    if os.path.exists(sf_path):
                        os.remove(sf_path)
                        overwatch.info(f'Removed old safetensors: {sf_file}')
                except Exception as e:
                    overwatch.warning(
                        f'Failed to remove checkpoint {old_ckpt}: {e}')

    def _resolve_lr_scheduler_cfg(self) -> Dict:
        if self.lr_scheduler_cfg is None:
            raise ValueError('runner.lr_scheduler must be provided.')
        return dict(self.lr_scheduler_cfg)

    def _setup_optimizer_and_scheduler(
        self,
        n_train_examples: int,
    ) -> None:
        """Setup optimizer and learning rate scheduler policy."""
        n_train_examples = math.ceil(
            n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            self.num_training_steps = (
                n_train_examples * self.max_epochs) // self.global_batch_size
        else:
            self.num_training_steps = self.max_steps

        scheduler_cfg = self._resolve_lr_scheduler_cfg()
        self.lr_scheduler = build_lr_scheduler_from_cfg(scheduler_cfg)
        self.lr_scheduler_policy_type = scheduler_cfg['type']
        self.optimizer, self.lr_scheduler = self.lr_scheduler.build(self)
        self._write_optimizer_group_trace()

    def _get_log_lr(self) -> float:
        if hasattr(self.lr_scheduler, 'get_log_lr'):
            return self.lr_scheduler.get_log_lr(self)
        return self.lr_scheduler.get_last_lr()[0]

    def run(self, vla_dataset, eval_dataset=None) -> None:
        """Train the VLA model."""
        if self._mixed_training is not None:
            return self._mixed_training.run(self, vla_dataset, eval_dataset)
        self._save_vlm_assets(vla_dataset)
        self._configure_dataset_sharding(vla_dataset)
        training_eval_dataset = (
            vla_dataset if eval_dataset is None else eval_dataset)
        set_distributed_context = getattr(
            vla_dataset, 'set_distributed_context', None)
        if callable(set_distributed_context):
            set_distributed_context(
                rank=self._data_parallel_rank(),
                world_size=self._data_parallel_world_size())
        # Setup dataloader
        sampler = torch.utils.data.distributed.DistributedSampler(
            vla_dataset,
            num_replicas=self._data_parallel_world_size(),
            rank=self._data_parallel_rank(),
            shuffle=not bool(getattr(vla_dataset, "deterministic_order", False)),
            # PyTorch defaults this to zero.  The native LingBot sampler uses
            # the training seed, so leaving the default silently changes the
            # sample trajectory even when all model/RNG settings match.
            seed=0 if self.seed is None else self.seed,
            drop_last=False) if self.sampler == 'distributed' else None

        use_workers = self.per_device_num_workers > 0
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.per_device_num_workers,
            worker_init_fn=worker_init_function,
            generator=self._build_dataloader_generator(),
            pin_memory=True,
            prefetch_factor=2 if use_workers else None,
            persistent_workers=use_workers)

        # Calculate steps per epoch
        self.steps_per_epoch = self._get_steps_per_epoch(vla_dataset)
        self._log_training_info(vla_dataset)
        self.resume()
        self._initialize_ema_parameters()

        # Dispatch to training mode specific loop
        self.vla.train()
        self.optimizer.zero_grad()
        self._active_dataloader = dataloader

        try:
            if self.training_mode == 'step_based':
                self._sync_step_based_epoch_with_global_step()
                return self._run_step_based(dataloader, sampler,
                                            training_eval_dataset)
            else:
                return self._run_epoch_based(dataloader, sampler,
                                            training_eval_dataset)
        finally:
            self._shutdown_dataloader(self._active_dataloader)
            self._active_dataloader = None
            gc.collect()

    def _configure_dataset_sharding(self, dataset: Any) -> None:
        """Configure the dataset sharding strategy before worker startup.

        Args:
            dataset: Training dataset to configure before worker startup.
        """
        strategy = self.dataset_sharding_strategy
        configure_sharding = getattr(dataset, 'configure_sharding', None)

        if strategy == 'round_robin':
            if callable(configure_sharding):
                configure_sharding(
                    strategy='round_robin',
                    batch_size=self.per_device_batch_size,
                )
        elif strategy == 'blockwise':
            if not callable(configure_sharding):
                raise TypeError(
                    '`dataset_sharding_strategy="blockwise"` requires a '
                    'dataset with `configure_sharding()` support.')
            configure_sharding(
                strategy='blockwise',
                batch_size=self.per_device_batch_size,
            )
        else:
            raise ValueError(
                '`dataset_sharding_strategy` must be either "round_robin" '
                f'or "blockwise", got {strategy!r}.')

    def _data_parallel_world_size(self) -> int:
        """World size used by data sampling; MCore runners override this."""
        return overwatch.world_size()

    def _data_parallel_rank(self) -> int:
        """Rank used by data sampling; MCore runners override this."""
        return overwatch.rank()

    def _next_batch(self, dataloader, dataloader_iter, sampler):
        """Fetch a micro-batch, restarting the iterator if it ends."""
        while True:
            if dataloader_iter is None:
                self._set_dataloader_epoch(dataloader, sampler)
                dataloader_iter = iter(dataloader)
                # Rebuild the iterator at the saved position.  The cursor is
                # measured in batches yielded by this rank's sampler.
                for _ in range(self._sampler_batch_offset):
                    try:
                        next(dataloader_iter)
                    except StopIteration:
                        self.current_epoch += 1
                        self._sampler_batch_offset = 0
                        dataloader_iter = None
                        break
                if dataloader_iter is None:
                    continue

            try:
                batch = next(dataloader_iter)
                self._sampler_batch_offset += 1
                return batch, dataloader_iter
            except StopIteration:
                # Preserve legacy VLA iterator/sampler epoch semantics.
                # Only VLM with a known step budget lets the outer loop
                # own this counter: filling its last accumulation window
                # can restart a finite loader, but must not count the same
                # epoch twice and end training early.
                if self.train_task != 'vlm' or not self.steps_per_epoch:
                    self.current_epoch += 1
                self._sampler_batch_offset = 0
                dataloader_iter = None

    def _set_dataloader_epoch(self, dataloader, sampler) -> None:
        """Synchronize sampler and iterable-dataset shuffle epochs.

        Args:
            dataloader: Training DataLoader whose dataset may expose
                ``set_epoch``.
            sampler: Optional distributed sampler.
        """
        if sampler:
            sampler.set_epoch(self.current_epoch)
        set_epoch = getattr(dataloader.dataset, 'set_epoch', None)
        if callable(set_epoch):
            set_epoch(self.current_epoch)

    def _run_accumulated_training_step(self, dataloader, dataloader_iter,
                                       sampler):
        """Run one optimizer step made of one or more micro-batches."""
        if self._mixed_training is not None:
            return self._mixed_training.step(self), dataloader_iter
        losses = []
        for micro_step in range(self.grad_accumulation_steps):
            batch, dataloader_iter = self._next_batch(dataloader,
                                                      dataloader_iter, sampler)
            should_step = micro_step == self.grad_accumulation_steps - 1
            # Skip the DDP/FSDP gradient all-reduce on non-final
            # micro-steps; gradients are synchronized only once when the
            # accumulated optimizer step is taken.
            with self._grad_sync_context(should_sync=should_step):
                loss = self._training_step(batch, should_step=should_step)
            losses.append(loss.detach())
        mean_loss = (
            losses[-1]
            if self.train_task == 'vlm' else torch.stack(losses).mean())
        return float(mean_loss.item()), dataloader_iter

    def _grad_sync_context(self, should_sync: bool):
        """Return the gradient synchronization context for a micro-step.

        For DDP/FSDP-wrapped models, ``no_sync()`` suppresses the gradient
        all-reduce during ``backward``. We only need to synchronize on the
        last micro-step of a gradient-accumulation window, which cuts the
        gradient communication volume by a factor of
        ``grad_accumulation_steps``. Falls back to a no-op context for the
        final micro-step, single-GPU runs, or any model lacking
        ``no_sync``.
        """
        if should_sync:
            return contextlib.nullcontext()
        no_sync = getattr(self.vla, 'no_sync', None)
        if callable(no_sync):
            return no_sync()
        return contextlib.nullcontext()

    def _run_step_based(self, dataloader, sampler,
                        training_eval_dataset) -> str:
        """Step-based training loop. Handles infinite dataloaders."""
        with tqdm(
                total=self.max_steps,
                desc=self.metric.get_status(),
                leave=False,
                disable=not overwatch.is_rank_zero(),
                initial=self.metric.global_step) as progress:

            dataloader_iter = None
            epoch_step_count = 0

            while self.metric.global_step < self.max_steps:
                loss, dataloader_iter = self._run_accumulated_training_step(
                    dataloader, dataloader_iter, sampler)
                self._loss_accumulator.append(loss)
                epoch_step_count += 1

                # Update metrics
                self.metric.commit(
                    global_step=self.metric.global_step + 1,
                    epoch=self.current_epoch,
                    lr=self._get_log_lr())
                if self._mixed_training is not None:
                    status = self.metric.push(
                        extra_metrics=self._mixed_training.step_metrics)
                else:
                    status = self.metric.push()
                progress.set_description(status, refresh=False)
                progress.update()
                if (self.evaluator is not None
                        and self.evaluator.should_run(self)):
                    self.evaluator.run(self, training_eval_dataset)

                # For infinite dataloaders: check epoch boundary by step count
                if (self.steps_per_epoch
                        and epoch_step_count >= self.steps_per_epoch):
                    self.current_epoch += 1
                    self._sampler_batch_offset = 0
                    epoch_step_count = 0
                    dataloader_iter = None

                # Save after advancing the data cursor.  A checkpoint records
                # the position of the *next* batch, so resume follows the
                # same epoch/cursor semantics as uninterrupted training.
                if self._should_save_step_checkpoint():
                    self._save_and_sync()

                # Parity runs must preserve the production scheduler horizon;
                # changing max_steps for a short diagnostic also changes LR.
                # Stop only after the requested update and trace have landed.
                parity_stop = os.environ.get("FLUXVLA_PARITY_STOP_AFTER_STEP")
                if (parity_stop is not None
                        and self.metric.global_step >= int(parity_stop)):
                    if overwatch.is_rank_zero():
                        overwatch.info(
                            "Parity diagnostic stopped after global step "
                            f"{self.metric.global_step}.")
                    break

        return self._get_checkpoint_path()

    def _sync_step_based_epoch_with_global_step(self) -> None:
        """Keep step-based epoch display derived from the global step.

        Step-based training does not persist dataloader iterator position, so
        after resume the least surprising epoch value is the completed number
        of full dataset passes implied by ``global_step``.
        """
        if (self.training_mode != 'step_based' or not self.steps_per_epoch
                or self.metric.global_step == 0):
            return

        expected_epoch = self.metric.global_step // self.steps_per_epoch
        if self.current_epoch == expected_epoch:
            return

        if overwatch.is_rank_zero():
            overwatch.warning(
                'Correcting step-based epoch from '
                f'{self.current_epoch} to {expected_epoch} based on '
                f'global_step={self.metric.global_step} and '
                f'steps_per_epoch={self.steps_per_epoch}.')
        self.current_epoch = expected_epoch
        # Checkpoints written before the cursor-ordering fix may contain a
        # non-zero offset from the previous epoch.  Once the epoch is derived
        # from global_step, that stale offset must not be applied again.
        self._sampler_batch_offset = 0
        if hasattr(self.metric, 'epoch'):
            self.metric.epoch = expected_epoch

    def _run_epoch_based(self, dataloader, sampler,
                         training_eval_dataset) -> str:
        """Epoch-based training with nested progress bars. Handles
            infinite dataloaders.

        Args:
            dataloader: The dataloader object.
            sampler: The sampler used in DataLoader.

        Returns:
            str: The path to the latest checkpoint.
        """
        with tqdm(
                total=self.max_epochs,
                desc='Epochs',
                leave=False,
                disable=not overwatch.is_rank_zero(),
                initial=self.current_epoch) as epoch_pbar:

            while self.current_epoch < self.max_epochs:
                self._set_dataloader_epoch(dataloader, sampler)
                dataloader_iter = iter(dataloader)
                epoch_step_count = 0
                iter_total = self.steps_per_epoch or len(dataloader)

                with tqdm(
                        total=iter_total,
                        desc=f'Epoch {self.current_epoch}',
                        leave=False,
                        disable=not overwatch.is_rank_zero()) as iter_pbar:

                    while True:
                        loss, dataloader_iter = \
                            self._run_accumulated_training_step(
                                dataloader, dataloader_iter, sampler)
                        self._loss_accumulator.append(loss)
                        epoch_step_count += 1

                        # Update metrics
                        self.metric.commit(
                            global_step=self.metric.global_step + 1,
                            epoch=self.current_epoch,
                            lr=self._get_log_lr())
                        iter_pbar.set_description(self.metric.push())
                        iter_pbar.update()
                        if (self.evaluator is not None
                                and self.evaluator.should_run(self)):
                            self.evaluator.run(self, training_eval_dataset)

                        # For infinite dataloaders: end epoch by step count
                        if (self.steps_per_epoch
                                and epoch_step_count >= self.steps_per_epoch):
                            break

                # Epoch completed
                self.current_epoch += 1
                self._sampler_batch_offset = 0
                epoch_pbar.update()

                # Save checkpoint at epoch end
                if self._should_save_epoch_checkpoint():
                    self._save_and_sync()

        return self._get_checkpoint_path()

    def _get_steps_per_epoch(self, vla_dataset) -> Optional[int]:
        """Calculate steps per epoch from dataset or estimate."""
        try:
            return math.ceil(len(vla_dataset) / self.global_batch_size)
        except (TypeError, AttributeError):
            return self._estimate_steps_per_epoch(vla_dataset, None)

    def _log_training_info(self, vla_dataset):
        """Log training configuration."""
        if not overwatch.is_rank_zero():
            return
        try:
            overwatch.info(f'Dataset length: {len(vla_dataset)}')
        except (TypeError, AttributeError):
            overwatch.info('Dataset length: unknown (infinite iterator)')
        overwatch.info(
            f'Training: mode={self.training_mode}, epochs={self.max_epochs}, '
            f'steps/epoch={self.steps_per_epoch}, '
            f'batch={self.global_batch_size} '
            f'({self.per_device_batch_size}x{overwatch.world_size()}'
            f'x{self.grad_accumulation_steps})')

    def _vla_accepts_kwarg(self, key: str) -> bool:
        """Return whether the wrapped VLA forward accepts ``key``."""
        cache = getattr(self, '_vla_accepts_kwarg_cache', {})
        if key in cache:
            return cache[key]

        module = self._unwrap_vla()
        signature = inspect.signature(module.forward)
        accepts = key in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values())
        cache[key] = accepts
        self._vla_accepts_kwarg_cache = cache
        return accepts

    @staticmethod
    def _collect_output_loss_metrics(output) -> Dict[str, torch.Tensor]:
        """Collect scalar loss components and policy diagnostics."""
        if not hasattr(output, 'items'):
            return {}

        metrics = {}
        reserved_keys = {'loss'}
        metric_aliases = {
            'action_accuracy': 'action_accuracy',
            'action_l1_loss': 'l1_loss',
        }
        for key, value in output.items():
            if key in reserved_keys:
                continue
            metric_key = metric_aliases.get(key)
            if metric_key is None and (key.startswith('loss_')
                                       or key.endswith('_loss')):
                metric_key = key
            if metric_key is None:
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[metric_key] = value
        return metrics

    def _reduce_training_metrics_for_metrics(
            self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return scalar output metrics in the backend's reporting scope."""
        return metrics

    def _training_step(self, batch, should_step: bool = True) -> torch.Tensor:
        """Execute single training step: forward, backward, optimize."""
        mixed = self._mixed_training
        task = mixed.active_task if mixed is not None else self.train_task
        if task not in ('vla', 'vlm'):
            raise ValueError('Select a task before a mixed training step.')
        memory_metrics = os.environ.get("FLUXVLA_MEMORY_METRICS") == "1"
        if memory_metrics and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
        self.lr_scheduler.prepare_step(self)
        batch = self._maybe_load_pi05_alignment_batch(batch)
        self._write_batch_identity_trace(batch)
        # OpenPI keeps observations/actions and master parameters in FP32.
        # Autocast is responsible only for BF16 transformer compute.
        batch_dtype = self.batch_dtype
        if batch_dtype is None:
            batch_dtype = (
                self.mixed_precision_dtype
                if self.enable_mixed_precision_training
                and not self.keep_params_fp32 else torch.float32)
        batch = self._prepare_batch(batch, self.device_id, batch_dtype)
        batch = self._prepare_model_training_batch(batch)
        if ('sample_weight' in batch
                and not self._vla_accepts_kwarg('sample_weight')):
            batch = dict(batch)
            batch.pop('sample_weight')
        self._maybe_capture_pi05_alignment_batch(batch)
        with torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_forward_autocast):
            if mixed is not None and task == 'vla':
                output = self.vla(
                    task='vla', return_loss_statistics=True, **batch)
            else:
                output: CausalLMOutputWithPast = (
                    self.vla(task='vlm', **batch)
                    if task == 'vlm' else self.vla(**batch))
            loss = output['loss']
            loss_metrics = self._collect_output_loss_metrics(output)

        if mixed is not None:
            loss = mixed.backward(self, output, should_step)
        elif self.train_task == 'vlm':
            totals = torch.stack([
                output[key].detach().double()
                for key in ('loss_sum', 'num_tokens', 'num_correct')
            ])
            if self._vlm_step_totals is None:
                self._vlm_step_totals = totals
            else:
                self._vlm_step_totals += totals
            output['loss_sum'].backward()
            if should_step:
                loss = self._finish_vlm_accumulation(output)
        else:
            metric_loss = self._reduce_training_loss_for_metrics(loss)
            loss_metrics = self._reduce_training_metrics_for_metrics(loss_metrics)
            self.metric.commit(loss=metric_loss, **loss_metrics)
            (loss / self.grad_accumulation_steps).backward()

        self._write_alignment_gradient_trace()
        self._write_alignment_param_trace("pre_clip")

        # Commit per-dataset metrics
        if overwatch.is_rank_zero() and all(k in output for k in [
                'action_accuracy_ds', 'action_l1_loss_ds', 'ds_names'
        ]):  # noqa: E501
            for ds, acc, l1 in zip(output['ds_names'],
                                   output['action_accuracy_ds'],
                                   output['action_l1_loss_ds']):
                self.metric.commit_for_dataset(
                    dataset_name=ds.decode(), action_accuracy=acc, l1_loss=l1)

        if not should_step:
            return loss.detach()

        # Gradient step with fallback on optimizer state mismatch
        if (os.environ.get('FLUXVLA_GRAD_NORM_DIAGNOSTICS') == '1'
                and self.metric.global_step == 0):
            self._log_gradient_norm_breakdown()
        grad_norm = self.clip_grad_norm()
        clip_coefficient = None
        capture_clip = os.environ.get("FLUXVLA_PARAM_TRACE") == "1"
        if grad_norm is not None:
            # FSDP2 returns a replicated DTensor whose local shard is not the
            # reportable global norm. All ranks must materialize it before
            # logging or deriving diagnostic clip coefficients.
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            grad_norm_value = torch.as_tensor(
                grad_norm, device=loss.device, dtype=torch.float32)
            if capture_clip and self.max_grad_norm is not None:
                clip_coefficient = min(
                    1.0,
                    float(self.max_grad_norm)
                    / (float(grad_norm_value.detach().item()) + 1e-6),
                )
            self.metric.commit(grad_norm=grad_norm_value)
            if clip_coefficient is not None:
                self.metric.commit(
                    clip_coefficient=loss.new_tensor(clip_coefficient))
        self._alignment_clip_coefficient = clip_coefficient
        try:
            self.optimizer.step()
        except RuntimeError as e:
            if mixed is not None:
                raise
            if 'size' in str(e).lower() or 'shape' in str(e).lower():
                self._reinit_optimizer()
                self.optimizer.step()
            else:
                raise
        self._update_ema_parameters()
        self.lr_scheduler.step(self)
        self._write_alignment_param_trace("post_update")
        if memory_metrics and torch.cuda.is_available():
            self.metric.commit(
                max_memory_allocated_gb=torch.as_tensor(
                    torch.cuda.max_memory_allocated(self.device_id) / (1024**3),
                    device=loss.device,
                    dtype=torch.float32))
        self.optimizer.zero_grad()

        # Custom hook for subclasses
        if hasattr(self, '_custom_training_step'):
            custom_loss = self._custom_training_step(batch, output, loss)
            if custom_loss is not None:
                loss = loss.detach().new_tensor(custom_loss)

        return loss

    def _maybe_load_pi05_alignment_batch(self, batch: Dict) -> Dict:
        """Replace an online batch with a frozen model-ready PI0.5 replay batch."""
        replay_dir = os.environ.get("PI05_ALIGNMENT_REPLAY_DIR")
        if not replay_dir:
            return batch

        import gzip

        step = int(self.metric.global_step)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        state = getattr(self, "_pi05_alignment_replay_state", None)
        if state is None or state["step"] != step:
            micro_step = 0
        else:
            micro_step = state["micro_step"] + 1
        if micro_step >= self.grad_accumulation_steps:
            raise RuntimeError(
                "PI0.5 alignment replay consumed more micro-batches than the "
                f"configured gradient accumulation ({self.grad_accumulation_steps}) "
                f"at optimizer step {step}."
            )
        self._pi05_alignment_replay_state = {
            "step": step,
            "micro_step": micro_step,
        }
        # Frozen captures are logical data-parallel streams. Reconstruct the
        # same stream ordering when a smaller physical DP group uses gradient
        # accumulation: rank r consumes r, r + DP, r + 2*DP, ... .
        source_rank = rank + micro_step * world_size
        path = Path(replay_dir) / f"step_{step:06d}_rank_{source_rank:02d}.pt.gz"
        if not path.is_file():
            raise FileNotFoundError(f"PI0.5 alignment replay batch not found: {path}")
        with gzip.open(path, "rb") as stream:
            artifact = torch.load(stream, map_location="cpu", weights_only=False)

        replay = {
            "images": artifact.get("images"),
            "img_masks": artifact.get("img_masks"),
            "lang_tokens": artifact.get("tokens"),
            "lang_masks": artifact.get("text_masks"),
            "states": artifact.get("states"),
            "actions": artifact["actions"],
            "noise": artifact.get("noise"),
            "time": artifact.get("time"),
            "action_masks": artifact.get("action_is_pad"),
        }
        return {key: value for key, value in replay.items() if value is not None}

    def _maybe_capture_pi05_alignment_batch(self, batch: Dict) -> None:
        """Capture a deterministic PI0.5 model-ready batch for backend replay.

        This is deliberately opt-in and runs immediately before the reference
        model forward.  When requested, flow noise/time are injected into the
        batch so the native MCore replay consumes exactly the same stochastic
        target.  The artifact format is the compact schema understood by
        ``FixedPI0IterableDataset``.
        """
        output_dir = os.environ.get('PI05_ALIGNMENT_CAPTURE_DIR')
        if not output_dir:
            return
        max_steps = int(os.environ.get('PI05_ALIGNMENT_CAPTURE_STEPS', '1'))
        step = int(getattr(self.metric, 'global_step', 0))
        if step >= max_steps or 'actions' not in batch:
            return
        module = self.vla.module if hasattr(self.vla, 'module') else self.vla
        accepts_flow = (
            self._vla_accepts_kwarg('noise')
            and self._vla_accepts_kwarg('time')
            and hasattr(module, 'sample_noise')
            and hasattr(module, 'sample_time')
        )
        if 'noise' not in batch and accepts_flow:
            seed = int(os.environ.get('PI05_ALIGNMENT_FLOW_SEED', '123'))
            generator = torch.Generator(device=batch['actions'].device)
            generator.manual_seed(seed + step)
            actions = batch['actions']
            batch['noise'] = torch.normal(
                mean=0.0, std=1.0, size=actions.shape, dtype=torch.float32,
                device=actions.device, generator=generator)
            # Isolate the CUDA stream used by the reference beta sampler, so
            # capturing does not perturb dropout or any later model RNG.
            previous_state = torch.cuda.get_rng_state(self.device_id)
            try:
                torch.cuda.manual_seed(seed + step + 1)
                batch['time'] = module.sample_time(actions.shape[0], actions.device)
            finally:
                torch.cuda.set_rng_state(previous_state, self.device_id)

        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, (list, tuple)):
                return type(value)(cpu(item) for item in value)
            return value

        import gzip
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get('RANK', '0'))
        path = Path(output_dir) / f'step_{step:06d}_rank_{rank:02d}.pt.gz'
        artifact = {
            'images': cpu(batch.get('images')),
            'img_masks': cpu(batch.get('img_masks')),
            'tokens': cpu(batch.get('lang_tokens', batch.get('tokens'))),
            'text_masks': cpu(batch.get('lang_masks', batch.get('text_masks'))),
            'states': cpu(batch.get('states')),
            'actions': cpu(batch['actions']),
            'noise': cpu(batch.get('noise')),
            'time': cpu(batch.get('time')),
            'action_is_pad': cpu(batch.get('action_masks', batch.get('action_is_pad'))),
        }
        with gzip.open(path, 'wb', compresslevel=1) as stream:
            torch.save(artifact, stream)

    def _reduce_training_loss_for_metrics(self, loss: torch.Tensor) -> torch.Tensor:
        """Return the loss value used for logging; backends may aggregate it."""
        return loss

    def _write_batch_identity_trace(self, batch: Dict) -> None:
        """Optionally persist stable sample identities for replay debugging."""
        if os.environ.get("FLUXVLA_BATCH_TRACE") != "1":
            return
        trace_dir = Path(
            os.environ.get("FLUXVLA_BATCH_TRACE_DIR", "work_dirs"))
        trace_dir.mkdir(parents=True, exist_ok=True)
        identity = {}
        for key in ("sample_index", "index", "episode_index", "current_index", "frame_index",
                    "timestamp"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                identity[key] = value.detach().cpu().tolist()
            elif value is not None:
                identity[key] = value
        row = {"step": int(self.metric.global_step + 1), "identity": identity}
        rank = overwatch.rank()
        path = trace_dir / f"batch_rank{rank}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_alignment_param_trace(self, phase: str) -> None:
        """Optionally record selected parameter/optimizer state summaries.

        This diagnostic is deliberately opt-in because each statistic performs
        a device synchronization.  It is intended for fixed-replay first-bad
        analysis at the optimizer boundary, not for performance measurement.
        """
        if os.environ.get("FLUXVLA_PARAM_TRACE") != "1":
            return
        pattern = os.environ.get(
            "FLUXVLA_PARAM_TRACE_PATTERN",
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"(\.layer)?\.(weight|bias)$",
        )
        matcher = re.compile(pattern)
        root = Path(os.environ.get("FLUXVLA_PARAM_TRACE_DIR", "work_dirs/param_trace"))
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        path = root / f"rank{rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        step = int(self.metric.global_step) + 1
        target_step = int(os.environ.get("FLUXVLA_PARAM_TRACE_STEP", "0"))
        if target_step and step != target_step:
            return
        snapshots = getattr(self, "_alignment_param_snapshots", None)
        if snapshots is None:
            snapshots = self._alignment_param_snapshots = {}
        rows = []
        tensor_trace_root = os.environ.get("FLUXVLA_OPTIMIZER_TENSOR_TRACE_DIR")
        materialize_full = os.environ.get("FLUXVLA_PARAM_TRACE_FULL") == "1"
        summary_only = os.environ.get("FLUXVLA_PARAM_SUMMARY_ONLY") == "1"

        def trace_value(value):
            """Materialize selected DTensors only for explicit parity tracing."""
            if value is None:
                return None
            if materialize_full and hasattr(value, "full_tensor"):
                value = value.full_tensor()
            elif hasattr(value, "to_local"):
                # Avoid perturbing clipping with diagnostic collectives. Paired
                # FSDP runs compare the same rank-local shard.
                value = value.to_local()
            return value.detach().float()

        tensor_snapshots = getattr(self, "_alignment_optimizer_tensor_snapshots", None)
        if tensor_snapshots is None:
            tensor_snapshots = self._alignment_optimizer_tensor_snapshots = {}
        tensor_rows = []
        for name, parameter in self.vla.named_parameters():
            canonical = name.removeprefix("module.").replace(".layer.", ".")
            if not matcher.search(canonical):
                continue
            grad = parameter.grad
            state = self.optimizer.state.get(parameter, {})
            parameter_trace = trace_value(parameter)
            grad_trace = trace_value(grad)
            row = {
                "step": step,
                "phase": phase,
                "name": canonical,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "parameter_norm": float(parameter_trace.norm().item()),
                "parameter_sum": float(parameter_trace.sum().item()),
                "parameter_abs_max": float(parameter_trace.abs().max().item()),
                "grad_norm": None if grad_trace is None else float(grad_trace.norm().item()),
                "grad_mean": None if grad_trace is None else float(grad_trace.mean().item()),
                "grad_sum": None if grad_trace is None else float(grad_trace.sum().item()),
                "grad_abs_max": None if grad_trace is None else float(grad_trace.abs().max().item()),
                "tp_sharded": bool(
                    getattr(parameter, "tensor_model_parallel", False)
                    or getattr(parameter, "_tp_sharded", False)
                ),
            }
            if phase == "pre_clip":
                # Keep the selected tensors on-device only for the interval
                # between pre-clip and post-update. This is diagnostic-only
                # and avoids changing the normal training memory footprint.
                if not summary_only:
                    snapshots[canonical] = parameter_trace.clone()
                if tensor_trace_root:
                    tensor_snapshots[canonical] = {
                        "parameter_before": parameter_trace.cpu().clone(),
                        "raw_grad": None
                        if grad_trace is None else grad_trace.cpu().clone(),
                    }
                    for state_name in ("exp_avg", "exp_avg_sq"):
                        value = state.get(state_name)
                        if isinstance(value, torch.Tensor):
                            tensor_snapshots[canonical][f"{state_name}_before"] = (
                                trace_value(value).cpu().clone()
                            )
            elif phase == "post_update":
                previous = snapshots.pop(canonical, None)
                parameter_bf16 = parameter_trace.to(torch.bfloat16).float()
                parameter_bf16_bits = (
                    parameter_bf16.to(torch.bfloat16)
                    .contiguous()
                    .view(torch.int16)
                    .to(torch.int64)
                    .bitwise_and(0xFFFF)
                )
                row.update({
                    "parameter_bf16_norm": float(parameter_bf16.norm().item()),
                    "parameter_bf16_sum": float(parameter_bf16.sum().item()),
                    "parameter_bf16_abs_max": float(
                        parameter_bf16.abs().max().item()),
                    "parameter_bf16_bits_sum": int(
                        parameter_bf16_bits.sum().item()),
                })
                row["parameter_delta_norm"] = (
                    None if previous is None else float(
                        (parameter_trace - previous).norm().item())
                )
                if tensor_trace_root:
                    tensor_snapshot = tensor_snapshots.pop(canonical, {})
                    tensor_snapshot.update(
                        {
                            "name": canonical,
                            "model_shape": list(parameter.shape),
                            "master_shape": list(parameter.shape),
                            "post_clip_grad": None
                            if grad is None
                            else grad_trace.cpu().clone(),
                            "parameter_after": parameter_trace.cpu().clone(),
                            "exp_avg": None,
                            "exp_avg_sq": None,
                        }
                    )
                    for state_name in ("exp_avg", "exp_avg_sq"):
                        value = state.get(state_name)
                        if isinstance(value, torch.Tensor):
                            tensor_snapshot[state_name] = trace_value(value).cpu().clone()
                    tensor_rows.append(tensor_snapshot)
            row["clip_coefficient"] = getattr(
                self, "_alignment_clip_coefficient", None)
            for state_name in ("exp_avg", "exp_avg_sq"):
                value = state.get(state_name)
                state_trace = trace_value(value)
                row[f"{state_name}_norm"] = (
                    None if state_trace is None else float(state_trace.norm().item())
                )
                row[f"{state_name}_sum"] = (
                    None if state_trace is None else float(state_trace.sum().item())
                )
                row[f"{state_name}_abs_max"] = (
                    None if state_trace is None else float(state_trace.abs().max().item())
                )
            rows.append(row)
        with path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if phase == "post_update" and tensor_trace_root:
            tensor_root = Path(tensor_trace_root)
            tensor_root.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backend": "fsdp",
                    "step": step,
                    "rank": rank,
                    "parameters": tensor_rows,
                },
                tensor_root / f"fsdp_step_{step:06d}_rank_{rank:02d}.pt",
            )

    def _write_alignment_gradient_trace(self) -> None:
        """Record deterministic per-parameter gradient summaries for X4 runs.

        This is intentionally separate from the existing parameter trace: it is
        emitted immediately after backward and before clipping, so a mismatch
        cannot be hidden by a later optimizer operation.  It is opt-in because
        iterating all parameters is synchronization-heavy.
        """
        if os.environ.get("FLUXVLA_GRAD_TRACE") != "1":
            return
        pattern = os.environ.get(
            "FLUXVLA_GRAD_TRACE_PATTERN",
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
            r"action_out_proj|action_in_proj|time_mlp|cond_proj|final_norm|norm)"
            r"(\.layer)?\.(weight|bias|dense\.weight|dense\.bias)$",
        )
        matcher = re.compile(pattern)
        root = Path(os.environ.get("FLUXVLA_GRAD_TRACE_DIR", "work_dirs/grad_trace"))
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        path = root / f"rank{rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        step = int(self.metric.global_step) + 1
        with path.open("a", encoding="utf-8") as stream:
            for name, parameter in self.vla.named_parameters():
                canonical = name.removeprefix("module.").replace(".layer.", ".")
                if not matcher.search(canonical):
                    continue
                grad = parameter.grad
                if grad is None:
                    row = {"step": step, "name": canonical, "shape": list(parameter.shape),
                           "grad": None}
                else:
                    value = grad.detach().float()
                    row = {
                        "step": step,
                        "name": canonical,
                        "shape": list(parameter.shape),
                        "grad_norm": float(value.norm().item()),
                        "grad_mean": float(value.mean().item()),
                        "grad_abs_max": float(value.abs().max().item()),
                        "grad_sum": float(value.sum().item()),
                    }
                stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _log_gradient_norm_breakdown(self, top_k: int = 20) -> None:
        """Log the largest per-parameter gradient norms for alignment runs."""
        if not overwatch.is_rank_zero():
            return
        norms = []
        for name, parameter in self.vla.named_parameters():
            if parameter.grad is not None:
                norms.append((name, parameter.grad.detach().float().norm().item()))
        norms.sort(key=lambda item: item[1], reverse=True)
        overwatch.info('Gradient norm breakdown (top %d): %s', top_k,
                       norms[:top_k])

    def _reinit_optimizer(self):
        """Reinitialize optimizer on state mismatch."""
        if overwatch.is_rank_zero():
            overwatch.warning('Optimizer state mismatch. Reinitializing.')
        last_lrs = self.lr_scheduler.get_last_lr()
        self.optimizer = self.lr_scheduler.build_optimizer(self)
        self._write_optimizer_group_trace()
        for group, lr in zip(self.optimizer.param_groups, last_lrs):
            group['lr'] = lr
        self.lr_scheduler.bind_optimizer(self.optimizer)
        self.lr_scheduler.prepare_step(self)
        self.optimizer_state_loaded = False

    def _write_optimizer_group_trace(self) -> None:
        """Write opt-in optimizer group membership for parity debugging.

        This is intentionally a metadata-only diagnostic.  It records the
        effective LR/weight decay and parameter names after FSDP wrapping, so
        expert-LR regexes can be checked without materializing optimizer
        tensors or changing the training path.
        """
        if os.environ.get("FLUXVLA_OPTIMIZER_GROUP_TRACE") != "1":
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        root = Path(
            os.environ.get(
                "FLUXVLA_OPTIMIZER_GROUP_TRACE_DIR",
                "work_dirs/optimizer_trace",
            ))
        root.mkdir(parents=True, exist_ok=True)
        names_by_id = {
            id(parameter): name.removeprefix("module.").replace(
                "._fsdp_wrapped_module.", ".")
            for name, parameter in self.vla.named_parameters()
        }
        groups = []
        for index, group in enumerate(self.optimizer.param_groups):
            names = [
                names_by_id.get(id(parameter), "<unnamed>")
                for parameter in group["params"]
            ]
            expert_names = [name for name in names if ".mlp.experts." in name]
            groups.append({
                "group": index,
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "betas": [float(value) for value in group.get("betas", ())],
                "eps": float(group.get("eps", 0.0)),
                "parameter_count": len(names),
                "expert_parameter_count": len(expert_names),
                "parameter_names": names,
            })
        payload = {
            "backend": "fsdp",
            "world_size": (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized() else 1),
            "groups": groups,
        }
        (root / "optimizer_groups.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _save_and_sync(self, loss_value: float = None):
        """Save checkpoint and synchronize.

        Uses averaged loss over the checkpoint interval if available.
        """
        # Use averaged loss if accumulated, otherwise use provided value
        if self._loss_accumulator:
            avg_loss = sum(self._loss_accumulator) / len(
                self._loss_accumulator)
            self._loss_accumulator.clear()
        else:
            avg_loss = loss_value

        self.save_checkpoint(self.metric.run_dir, self.metric.global_step,
                             self.current_epoch, avg_loss)
        dist.barrier()

    def _get_checkpoint_path(self) -> str:
        """Get latest checkpoint path."""
        return os.path.join(self.metric.run_dir, 'checkpoints',
                            'latest-checkpoint.pt')
