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
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines.utils import check_bloat16_supported
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import worker_init_function
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
        reduce_in_full_precision (bool, optional): Reduce in full precision.
            Defaults to True.
        mixed_precision_dtype (str, optional): Data type for mixed
            precision training. Defaults to 'bf16'.
        sharding_strategy (str, optional): Sharding strategy for
            distributed training. Defaults to 'full-shard'.
    """

    def __init__(self,
                 cfg: dict,
                 device_id: int,
                 collator: Dict,
                 sampler: str,
                 metric: Dict,
                 optimizer: Optional[Dict] = None,
                 max_epochs: int = None,
                 max_steps: Optional[int] = None,
                 save_epoch_interval: int = 1,
                 save_iter_interval: int = 10000,
                 max_keep_ckpts: int = 2,
                 lr_scheduler: Dict = None,
                 enable_gradient_checkpointing: bool = True,
                 enable_mixed_precision_training: bool = True,
                 reduce_in_full_precision: bool = True,
                 mixed_precision_dtype: str = 'bf16',
                 grad_accumulation_steps: int = 1,
                 evaluator: Optional[Dict] = None,
                 tokenizer: Optional[Dict] = None,
                 resume_from: Optional[str] = None):
        from ..utils.builder import (build_collator_from_cfg,
                                     build_metric_from_cfg, build_vla_from_cfg)

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
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.per_device_batch_size = cfg.train_dataloader.per_device_batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
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
            assert check_bloat16_supported(), \
                'BFloat16 is not supported on this hardware; unset `mixed_precision`'  # noqa: E501

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
        checkpoint_info = torch.load(self.resume_from)

        # Restore model state (delegated to subclasses for FSDP/DDP-specific
        # handling)
        if 'model' in checkpoint_info:
            self._load_model_state(checkpoint_info['model'])

        # Restore training state
        if 'global_step' in checkpoint_info:
            self.metric.global_step = checkpoint_info['global_step']
        if 'epoch' in checkpoint_info:
            self.current_epoch = checkpoint_info['epoch']

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
                    if overwatch.is_rank_zero():
                        overwatch.warning(
                            'Failed to load optimizer state. '
                            'Training will continue with fresh optimizer '
                            'state.')
            except Exception as e:
                if overwatch.is_rank_zero():
                    overwatch.warning(
                        f'Error loading optimizer state: {e}. '
                        f'Training will continue with fresh optimizer state.')
            finally:
                # Clean up temporary instance variable
                self._current_checkpoint_info = None
                # Ensure all ranks synchronize even if loading failed
                # This prevents deadlock if some ranks succeed and others fail
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
                overwatch.warning(f'Failed to load scheduler state: {e}')

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Resumed training from step {self.metric.global_step}, '
                f'epoch {self.current_epoch}')
        dist.barrier()

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

    @staticmethod
    def _build_dataloader_generator() -> Optional[torch.Generator]:
        seed = os.environ.get('EXPERIMENT_GLOBAL_SEED')
        if seed is None:
            return None

        generator = torch.Generator()
        generator.manual_seed(int(seed) + overwatch.rank())
        return generator

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

    def _get_log_lr(self) -> float:
        if hasattr(self.lr_scheduler, 'get_log_lr'):
            return self.lr_scheduler.get_log_lr(self)
        return self.lr_scheduler.get_last_lr()[0]

    def run(self, vla_dataset, eval_dataset=None) -> None:
        """Train the VLA model."""
        training_eval_dataset = (
            vla_dataset if eval_dataset is None else eval_dataset)
        # Setup dataloader
        sampler = torch.utils.data.distributed.DistributedSampler(
            vla_dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
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

        # Dispatch to training mode specific loop
        self.vla.train()
        self.optimizer.zero_grad()
        self._active_dataloader = dataloader

        try:
            if self.training_mode == 'step_based':
                return self._run_step_based(dataloader, sampler,
                                            training_eval_dataset)
            else:
                return self._run_epoch_based(dataloader, sampler,
                                             training_eval_dataset)
        finally:
            self._shutdown_dataloader(self._active_dataloader)
            self._active_dataloader = None
            gc.collect()

    def _next_batch(self, dataloader, dataloader_iter, sampler):
        """Fetch a micro-batch, restarting the epoch if the iterator ends."""
        while True:
            if dataloader_iter is None:
                if sampler:
                    sampler.set_epoch(self.current_epoch)
                dataloader_iter = iter(dataloader)

            try:
                return next(dataloader_iter), dataloader_iter
            except StopIteration:
                self.current_epoch += 1
                dataloader_iter = None

    def _run_accumulated_training_step(self, dataloader, dataloader_iter,
                                       sampler):
        """Run one optimizer step made of one or more micro-batches."""
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
        mean_loss = torch.stack(losses).mean()
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
                progress.set_description(self.metric.push(), refresh=False)
                progress.update()
                if (self.evaluator is not None
                        and self.evaluator.should_run(self)):
                    self.evaluator.run(self, training_eval_dataset)

                # Save checkpoint
                if self._should_save_step_checkpoint():
                    self._save_and_sync()

                # For infinite dataloaders: check epoch boundary by step count
                if (self.steps_per_epoch
                        and epoch_step_count >= self.steps_per_epoch):
                    self.current_epoch += 1
                    dataloader_iter = None

        return self._get_checkpoint_path()

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
                if sampler:
                    sampler.set_epoch(self.current_epoch)

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

        module = self.vla.module if hasattr(self.vla, 'module') else self.vla
        signature = inspect.signature(module.forward)
        accepts = key in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values())
        cache[key] = accepts
        self._vla_accepts_kwarg_cache = cache
        return accepts

    @staticmethod
    def _collect_output_loss_metrics(output) -> Dict[str, torch.Tensor]:
        """Collect scalar loss components returned by a VLA forward pass."""
        if not hasattr(output, 'items'):
            return {}

        metrics = {}
        reserved_keys = {'loss'}
        for key, value in output.items():
            if key in reserved_keys:
                continue
            if not (key.startswith('loss_') or key.endswith('_loss')):
                continue
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = value
        return metrics

    def _training_step(self, batch, should_step: bool = True) -> torch.Tensor:
        """Execute single training step: forward, backward, optimize."""
        self.lr_scheduler.prepare_step(self)
        batch = self._prepare_batch(
            batch, self.device_id, self.mixed_precision_dtype
            if self.enable_mixed_precision_training else None)
        if ('sample_weight' in batch
                and not self._vla_accepts_kwarg('sample_weight')):
            batch = dict(batch)
            batch.pop('sample_weight')
        with torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision_training):
            output: CausalLMOutputWithPast = self.vla(**batch)
            loss = output['loss']
            loss_metrics = self._collect_output_loss_metrics(output)

        self.metric.commit(loss=loss, **loss_metrics)
        (loss / self.grad_accumulation_steps).backward()

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
        self.clip_grad_norm()
        try:
            self.optimizer.step()
        except RuntimeError as e:
            if 'size' in str(e).lower() or 'shape' in str(e).lower():
                self._reinit_optimizer()
                self.optimizer.step()
            else:
                raise
        self.lr_scheduler.step(self)
        self.optimizer.zero_grad()

        # Custom hook for subclasses
        if hasattr(self, '_custom_training_step'):
            custom_loss = self._custom_training_step(batch, output, loss)
            if custom_loss is not None:
                loss = loss.detach().new_tensor(custom_loss)

        return loss

    def _reinit_optimizer(self):
        """Reinitialize optimizer on state mismatch."""
        if overwatch.is_rank_zero():
            overwatch.warning('Optimizer state mismatch. Reinitializing.')
        last_lrs = self.lr_scheduler.get_last_lr()
        self.optimizer = self.lr_scheduler.build_optimizer(self)
        for group, lr in zip(self.optimizer.param_groups, last_lrs):
            group['lr'] = lr
        self.lr_scheduler.bind_optimizer(self.optimizer)
        self.lr_scheduler.prepare_step(self)
        self.optimizer_state_loaded = False

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
