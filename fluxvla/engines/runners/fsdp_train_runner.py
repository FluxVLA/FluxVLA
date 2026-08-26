# Origin: Modified from
# Upstream-Repo: openvla/openvla
# Upstream-Path: prismatic/training/strategies/fsdp.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

import copy
import gc
import math
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)

from ..utils import initialize_overwatch
from ..utils.name_map import str_to_dtype
from ..utils.root import RUNNERS
from .base_train_runner import BaseTrainRunner

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class FSDPTrainRunner(BaseTrainRunner):
    """FSDP Runner for training VLMs with Fully Sharded Data Parallelism.
    This class extends the BaseTrainRunner and implements the
    setup and training process for FSDP.
    It initializes the FSDP strategy, sets up the optimizer and learning rate
    scheduler, and handles gradient checkpointing.
    It also provides a method to save checkpoints.

    Args:
        cfg (dict): Configuration dictionary for the runner.
        stage (str): Stage of training (e.g., 'vla-train', 'vla-train').
        epochs (int): Number of training epochs.
        max_steps (int): Maximum number of training steps.
        optimizer (Dict): Optimizer configuration.
        max_grad_norm (int): Maximum gradient norm for clipping.
        collator (Dict): Collator object for batching data.
        metric (Dict): Metric object for evaluation.
        save_iter_interval (int, optional): Interval for saving checkpoints
            based on iterations. Defaults to 10000.
        save_epoch_interval (int, optional): Interval for saving checkpoints
            based on epochs. Defaults to 1.
        max_keep_ckpts (int, optional): Maximum number of checkpoints to keep.
            Defaults to 2.
        lr_scheduler (Dict): Learning rate scheduler policy configuration.
        enable_gradient_checkpointing (bool, optional): Enable gradient
            checkpointing. Defaults to True.
        enable_mixed_precision_training (bool, optional): Enable mixed
            precision training. Defaults to True.
        reduce_in_full_precision (bool, optional): Reduce in full precision.
            Defaults to True.
        mixed_precision_dtype (str, optional): Data type for mixed precision
            training.  Defaults to 'bf16'.
        keep_params_fp32 (bool, optional): Keep master parameters and floating
            batch inputs in FP32 while autocast controls compute precision.
            Defaults to False.
        deterministic_algorithms (bool, optional): Require deterministic
            PyTorch/CUDA kernels, including SDPA backward. Defaults to False.
        sharding_strategy (str, optional): Sharding strategy for FSDP.
            Defaults to 'hybrid-shard'.
        fsdp_wrap_policy (str, optional): FSDP auto-wrap granularity. ``model``
            uses the model's existing policy, ``execution-block`` uses an
            execution-aware policy supplied by the model, and ``root`` wraps
            only the root module. Defaults to ``model``.
    """

    def __init__(self,
                 cfg: dict,
                 max_grad_norm: int,
                 collator: Dict,
                 sampler: str,
                 metric: Dict,
                 dataset_sharding_strategy: str = 'round_robin',
                 optimizer: Optional[Dict] = None,
                 max_epochs: int = None,
                 max_steps: int = None,
                 save_epoch_interval: int = 1,
                 save_iter_interval: int = 10000,
                 max_keep_ckpts: int = 2,
                 lr_scheduler: Optional[Dict] = None,
                 enable_gradient_checkpointing: bool = True,
                 enable_mixed_precision_training: bool = True,
                 reduce_in_full_precision: bool = True,
                 mixed_precision_dtype: str = 'bf16',
                 keep_params_fp32: bool = False,
                 grad_accumulation_steps: int = 1,
                 deterministic_algorithms: bool = False,
                 ema_decay: Optional[float] = None,
                 seed: Optional[int] = None,
                 evaluator: Optional[Dict] = None,
                 sharding_strategy: str = 'hybrid-shard',
                 fsdp_wrap_policy: str = 'model',
                 pre_fsdp_param_dtype: Optional[str] = None,
                 change_key_name: bool = False,
                 tokenizer: Optional[Dict] = None,
                 resume_from: Optional[str] = None,
                 args=None,
                 *_unused_args,
                 **kwargs) -> None:
        if kwargs:
            fields = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected runner config field(s): {fields}')
        device_id = overwatch.local_rank()
        super().__init__(
            cfg=cfg,
            device_id=device_id,
            collator=collator,
            sampler=sampler,
            metric=metric,
            dataset_sharding_strategy=dataset_sharding_strategy,
            optimizer=optimizer,
            max_epochs=max_epochs,
            max_steps=max_steps,
            save_epoch_interval=save_epoch_interval,
            save_iter_interval=save_iter_interval,
            max_keep_ckpts=max_keep_ckpts,
            lr_scheduler=lr_scheduler,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            keep_params_fp32=keep_params_fp32,
            grad_accumulation_steps=grad_accumulation_steps,
            deterministic_algorithms=deterministic_algorithms,
            ema_decay=ema_decay,
            seed=seed,
            evaluator=evaluator,
            tokenizer=tokenizer,
            resume_from=resume_from)
        self.cfg = cfg
        self.args = args
        self.max_grad_norm = max_grad_norm
        self.sharding_strategy = sharding_strategy
        self.pre_fsdp_param_dtype = (
            str_to_dtype(pre_fsdp_param_dtype)
            if pre_fsdp_param_dtype is not None else None)
        if self.sharding_strategy == 'global-shard-grad-op':
            # Use the default global process group. Unlike the private
            # hybrid Zero2 strategy below, this does not create one extra
            # inter-node communicator per local rank.
            self.fsdp_sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif self.sharding_strategy in ('shard-grad-op', 'hybrid-shard-zero2'):
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif self.sharding_strategy == 'full-shard':
            self.fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self.sharding_strategy == 'hybrid-shard':
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        elif self.sharding_strategy == 'no-shard':
            self.fsdp_sharding_strategy = ShardingStrategy.NO_SHARD
        else:
            raise ValueError(
                f'FSDP Sharding Strategy {sharding_strategy} is not supported!'
            )
        if fsdp_wrap_policy not in ('model', 'execution-block', 'root'):
            raise ValueError(
                'FSDP wrap policy must be one of model, execution-block, '
                f'or root, got {fsdp_wrap_policy!r}.')
        self.fsdp_wrap_policy = fsdp_wrap_policy
        self.change_key_name = change_key_name
        self.fsdp_state_dict_type = StateDictType.FULL_STATE_DICT
        self.fsdp_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True)

    @classmethod
    def _move_checkpoint_tensors_to_cpu(cls, value):
        """Build a CPU checkpoint tree without mutating live state."""
        if isinstance(value, torch.Tensor):
            if value.device.type == 'cpu':
                return value
            return value.detach().cpu()
        if isinstance(value, dict):
            converted = copy.copy(value)
            converted.clear()
            for key, item in value.items():
                converted[key] = cls._move_checkpoint_tensors_to_cpu(item)
            if hasattr(value, '_metadata'):
                converted._metadata = cls._move_checkpoint_tensors_to_cpu(
                    value._metadata)
            return converted
        if isinstance(value, list):
            return [
                cls._move_checkpoint_tensors_to_cpu(item) for item in value
            ]
        if isinstance(value, tuple):
            converted = tuple(
                cls._move_checkpoint_tensors_to_cpu(item) for item in value)
            if hasattr(value, '_fields'):
                return type(value)(*converted)
            if type(value) is not tuple:
                try:
                    return type(value)(converted)
                except TypeError:
                    pass
            return converted
        return value

    def _format_model_state_dict(self, state_dict):
        if not self.change_key_name:
            return state_dict
        formatted = {
            module_key: OrderedDict()
            for module_key in self.all_module_keys
        }
        for key, parameter in state_dict.items():
            for module_key in formatted:
                prefix = f'{module_key}.'
                if key.startswith(prefix):
                    formatted[module_key][key.removeprefix(prefix)] = parameter
        return formatted

    def _resolve_fsdp_ignored_modules(self) -> list[nn.Module]:
        """Resolve model-owned, fully frozen modules excluded from FSDP."""
        getter = getattr(self.vla, 'get_fsdp_ignored_modules', None)
        modules = list(getter()) if callable(getter) else []
        deduped = []
        seen = set()
        for module in modules:
            if not isinstance(module, nn.Module):
                raise TypeError('get_fsdp_ignored_modules() must return '
                                'torch.nn.Module instances.')
            if id(module) in seen:
                continue
            if any(param.requires_grad for param in module.parameters()):
                raise ValueError('FSDP ignored modules must be fully frozen; '
                                 f'got trainable {type(module).__name__}.')
            deduped.append(module)
            seen.add(id(module))
        return deduped

    def run_training_eval(
        self,
        batch: Dict[str, Any],
        num_inference_steps: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Execute training evaluation through the root FSDP forward.

        FSDP materializes root-owned parameters only around the wrapper's
        ``forward`` call, so training evaluation must use that entry point.

        Args:
            batch: Collated evaluation batch.
            num_inference_steps: Number of diffusion inference steps.
            seed: Evaluation random seed.

        Returns:
            Training-evaluation output dictionary.
        """
        return self.vla(
            forward_mode='training_eval',
            training_eval_batch=batch,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        """Saves the checkpoint of the model.

        Args:
            run_dir (Path): Directory to save the checkpoint.
            global_step (int): Current global step.
            epoch (int): Current epoch.
            train_loss (Optional[float], optional): Training loss.
                Defaults to None.
        """
        assert isinstance(self.vla, FSDP), \
            'FSDPStrategy.save_checkpoint assumes VLA is \
                already wrapped in FSDP!'

        if hasattr(self.vla._fsdp_wrapped_module, 'llm_backbone'):
            if hasattr(self.vla._fsdp_wrapped_module.llm_backbone, 'config'):
                self.vla._fsdp_wrapped_module.llm_backbone.config.to_json_file(  # noqa: E501
                    os.path.join(run_dir, 'llm_backbone_config.json'))
        if hasattr(self.vla._fsdp_wrapped_module, 'vlm_backbone'):
            if hasattr(self.vla._fsdp_wrapped_module.vlm_backbone, 'config'):
                self.vla._fsdp_wrapped_module.vlm_backbone.config.to_json_file(  # noqa: E501
                    os.path.join(run_dir, 'vlm_backbone_config.json'))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(run_dir, 'tokenizer'))
        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(self.vla, self.fsdp_state_dict_type,
                                  self.fsdp_save_policy):
            full_vla_state_dict = self._move_checkpoint_tensors_to_cpu(
                self.vla.state_dict())
            train_model_state_dicts = None
            if getattr(self, '_ema_params', None) is not None:
                with self._use_ema_parameters():
                    full_ema_state_dict = (
                        self._move_checkpoint_tensors_to_cpu(
                            self.vla.state_dict()))
                train_model_state_dicts = self._format_model_state_dict(
                    full_vla_state_dict)
                model_state_dicts = self._format_model_state_dict(
                    full_ema_state_dict)
            else:
                model_state_dicts = self._format_model_state_dict(
                    full_vla_state_dict)

            # Get full optimizer state dict for FSDP
            # FSDP shards optimizer states, so we need to gather the full state
            # IMPORTANT: Ensure all ranks are synchronized before gathering
            # optimizer state
            # This prevents AssertionError about different step values across
            # ranks
            # First barrier: ensure all ranks reach this point
            dist.barrier()

            # For FSDP, we need to ensure optimizer states are synchronized
            # before calling full_optim_state_dict
            # This is critical after resume, as different ranks might have
            # different optimizer states if loading failed on some ranks
            if self.optimizer is not None:
                # Ensure all ranks have completed the same number of optimizer
                # steps by synchronizing before gathering the full state
                dist.barrier()
                full_optimizer_state_dict = FSDP.full_optim_state_dict(
                    self.vla, self.optimizer)
                full_optimizer_state_dict = (
                    self._move_checkpoint_tensors_to_cpu(
                        full_optimizer_state_dict))
            else:
                full_optimizer_state_dict = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Save on rank zero *only*
            if overwatch.is_rank_zero():
                checkpoint_dir = os.path.join(run_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                if train_loss is None:
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt'  # noqa: E231, E501
                    )
                else:
                    checkpoint_path = (
                        os.path.join(
                            checkpoint_dir,
                            f'step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt'  # noqa: E231, E501
                        ))

                # Prepare checkpoint dictionary
                checkpoint_dict = {
                    'model': model_state_dicts,
                    'global_step': global_step,
                    'epoch': epoch,
                }
                if train_model_state_dicts is not None:
                    checkpoint_dict['train_model'] = train_model_state_dicts
                    checkpoint_dict['ema_decay'] = self.ema_decay

                # Save scheduler state
                if self.lr_scheduler is not None:
                    checkpoint_dict[
                        'scheduler_state_dict'] = self.lr_scheduler.state_dict(
                        )

                # Save full optimizer state dict (only on rank 0)
                if full_optimizer_state_dict is not None:
                    checkpoint_dict[
                        'optimizer_state_dict'] = full_optimizer_state_dict

                # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
                torch.save(checkpoint_dict, checkpoint_path)

                # Save model weights as safetensors for fast loading
                safetensors_path = checkpoint_path.replace(
                    '.pt', '.safetensors')
                self._save_model_safetensors(model_state_dicts,
                                             safetensors_path)
                overwatch.info(f'Saved safetensors at: {safetensors_path}')

                # Create symlink to latest checkpoint
                latest_ckpt_link = os.path.join(checkpoint_dir,
                                                'latest-checkpoint.pt')
                if os.path.islink(latest_ckpt_link) or os.path.exists(
                        latest_ckpt_link):
                    os.remove(latest_ckpt_link)
                os.symlink(os.path.abspath(checkpoint_path), latest_ckpt_link)

                latest_sf_link = os.path.join(checkpoint_dir,
                                              'latest-checkpoint.safetensors')
                if os.path.islink(latest_sf_link) or os.path.exists(
                        latest_sf_link):
                    os.remove(latest_sf_link)
                os.symlink(os.path.abspath(safetensors_path), latest_sf_link)

                self._cleanup_old_checkpoints(checkpoint_dir)

        if overwatch.is_rank_zero():
            del checkpoint_dict
        del full_vla_state_dict
        del model_state_dicts
        del full_optimizer_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_setup(self, n_train_examples: int) -> None:
        self.vla.from_pretrained()
        torch.cuda.set_device(device_id := self.device_id)  # noqa: F841
        torch.cuda.empty_cache()

        self.vla.freeze_backbones()
        fsdp_ignored_modules = self._resolve_fsdp_ignored_modules()

        is_no_shard = (
            self.fsdp_sharding_strategy == ShardingStrategy.NO_SHARD)

        if is_no_shard or self.fsdp_wrap_policy == 'root':
            vla_fsdp_wrapping_policy = None
        elif self.fsdp_wrap_policy == 'execution-block':
            policy_getter = getattr(
                self.vla, 'get_fsdp_execution_block_wrapping_policy', None)
            if not callable(policy_getter):
                raise ValueError(
                    f'{type(self.vla).__name__} does not provide an '
                    'execution-block FSDP wrapping policy.')
            vla_fsdp_wrapping_policy = policy_getter()
        else:
            vla_fsdp_wrapping_policy = self.vla.get_fsdp_wrapping_policy()

        # Assemble the Default FSDP Mixed Precision Policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:  # noqa: E501
            param_dtype = (None if self.keep_params_fp32 else torch.bfloat16)
            if is_no_shard:
                fsdp_precision_policy = MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16)
            elif not self.reduce_in_full_precision:
                fsdp_precision_policy = MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16)
            else:
                fsdp_precision_policy = MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32)
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32)

        if self.pre_fsdp_param_dtype is not None:
            target_dtype = self.pre_fsdp_param_dtype
        elif is_no_shard and not self.keep_params_fp32:
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32
        ignored_param_ids = {
            id(param)
            for module in fsdp_ignored_modules
            for param in module.parameters()
        }
        for name, param in self.vla.named_parameters():
            if id(param) in ignored_param_ids:
                continue
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
        overwatch.info(
            f'Unified FSDP-managed model parameters to {target_dtype}; '
            f'kept {len(ignored_param_ids):,} frozen parameter tensors in '
            'their source dtype.',
            ctx_level=1)

        # FSDP does not own or move ignored modules. Keep them replicated on
        # each rank in their checkpoint dtype (BF16 for DiT4DiT), matching the
        # source ZeRO-2 execution and avoiding repeated parameter all-gathers.
        current_device = torch.device('cuda', torch.cuda.current_device())
        for module in fsdp_ignored_modules:
            module.to(device=current_device)

        # Collect checkpoint layer classes BEFORE FSDP wrapping
        checkpoint_layer_classes = set()
        vlm_has_hf_checkpointing = False
        if self.enable_gradient_checkpointing:
            if hasattr(self, 'llm_transformer_layer_cls') \
                    and self.llm_transformer_layer_cls is not None:
                checkpoint_layer_classes.add(self.llm_transformer_layer_cls)

            try:
                from timm.models.vision_transformer import Block as VisionBlock
                checkpoint_layer_classes.add(VisionBlock)
            except ImportError:
                pass

            if hasattr(self.vla,
                       'vlm_backbone') and self.vla.vlm_backbone is not None:
                if hasattr(self.vla.vlm_backbone,
                           'enable_gradient_checkpointing'):
                    vlm_has_hf_checkpointing = True
                elif hasattr(self.vla.vlm_backbone, 'transformer_layer_cls'):
                    checkpoint_layer_classes.add(
                        self.vla.vlm_backbone.transformer_layer_cls)

            if hasattr(self.vla,
                       'llm_expert') and self.vla.llm_expert is not None:
                if hasattr(self.vla.llm_expert, 'transformer_layer_cls'):
                    checkpoint_layer_classes.add(
                        self.vla.llm_expert.transformer_layer_cls)

        self.vla = FSDP(
            self.vla,
            auto_wrap_policy=vla_fsdp_wrapping_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            ignored_modules=(fsdp_ignored_modules or None),
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )
        fsdp_unit_count = sum(
            isinstance(module, FSDP) for module in self.vla.modules())
        ignored_numel = sum(param.numel() for module in fsdp_ignored_modules
                            for param in module.parameters())
        overwatch.info(
            f'FSDP wrapping created {fsdp_unit_count} units '
            f'(policy={self.fsdp_wrap_policy}); kept '
            f'{ignored_numel / 1e9:.2f}B frozen parameters replicated.',
            ctx_level=1)

        # Apply Gradient Checkpointing AFTER FSDP wrapping
        if self.enable_gradient_checkpointing:
            # Enable HuggingFace gradient checkpointing for VLM backbone
            # This must be done AFTER FSDP wrapping
            if vlm_has_hf_checkpointing:
                # Find the vlm_backbone module within FSDP-wrapped model
                # Check for vlm_backbone attribute first (more specific)
                for name, module in self.vla.named_modules():
                    if 'vlm_backbone' in name and hasattr(
                            module, 'enable_gradient_checkpointing'):
                        module.enable_gradient_checkpointing()
                        overwatch.info(
                            f'VLM backbone ({name}) uses HuggingFace gradient '
                            'checkpointing (enabled after FSDP)',
                            ctx_level=1)
                        break

            # Apply PyTorch checkpoint wrapper for non-HF layers
            if checkpoint_layer_classes:
                non_reentrant_wrapper = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT)

                def check_fn(submodule: nn.Module) -> bool:
                    for layer_cls in checkpoint_layer_classes:
                        if isinstance(submodule, layer_cls):
                            return True
                    return False

                apply_activation_checkpointing(
                    self.vla,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn)

        # Barrier =>> Sharding takes a minute?
        dist.barrier()
        # Create Optimizer and LR Scheduler
        # Use base class method to setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(n_train_examples)

        # Calculate values for logging
        n_train_examples_rounded = math.ceil(
            n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples_rounded *
                                  self.max_epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps
        scheduler_type = self.lr_scheduler_cfg.get('type', 'unknown')
        if 'warmup_ratio' in self.lr_scheduler_cfg:
            warmup_ratio = self.lr_scheduler_cfg['warmup_ratio']
            warmup_info = (
                f'{int(num_training_steps * warmup_ratio)} ({warmup_ratio})')
        elif 'warmup_steps' in self.lr_scheduler_cfg:
            warmup_info = f"{self.lr_scheduler_cfg['warmup_steps']} steps"
        else:
            warmup_info = '0'
        # Finalize Setup =>> Log!
        overwatch.info(
            f'FSDP {self.sharding_strategy} Strategy '
            f'(wrap={self.fsdp_wrap_policy}) =>> Finalized Training Setup:\n'  # noqa: E231, E501
            f'         |-> Global (Effective) Batch Size = {self.global_batch_size}\n'  # noqa: E221, E501
            f'         |-> Per-Device Batch Size = {self.per_device_batch_size}\n'  # noqa: E221, E501
            f'         |-> Distributed World Size = {overwatch.world_size()}\n'  # noqa: E221, E501
            f'         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n'  # noqa: E221, E501
            f'         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n'  # noqa: E221, E501
            f'         |-> Deterministic Algorithms = {self.deterministic_algorithms}\n'  # noqa: E221, E501
            f'         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n'  # noqa: E221, E501
            f'                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n'  # noqa: E221, E501
            f'                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n'  # noqa: E221, E501
            f'                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n'  # noqa: E221, E501
            f"         |-> Optimizer = {self.optimizer_cfg['type']}\n"  # noqa: E221, E501
            f"         |-> Default Optimizer LR = {self.optimizer_cfg['lr']}\n"  # noqa: E221, E501
            f"         |-> Optimizer Weight Decay = {self.optimizer_cfg.get('weight_decay')}\n"  # noqa: E221, E501
            f'         |-> LR Scheduler Type = {scheduler_type}\n'  # noqa: E221, E501
            f'         |-> LR Scheduler Warm-up = {warmup_info}\n'  # noqa: E221, E501
            f'         |-> Dataset Size = {n_train_examples} Examples\n'  # noqa: E221, E501
            f'         |-> Max Steps = {num_training_steps}\n\n'  # noqa: E221, E501
        )

    def clip_grad_norm(self) -> None:
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*  # noqa: E501
        self.vla.clip_grad_norm_(max_norm=self.max_grad_norm)

    def _load_model_state(self, checkpoint_model_state: dict) -> None:
        """Load FSDP model state from checkpoint.

        Args:
            checkpoint_model_state (dict): Model state dict from checkpoint.
        """
        if overwatch.is_rank_zero():
            overwatch.info('Loading FSDP model state')

        # Synchronize all ranks before loading model state
        dist.barrier()

        # Load model state dict using FSDP state_dict_type
        with FSDP.state_dict_type(self.vla, self.fsdp_state_dict_type,
                                  self.fsdp_save_policy):
            # Handle both dict format (when change_key_name is True) and
            # direct state_dict format
            if self.change_key_name and isinstance(checkpoint_model_state,
                                                   dict):
                # Reconstruct full state dict from module keys
                full_state_dict = OrderedDict()
                for mkey, mstate_dict in checkpoint_model_state.items():
                    for key, param in mstate_dict.items():
                        full_state_dict[f'{mkey}.{key}'] = param
                checkpoint_model_state = full_state_dict

            # Load the state dict
            self.vla.load_state_dict(checkpoint_model_state, strict=False)

        # Synchronize after loading
        dist.barrier()

        if overwatch.is_rank_zero():
            overwatch.info('FSDP model state restored from checkpoint')

    def _load_optimizer_state(self, checkpoint_optimizer_state: dict) -> bool:
        """Load FSDP optimizer state from checkpoint.

        Args:
            checkpoint_optimizer_state (dict): Full optimizer state dict from
                checkpoint.

        Returns:
            bool: True if optimizer state was successfully loaded,
                False otherwise.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if overwatch.is_rank_zero():
            overwatch.info('Loading FSDP optimizer state')

        # Synchronize all ranks before loading optimizer state
        # This ensures all ranks are at the same point
        dist.barrier()

        # Load full optimizer state dict on rank 0, then shard it
        full_osd = checkpoint_optimizer_state

        # Use the new API if available, otherwise fall back to deprecated API
        try:
            # New API: optim_state_dict_to_load
            sharded_osd = FSDP.optim_state_dict_to_load(
                full_osd, self.vla, self.optimizer)
        except (AttributeError, TypeError):
            # Fall back to deprecated API for older PyTorch versions
            sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, self.vla)

        # Load the sharded optimizer state dict
        # All ranks must load the state to keep them synchronized
        self.optimizer.load_state_dict(sharded_osd)
        self.optimizer_state_loaded = True

        # Synchronize after loading to ensure all ranks have loaded
        dist.barrier()

        if overwatch.is_rank_zero():
            overwatch.info('FSDP optimizer state restored from checkpoint')

        return True
