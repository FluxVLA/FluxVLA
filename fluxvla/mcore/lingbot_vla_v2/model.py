"""MegatronModule ownership boundary for the LingBot-VLA-v2 policy."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
from megatron.core.transformer.module import MegatronModule


class LingBotVLA2MCoreModel(MegatronModule):
    """Run the certified FluxVLA LingBot adapter in Megatron's lifecycle.

    P0 deliberately keeps the official LingBot computation graph intact.  MCore
    owns DDP, forward/backward scheduling, optimizer state and checkpointing;
    later TP work can replace selected decoder blocks behind this same contract.
    """

    def __init__(
        self,
        config,
        adapter: torch.nn.Module,
        *,
        preserve_fsdp2_fp32_modules: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.adapter = adapter
        self.input_tensor = None
        # Native FSDP2 keeps parameter storage in FP32 and casts the compute
        # view to BF16. Megatron's stock Float16Module calls model.bfloat16()
        # in-place, which would round the loaded checkpoint before step 1.
        self._preserve_all_fp32_params = (
            os.environ.get("FLUXVLA_MCORE_PRESERVE_FP32_PARAMS", "0") == "1"
        )
        # Native production has two independent precision switches. The
        # training-side FSDP2 policy keeps ``basic_modules`` (action decoder
        # and AdaNorm units) in FP32, while model ``action_fp32`` only promotes
        # flow inputs. Do not bind parameter storage to ``action_fp32``: the
        # production contract has action_fp32=False and still creates FP32
        # FSDP2 units for these modules.
        fp32_parameter_ids = {
            id(parameter)
            for module in (
                adapter.get_fsdp2_full_precision_modules()
                if preserve_fsdp2_fp32_modules
                else ()
            )
            for parameter in module.parameters()
        }
        self._fp32_parameter_names = {
            name
            for name, parameter in self.named_parameters()
            if id(parameter) in fp32_parameter_ids
        }
        # FSDP2's MixedPrecisionPolicy casts managed parameters and forward
        # inputs, but does not blanket-cast model buffers. Preserve floating
        # buffers (notably rotary inv_freq) together with this compatibility
        # contract; Megatron's Float16Module would otherwise cast them to BF16
        # and create a first-bad attention activation despite equal weights.
        self._preserved_buffer_names = (
            {
                name
                for name, buffer in self.named_buffers()
                if buffer.is_floating_point()
            }
            if preserve_fsdp2_fp32_modules
            else set()
        )
        # The native adapter may carry ``allreduce=False`` markers from its
        # FSDP runtime.  They are foreign metadata when EP is disabled, but
        # under MCore EP the local routed-expert shards deliberately use the
        # same marker so DDP assigns them to the expert-DP buffer group.
        if config.expert_model_parallel_size == 1:
            for parameter in self.parameters():
                parameter.allreduce = True

    def _apply(self, fn, recurse: bool = True):
        """Preserve native action-basic FP32 parameters under ``--bf16``.

        Megatron's Float16Module calls ``model.bfloat16()``.  Native FSDP2 casts
        the VLM/root units to BF16 for compute but leaves ``_no_split_modules``
        action units in FP32.  Retaining those parameter storages reproduces the
        same mixed model/master-parameter boundary.
        """
        probe = torch.empty(0, dtype=torch.float32)
        try:
            target_dtype = fn(probe).dtype
        except Exception:
            target_dtype = None
        preserved = {}
        preserved_buffers = {}
        if target_dtype in (torch.float16, torch.bfloat16):
            preserved = {
                name: parameter.data
                for name, parameter in self.named_parameters()
                if name in self._fp32_parameter_names
                and parameter.dtype == torch.float32
            }
            for name, parameter in self.named_parameters():
                if parameter.dtype == torch.float32 and name not in preserved:
                    parameter._fluxvla_fp32_init = parameter.detach().cpu().clone()
            preserved_buffers = {
                name: buffer
                for name, buffer in self.named_buffers()
                if name in self._preserved_buffer_names
            }
        result = super()._apply(fn, recurse=recurse)
        if preserved:
            for name, parameter in self.named_parameters():
                original = preserved.get(name)
                if original is not None:
                    parameter.data = original.to(device=parameter.device)
        if preserved_buffers:
            modules = dict(self.named_modules())
            for name, original in preserved_buffers.items():
                module_name, _, buffer_name = name.rpartition(".")
                module = modules[module_name]
                module._buffers[buffer_name] = original.to(
                    device=module._buffers[buffer_name].device
                )
        return result

    def bfloat16(self):
        """Keep FP32 master storage for the Native FSDP2 parity path."""
        if self._preserve_all_fp32_params:
            return self
        return super().bfloat16()

    def half(self):
        """Keep FP32 master storage for the Native FSDP2 parity path."""
        if self._preserve_all_fp32_params:
            return self
        return super().half()

    def set_input_tensor(self, input_tensor) -> None:
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    def forward(self, **batch: Any) -> Mapping[str, Any]:
        # Native FSDP executes the complete policy under a BF16 autocast
        # region, including the path whose parameter storage has already been
        # converted to BF16.  Keeping autocast conditional on the FP32-master
        # diagnostic path made the normal MCore BF16 route execute softmax,
        # normalization, and other autocast-sensitive operators at a different
        # precision.  Preserve one compute contract for both storage modes.
        # Native FSDP prepares every floating-point batch field as BF16 before
        # entering the root autocast region.  Autocast alone does not change
        # tensor storage, so leaving MCore's loader outputs in FP32 changes
        # preprocessing/embedding paths that consume the input directly.
        batch = self._prepare_native_batch(batch)
        use_bf16_autocast = (
            torch.cuda.is_available()
            and os.environ.get("FLUXVLA_MCORE_DISABLE_AUTOCAST", "0") != "1"
        )
        if use_bf16_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self.adapter(**batch)
        return self.adapter(**batch)

    def _prepare_native_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Match Native runner's BF16 input cast without touching integer masks."""
        def convert(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    return value.to(dtype=torch.bfloat16)
                return value
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, tuple):
                return tuple(convert(item) for item in value)
            return value

        prepared = {key: convert(value) for key, value in batch.items()}
        prepare = getattr(self.adapter, "prepare_training_batch", None)
        return prepare(prepared) if callable(prepare) else prepared

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets=(),
        metadata=None,
    ):
        """Expose replicated P0 tensors to MCore distributed checkpointing."""
        del metadata
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {},
            sharded_offsets,
        )


__all__ = ["LingBotVLA2MCoreModel"]
