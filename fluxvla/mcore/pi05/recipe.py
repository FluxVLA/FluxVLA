"""Native MCore training recipe for PI0.5."""

from __future__ import annotations

from megatron.core.enums import ModelType

from ..checkpointing import install_mixed_precision_optimizer_checkpoint_adapter
from ..contracts import MCoreModelSpec
from ..diagnostics import (
    install_fsdp_grad_norm_contract,
    install_fsdp_grad_reduce_contract,
    install_optimizer_alignment_trace,
)
from .dataset import build_pi0_datasets
from .forward_step import forward_step
from .provider import get_pi05_args, model_provider
from .scheduler import install_pi05_scheduler_builder


def _dataset_provider(num_samples):
    from megatron.training import get_args

    return build_pi0_datasets(get_args())


PI05_SPEC = MCoreModelSpec(
    name="pi05",
    model_type=ModelType.encoder_or_decoder,
    model_provider=model_provider,
    dataset_provider=_dataset_provider,
    forward_step=forward_step,
    args_provider=get_pi05_args,
    setup_hooks=(
        install_mixed_precision_optimizer_checkpoint_adapter,
        install_pi05_scheduler_builder,
        install_fsdp_grad_norm_contract,
        install_optimizer_alignment_trace,
        install_fsdp_grad_reduce_contract,
    ),
    args_defaults={
        "tokenizer_type": "NullTokenizer",
        "vocab_size": 1,
        "dataloader_type": "external",
    },
)


__all__ = ["PI05_SPEC"]
