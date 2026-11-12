"""Native MCore training recipe for LingBot-VLA-v2."""

from __future__ import annotations

from megatron.core.enums import ModelType

from ..checkpointing import install_mixed_precision_optimizer_checkpoint_adapter
from ..contracts import MCoreModelSpec
from ..diagnostics import install_optimizer_alignment_trace
from .dataset import build_lingbot_datasets
from .forward_step import forward_step
from .optimizer import (
    install_lingbot_fsdp2_reduction_order,
    install_lingbot_optimizer_builder,
)
from .provider import get_lingbot_args, model_provider


def _dataset_provider(_num_samples):
    from megatron.training import get_args

    return build_lingbot_datasets(get_args())


LINGBOT_VLA_V2_SPEC = MCoreModelSpec(
    name="lingbot-vla-v2",
    model_type=ModelType.encoder_or_decoder,
    model_provider=model_provider,
    dataset_provider=_dataset_provider,
    forward_step=forward_step,
    args_provider=get_lingbot_args,
    setup_hooks=(
        install_mixed_precision_optimizer_checkpoint_adapter,
        install_lingbot_optimizer_builder,
        install_lingbot_fsdp2_reduction_order,
        install_optimizer_alignment_trace,
    ),
    args_defaults={
        "tokenizer_type": "NullTokenizer",
        "vocab_size": 1,
        "dataloader_type": "external",
    },
)


__all__ = ["LINGBOT_VLA_V2_SPEC"]
