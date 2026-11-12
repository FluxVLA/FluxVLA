"""Common Megatron pretrain launcher for Native MCore model recipes."""

from __future__ import annotations

from .contracts import MCoreModelSpec


def run_mcore(spec: MCoreModelSpec) -> None:
    """Run a validated model spec through Megatron's native lifecycle."""
    if not isinstance(spec, MCoreModelSpec):
        raise TypeError(f"expected MCoreModelSpec, got {type(spec).__name__}")

    from megatron.training import pretrain

    for hook in spec.setup_hooks:
        hook()

    def dataset_provider(num_samples):
        return spec.dataset_provider(num_samples)

    dataset_provider.is_distributed = True
    pretrain(
        dataset_provider,
        spec.model_provider,
        spec.model_type,
        spec.forward_step,
        extra_args_provider=spec.args_provider,
        args_defaults=dict(spec.args_defaults),
    )
