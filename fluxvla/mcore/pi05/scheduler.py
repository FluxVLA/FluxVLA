"""PI0.5 learning-rate contract adapters for Megatron training."""

from __future__ import annotations

from typing import Callable


def apply_initial_lr_contract(scheduler, contract: str) -> None:
    """Apply the source FluxVLA scheduler's fresh-run LR semantics.

    FluxVLA's Hugging Face cosine scheduler explicitly starts optimizer
    parameter groups at zero and advances the schedule after the first update.
    Megatron starts at ``get_lr(num_steps=0)`` instead.  Only the parameter-group
    LR is changed here so the first real gradient still initializes Adam state
    and the scheduler remains at step zero.
    """
    if contract == "megatron":
        return
    if contract != "fluxvla":
        raise ValueError(f"Unsupported PI0.5 LR scheduler contract: {contract}")
    if scheduler.num_steps != 0:
        raise RuntimeError(
            "FluxVLA initial LR contract can only be applied to a fresh scheduler"
        )
    for param_group in scheduler.optimizer.param_groups:
        param_group["lr"] = 0.0


def install_pi05_scheduler_builder() -> None:
    """Install the PI0.5 scheduler adapter at Megatron's training boundary."""
    from megatron.training import get_args
    from megatron.training import training as megatron_training

    current_builder = megatron_training.get_optimizer_param_scheduler
    if getattr(current_builder, "_fluxvla_pi05_adapter", False):
        return

    upstream_builder: Callable = current_builder

    def build_pi05_optimizer_param_scheduler(optimizer):
        scheduler = upstream_builder(optimizer)
        contract = getattr(get_args(), "pi0_lr_scheduler_contract", "fluxvla")
        apply_initial_lr_contract(scheduler, contract)
        return scheduler

    build_pi05_optimizer_param_scheduler._fluxvla_pi05_adapter = True
    megatron_training.get_optimizer_param_scheduler = build_pi05_optimizer_param_scheduler
