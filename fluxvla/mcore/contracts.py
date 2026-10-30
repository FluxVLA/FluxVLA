"""Public contracts shared by Native Megatron-Core model recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class MCoreModelSpec:
    """Callbacks required to run one model with Megatron's native trainer.

    Model-specific code owns the callbacks and their argument namespace.  The
    launcher owns only Megatron lifecycle wiring, which lets another model
    reuse the same distributed initialization and checkpoint/scheduler hooks.
    """

    name: str
    model_type: Any
    model_provider: Callable[..., Any]
    dataset_provider: Callable[..., Any]
    forward_step: Callable[..., Any]
    args_provider: Callable[..., Any]
    setup_hooks: tuple[Callable[[], Any], ...] = ()
    args_defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("MCore model spec name must be non-empty")
        for field_name in (
            "model_provider",
            "dataset_provider",
            "forward_step",
            "args_provider",
        ):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"MCore spec field {field_name!r} must be callable")
        if not isinstance(self.args_defaults, Mapping):
            raise TypeError("MCore spec args_defaults must be a mapping")
