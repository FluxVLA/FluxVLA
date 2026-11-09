"""Topology contracts shared by Native MCore model providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCoreTopology:
    tp: int = 1
    pp: int = 1
    sp: bool = False
    dp: int = 1
    cp: int = 1
    ep: int = 1


def validate_mcore_topology(
    topology: MCoreTopology,
    *,
    allow_tp: bool = True,
    allow_sp: bool = False,
    allow_pp: bool = False,
    allow_cp: bool = False,
    allow_ep: bool = False,
) -> None:
    """Reject unsupported parallel dimensions instead of silently replicating."""
    checks = (
        ("TP", topology.tp > 1, allow_tp),
        ("SP", topology.sp, allow_sp),
        ("PP", topology.pp > 1, allow_pp),
        ("CP", topology.cp > 1, allow_cp),
        ("EP", topology.ep > 1, allow_ep),
    )
    unsupported = [name for name, enabled, allowed in checks if enabled and not allowed]
    if unsupported:
        raise NotImplementedError(
            "Unsupported Native MCore parallel dimensions: " + ", ".join(unsupported)
        )


__all__ = ["MCoreTopology", "validate_mcore_topology"]
