"""Reusable Native MCore tensor-parallel building blocks.

Model packages (PI0.5, LingBot, and future VLAs) own their block topology;
this package owns only backend primitives and topology validation.
"""

from .linear import MCoreTPLinear, mcore_tp_config
from .activation import SequenceParallelMLP
from .layers import ColumnParallelLinear, RowParallelLinear
from .topology import MCoreTopology, validate_mcore_topology

__all__ = [
    "MCoreTPLinear",
    "SequenceParallelMLP",
    "MCoreTopology",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "mcore_tp_config",
    "validate_mcore_topology",
]
