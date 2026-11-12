"""MCore layer exports shared by model-specific TP blocks."""

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

__all__ = ["ColumnParallelLinear", "RowParallelLinear"]
