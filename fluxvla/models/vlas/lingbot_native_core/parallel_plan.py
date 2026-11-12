"""Small local parallel-plan value object used by the native model contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParallelPlan:
    ep_plan: dict[str, Any] = field(default_factory=dict)
