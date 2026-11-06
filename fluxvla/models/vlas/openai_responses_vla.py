# Copyright 2026 Limx Dynamics
"""Compatibility names for policies now exposed through openai_vla."""

from typing import Sequence

from fluxvla.engines import VLAS
from .openai_vla import (
    DEFAULT_LIBERO_SYSTEM_PROMPT,  # noqa: F401
    DEFAULT_OLI_SYSTEM_PROMPT,
    DEFAULT_ROBOCASA_SYSTEM_PROMPT,  # noqa: F401
    OpenAILiberoVLA,
    OpenAIOliVLA,
    OpenAIRobocasaVLA)

__all__ = [
    'DEFAULT_LIBERO_SYSTEM_PROMPT',
    'DEFAULT_OLI_SYSTEM_PROMPT',
    'DEFAULT_ROBOCASA_SYSTEM_PROMPT',
    'OpenAIResponsesVLA',
    'OpenAIResponsesOliVLA',
    'OpenAIResponsesRobocasaVLA',
]

OpenAIResponsesVLA = OpenAILiberoVLA
OpenAIResponsesRobocasaVLA = OpenAIRobocasaVLA
VLAS.register_module(name='OpenAIResponsesVLA', module=OpenAILiberoVLA)
VLAS.register_module(
    name='OpenAIResponsesRobocasaVLA', module=OpenAIRobocasaVLA)


@VLAS.register_module()
class OpenAIResponsesOliVLA(OpenAIOliVLA):
    """Preserve the former OLI constructor's positional motion limits."""

    def __init__(self,
                 max_wrist_delta: float = 0.08,
                 max_head_delta: float = 0.03,
                 max_foot_delta: float = 0.08,
                 max_base_xy_delta: float = 0.10,
                 max_base_height_delta: float = 0.08,
                 max_rotation_delta_deg: float = 15.0,
                 max_base_tilt_deg: float = 15.0,
                 trace_dir: str = None,
                 trace_console: bool = True,
                 workspace_bounds: Sequence[Sequence[float]] = ((-0.30, 0.80),
                                                                (-0.65, 0.65),
                                                                (-0.30, 1.50)),
                 system_prompt: str = DEFAULT_OLI_SYSTEM_PROMPT,
                 **kwargs) -> None:
        super().__init__(
            max_wrist_delta=max_wrist_delta,
            max_head_delta=max_head_delta,
            max_foot_delta=max_foot_delta,
            max_base_xy_delta=max_base_xy_delta,
            max_base_height_delta=max_base_height_delta,
            max_rotation_delta_deg=max_rotation_delta_deg,
            max_base_tilt_deg=max_base_tilt_deg,
            trace_dir=trace_dir,
            trace_console=trace_console,
            workspace_bounds=workspace_bounds,
            system_prompt=system_prompt,
            **kwargs)
