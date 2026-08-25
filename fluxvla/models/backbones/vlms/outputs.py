# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared output contract for interchangeable VLM backbones."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import torch


@dataclass
class VLMBackboneOutput:
    """Canonical feature output consumed by continuous-action VLAs.

    ``hidden_states`` remains available for Hugging Face-style callers, while
    ``last_hidden_state`` is the stable FluxVLA action-head input. Backbone-
    owned training objectives and optional diagnostics are carried in generic
    dictionaries instead of adding model-specific fields to the VLA wrapper.
    """

    last_hidden_state: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    auxiliary_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    auxiliary_outputs: Dict[str, Any] = field(default_factory=dict)
    hidden_states: Optional[Sequence[torch.Tensor]] = None

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.last_hidden_state):
            raise TypeError('last_hidden_state must be a torch.Tensor, got '
                            f'{type(self.last_hidden_state)!r}.')
        if self.hidden_states is None:
            self.hidden_states = [self.last_hidden_state]
        elif len(self.hidden_states) == 0:
            raise ValueError('hidden_states must not be empty when provided.')


def normalize_vlm_backbone_output(output: Any) -> VLMBackboneOutput:
    """Convert canonical, Hugging Face, and legacy tuple outputs.

    Existing FluxVLA VLM backbones return ``(features, mask, auxiliary)``.
    Keeping that adapter here lets them coexist with the canonical output
    while individual backbones migrate without changing experiment behavior.
    """
    if isinstance(output, VLMBackboneOutput):
        return output

    if torch.is_tensor(output):
        return VLMBackboneOutput(last_hidden_state=output)

    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError('VLM backbone returned an empty sequence.')
        auxiliary_outputs = {}
        if len(output) > 2 and output[2] is not None:
            auxiliary_outputs['legacy_auxiliary_output'] = output[2]
        return VLMBackboneOutput(
            last_hidden_state=output[0],
            attention_mask=output[1] if len(output) > 1 else None,
            auxiliary_outputs=auxiliary_outputs,
        )

    if isinstance(output, Mapping):
        hidden_states = output.get('hidden_states')
        last_hidden_state = output.get('last_hidden_state')
        if last_hidden_state is None and hidden_states is not None:
            if len(hidden_states) == 0:
                raise ValueError('VLM hidden_states must not be empty.')
            last_hidden_state = hidden_states[-1]
        if last_hidden_state is None:
            raise ValueError('VLM backbone output mapping must contain '
                             '`last_hidden_state` or `hidden_states`.')
        return VLMBackboneOutput(
            last_hidden_state=last_hidden_state,
            attention_mask=output.get('attention_mask'),
            auxiliary_losses=dict(output.get('auxiliary_losses') or {}),
            auxiliary_outputs=dict(output.get('auxiliary_outputs') or {}),
            hidden_states=hidden_states,
        )

    hidden_states = getattr(output, 'hidden_states', None)
    last_hidden_state = getattr(output, 'last_hidden_state', None)
    if last_hidden_state is None and hidden_states is not None:
        if len(hidden_states) == 0:
            raise ValueError('VLM hidden_states must not be empty.')
        last_hidden_state = hidden_states[-1]
    if last_hidden_state is None:
        raise TypeError('Unsupported VLM backbone output type '
                        f'{type(output)!r}.')

    auxiliary_losses = dict(getattr(output, 'auxiliary_losses', None) or {})
    # Compatibility with the pre-contract Cosmos25 output. This path is also
    # useful for old checkpoints/tests that provide a lightweight mock object.
    future_video_loss = getattr(output, 'future_video_loss', None)
    if future_video_loss is not None:
        auxiliary_losses.setdefault('future_video_loss', future_video_loss)

    auxiliary_outputs = dict(getattr(output, 'auxiliary_outputs', None) or {})
    pred_future_video = getattr(output, 'pred_future_video', None)
    if pred_future_video is not None:
        auxiliary_outputs.setdefault('pred_future_video', pred_future_video)

    return VLMBackboneOutput(
        last_hidden_state=last_hidden_state,
        attention_mask=getattr(output, 'attention_mask', None),
        auxiliary_losses=auxiliary_losses,
        auxiliary_outputs=auxiliary_outputs,
        hidden_states=hidden_states,
    )


__all__ = ['VLMBackboneOutput', 'normalize_vlm_backbone_output']
