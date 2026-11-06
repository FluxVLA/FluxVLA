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
"""Local RTC inference runner for a dual-arm reBot."""

import time

import numpy as np
import torch

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from ..utils.trajectory_utils import resample_remaining
from .rebot_dual_inference_runner import RebotDualInferenceRunner

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class RebotDualRTCInferenceRunner(RebotDualInferenceRunner):
    """Condition predictions on the previous chunk's remaining actions."""

    def __init__(self, rtc_config=None, *args, **kwargs):
        self.rtc_config = rtc_config or {}
        self._rtc_action_scale = None
        super().__init__(*args, **kwargs)

        method = self.rtc_config.get('method', 'prefix')
        if method not in {'prefix', 'guidance'}:
            raise ValueError(f'Unsupported RTC method: {method}')
        prefix_len = self.rtc_config.get('prefix_len')
        if prefix_len is not None and (not isinstance(prefix_len, int)
                                       or prefix_len < 0):
            raise ValueError('RTC prefix_len must be None or nonnegative')
        if self.rtc_config.get('enabled', False) and self._use_remote:
            raise ValueError('reBot RTC currently requires local inference')

    def run(self, initial_instruction='Complete the requested bimanual task.'):
        """Run inference with the context required by the RTC method."""
        import rospy

        use_vjp = (
            self.rtc_config.get('enabled', False)
            and self.rtc_config.get('method', 'prefix') == 'guidance'
            and self.rtc_config.get('use_vjp', False))
        mode_context = torch.no_grad if use_vjp else torch.inference_mode

        with mode_context():
            while not rospy.is_shutdown():
                self._run_episode(initial_instruction)

    def _rebase_prefix(self, remaining, previous_state, current_state):
        """Preserve absolute targets when relative actions change anchor."""
        denormalizer = self.denormalize_action
        mask = getattr(denormalizer, 'delta_action_mask', None)
        if mask is None or not np.any(mask):
            return remaining
        if getattr(denormalizer, 'clip_normalized_action', False):
            raise ValueError('RTC delta rebasing requires unclipped action '
                             'denormalization')

        if self._rtc_action_scale is None:
            probes = np.stack((np.zeros(remaining.shape[-1], dtype=np.float32),
                               np.ones(remaining.shape[-1], dtype=np.float32)))
            decoded = denormalizer(
                dict(action=probes[None], state=np.zeros_like(previous_state)))
            scale = decoded[1] - decoded[0]
            if not np.all(np.isfinite(scale[mask])) or np.any(
                    scale[mask] == 0):
                raise ValueError('RTC delta-action scales must be finite and '
                                 'nonzero')
            self._rtc_action_scale = scale

        shift = np.asarray(previous_state) - np.asarray(current_state)
        permutation = getattr(denormalizer, 'state_permutation', None)
        if permutation is not None:
            shift = shift[permutation]
        indices = np.flatnonzero(mask)
        rebased = remaining.copy()
        rebased[:, indices] += shift[indices] / self._rtc_action_scale[indices]
        return rebased

    def _predict_action(self, inputs):
        """Inject the remaining previous actions into the model inputs."""
        if not self.rtc_config.get('enabled', False):
            return super()._predict_action(inputs)

        previous = self._prev_ctx
        context = self._action_ctx
        context.inference_start = time.time()
        prefix_len = 0

        if previous is not None:
            offset = (context.inference_start -
                      previous.action_timestamp) / self.dt
            remaining = resample_remaining(previous.raw_actions[0], offset)
            prefix_len = self.rtc_config.get('prefix_len')
            if prefix_len is None:
                prefix_len = int(previous.inference_elapsed *
                                 self.publish_rate)
            prefix_len = min(prefix_len, len(remaining))
            if prefix_len > 0:
                remaining = self._rebase_prefix(remaining, previous.state,
                                                context.state)
                inputs['prev_actions'] = torch.from_numpy(remaining[None]).to(
                    device=inputs['states'].device,
                    dtype=inputs['states'].dtype)
                inputs['prefix_len'] = prefix_len
                inputs['rtc_config'] = self.rtc_config

        raw_action = self.vla.predict_action(**inputs)
        context.raw_actions = raw_action.detach().float().cpu().numpy()
        context.inference_elapsed = time.time() - context.inference_start
        method = self.rtc_config.get('method', 'prefix')
        elapsed = format(context.inference_elapsed, '.3f')
        overwatch.info(f'RTC {method}: prefix_len={prefix_len}, '
                       f'inference_s={elapsed}')
        return raw_action
