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

import time

import torch

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from ..utils.trajectory_utils import resample_remaining
from .ur_inference_runner import URInferenceRunner

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class URRTCInferenceRunner(URInferenceRunner):
    """UR inference runner with RTC prefix conditioning.

    RTC (Real-Time Chunking) conditions each new action chunk on the unexecuted
    suffix of the previous raw model prediction. ``execute_horizon`` controls
    how many denormalized actions are sent before requesting the next chunk;
    when it is ``None`` the whole chunk is executed and RTC has no remaining
    prefix to reuse.
    """

    def __init__(self,
                 rtc_config: dict = None,
                 execute_horizon: int = None,
                 *args,
                 **kwargs):
        self.rtc_config = rtc_config
        self.execute_horizon = execute_horizon
        if self.rtc_config and self.rtc_config.get('enabled', False):
            assert self.rtc_config.get('method', 'prefix') != 'guidance', (
                'Guidance RTC is not supported by URRTCInferenceRunner.')
        super().__init__(*args, **kwargs)
        self.dt = 1.0 / self.publish_rate

    def run(self,
            initial_instruction:
            str = 'place it in the brown paper bag with right arm'):
        """Run inference with the context required by the selected RTC mode."""
        import rospy

        overwatch.info('Starting UR RTC inference runner')

        use_vjp = (
            self.rtc_config and self.rtc_config.get('enabled', False)
            and self.rtc_config.get('method', 'prefix') == 'guidance'
            and self.rtc_config.get('use_vjp', False))
        mode_context = torch.no_grad if use_vjp else torch.inference_mode

        with mode_context():
            while not rospy.is_shutdown():
                self._run_episode(initial_instruction)

    def _predict_action(self, inputs: dict):
        """Predict a chunk, optionally injecting previous chunk prefix."""
        ctx = self._action_ctx
        ctx.inference_start = time.time()

        if self._use_remote:
            raw_action = super()._predict_action(inputs)
            ctx.inference_elapsed = time.time() - ctx.inference_start
            ctx.raw_actions = raw_action.cpu().numpy()
            return raw_action

        prev = self._prev_ctx
        if (prev is not None and self.rtc_config
                and self.rtc_config.get('enabled', False)
                and hasattr(prev, 'action_timestamp')
                and hasattr(prev, 'raw_actions')):
            offset = (ctx.inference_start - prev.action_timestamp) / self.dt
            remaining = resample_remaining(prev.raw_actions[0], offset)[None]
            prefix_len = self.rtc_config.get('prefix_len')
            if prefix_len is None:
                prefix_len = int(prev.inference_elapsed * self.publish_rate)
            prefix_len = min(prefix_len, remaining.shape[1])

            if prefix_len > 0:
                inputs['prev_actions'] = torch.from_numpy(remaining).to(
                    device=inputs['states'].device,
                    dtype=inputs['states'].dtype)
                inputs['prefix_len'] = prefix_len
                inputs['rtc_config'] = self.rtc_config

        raw_action = self.vla.predict_action(**inputs)
        ctx.inference_elapsed = time.time() - ctx.inference_start
        ctx.raw_actions = raw_action.cpu().numpy()
        return raw_action

    def _execute_actions(self, actions, rate):
        """Execute the action horizon and record when execution starts."""
        if self.disable_puppet_arm:
            return

        ctx = self._action_ctx
        ctx.action_timestamp = time.time()

        if self.execute_horizon is not None:
            actions = actions[:self.execute_horizon]
        # print("pass execute")
        for action in actions:
            self.ros_operator.servoj(action[:6])
            self.ros_operator.movegrip(action[6])
            rate.sleep()
