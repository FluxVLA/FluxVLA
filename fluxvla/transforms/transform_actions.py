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

from typing import Dict, List

import numpy as np

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class ProcessLiberoActions:

    def __init__(self, mask: List[bool] = None) -> None:
        """ProcessLiberoActions is a transform
        that modifies the actions in the data
        by subtracting the state values from
        the actions based on a mask.

        Args:
            mask (List[bool], optional): A list
                indicating which dimensions
                of the state should be subtracted from
                the actions.
                If None, no subtraction is performed.
        """
        self.mask = np.asarray(mask, dtype=bool)

    def __call__(self, data: Dict) -> Dict:
        if 'actions' not in data or self.mask is None:
            return data

        states, actions = data['states'], data['actions']
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, states[..., :dims], 0), axis=-2)
        data['actions'] = actions

        return data


@TRANSFORMS.register_module()
class XVLAEE6DTo20D:
    """Convert X-VLA EE6D state/action arrays to the canonical 20D layout."""

    def __init__(self,
                 state_key: str = 'states',
                 action_key: str = 'actions',
                 target_dim: int = 20,
                 binarize_gripper: bool = True,
                 gripper_threshold: float = 0.0) -> None:
        if target_dim != 20:
            raise ValueError(
                f'{self.__class__.__name__} currently supports only '
                f'target_dim=20, got {target_dim}.')
        self.state_key = state_key
        self.action_key = action_key
        self.target_dim = target_dim
        self.binarize_gripper = binarize_gripper
        self.gripper_threshold = gripper_threshold

    def _convert(self, value: np.ndarray, key: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape[-1] == self.target_dim:
            out = arr.copy()
            if self.binarize_gripper:
                out[..., 9:10] = (out[..., 9:10]
                                  > self.gripper_threshold).astype(np.float32)
                out[..., 19:20] = (out[..., 19:20]
                                   > self.gripper_threshold).astype(np.float32)
            return out

        if arr.shape[-1] != 10:
            raise ValueError(
                f'{self.__class__.__name__} expects {key} last dimension '
                f'to be 10 or {self.target_dim}, got {arr.shape}.')

        out_shape = arr.shape[:-1] + (self.target_dim, )
        out = np.zeros(out_shape, dtype=np.float32)
        out[..., :9] = arr[..., :9]
        if self.binarize_gripper:
            out[..., 9:10] = (arr[..., 9:10]
                              > self.gripper_threshold).astype(np.float32)
        else:
            out[..., 9:10] = arr[..., 9:10]
        return out

    def __call__(self, data: Dict) -> Dict:
        if self.state_key in data:
            data[self.state_key] = self._convert(data[self.state_key],
                                                 self.state_key)
        if self.action_key in data:
            data[self.action_key] = self._convert(data[self.action_key],
                                                  self.action_key)
        return data
