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
"""Raw simulator observation adapters for API-hosted policies."""

from typing import Any, Dict, List

import numpy as np
from PIL import Image

from fluxvla.engines import DATASETS


@DATASETS.register_module()
class OpenAILiberoEvalDataset:
    """Prepare raw images and proprioception for ``OpenAIResponsesVLA``.

    Unlike local VLA datasets, this adapter performs no tokenization,
    normalization, tensor conversion, or CUDA transfer. The API policy owns
    image encoding and returns native LIBERO control actions.
    """

    def __init__(self,
                 norm_stats: Any = None,
                 task_suite_name: str = None,
                 norm_stats_key: str = None,
                 img_keys: List[str] = None,
                 resize_size: int = 256) -> None:
        del norm_stats, task_suite_name, norm_stats_key
        self.img_keys = img_keys or [
            'agentview_image', 'robot0_eye_in_hand_image'
        ]
        self.resize_size = resize_size

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)[::-1, ::-1].copy()
        if self.resize_size is not None:
            if isinstance(self.resize_size, int):
                size = (self.resize_size, self.resize_size)
            else:
                height, width = self.resize_size
                size = (width, height)
            image = np.asarray(
                Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
        return image

    def __call__(self, inputs: Dict[str, Any]):
        images = []
        for image_key in self.img_keys:
            if image_key not in inputs:
                raise KeyError(f'Missing image key: {image_key!r}')
            images.append(self._prepare_image(inputs[image_key]))

        batch = {
            'images': images,
            'image_names': list(self.img_keys),
            'task_description': inputs['task_description'],
            'eef_position': np.asarray(inputs['robot0_eef_pos']).copy(),
            'eef_quaternion': np.asarray(inputs['robot0_eef_quat']).copy(),
            'gripper_position':
            np.asarray(inputs['robot0_gripper_qpos']).copy(),
            'joint_position': np.asarray(inputs['robot0_joint_pos']).copy(),
            'reset_history': bool(inputs.get('is_new_episode', False)),
        }
        return batch, images[0]


@DATASETS.register_module()
class OpenAIRobocasaEvalDataset:
    """Prepare native RoboCasa observations for an API-hosted policy.

    The adapter deliberately keeps images and robot state on CPU. The remote
    policy owns image encoding and returns actions in the native 29D GR1
    action order used by :class:`RobocasaEvalRunner`.
    """

    _STATE_KEYS = (
        'state.left_arm',
        'state.left_hand',
        'state.right_arm',
        'state.right_hand',
        'state.waist',
    )

    def __init__(self,
                 norm_stats: Any = None,
                 unnorm_key: str = None,
                 img_keys: List[str] = None,
                 resize_size: int = 512) -> None:
        del norm_stats, unnorm_key
        self.img_keys = img_keys or [
            'video.ego_view_bg_crop_pad_res256_freq20'
        ]
        self.resize_size = resize_size
        self.last_raw_state = None
        self.last_debug = {}

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).copy()
        if self.resize_size is not None:
            if isinstance(self.resize_size, int):
                size = (self.resize_size, self.resize_size)
            else:
                height, width = self.resize_size
                size = (width, height)
            image = np.asarray(
                Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
        return image

    def __call__(self, inputs: Dict[str, Any]):
        images = []
        for image_key in self.img_keys:
            if image_key not in inputs:
                raise KeyError(f'Missing image key: {image_key!r}')
            images.append(self._prepare_image(inputs[image_key]))

        states = {}
        for state_key in self._STATE_KEYS:
            if state_key not in inputs:
                raise KeyError(f'Missing state key: {state_key!r}')
            states[state_key.removeprefix('state.')] = np.asarray(
                inputs[state_key], dtype=np.float32).copy()

        self.last_raw_state = np.concatenate([
            states['left_arm'], states['left_hand'], states['right_arm'],
            states['right_hand'], states['waist']
        ])
        task_description = str(inputs.get('task_description', ''))
        self.last_debug = {'task_description': task_description}
        batch = {
            'images': images,
            'image_names': list(self.img_keys),
            'task_description': task_description,
            **states,
            'reset_history': bool(inputs.get('is_new_episode', False)),
        }
        return batch, images[0]
