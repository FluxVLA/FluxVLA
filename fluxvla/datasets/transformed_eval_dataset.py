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

from typing import Any, Dict, List

from fluxvla.engines import DATASETS, build_transform_from_cfg


@DATASETS.register_module()
class TransformedEvalDataset:
    """Build an evaluation batch directly from a transform pipeline."""

    eval_context_keys = ()

    def __init__(self, transforms: List[Dict], batch_keys: List[str]) -> None:
        if not transforms:
            raise ValueError('transforms must contain at least one transform')
        if not batch_keys:
            raise ValueError('batch_keys must contain at least one key')
        self.transforms = [
            build_transform_from_cfg(transform) for transform in transforms
        ]
        self.batch_keys = tuple(batch_keys)

    def __call__(self, inputs: Dict[str, Any]):
        data = dict(inputs)
        is_new_episode = bool(data.get('is_new_episode', False))
        for transform in self.transforms:
            data = transform(data)

        missing_keys = [key for key in self.batch_keys if key not in data]
        if missing_keys:
            raise KeyError(
                f'Transform pipeline did not produce batch keys: '
                f'{missing_keys}')

        batch = {key: data[key] for key in self.batch_keys}
        batch['reset_history'] = is_new_episode
        return batch, data.get('replay_img')