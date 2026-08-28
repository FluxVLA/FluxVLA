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
"""RoboDojo evaluation dataset.

Converts an online RoboDojo observation dict into the model input batch
format. Mirrors :class:`RobocasaEvalDataset`: run the configured transform
chain, then assemble a batch compatible with the FluxVLA model. Unlike the
LIBERO eval pipeline this dataset is single-frame and has no image buffer,
padding, or frame splitting; statistics are injected under ``norm_stats``
(the key :class:`StateFromInputs` reads).
"""

import json
from typing import Any, Dict, List

import numpy as np
import torch

from fluxvla.engines.utils.root import DATASETS


@DATASETS.register_module()
class RoboDojoEvalDataset:
    """Convert a RoboDojo observation dict into a model input batch.

    Args:
        norm_stats (str | Dict | None): Normalization statistics path or dict.
        unnorm_key (str): Key inside ``norm_stats`` for this benchmark.
        transforms (List[Dict] | None): Transform config list applied in
            order (e.g. ProcessEvalInputs, StateFromInputs, prompt, image).
    """

    def __init__(self,
                 norm_stats: Any = None,
                 unnorm_key: str = 'robodojo_arx_x5',
                 transforms: List[Dict] = None,
                 **kwargs) -> None:
        from fluxvla.engines import build_transform_from_cfg

        self.transforms = [
            build_transform_from_cfg(t) for t in (transforms or [])
        ]
        self.unnorm_key = unnorm_key
        if isinstance(norm_stats, str) and norm_stats:
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats

    def __call__(self, inputs: Dict[str, Any]) -> tuple:
        """Convert one observation into ``(batch, None)`` (no replay)."""
        data = dict(inputs)

        # Inject statistics for StateFromInputs.
        if self.norm_stats is not None and self.unnorm_key in self.norm_stats:
            data['norm_stats'] = self.norm_stats[self.unnorm_key]

        for transform in self.transforms:
            data = transform(data)

        assert 'lang_tokens' in data and 'lang_masks' in data, \
            'Prompt transform must provide lang_tokens and lang_masks'
        tokens = torch.tensor(data['lang_tokens'])
        token_mask = data['lang_masks'].tolist() if hasattr(
            data['lang_masks'], 'tolist') else list(data['lang_masks'])

        # TransformImage produces the CHW (3 * num_imgs, H, W) tensor layout
        # the model expects; no numpy/HWC conversion is needed here.
        pixel_values = data['pixel_values']
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                'pixel_values must be a torch tensor (TransformImage output), '
                f'got {type(pixel_values)}')
        num_imgs = pixel_values.shape[0] // 3
        img_masks = data.get('img_masks', [True] * num_imgs)
        img_masks = list(img_masks)

        batch = dict(
            images=pixel_values.cuda().unsqueeze(0),
            img_masks=torch.tensor([img_masks]).cuda(),
            lang_tokens=tokens.unsqueeze(0).cuda(),
            lang_masks=torch.tensor(token_mask).unsqueeze(0).cuda(),
        )
        if 'states' in data:
            batch['states'] = torch.from_numpy(
                data['states']).bfloat16().cuda().unsqueeze(0)
        if 'embodiment_ids' in data:
            batch['embodiment_ids'] = torch.from_numpy(
                np.asarray(data['embodiment_ids'])).int().cuda().unsqueeze(0)
        batch['reset_history'] = bool(data.get('is_new_episode', False))
        return batch, None
