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
"""Collator for VLM SFT: pad sequences and batch pixel_values / image_grid_thw
so that the batch can be passed to HuggingFace VLM forward.
"""

from typing import Any, Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from fluxvla.engines import COLLATORS


@COLLATORS.register_module()
class VLMSFTCollator:
    """Collate VLM SFT samples into a batch.

    Pads input_ids, attention_mask, labels to the max length in the batch.
    Stacks or concatenates pixel_values and image_grid_thw when present
    (e.g. Qwen3-VL). Optional keys (pixel_values_videos, video_grid_thw)
    are concatenated if present.

    Args:
        pad_token_id (int): Token id for padding input_ids.
        ignore_index (int): Value for padding labels (default -100).
    """

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int = -100,
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(
        self,
        instances: Sequence[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        if not instances:
            raise ValueError('VLMSFTCollator received empty batch.')

        # Pad sequences: list of (L,) tensors -> (B, max_L)
        input_ids = [
            x['input_ids'].squeeze(0)
            if x['input_ids'].dim() > 1 else x['input_ids'] for x in instances
        ]
        labels = [
            x['labels'].squeeze(0) if x['labels'].dim() > 1 else x['labels']
            for x in instances
        ]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index)

        attention_mask_list = [x.get('attention_mask') for x in instances]
        if attention_mask_list[0] is not None:
            attention_mask_list = [
                a.squeeze(0) if a.dim() > 1 else a for a in attention_mask_list
            ]
            attention_mask = pad_sequence(
                attention_mask_list, batch_first=True, padding_value=0)
        else:
            attention_mask = (input_ids != self.pad_token_id).long()

        out = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        # Optional: pixel_values, image_grid_thw (Qwen3-VL style)
        # HF expects image_grid_thw 2D (num_images, 3). Single-image
        # samples: (3,) -> unsqueeze to (1, 3) before concat to avoid (B*3,).
        pv_list = [
            x.get('pixel_values') for x in instances
            if x.get('pixel_values') is not None
        ]
        if pv_list:
            for i, pv in enumerate(pv_list):
                if pv.dim() == 3:
                    pv_list[i] = pv.unsqueeze(0)
            out['pixel_values'] = torch.cat(pv_list, dim=0)
            ig_list = [
                x.get('image_grid_thw') for x in instances
                if x.get('image_grid_thw') is not None
            ]
            if ig_list:
                for i, ig in enumerate(ig_list):
                    if ig.dim() == 1:
                        ig_list[i] = ig.unsqueeze(0)  # (3,) -> (1, 3)
                out['image_grid_thw'] = torch.cat(ig_list, dim=0)

        # Optional: pixel_values_videos, video_grid_thw
        pvv_list = [
            x.get('pixel_values_videos') for x in instances
            if x.get('pixel_values_videos') is not None
        ]
        if pvv_list:
            out['pixel_values_videos'] = torch.cat(pvv_list, dim=0)
            vg_list = [
                x.get('video_grid_thw') for x in instances
                if x.get('video_grid_thw') is not None
            ]
            if vg_list:
                out['video_grid_thw'] = torch.cat(vg_list, dim=0)

        return out
