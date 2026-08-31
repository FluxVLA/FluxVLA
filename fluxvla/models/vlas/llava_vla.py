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

from pathlib import Path
from typing import Dict, List, Optional

import torch

from fluxvla.engines import VLAS, initialize_overwatch
from fluxvla.models.backbones.vlms.outputs import VLMBackboneOutput
from .base_vla import BaseVLA
from .continuous_action_utils import (add_auxiliary_losses,
                                      normalize_action_head_output)

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class LlavaVLA(BaseVLA):
    """
    Continuous-action VLA wrapper for LLaVA-style architectures.

    The class inherits directly from :class:`BaseVLA`. It supports either a
    combined ``vlm_backbone`` or separate vision/LLM/projector modules, and
    forwards the resulting hidden states to a continuous-action head.

    Args:
        vla_head (Dict): Configuration dictionary for the VLA head.
        vision_backbone (Dict, optional): Configuration dictionary for the
            vision backbone. Defaults to None.
        llm_backbone (Dict, optional): Configuration dictionary for the LLM
            backbone. Defaults to None.
        vlm_backbone (Dict, optional): Configuration dictionary for the VLM
            backbone. Defaults to None.
        projector (Dict, optional): Configuration dictionary for the projector.
            Defaults to None.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training. Defaults to True.
        freeze_vision_backbone (bool): Whether to freeze the weight of the
            vision backbone. Defaults to True.
        freeze_llm_backbone (bool): Whether to freeze the weight of the LLM
            backbone. Defaults to True.
        freeze_projector (bool): Whether to freeze the weight of the projector.
            Defaults to False.
        freeze_vlm_backbone (bool): Whether to freeze the weight of the VLM
            backbone. Defaults to False.
        vision_backbone_fp32 (bool): Whether to use fp32 for training the
            vision backbone. Defaults to False.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer of the
            model. Defaults to False.
        ignore_index (int): The index to ignore in the loss computation.
            Defaults to -100.
        pretrained_name_or_path (Path, optional): Path to the pretrained model.
            Defaults to None.
        freeze_weights (bool): Whether to freeze the weights of the model.
            Defaults to False.
        norm_stats (Dict, optional): Normalization statistics for the model.
            Defaults to None.
        name_mapping (Dict, optional): Mapping of names for the model.
            Defaults to None.
    """

    def __init__(self,
                 vla_head: Dict,
                 vision_backbone: Dict = None,
                 llm_backbone: Dict = None,
                 vlm_backbone: Dict = None,
                 projector: Dict = None,
                 enable_mixed_precision_training: bool = True,
                 freeze_vision_backbone: bool = True,
                 freeze_llm_backbone: bool = True,
                 freeze_projector: bool = False,
                 freeze_vlm_backbone: bool = False,
                 vision_backbone_fp32: bool = False,
                 unfreeze_last_layer: bool = False,
                 ignore_index: int = -100,
                 freeze_weights: bool = False,
                 norm_stats: Dict = None,
                 pretrained_name_or_path: Path = None,
                 name_mapping: Dict = None,
                 strict_mapping: bool = False,
                 freeze_vla_head: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            projector=projector,
            vlm_backbone=vlm_backbone,
            vla_head=vla_head,
            enable_mixed_precision_training=enable_mixed_precision_training,
            freeze_vision_backbone=freeze_vision_backbone,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_projector=freeze_projector,
            freeze_vlm_backbone=freeze_vlm_backbone,
            freeze_vla_head=freeze_vla_head,
            vision_backbone_fp32=vision_backbone_fp32,
            unfreeze_last_layer=unfreeze_last_layer,
            ignore_index=ignore_index,
            norm_stats=norm_stats,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping)
        # Keep the same runner-facing module metadata and initialization state
        # that LlavaVLA previously received through OpenVLA.
        self.all_module_keys = [
            'vision_backbone', 'llm_backbone', 'projector', 'head'
        ]
        self.string2idx = {}

    def forward(self,
                lang_tokens: Optional[torch.LongTensor] = None,
                lang_masks: Optional[torch.Tensor] = None,
                images: Optional[torch.FloatTensor] = None,
                states: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                multimodal_indices: Optional[torch.LongTensor] = None,
                actions: Optional[torch.FloatTensor] = None,
                dataset_names: Optional[List[str]] = None,
                img_masks: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                embodiment_ids: Optional[torch.LongTensor] = None,
                action_masks: Optional[torch.Tensor] = None,
                *args,
                **kwargs) -> Dict:
        """
        Forward pass for the LlavaVLA model.
        """

        backbone_output = None
        # Check if the model has a VLM backbone and use it if available.
        if hasattr(self, 'vlm_backbone') and self.vlm_backbone is not None:
            backbone_output = self.vlm_backbone(
                images=images,
                lang_tokens=lang_tokens,
                img_masks=img_masks,
                lang_masks=lang_masks,
                image_grid_thw=image_grid_thw)
            if isinstance(backbone_output, VLMBackboneOutput):
                last_hidden_state = backbone_output.last_hidden_state
                fused_attention_mask = backbone_output.attention_mask
                auxiliary_losses = backbone_output.auxiliary_losses
            else:
                last_hidden_state, fused_attention_mask, _ = backbone_output
                auxiliary_losses = {}
        else:
            output, fused_attention_mask = self.forward_model(
                input_ids=lang_tokens,
                attention_mask=lang_masks,
                pixel_values=images,
                labels=labels,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                multimodal_indices=multimodal_indices,
            )
            if 'hidden_states' in output:
                last_hidden_state = output['hidden_states'][-1]
            else:
                assert 'last_hidden_state' in output, \
                    'Output must contain either hidden_states or last_hidden_state.'  # noqa: E501
                last_hidden_state = output['last_hidden_state']
            auxiliary_losses = {}
        head_kwargs = dict(
            input_features=last_hidden_state,
            states=states,
            attention_mask=fused_attention_mask,
            actions=actions,
            action_masks=action_masks,
            embodiment_ids=embodiment_ids,
            sample_weight=kwargs.get('sample_weight'))
        if isinstance(backbone_output, VLMBackboneOutput):
            image_mask = backbone_output.auxiliary_outputs.get('image_mask')
            if image_mask is not None:
                head_kwargs['image_mask'] = image_mask
        ret_dict = self.vla_head(**head_kwargs)
        if not auxiliary_losses and not torch.is_tensor(ret_dict):
            return ret_dict
        return add_auxiliary_losses(
            normalize_action_head_output(ret_dict), auxiliary_losses)

    def predict_action(self,
                       images: List[torch.Tensor],
                       lang_tokens: torch.Tensor,
                       states: torch.Tensor,
                       img_masks: Optional[torch.Tensor] = None,
                       lang_masks: Optional[torch.Tensor] = None,
                       image_grid_thw: Optional[torch.Tensor] = None,
                       embodiment_ids: Optional[torch.Tensor] = None,
                       prev_actions: Optional[torch.Tensor] = None,
                       prefix_len: int = 0,
                       rtc_config: Optional[Dict] = None,
                       seed: Optional[int] = None,
                       *args,
                       **kwargs):
        backbone_output = None
        if hasattr(self, 'vlm_backbone') and self.vlm_backbone is not None:
            backbone_output = self.vlm_backbone(
                images=images,
                lang_tokens=lang_tokens,
                img_masks=img_masks,
                lang_masks=lang_masks,
                image_grid_thw=image_grid_thw)
            if isinstance(backbone_output, VLMBackboneOutput):
                last_hidden_state = backbone_output.last_hidden_state
                fused_attention_mask = backbone_output.attention_mask
            else:
                last_hidden_state, fused_attention_mask, _ = backbone_output
        else:
            output, fused_attention_mask = self.forward_model(
                input_ids=lang_tokens,
                attention_mask=lang_masks,
                pixel_values=images)
            if 'hidden_states' in output:
                last_hidden_state = output['hidden_states'][-1]
            else:
                assert 'last_hidden_state' in output, \
                    'Output must contain either hidden_states or last_hidden_state.'  # noqa: E501
                last_hidden_state = output['last_hidden_state']
        predict_kwargs = dict(
            input_features=last_hidden_state,
            states=states,
            attention_mask=fused_attention_mask,
            embodiment_ids=embodiment_ids,
            prev_actions=prev_actions,
            prefix_len=prefix_len,
            rtc_config=rtc_config)
        if isinstance(backbone_output, VLMBackboneOutput):
            image_mask = backbone_output.auxiliary_outputs.get('image_mask')
            if image_mask is not None:
                predict_kwargs['image_mask'] = image_mask
        if seed is not None:
            predict_kwargs['seed'] = seed
        pred_actions = self.vla_head.predict_action(**predict_kwargs)
        return pred_actions.float()
