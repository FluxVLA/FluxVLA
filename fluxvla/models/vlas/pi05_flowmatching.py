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

from fluxvla_export.components.pi_policy import PI05Computation

from fluxvla.engines import VLAS
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .pi0_flowmatching import PI0FlowMatching

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI05FlowMatching(PI05Computation, PI0FlowMatching):
    """PI0 Flow Matching Model for Vision-Language Alignment.
    Implemented based on https://arxiv.org/abs/2504.16054

    This model is designed to handle vision-language alignment tasks
    using flow matching techniques, leveraging a vision backbone,
    language model backbone, projector, and a VLA head.

    Args:
        state_proj (Dict): Configuration dictionary for the state
            projector.
        action_in_proj (Dict): Configuration dictionary for the action
            input projector.
        action_out_proj (Dict): Configuration dictionary for the action
            output projector.
        action_time_mlp_in (Dict): Configuration dictionary for the action
            time MLP input.
        action_time_mlp_out (Dict): Configuration dictionary for the action
            time MLP output.
        vlm_backbone (str): Identifier for the vision-language model backbone.
        vla_head (str): Identifier for the vision-language alignment head.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
        freeze_vision_backbone (bool): Whether to freeze the vision backbone.
        freeze_llm_backbone (bool): Whether to freeze the language model
            backbone.
        freeze_projector (bool): Whether to freeze the projector.
        vision_backbone_fp32 (bool): Whether to use FP32 for the vision
            backbone.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer
            of the model.
        ignore_index (int): Index to ignore in loss calculations.
        norm_stats (Dict, optional): Normalization statistics for the model.
        **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(self, **kwargs):
        assert kwargs.get('state_proj') is None, (
            'PI05FlowMatching does not use state_proj.')
        super().__init__(**kwargs)
