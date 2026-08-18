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

from types import SimpleNamespace

import torch

from fluxvla.models.third_party_models.cosmos3.data.vfm import sequence_packing
from fluxvla.models.vlas.cosmos3_flowmatching_training_mixins import \
    Cosmos3SequenceMixin

SequencePlan = sequence_packing.SequencePlan


class _LightweightCosmos3Packer(Cosmos3SequenceMixin):
    device = torch.device('cpu')
    special_tokens = {
        'eos_token_id': 1,
        'start_of_generation': 2,
        'end_of_generation': 3,
    }
    latent_patch_size = 1
    position_embedding_type = 'unified_3d_mrope'
    unified_3d_mrope_reset_spatial_ids = True
    unified_3d_mrope_temporal_modality_margin = 0
    enable_fps_modulation = False
    base_fps = 24.0
    max_action_dim = 4
    vision_vae = SimpleNamespace(temporal_compression_factor=4)


def test_cosmos3_sequence_packer_consumes_collated_action_inputs():
    packer = _LightweightCosmos3Packer()
    sequence_plan = SequencePlan(
        has_text=True,
        has_vision=False,
        has_action=True,
        condition_frame_indexes_action=[],
    )

    packed = packer._pack_generation_data(
        sequence_plans=[sequence_plan],
        text_token_ids=[[4, 5]],
        vision_tokens=None,
        action_tokens=[torch.zeros(2, 4)],
        embodiment_ids=[torch.tensor(5)],
        raw_action_dim=[torch.tensor(2)],
        timesteps=torch.tensor([[10.]]),
        fps_vision=None,
        fps_action=None,
    )

    assert packed.text_ids.tolist() == [4, 5, 1, 2]
    assert packed.text_indexes.tolist() == [0, 1, 2, 3]
    assert packed.action.sequence_indexes.tolist() == [4, 5]
    assert packed.action.timesteps.tolist() == [10.0, 10.0]
    assert packed.action.domain_id[0].item() == 5
    assert packed.action.raw_action_dim[0].item() == 2
    assert packed.action.condition_mask[0].tolist() == [[0.0], [0.0]]


def test_cosmos3_sequence_packer_keeps_conditioned_action_out_of_mse_loss():
    packer = _LightweightCosmos3Packer()
    sequence_plan = SequencePlan(
        has_text=True,
        has_vision=False,
        has_action=True,
        condition_frame_indexes_action=[0],
    )

    packed = packer._pack_generation_data(
        sequence_plans=[sequence_plan],
        text_token_ids=[[4]],
        vision_tokens=None,
        action_tokens=[torch.ones(2, 4)],
        embodiment_ids=[torch.tensor(5)],
        raw_action_dim=[torch.tensor(4)],
        timesteps=torch.tensor([[20.]]),
        fps_vision=None,
        fps_action=None,
    )

    assert packed.action.condition_mask[0].tolist() == [[1.0], [0.0]]
    assert packed.action.noisy_frame_indexes[0].tolist() == [1]
    assert packed.action.mse_loss_indexes.tolist() == [4]
