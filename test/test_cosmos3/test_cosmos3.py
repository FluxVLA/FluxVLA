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

# isort: skip_file

import json

import numpy as np
import pytest
import torch

from fluxvla.collators.dict_collator import DictCollator
from fluxvla.models.vlas.cosmos3_flowmatching_utils import (
    _get_vision_data_resolution, resolve_inference_config, resolve_torch_dtype,
    resolve_training_config)
from fluxvla.models.vlas.cosmos3_flowmatching_training_mixins import (
    Cosmos3LossMixin, Cosmos3ScheduleMixin)
from fluxvla.transforms.transform_cosmos3 import (BuildCosmos3Sequence,
                                                  ProcessCosmos3Prompt,
                                                  build_sequence_plan_from_mode
                                                  )


class DummyTokenizer:

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        return list(range(10))


class DummyLossModel(torch.nn.Module, Cosmos3LossMixin):

    def __init__(self, normalize_loss_by_active=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.rectified_flow_training_config = {
            'normalize_loss_by_active': normalize_loss_by_active,
        }


class DummyScheduleModel(Cosmos3ScheduleMixin):

    def __init__(self):
        self.device = torch.device('cpu')
        self.rectified_flow_training_config = {}


def test_dict_collator_preserves_cosmos3_per_sample_fields():
    collator = DictCollator(
        keys=['images', 'embodiment_ids'],
        meta_keys=['text_token_ids', 'sequence_plan', 'task_description'])
    plans = [object(), object()]
    batch = [
        {
            'images': np.ones((1, 2), dtype=np.float32),
            'text_token_ids': [1, 2],
            'sequence_plan': plans[0],
            'task_description': 'pick',
            'embodiment_ids': np.array(5, dtype=np.int64),
        },
        {
            'images': np.zeros((1, 2), dtype=np.float32),
            'text_token_ids': np.array([3], dtype=np.int64),
            'sequence_plan': plans[1],
            'task_description': 'place',
            'embodiment_ids': np.array(6, dtype=np.int64),
        },
    ]

    result = collator(batch)

    assert result['images'].shape == (2, 1, 2)
    assert result['text_token_ids'][0] == [1, 2]
    assert result['text_token_ids'][1].tolist() == [3]
    assert result['sequence_plan'] == plans
    assert result['task_description'] == ['pick', 'place']
    assert result['embodiment_ids'].tolist() == [5, 6]


@pytest.mark.parametrize(
    ('mode', 'vision_indexes', 'action_indexes', 'action_offset'), [
        ('image2video', [0], [], 1),
        ('forward_dynamics', [0], [0, 1, 2, 3], 1),
        ('inverse_dynamics', [0, 1], [], 1),
        ('wam', [0], [], 1),
    ])
def test_sequence_plan_modes(mode, vision_indexes, action_indexes,
                             action_offset):
    plan = build_sequence_plan_from_mode(
        mode, video_length=5, action_length=4, video_temporal_downsample=4)

    assert plan.has_action is (mode != 'image2video')
    assert plan.condition_frame_indexes_vision == vision_indexes
    assert plan.condition_frame_indexes_action == action_indexes
    assert plan.action_start_frame_offset == action_offset


def test_build_cosmos3_sequence_prepends_state_and_history():
    transform = BuildCosmos3Sequence(
        mode='wam',
        frame_window_size=5,
        raw_action_dim=2,
        prepend_state_to_action=True)
    data = {
        'actions':
        np.array([[1., 2., 0.], [3., 4., 0.], [5., 6., 0.], [7., 8., 0.]],
                 dtype=np.float32),
        'states':
        np.array([9., 8., 0.], dtype=np.float32),
        'history_action':
        np.array([[7., 6., 0.]], dtype=np.float32),
        'embodiment_ids':
        np.array(5),
    }

    result = transform(data)

    assert result['actions'].tolist() == [[7., 6., 0.], [9., 8., 0.],
                                          [1., 2., 0.], [3., 4., 0.],
                                          [5., 6., 0.], [7., 8., 0.]]
    assert 'history_action' not in result
    assert result['raw_action_dim'].item() == 2
    assert result['sequence_plan'].condition_frame_indexes_action == [0, 1]
    assert result['sequence_plan'].action_start_frame_offset == -1
    assert result['conditioning_fps'].item() == pytest.approx(15.)


def test_build_cosmos3_image2video_removes_action_fields():
    transform = BuildCosmos3Sequence(mode='image2video', frame_window_size=5)
    result = transform({
        'actions': np.ones((4, 2)),
        'raw_action_dim': np.array(2),
        'embodiment_ids': np.array(5),
        'action_fps': np.array(15.),
    })

    assert result['sequence_plan'].has_action is False
    assert result['sequence_plan'].condition_frame_indexes_vision == [0]
    assert 'actions' not in result
    assert 'raw_action_dim' not in result
    assert 'embodiment_ids' not in result
    assert 'action_fps' not in result


def test_process_cosmos3_prompt_appends_metadata_and_truncates():
    transform = ProcessCosmos3Prompt(
        tokenizer={'type': 'unused'},
        max_len=3,
        use_system_prompt=True,
        action_metadata={
            'append_viewpoint': True,
            'append_duration_fps': True,
            'append_resolution': True,
        },
        output_attention_mask_key='lang_masks',
    )
    tokenizer = DummyTokenizer()
    transform._tokenizer = tokenizer

    result = transform({
        'task_description': 'Move the block',
        'viewpoint': 'wrist_view',
        'conditioning_fps': np.array(10., dtype=np.float32),
        'num_frames': np.array(20),
        'image_size': np.array([480, 640]),
    })

    assert 'wrist-mounted camera' in result['task_description']
    assert '2.0 seconds long and is of 10 FPS' in result['task_description']
    assert '480x640 resolution' in result['task_description']
    assert result['text_token_ids'].tolist() == [0, 1, 2]
    assert result['lang_masks'].tolist() == [True, True, True]
    conversations, kwargs = tokenizer.calls[0]
    assert conversations[0]['role'] == 'system'
    assert conversations[1]['content'] == result['task_description']
    assert kwargs['add_vision_id'] is False


def test_process_cosmos3_prompt_formats_action_json():
    transform = ProcessCosmos3Prompt(
        tokenizer={'type': 'unused'},
        action_metadata={'viewpoint': 'ego_view'},
        format_prompt_as_json=True,
    )
    transform._tokenizer = DummyTokenizer()

    result = transform({
        'task_description': 'open drawer',
        'conditioning_fps': np.array(15.),
        'frame_window_size': np.array(30),
        'video_height': np.array(480),
        'video_width': np.array(640),
        'action_horizon': np.array(29),
    })
    prompt = json.loads(result['task_description'])

    assert prompt['fps'] == 15.
    assert prompt['resolution'] == {'H': 480, 'W': 640}
    assert prompt['aspect_ratio'] == '4,3'
    assert prompt['actions'][0]['description'] == 'open drawer.'
    assert prompt['actions'][0]['idle_frame'] == '0 out of 29.'


def test_flow_config_resolution_and_resolution_boundaries():
    training = resolve_training_config({
        'train_time_vision_distribution': 'uniform',
        'train_time_weight': 'reweighting',
    })

    assert training['train_time_video_distribution'] == 'uniform'
    assert training['train_time_vision_distribution'] == 'uniform'
    assert training['train_time_weight'] == 'uniform'
    assert resolve_torch_dtype('bf16') is torch.bfloat16
    assert resolve_torch_dtype(torch.float32) is torch.float32
    assert _get_vision_data_resolution((256, 1024)) == '256'
    assert _get_vision_data_resolution((640, 1280)) == '480'
    assert _get_vision_data_resolution((960, 1280)) == '720'
    assert resolve_inference_config({'scheduler_type':
                                     'UNIPC'})['scheduler_type'] == 'unipc'

    with pytest.raises(ValueError, match='Unsupported Cosmos3 vision'):
        _get_vision_data_resolution((961, 1024))
    with pytest.raises(ValueError, match='scheduler_type'):
        resolve_inference_config({'scheduler_type': 'euler'})


def test_cosmos3_loss_ignores_conditioned_action_tokens():
    model = DummyLossModel(normalize_loss_by_active=True)
    loss = model._compute_action_loss(
        [torch.tensor([[3., 1.], [5., 1.]])],
        [torch.tensor([[1., 1.], [1., 1.]])],
        [torch.tensor(1)],
        [torch.tensor([[1.], [0.]])],
    )

    assert loss.item() == pytest.approx(16.)


def test_cosmos3_schedule_resolves_scalar_dict_and_dynamic_shifts():
    model = DummyScheduleModel()

    scalar = model._resolve_flow_shifts(2.5, batch_size=2, for_action=False)
    by_resolution = model._resolve_flow_shifts(
        {
            '256': 3,
            '480': 5,
        },
        batch_size=2,
        for_action=False,
        vision_resolutions=['256', '480'])
    dynamic = model._resolve_flow_shifts(
        {'dynamic_shift_base_num_tokens_video': 100},
        batch_size=2,
        for_action=False,
        num_tokens=[100, 400],
    )

    assert scalar.tolist() == [2.5, 2.5]
    assert by_resolution.tolist() == [3., 5.]
    assert dynamic.tolist() == [1., 2.]
    with pytest.raises(ValueError, match='vision-only'):
        model._resolve_flow_shifts({'256': 3}, batch_size=1, for_action=True)
