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

from mmengine import Config

NANO_CONFIG = 'configs/cosmos3/cosmos3nano_libero_10_full_finetune.py'
EDGE_CONFIG = 'configs/cosmos3/cosmos3edge_libero_10_full_finetune.py'


def _get_transform(transforms, transform_type):
    return next(transform for transform in transforms
                if transform.type == transform_type)


def test_cosmos3_nano_libero_training_contract():
    cfg = Config.fromfile(NANO_CONFIG)

    assert cfg.model.type == 'Cosmos3FlowMatching'
    assert cfg.model.max_action_dim == 64
    assert cfg.model.action_horizon == 16
    assert cfg.model.rectified_flow_training_config.shift == {
        '256': 3,
        '480': 5,
        '720': 10,
    }
    assert (
        cfg.model.rectified_flow_training_config.independent_action_schedule is
        False)
    assert cfg.model.rectified_flow_training_config.action_loss_weight == 10.0

    assert cfg.train_dataloader.per_device_batch_size == 16
    assert cfg.runner.max_steps == 2_000
    assert cfg.runner.grad_accumulation_steps == 8
    assert cfg.runner.lr_scheduler.warmup_steps == 500
    assert cfg.eval.model_family == 'cosmos3'
    assert cfg.eval.eval_chunk_size == 16


def test_cosmos3_nano_data_contract_matches_model_inputs():
    cfg = Config.fromfile(NANO_CONFIG)
    transforms = cfg.train_dataloader.dataset.datasets.transforms
    prompt = _get_transform(transforms, 'ProcessCosmos3Prompt')
    sequence = _get_transform(transforms, 'BuildCosmos3Sequence')
    collator = cfg.runner.collator

    assert prompt.max_len == 512
    assert prompt.format_prompt_as_json is True
    assert prompt.action_metadata.frame_window_size == 17
    assert prompt.action_metadata.conditioning_fps == 20.0
    assert (prompt.tokenizer.model_path ==
            './checkpoints/Cosmos3-Nano/text_tokenizer')

    assert sequence.mode == 'wam'
    assert sequence.frame_window_size == 17
    assert sequence.raw_action_dim == 10
    assert sequence.prepend_state_to_action is False

    assert collator.type == 'DictCollator'
    assert 'text_token_ids' in collator.meta_keys
    assert 'sequence_plan' in collator.meta_keys
    assert set(collator['keys']) >= {
        'images', 'actions', 'embodiment_ids', 'raw_action_dim',
        'conditioning_fps', 'action_fps'
    }


def test_cosmos3_edge_keeps_nano_action_sequence_contract():
    nano = Config.fromfile(NANO_CONFIG)
    edge = Config.fromfile(EDGE_CONFIG)
    edge_transforms = edge.train_dataloader.dataset.datasets.transforms
    edge_sequence = _get_transform(edge_transforms, 'BuildCosmos3Sequence')

    assert edge.model.type == 'Cosmos3FlowMatching'
    assert edge.model.max_action_dim == nano.model.max_action_dim
    assert edge.model.action_horizon == nano.model.action_horizon
    assert edge_sequence.mode == 'wam'
    assert edge_sequence.frame_window_size == 17
    assert edge_sequence.raw_action_dim == 10
    assert edge.runner.collator.type == 'DictCollator'
    assert edge.eval.model_family == 'cosmos3'
