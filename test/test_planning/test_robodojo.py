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
"""RoboDojo config contracts; no Themis plugin or simulator required."""

from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/hierarchical/robodojo_pi05_gpt6.py'


@pytest.fixture
def config(monkeypatch):
    for key in ('PI05_ROBODOJO_CHECKPOINT_DIR', 'PI05_ROBODOJO_CHECKPOINT',
                'ROBODOJO_SUBTASK_DATA_ROOT', 'OPENAI_PLANNER_MODEL'):
        monkeypatch.delenv(key, raising=False)
    return Config.fromfile(CONFIG)


def test_checkpoint_and_preprocessing_contract(config):
    cfg = config
    assert cfg.inference_model.name_mapping == {}
    assert cfg.inference_model.ori_action_dim == 14
    assert cfg.themis.evaluation.execute_horizon == 8
    assert cfg.model.pretrained_name_or_path == cfg.themis.server.ckpt_path
    assert cfg.themis.server.ckpt_path.endswith(
        'step-060000-epoch-04-loss=0.0011.safetensors')
    assert cfg.eval.denormalize_action.delta_action_mask == ([True] * 6 +
                                                             [False] +
                                                             [True] * 6 +
                                                             [False])
    prompt = next(transform for transform in cfg.eval.dataset.transforms
                  if transform.type == 'ProcessPrompts')
    assert prompt.tokenizer.model_path == str(
        Path(cfg.themis.server.norm_stats_path).parent / 'tokenizer')


def test_full_schedule_inherits_standard_benchmark(config):
    baseline = Config.fromfile(
        ROOT / 'configs/pi05/pi05_paligemma_robodojo_full_finetune.py')
    evaluation = config.themis.evaluation
    assert evaluation.environment.task_name == 'all'
    assert evaluation.episodes_per_task == 50
    assert evaluation.max_episode_steps == 2000
    assert evaluation.episodes_per_task_overrides == (
        baseline.themis.evaluation.episodes_per_task_overrides)
    assert config.eval.num_trials_per_task == baseline.eval.num_trials_per_task
    assert config.eval.num_trials_per_task_overrides == (
        baseline.eval.num_trials_per_task_overrides)


def test_ordinary_themis_protocol_and_top_level_policy(config):
    evaluation = config.themis.evaluation
    client = evaluation.policy_client
    assert client.type == 'FluxVLAPolicyClient'
    assert not client.get('send_policy_context', False)
    assert not client.get('send_environment_feedback', False)
    assert evaluation.environment.type == 'RoboDojoEnvironment'
    assert config.themis.server.mode == 'stateless'
    assert config.themis.server.runner.policy_execution == 'per_request'
    assert 'custom_imports' not in config.themis
    assert 'policy' not in config.eval
    policy = config.policy
    assert policy.type == 'HierarchicalPolicy'
    assert 'skill_catalog' not in policy
    assert policy.subtask_source.data_root == (
        '/mnt/data/oss/datasets/RoboDojo_lerobot_v3')
    assert policy.subtask_source.subtask_prefix == 'dense'
    assert policy.subtask_source.max_candidates == 4096
    aliases = policy.subtask_source.task_aliases
    assert aliases == {
        'Pick up the broom, hand it over to the right hand, then use '
        'the dustpan to sweep the objects.':
        'Pick up the broom, hand it over to the right hand, then use '
        'the dustpan to sweep the blocks.'
    }
    assert policy.replan_every_chunks == 1
    assert not policy.planner.allow_request_finish
    assert policy.completion_mode == policy.planner.completion_mode == (
        'disabled')
    assert policy.monitor is policy.goal_conditions is None
    assert policy.planner_image_keys == policy.planner.image_keys
    assert len(policy.planner_image_keys) == 3


def test_candidate_data_path_override(monkeypatch):
    monkeypatch.setenv('ROBODOJO_SUBTASK_DATA_ROOT', '/tmp/training-data')
    cfg = Config.fromfile(CONFIG)
    assert cfg.policy.subtask_source.data_root == '/tmp/training-data'


@pytest.mark.parametrize('override', [None, '', 'test-planner'])
def test_planner_model_default_and_override(monkeypatch, override):
    if override is None:
        monkeypatch.delenv('OPENAI_PLANNER_MODEL', raising=False)
    else:
        monkeypatch.setenv('OPENAI_PLANNER_MODEL', override)
    cfg = Config.fromfile(CONFIG)
    assert cfg.policy.planner.model == (override or 'gpt-5.6-sol')


def test_debug_schedule_and_output_paths_are_runtime_overrides(config):
    reporting = config.themis.server.evaluation_reporting
    assert 'result_output_dir' not in reporting
    assert 'video_dir' not in config.themis.evaluation.environment
    config.merge_from_dict({
        'themis.evaluation.environment.task_name': ['stack_blocks', 'push_T'],
        'themis.evaluation.episodes_per_task':
        5,
        'themis.evaluation.episodes_per_task_overrides._delete_':
        True,
        'themis.evaluation.max_episode_steps':
        550,
        'themis.server.evaluation_reporting.result_output_dir':
        '/tmp/rollout',
        'eval.num_trials_per_task':
        5,
        'eval.num_trials_per_task_overrides._delete_':
        True,
    })
    assert config.themis.evaluation.environment.task_name == [
        'stack_blocks', 'push_T'
    ]
    assert config.themis.evaluation.episodes_per_task == 5
    assert config.themis.evaluation.episodes_per_task_overrides == {}
    assert config.themis.evaluation.max_episode_steps == 550
    assert config.themis.server.evaluation_reporting.result_output_dir == (
        '/tmp/rollout')
    assert config.eval.num_trials_per_task == 5
    assert config.eval.num_trials_per_task_overrides == {}
