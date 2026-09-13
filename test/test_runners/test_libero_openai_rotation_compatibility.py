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

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluxvla.datasets.openai_eval_dataset import OpenAILiberoEvalDataset
from fluxvla.engines.runners import libero_eval_runner as module
from fluxvla.engines.runners.libero_eval_runner import LiberoEvalRunner
from fluxvla.models.vlas.openai_responses_vla import OpenAIResponsesVLA
from fluxvla.transforms.normalize import IdentityLiberoAction


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('sign', [-1, 1])
def test_existing_runner_delivers_rotation_without_special_handling(
        monkeypatch, tmp_path, axis, sign):
    """Use the original run loop; fake only HTTP, physics and distribution."""
    observations = {
        'agentview_image': np.zeros((16, 16, 3), dtype=np.uint8),
        'robot0_eye_in_hand_image': np.zeros((16, 16, 3), dtype=np.uint8),
        'robot0_eef_pos': np.zeros(3),
        'robot0_eef_quat': np.array([0, 0, 0, 1]),
        'robot0_gripper_qpos': np.zeros(2),
        'robot0_joint_pos': np.zeros(7),
    }

    class FakeEnv:
        closed = False

        def reset(self):
            self.actions = []

        def set_init_state(self, _):
            return observations.copy()

        def step(self, action):
            self.actions.append(action)
            return observations.copy(), 0, len(self.actions) == 10, {}

        def close(self):
            self.closed = True

    env = FakeEnv()
    task = SimpleNamespace(language='align the wrist')
    suite = SimpleNamespace(
        n_tasks=1,
        get_task=lambda _: task,
        get_task_init_states=lambda _: [None])
    benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {'libero_10': lambda: suite})
    monkeypatch.setattr(module, '_get_libero_benchmark', lambda: benchmark)
    monkeypatch.setattr(module, 'get_libero_env', lambda *args, **kwargs:
                        (env, task.language))
    monkeypatch.setattr(module.overwatch, 'rank', lambda: 0)
    monkeypatch.setattr(module.overwatch, 'world_size', lambda: 1)
    monkeypatch.setattr(module.torch.cuda, 'current_device', lambda: 'cpu')
    for name in ('broadcast_object_list', 'all_reduce', 'barrier'):
        monkeypatch.setattr(module.dist, name, lambda *args, **kwargs: None)

    model = OpenAIResponsesVLA(action_horizon=8)
    rotation = np.eye(3)[axis] * sign * 0.2
    monkeypatch.setattr(
        model, '_post_json', lambda _: {
            'output': [{
                'type':
                'function_call',
                'name':
                'move_to',
                'call_id':
                'rotate',
                'arguments':
                json.dumps({
                    'targets': {
                        'rotation_delta': rotation.tolist()
                    },
                    'note': 'rotate'
                }),
            }]
        })
    runner = object.__new__(LiberoEvalRunner)
    attributes = dict(
        cfg=SimpleNamespace(filename=None),
        ckpt_path=None,
        task_suite_name='libero_10',
        task_ids=None,
        model_family='test',
        num_trials_per_task=1,
        num_steps_wait=2,
        max_steps=8,
        seed=7,
        eval_chunk_size=8,
        resize_size=256,
        num_inference_steps=None,
        inference_seed=None,
        model_build_device='cpu',
        model_build_dtype=None,
        mixed_precision_dtype=torch.bfloat16,
        enable_mixed_precision_training=False,
        eval_shard_strategy='episode',
        run_id_suffix=None,
        output_dir=str(tmp_path),
        result_output_dir=None,
        result_gpu_id=0,
        norm_stats_key=None,
        preprocess_every_step=False,
        save_rollout_videos=False,
        save_failed_rollout_videos=False,
        save_multi_view_rollout_videos=False,
        rollout_dir=None,
        dataset=OpenAILiberoEvalDataset(resize_size=16),
        denormalize_action=IdentityLiberoAction(),
        vla=model,
    )
    for name, value in attributes.items():
        setattr(runner, name, value)
    runner.run()

    assert env.closed
    assert len(env.actions) == 10  # Two settling steps + eight native actions.
    assert model._llm_calls == 1
    expected = np.zeros(7)
    expected[3 + axis] = sign * 0.25
    expected[6] = -1
    np.testing.assert_allclose(env.actions[2:], np.tile(expected, (8, 1)))
    summary = json.loads((Path(runner.run_dir) / 'summary.json').read_text())
    assert summary['suite_stats']['libero_10']['total_successes'] == 1
