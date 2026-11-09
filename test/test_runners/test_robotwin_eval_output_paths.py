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

from fluxvla.engines.runners import robotwin_eval_runner
from fluxvla.engines.runners.robotwin_eval_runner import RobotwinEvalRunner


def test_robotwin_eval_run_dir_uses_checkpoint_stem(tmp_path):
    ckpt_root = RobotwinEvalRunner._build_ckpt_root(
        '/tmp/checkpoints/step-10.pt', output_dir=tmp_path)
    run_id = RobotwinEvalRunner._build_run_id('model', 'date')
    run_dir = str(ckpt_root / run_id)

    assert run_dir == str(tmp_path / 'eval_runs' / 'step-10' /
                          'EVAL-robotwin-model-date')


def test_robotwin_eval_run_dir_defaults_to_checkpoint_parent(tmp_path):
    ckpt_root = RobotwinEvalRunner._build_ckpt_root(
        str(tmp_path / 'checkpoints' / 'step-10.pt'))
    run_id = RobotwinEvalRunner._build_run_id('model', 'date')
    run_dir = str(ckpt_root / run_id)

    assert run_dir == str(tmp_path / 'eval_runs' / 'step-10' /
                          'EVAL-robotwin-model-date')


def test_robotwin_setup_binds_rank_cuda_device(tmp_path, monkeypatch):
    runner = object.__new__(RobotwinEvalRunner)
    runner.seed = 7
    runner.device_id = 1
    runner.rank = 0
    runner.world_size = 1
    runner.run_dir = str(tmp_path / 'eval_runs')
    runner.ckpt_path = str(tmp_path / 'step-10.pt')
    runner.robotwin_root = tmp_path / 'RoboTwin'
    runner.task_suite_name = 'clean'
    runner.task_list = ['beat_block_hammer']
    runner.local_task_list = runner.task_list
    runner.eval_chunk_size = 16
    runner.num_trials_per_task = 100

    Path(runner.ckpt_path).touch()
    eval_script = runner.robotwin_root / 'script' / 'eval_policy.py'
    eval_script.parent.mkdir(parents=True)
    eval_script.touch()

    selected_devices = []
    policy_args = []
    monkeypatch.setattr(robotwin_eval_runner, 'set_seed_everywhere',
                        lambda _: None)
    monkeypatch.setattr(robotwin_eval_runner.torch.cuda, 'is_available',
                        lambda: True)
    monkeypatch.setattr(robotwin_eval_runner.torch.cuda, 'set_device',
                        selected_devices.append)
    monkeypatch.setattr(robotwin_eval_runner, 'get_model', policy_args.append)
    monkeypatch.setattr(runner, '_log_rank', lambda _: None)
    monkeypatch.setattr(runner, '_load_task_progress', lambda: {})
    monkeypatch.setattr(runner, '_write_summary_json', lambda _: None)
    monkeypatch.setattr(runner, '_setup_vulkan_icd', lambda: None)
    monkeypatch.setattr(runner, '_base_policy_args',
                        lambda: dict(device=f'cuda:{runner.device_id}'))

    runner.run_setup()

    assert selected_devices == [1]
    assert policy_args == [dict(device='cuda:1')]
