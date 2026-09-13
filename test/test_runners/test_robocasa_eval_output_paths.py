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

from mmengine import Config

from fluxvla.engines import build_runner_from_cfg
from fluxvla.engines.runners import base_eval_runner, robocasa_eval_runner
from fluxvla.engines.runners.robocasa_eval_runner import RobocasaEvalRunner


class _CpuVLA:

    def eval(self):
        return self

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self


def test_robocasa_eval_run_dir_uses_checkpoint_stem(tmp_path):
    run_dir = RobocasaEvalRunner._build_run_dir(
        '/tmp/checkpoints/step-10.pt',
        'EVAL-robocasa-model-date',
        output_dir=tmp_path,
        inference_tag='ignored-model')

    assert run_dir == str(tmp_path / 'eval_runs' / 'step-10' /
                          'EVAL-robocasa-model-date')


def test_robocasa_eval_run_dir_uses_model_for_checkpoint_free_eval(tmp_path):
    run_dir = RobocasaEvalRunner._build_run_dir(
        None,
        'EVAL-robocasa-gpt6-astra-date',
        output_dir=tmp_path,
        inference_tag='gpt6-astra')

    assert run_dir == str(tmp_path / 'eval_runs' / 'gpt6-astra' /
                          'EVAL-robocasa-gpt6-astra-date')


def test_robocasa_runner_builds_without_checkpoint_or_stats(monkeypatch):
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'local_rank', lambda: 0, raising=False)
    monkeypatch.setattr(
        base_eval_runner.overwatch, 'distributed_state', None, raising=False)
    cfg = Config.fromfile(
        Path(__file__).resolve().parents[2] /
        'configs/openai/gpt6_astra_robocasa_inference.py')
    runner_cfg = cfg.eval.copy()
    runner_cfg.cfg = cfg
    runner_cfg.ckpt_path = None

    runner = build_runner_from_cfg(runner_cfg)
    runner.run_setup()

    assert runner.ckpt_path is None
    assert runner.vla.__class__.__name__ == 'OpenAIResponsesRobocasaVLA'
    assert runner.dataset.__class__.__name__ == 'OpenAIRobocasaEvalDataset'
    assert runner.denormalize_action.__class__.__name__ == \
        'IdentityRobocasaAction'
    assert next(runner.vla.buffers()).device.type == 'cpu'


def test_robocasa_cpu_model_still_binds_rank_cuda_device(monkeypatch):
    runner = object.__new__(RobocasaEvalRunner)
    runner.seed = 7
    runner.device_id = 1
    runner.model_build_device = 'cpu'
    runner.vla = _CpuVLA()

    selected_devices = []
    monkeypatch.setattr(robocasa_eval_runner, 'set_seed_everywhere',
                        lambda _: None)
    monkeypatch.setattr(robocasa_eval_runner.torch.cuda, 'is_available',
                        lambda: True)
    monkeypatch.setattr(robocasa_eval_runner.torch.cuda, 'set_device',
                        selected_devices.append)

    runner.run_setup()

    assert selected_devices == [1]
    assert runner.vla.to_kwargs == {'device': 'cpu'}
