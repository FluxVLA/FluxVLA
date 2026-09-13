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

from fluxvla.engines.runners import libero_eval_runner
from fluxvla.engines.runners.libero_eval_runner import LiberoEvalRunner


class _CpuVLA:

    def eval(self):
        return self

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self


def test_libero_eval_run_dir_uses_checkpoint_stem(tmp_path):
    run_dir = LiberoEvalRunner._build_run_dir(
        '/tmp/checkpoints/step-10.pt',
        'EVAL-libero_object-model-date',
        output_dir=tmp_path,
        inference_tag='ignored-model')

    assert run_dir == str(tmp_path / 'eval_runs' / 'step-10' /
                          'EVAL-libero_object-model-date')


def test_libero_eval_run_dir_uses_model_for_checkpoint_free_eval(tmp_path):
    run_dir = LiberoEvalRunner._build_run_dir(
        None,
        'EVAL-libero_object-gpt6-astra-date',
        output_dir=tmp_path,
        inference_tag='gpt6-astra')

    assert run_dir == str(tmp_path / 'eval_runs' / 'gpt6-astra' /
                          'EVAL-libero_object-gpt6-astra-date')


def test_libero_cpu_model_still_binds_rank_cuda_device(monkeypatch):
    runner = object.__new__(LiberoEvalRunner)
    runner.seed = 7
    runner.device_id = 1
    runner.model_build_device = 'cpu'
    runner.vla = _CpuVLA()

    selected_devices = []
    monkeypatch.setattr(libero_eval_runner, 'set_seed_everywhere',
                        lambda _: None)
    monkeypatch.setattr(libero_eval_runner.torch.cuda, 'is_available',
                        lambda: True)
    monkeypatch.setattr(libero_eval_runner.torch.cuda, 'set_device',
                        selected_devices.append)

    runner.run_setup()

    assert selected_devices == [1]
    assert runner.vla.to_kwargs == {'device': 'cpu'}
