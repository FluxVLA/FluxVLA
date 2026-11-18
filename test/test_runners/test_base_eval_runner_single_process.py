"""Direct eval and manager workers must work without a distributed logger."""

from types import SimpleNamespace

import pytest
import torch

from fluxvla.engines.runners import base_eval_runner
from fluxvla.engines.runners.base_eval_runner import BaseEvalRunner
from fluxvla.engines.utils.overwatch import PureOverwatch


def _initialize_common_attrs(monkeypatch, logger):
    monkeypatch.setattr(base_eval_runner, 'overwatch', logger)
    runner = BaseEvalRunner()
    runner.set_common_eval_attrs({}, 7, 'checkpoint.pt', 'smolvla', 'bf16',
                                 True)
    return runner


def test_direct_eval_uses_visible_gpu_zero_without_distributed_state(monkeypatch):
    runner = _initialize_common_attrs(monkeypatch,
                                     PureOverwatch('single_process_eval'))
    assert runner.device_id == 0
    assert runner.distributed_state is None


def test_distributed_eval_preserves_local_rank_and_state(monkeypatch):
    state = object()
    logger = SimpleNamespace(local_rank=lambda: 3, distributed_state=state)
    runner = _initialize_common_attrs(monkeypatch, logger)
    assert runner.device_id == 3
    assert runner.distributed_state is state


@pytest.mark.parametrize('build_dtype', [None, torch.float32])
def test_libero_setup_retains_explicit_parameter_precision(monkeypatch,
                                                         build_dtype):
    from fluxvla.engines.runners import libero_eval_runner

    class CPUModel(torch.nn.Linear):
        def to(self, *, device, dtype):
            # Exercise the real dtype conversion without requiring a GPU.
            return super().to(dtype=dtype)

    runner = libero_eval_runner.LiberoEvalRunner.__new__(
        libero_eval_runner.LiberoEvalRunner)
    runner.vla = CPUModel(1, 1, bias=False)
    with torch.no_grad():
        runner.vla.weight.fill_(1.0001)
    original = runner.vla.weight.detach().clone()
    runner.seed = 7
    runner.device_id = 0
    runner.model_build_device = None
    runner.model_build_dtype = build_dtype
    runner.enable_mixed_precision_training = True
    runner.mixed_precision_dtype = torch.bfloat16
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(libero_eval_runner, 'set_seed_everywhere', lambda _: None)
    runner.run_setup()
    expected = build_dtype or torch.bfloat16
    assert runner.vla.weight.dtype == expected
    assert torch.equal(runner.vla.weight, original.to(expected))
    if build_dtype is not None:
        assert torch.equal(runner.vla.weight, original)
