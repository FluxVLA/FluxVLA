import inspect

import torch

from fluxvla.engines.runners.fsdp2_grad_clip import clip_grad_norm_fsdp2
from fluxvla.engines.runners.fsdp_train_runner import FSDPTrainRunner


def test_fsdp2_clip_uses_global_p_norm_without_distributed_runtime():
    first = torch.nn.Parameter(torch.zeros(3))
    second = torch.nn.Parameter(torch.zeros(4))
    first.grad = torch.tensor([3.0, 0.0, 0.0])
    second.grad = torch.tensor([0.0, 4.0, 0.0, 0.0])

    norm = clip_grad_norm_fsdp2([first, second], max_norm=1.0)

    assert torch.allclose(norm, torch.tensor(5.0))
    assert torch.allclose(first.grad, torch.tensor([0.6, 0.0, 0.0]))
    assert torch.allclose(second.grad, torch.tensor([0.0, 0.8, 0.0, 0.0]))


def test_fsdp2_grad_norm_input_trace_is_step_gated_and_rank_local():
    source = inspect.getsource(FSDPTrainRunner._write_grad_norm_input_trace)

    assert "FLUXVLA_GRAD_NORM_INPUT_TRACE_STEP" in source
    assert 'hasattr(local_grad, "to_local")' in source
    assert '"grad_norms"' in source
    assert '"grad_placements"' in source
    assert '"grad_device_mesh_shape"' in source


def test_fsdp2_clip_trace_captures_dtensor_norm_intermediates():
    source = inspect.getsource(FSDPTrainRunner.clip_grad_norm)

    assert "per_parameter_norm_metadata" in source
    assert "stacked_norms_metadata" in source
    assert "computed_total_norm_metadata" in source
    assert "clip_grad_module._get_total_norm = traced_get_total_norm" in source
