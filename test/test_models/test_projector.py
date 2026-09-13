# Copyright 2026 Limx Dynamics
"""Exercise real projectors on CPU, without TensorFlow or checkpoints."""

import pytest
import torch

from fluxvla.engines import build_projector_from_cfg


@pytest.mark.parametrize('config', [
    dict(type='LinearProjector', in_dim=4, out_dim=8),
    dict(type='MLPProjector', vision_dim=4, llm_dim=8),
    dict(type='FusedMLPProjector', fused_vision_dim=4, llm_dim=8),
])
def test_projector_forward_backward_and_reload(config):
    projector = build_projector_from_cfg(config)
    inputs = torch.linspace(-1, 1, 24).reshape(2, 3, 4).requires_grad_()
    output = projector(inputs)

    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    for parameter in projector.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    restored = build_projector_from_cfg(config)
    restored.load_state_dict(projector.state_dict(), strict=True)
    torch.testing.assert_close(restored(inputs), output, rtol=0, atol=0)
