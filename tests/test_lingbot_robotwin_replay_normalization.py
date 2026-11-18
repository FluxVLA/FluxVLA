import json

import torch

from tools.lingbot_make_real_batch import _normalize_robotwin


def test_robotwin_normalization_uses_native_55d_layout(tmp_path):
    stats = {
        "observation.state.arm.position": {"q01": [0.0] * 12, "q99": [1.0] * 12},
        "observation.state.effector.position": {"q01": [0.0] * 2, "q99": [1.0] * 2},
        "action.arm.position": {"q01": [0.0] * 12, "q99": [1.0] * 12},
        "action.effector.position": {"q01": [0.0] * 2, "q99": [1.0] * 2},
    }
    raw = torch.arange(14, dtype=torch.float32).reshape(1, 14)
    output = _normalize_robotwin(raw, stats, "action")
    assert output.shape == (1, 55)
    assert torch.allclose(
        output[:, :12], torch.cat((raw[:, :6], raw[:, 7:13]), dim=-1) * 2 - 1
    )
    assert torch.equal(output[:, 12:28], torch.zeros(1, 16))
    assert torch.allclose(output[:, 28:30], raw[:, [6, 13]] * 2 - 1)
    assert torch.equal(output[:, 30:], torch.zeros(1, 25))
