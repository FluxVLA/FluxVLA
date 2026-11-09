import pytest
import torch

import fluxvla.ops.fused_moe as fused_moe_module

from fluxvla.ops.fused_moe import FusedMoeExpertFunction
from fluxvla.ops.robby_moe import robby_moe_forward


def test_fused_moe_backward_trace_is_step_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FLUXVLA_LINGBOT_FUSED_MOE_BACKWARD_TRACE_DIR", str(tmp_path)
    )
    monkeypatch.setenv("LINGBOT_FUSED_MOE_CALLS_PER_STEP", "2")
    monkeypatch.setenv("LINGBOT_FUSED_MOE_BACKWARD_TRACE_STEP", "2")
    monkeypatch.setattr(fused_moe_module, "_backward_trace_call_index", 0)
    values = tuple(torch.randn(2, 3) for _ in range(3))

    for _ in range(4):
        fused_moe_module._trace_backward_outputs(*values)

    paths = sorted(tmp_path.glob("*.pt"))
    assert [path.name for path in paths] == [
        "flux_step000002_rank0_call001_layer001.pt",
        "flux_step000002_rank0_call002_layer000.pt",
    ]
    payload = torch.load(paths[0], weights_only=True)
    assert payload["backend"] == "fluxvla"
    assert payload["step"] == 2
    torch.testing.assert_close(payload["grad_gate_weight"], values[1])


def test_moe_route_weight_and_topk_reduction_contract():
    torch.manual_seed(3)
    tokens, hidden, intermediate, experts, top_k = 5, 7, 11, 4, 2
    x = torch.randn(tokens, hidden, dtype=torch.float32)
    gate = torch.randn(experts, intermediate, hidden, dtype=torch.float32)
    up = torch.randn_like(gate)
    down = torch.randn(experts, hidden, intermediate, dtype=torch.float32)
    selected = torch.tensor([[2, 0], [1, 3], [0, 2], [3, 1], [2, 1]])
    weights = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4], [0.1, 0.9], [0.8, 0.2], [0.35, 0.65]],
        dtype=x.dtype,
    )

    expected_slots = torch.zeros(tokens, top_k, hidden)
    for expert_id in range(experts):
        token_ids, slot_ids = (selected == expert_id).nonzero(as_tuple=True)
        if token_ids.numel() == 0:
            continue
        values = torch.nn.functional.silu(x[token_ids] @ gate[expert_id].T)
        values = values * (x[token_ids] @ up[expert_id].T)
        values = values * weights[token_ids, slot_ids, None]
        expected_slots[token_ids, slot_ids] = values @ down[expert_id].T
    expected = expected_slots.float().sum(dim=1).to(x.dtype)

    actual = robby_moe_forward(x, weights, selected, gate, up, down)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_moe_fallback_preserves_gradients():
    torch.manual_seed(5)
    x = torch.randn(4, 6, requires_grad=True)
    gate = torch.randn(3, 8, 6, requires_grad=True)
    up = torch.randn(3, 8, 6, requires_grad=True)
    down = torch.randn(3, 6, 8, requires_grad=True)
    selected = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2]])
    weights = torch.tensor(
        [[0.4, 0.6], [0.7, 0.3], [0.2, 0.8], [0.5, 0.5]], dtype=x.dtype
    )

    loss = robby_moe_forward(x, weights, selected, gate, up, down).square().mean()
    loss.backward()
    for tensor in (x, gate, up, down):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton grouped MoE requires CUDA")
def test_training_fused_moe_preserves_bf16_gradient_contract():
    torch.manual_seed(17)
    experts, tokens, hidden, intermediate, top_k = 4, 9, 64, 96, 2
    device = torch.device("cuda")
    routing_weights = torch.randn(
        tokens, top_k, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    selected_experts = torch.tensor(
        [[0, 1], [2, 0], [1, 3], [3, 2], [0, 3], [1, 2], [2, 3], [0, 2], [3, 1]],
        device=device,
    )
    hidden_states = torch.randn(
        tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    gate = torch.randn(
        experts, intermediate, hidden, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    up = torch.randn_like(gate, requires_grad=True)
    down = torch.randn(
        experts, hidden, intermediate, device=device, dtype=torch.bfloat16, requires_grad=True
    )

    output = FusedMoeExpertFunction.apply(
        experts,
        routing_weights,
        selected_experts,
        hidden_states,
        gate,
        up,
        down,
    )
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert output.shape == hidden_states.shape
    for tensor in (routing_weights, hidden_states, gate, up, down):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
