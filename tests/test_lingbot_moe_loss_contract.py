import torch

from fluxvla.models.vlas.lingbot_native_core.modeling_lingbot_vla_v2 import (
    _register_router_logit_grad_trace,
)

from fluxvla.ops.triton_moe_loss import triton_sequence_wise_balance_loss


def _reference(logits, *, top_k, lengths):
    losses = []
    offset = 0
    for length in lengths:
        sequence = logits[offset : offset + length].float()
        scores = sequence.sigmoid()
        probabilities = scores / scores.sum(dim=-1, keepdim=True)
        with torch.no_grad():
            assignments = sequence.topk(top_k, dim=-1).indices
            counts = torch.zeros(sequence.shape[-1], dtype=torch.float32)
            counts.scatter_add_(
                0,
                assignments.reshape(-1),
                torch.ones(assignments.numel(), dtype=torch.float32),
            )
            load = counts * (float(sequence.shape[-1]) / top_k / length)
        losses.append((load * probabilities.mean(dim=0)).sum())
        offset += length
    return torch.stack(losses).mean()


def test_sequence_wise_sigmoid_loss_and_gradient_match_native_contract():
    first = torch.tensor(
        [[-1.0, 0.2, 1.3, 0.7], [0.8, -0.3, 0.1, 1.1], [0.4, 0.9, -0.8, 0.2]],
        requires_grad=True,
    )
    second = first.detach().clone().requires_grad_(True)

    actual = triton_sequence_wise_balance_loss(
        router_logits_list=(first,),
        top_k=2,
        seq_lengths=[2, 1],
        score_func="sigmoid",
    )[0]
    expected = _reference(second, top_k=2, lengths=[2, 1])
    actual.backward()
    expected.backward()

    assert torch.equal(actual, expected)
    assert torch.equal(first.grad, second.grad)


def test_sequence_wise_loss_returns_one_value_per_layer():
    logits = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    losses = triton_sequence_wise_balance_loss(
        router_logits_list=(logits, logits + 0.5),
        top_k=2,
        seq_lengths=[3, 3],
        score_func="sigmoid",
    )

    assert len(losses) == 2


def test_sequence_wise_loss_matches_native_kth_boundary_tie_semantics():
    # Experts 2 and 3 tie at the second selection round.  The native Triton
    # kernel assigns each half a count; torch.topk would select only one.
    actual_logits = torch.tensor([[3.0, 2.0, 1.0, 1.0]], requires_grad=True)
    expected_logits = actual_logits.detach().clone().requires_grad_(True)

    actual = triton_sequence_wise_balance_loss(
        router_logits_list=(actual_logits,),
        top_k=3,
        seq_lengths=[1],
        score_func="sigmoid",
    )[0]
    probabilities = expected_logits.sigmoid()
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    native_load = torch.tensor([4 / 3, 4 / 3, 2 / 3, 2 / 3])
    expected = (native_load * probabilities[0]).sum()

    actual.backward()
    expected.backward()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_logits.grad, expected_logits.grad)


def test_router_logit_grad_trace_records_only_the_target_step(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FLUXVLA_LINGBOT_ROUTER_GRAD_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("LINGBOT_ROUTER_GRAD_TRACE_STEP", "2")
    skipped = torch.randn(3, 4, requires_grad=True)
    traced = torch.randn(3, 4, requires_grad=True)

    _register_router_logit_grad_trace((skipped,), step=1)
    _register_router_logit_grad_trace((traced,), step=2)
    traced.square().sum().backward()

    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    payload = torch.load(paths[0], weights_only=True)
    assert payload["layer"] == 0
    assert torch.equal(payload["router_logits_grad"], 2 * traced.detach())
