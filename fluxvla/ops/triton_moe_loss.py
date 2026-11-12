"""LingBot-compatible MoE auxiliary losses.

The production model uses a sequence-wise, layer-wise balance loss.  The
counting path is non-differentiable, so a vectorized PyTorch implementation is
equivalent to the native Triton count kernel while preserving the exact
gradient-bearing sigmoid/softmax computation graph.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _native_tie_aware_topk_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Match the native Triton router-counting tie semantics.

    The native kernel selects one distinct score level per round.  When several
    experts share that level, their single count is divided evenly.  ``topk``
    instead chooses an arbitrary subset at a Kth-boundary tie, which changes the
    otherwise detached load factor and therefore the auxiliary-loss gradient.
    """
    remaining = logits
    mask = torch.zeros_like(logits, dtype=torch.float32)
    for _ in range(top_k):
        maximum = remaining.max(dim=-1, keepdim=True).values
        is_maximum = remaining == maximum
        tie_count = is_maximum.sum(dim=-1, keepdim=True).to(torch.float32)
        mask.add_(is_maximum.to(torch.float32) / tie_count)
        remaining = remaining.masked_fill(is_maximum, float("-inf"))
    return mask


def _sequence_wise_balance_loss(
    router_logits_list,
    top_k: int,
    seq_lengths,
    padding_len: int,
    score_func: str,
):
    valid_logits = [logits for logits in router_logits_list if logits is not None]
    if not valid_logits:
        return []

    all_logits = []
    valid_tokens_per_layer = None
    for logits in valid_logits:
        logits = logits.to(dtype=torch.float32)
        if padding_len > 0:
            logits = logits[: logits.shape[0] - padding_len]
        all_logits.append(logits)
        if valid_tokens_per_layer is None:
            valid_tokens_per_layer = logits.shape[0]

    if any(logits.shape[0] != valid_tokens_per_layer for logits in all_logits):
        return [
            _sequence_wise_balance_loss(
                (logits,), top_k, seq_lengths, 0, score_func
            )[0]
            for logits in all_logits
        ]

    lengths = list(seq_lengths or [valid_tokens_per_layer])
    device = all_logits[0].device
    num_layers = len(all_logits)
    num_experts = all_logits[0].shape[1]
    num_sequences = len(lengths)
    segment_lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    all_segment_lengths = segment_lengths.repeat(num_layers)
    segment_ids = torch.repeat_interleave(
        torch.arange(num_layers * num_sequences, device=device),
        all_segment_lengths,
    )
    concatenated = torch.cat(all_logits, dim=0)

    if score_func == "sigmoid":
        scores = concatenated.sigmoid()
        probabilities = scores / scores.sum(dim=-1, keepdim=True)
    else:
        probabilities = F.softmax(concatenated, dim=-1)

    expanded_segment_ids = segment_ids.unsqueeze(1).expand(-1, num_experts)
    probability_sums = torch.zeros(
        num_layers * num_sequences,
        num_experts,
        device=device,
        dtype=torch.float32,
    )
    probability_sums.scatter_add_(0, expanded_segment_ids, probabilities)
    probability_means = probability_sums / all_segment_lengths.unsqueeze(1).float().clamp(min=1)

    # The native Triton kernel counts K distinct score levels.  At a tie it
    # distributes one count across every tied expert instead of choosing an
    # arbitrary expert as torch.topk does.
    with torch.no_grad():
        token_expert_mask = _native_tie_aware_topk_mask(concatenated, top_k)
        count_sums = torch.zeros_like(probability_sums)
        count_sums.scatter_add_(0, expanded_segment_ids, token_expert_mask)
        load = count_sums * (
            (float(num_experts) / top_k)
            / all_segment_lengths.unsqueeze(1).float().clamp(min=1)
        )

    losses = (load * probability_means).sum(dim=-1)
    losses = losses.reshape(num_layers, num_sequences).mean(dim=1)
    return list(losses.unbind(0))


def triton_sequence_wise_balance_loss(
    router_logits: torch.Tensor | None = None,
    num_experts: int | None = None,
    top_k: int = 2,
    router_logits_list=None,
    seq_lengths=None,
    padding_len=0,
    score_func="softmax",
    *args,
    **kwargs,
) -> torch.Tensor:
    if router_logits_list is not None:
        return _sequence_wise_balance_loss(
            router_logits_list,
            top_k=top_k,
            seq_lengths=seq_lengths,
            padding_len=padding_len,
            score_func=score_func,
        )
    if router_logits is None or num_experts is None:
        raise ValueError("router_logits and num_experts are required")
    probabilities = router_logits.float().softmax(dim=-1)
    assignments = probabilities.topk(top_k, dim=-1).indices
    counts = torch.zeros(num_experts, device=router_logits.device, dtype=probabilities.dtype)
    counts.scatter_add_(0, assignments.reshape(-1), torch.ones_like(assignments, dtype=probabilities.dtype).reshape(-1))
    counts = counts / max(1, assignments.numel())
    load = probabilities.mean(dim=tuple(range(probabilities.ndim - 1)))
    return (load * counts).sum() * float(num_experts)
