# Copyright 2026 Limx Dynamics
"""Causal language loss shared by standalone VLMs and VLA policies."""

import torch
import torch.nn.functional as F


def causal_lm_loss(logits, labels, ignore_index=-100):
    """Return a differentiable sum and detached sufficient statistics.

    Labels are unshifted. Visual/prompt/padding positions must already be
    masked. The runner normalizes summed gradients across all ranks and
    micro-batches, rather than averaging means of differently sized answers.
    An empty local target is legal; a globally empty step is rejected by the
    runner after all ranks have participated in backward.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError('Causal LM logits and labels must align as (B,L,V) '
                         'and (B,L).')
    if labels.dtype != torch.long or labels.shape[1] < 2:
        raise ValueError('Causal LM labels must be int64 with length >= 2.')
    targets = labels[:, 1:].to(logits.device)
    predictions = logits[:, :-1]
    valid = targets.ne(ignore_index)
    loss_sum = F.cross_entropy(
        predictions[valid].float(),
        targets[valid],
        ignore_index=ignore_index,
        reduction='sum')
    num_tokens = valid.sum()
    with torch.no_grad():
        num_correct = ((predictions.argmax(-1) == targets) & valid).sum()
    return dict(
        loss=loss_sum / num_tokens.clamp_min(1),
        loss_sum=loss_sum,
        num_tokens=num_tokens.detach(),
        num_correct=num_correct.detach())
