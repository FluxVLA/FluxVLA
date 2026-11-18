"""Import-only flash-attn shim for parity diagnostics.

The diagnostic path uses eager attention.  This shim prevents an incompatible
system flash-attn extension from being imported before the model is built.
"""

import torch


def _eager(q, k, v, causal=False, **kwargs):
    if k.shape[-2] != q.shape[-2]:
        k = k.repeat_interleave(q.shape[-2] // k.shape[-2], dim=-2)
        v = v.repeat_interleave(q.shape[-2] // v.shape[-2], dim=-2)
    scores = torch.matmul(q.transpose(0, 1), k.transpose(0, 1).transpose(-2, -1))
    scores = scores * (q.shape[-1] ** -0.5)
    if causal:
        mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return torch.matmul(scores.softmax(dim=-1), v.transpose(0, 1)).transpose(0, 1)


def flash_attn_func(q, k, v, causal=False, **kwargs):
    return _eager(q, k, v, causal=causal, **kwargs)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
    outputs = []
    for index in range(cu_seqlens_q.numel() - 1):
        qs, qe = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        ks, ke = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        outputs.append(_eager(q[qs:qe], k[ks:ke], v[ks:ke], **kwargs))
    return torch.cat(outputs, dim=0)
