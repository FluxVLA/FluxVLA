"""PI05se subtask-recovery primitives independent of the training runtime."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def clean_prompt_text(value: str) -> str:
    """Match the PI05se processor's task/caption normalization."""
    return value.strip().replace("_", " ").replace("\n", " ")


def build_low_level_task_prompt(task: str, state: str, subtask: str | None = None) -> str:
    prompt = f"Task: {clean_prompt_text(task)}, State: {state};\n"
    if subtask is not None:
        prompt += f"Subtask: {clean_prompt_text(subtask)}\n"
    return prompt + "Action: "


def derive_subtask_spans(tokenizer, tasks, states, subtasks, tokens, masks):
    """Derive PI05se CE spans using the reference ``[prefix_len + 1, caption_len + 1)`` contract.

    The end includes the newline immediately after the caption, which is the
    generation stop target.  Keep this implementation byte-for-byte aligned
    with ``PI05sePolicy._subtask_token_spans`` in pi05_training_0811.
    """
    spans: list[tuple[int, int] | None] = [None] * len(tasks)
    rows = [i for i, caption in enumerate(subtasks) if caption]
    if not rows:
        return spans
    prefixes = [f"Task: {clean_prompt_text(tasks[i])}, State: {states[i]};\nSubtask:" for i in rows]
    captions = [" " + clean_prompt_text(subtasks[i]) for i in rows]
    prefix_ids = tokenizer(prefixes, add_special_tokens=False)["input_ids"]
    caption_ids = tokenizer(captions, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer.encode("\nAction: ", add_special_tokens=False)
    for piece_row, batch_row in enumerate(rows):
        prefix_len_with_bos = 1 + len(prefix_ids[piece_row])
        caption_len_with_stop = len(caption_ids[piece_row]) + 1
        start = prefix_len_with_bos
        end = prefix_len_with_bos + caption_len_with_stop
        expected = [tokenizer.bos_token_id] + prefix_ids[piece_row] + caption_ids[piece_row] + suffix_ids
        actual = tokens[batch_row, : int(masks[batch_row].sum())].tolist()
        if actual != expected:
            raise AssertionError(
                f"PI05se token contract drift in row {batch_row}: expected {expected}, got {actual}"
            )
        spans[batch_row] = (start, end)
    return spans


def compute_subtask_ce(
    prefix_hidden: Tensor,
    tokens: Tensor,
    subtask_spans: Sequence[tuple[int, int] | None],
    image_token_count: int,
    lm_head_weight: Tensor,
) -> Tensor | None:
    """Compute PI05se CE: mean per caption, then mean over captioned rows."""
    if len(subtask_spans) != tokens.shape[0]:
        raise ValueError("subtask_spans length must equal batch size")
    row_losses = []
    for row, span in enumerate(subtask_spans):
        if span is None:
            continue
        lo, hi = span
        if not (0 < lo < hi <= tokens.shape[1]):
            raise ValueError(f"invalid subtask span [{lo}, {hi}) for token length {tokens.shape[1]}")
        hidden = prefix_hidden[row, image_token_count + lo - 1 : image_token_count + hi - 1]
        logits = F.linear(hidden.to(lm_head_weight.dtype), lm_head_weight).float()
        row_losses.append(F.cross_entropy(logits, tokens[row, lo:hi], reduction="mean"))
    return torch.stack(row_losses).mean() if row_losses else None


def combine_pi05se_losses(flow_loss: Tensor, subtask_ce: Tensor | None, weight: float) -> Tensor:
    """Keep the legacy flow-only path identical when CE is disabled or absent."""
    if weight == 0.0 or subtask_ce is None:
        return flow_loss
    return flow_loss + weight * subtask_ce


def joint_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Tensor,
    scale: float,
    prefix_length: int,
    knowledge_insulation: bool,
    attention_implementation: str = "sdpa",
) -> Tensor:
    """Attention with detached prefix K/V only for suffix queries when KI is enabled."""
    if k.shape[-3] != q.shape[-3]:
        if q.shape[-3] % k.shape[-3]:
            raise ValueError("query head count must be divisible by key/value head count")
        batch, kv_heads, seq_len, head_dim = k.shape
        repetitions = q.shape[-3] // kv_heads
        k = k[:, :, None, :, :].expand(
            batch, kv_heads, repetitions, seq_len, head_dim
        ).reshape(batch, kv_heads * repetitions, seq_len, head_dim)
        v = v[:, :, None, :, :].expand(
            batch, kv_heads, repetitions, seq_len, head_dim
        ).reshape(batch, kv_heads * repetitions, seq_len, head_dim)

    if not knowledge_insulation or prefix_length == 0 or prefix_length == q.shape[-2]:
        if attention_implementation == "sdpa":
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, scale=scale
            )
        if attention_implementation != "eager":
            raise ValueError(
                f"unsupported attention implementation: {attention_implementation}"
            )
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + attention_mask[..., : k.shape[-2]]
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probabilities, v)

    prefix_q = q[..., :prefix_length, :]
    suffix_q = q[..., prefix_length:, :]
    prefix_mask = attention_mask[..., :prefix_length, :]
    suffix_mask = attention_mask[..., prefix_length:, :]

    prefix_scores = torch.matmul(prefix_q, k.transpose(-2, -1)) * scale + prefix_mask
    prefix_probs = torch.softmax(prefix_scores, dim=-1, dtype=torch.float32).to(q.dtype)
    prefix_out = torch.matmul(prefix_probs, v)

    insulated_k = torch.cat((k[..., :prefix_length, :].detach(), k[..., prefix_length:, :]), dim=-2)
    insulated_v = torch.cat((v[..., :prefix_length, :].detach(), v[..., prefix_length:, :]), dim=-2)
    suffix_scores = torch.matmul(suffix_q, insulated_k.transpose(-2, -1)) * scale + suffix_mask
    suffix_probs = torch.softmax(suffix_scores, dim=-1, dtype=torch.float32).to(q.dtype)
    suffix_out = torch.matmul(suffix_probs, insulated_v)
    return torch.cat((prefix_out, suffix_out), dim=-2)
