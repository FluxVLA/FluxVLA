"""PI0/PI0.5 embedders for Pai-Megatron-Patch pre-process stage.

These modules run only on the first pipeline-parallel stage (pre_process=True).
They are NOT tensor-parallel because:
  - SigLIP (vision encoder) is replicated across TP ranks — it is small and
    only runs once per forward pass.
  - The action/state MLP layers are tiny (1024-dim) and their cost is negligible.
  - Their outputs (full hidden states) feed into the TP-sharded transformer
    layers, which internally split by heads — no TP coordination needed here.

PI0 vs PI0.5:
  - PI0PrefixEmbedder: identical for both variants.
  - PI0SuffixEmbedder: PI0 variant prepends a state token (explicit token);
    PI0.5 variant produces adarms_cond (state + time conditioning for AdaRMS).
"""

import math
import os
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from fluxvla.models.vision_alignment_trace import trace_siglip_layers


def _profile_range(name: str):
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if os.environ.get("PI05_PROFILE_MARKERS") == "1" and rank == 0:
        return torch.profiler.record_function(name)
    return nullcontext()


def _trace_vision_backward(name: str, tensor: Tensor) -> None:
    """Capture a vision activation and its gradient for alignment diagnosis."""
    trace_dir = os.environ.get("PI05_VISION_BACKWARD_TRACE_DIR")
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if not trace_dir or rank != 0 or not tensor.requires_grad:
        return

    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path / f"native_{name}_value.pt")

    def _save(gradient: Tensor) -> Tensor:
        torch.save(gradient.detach().cpu(), path / f"native_{name}_grad.pt")
        return gradient

    tensor.register_hook(_save)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (kept local to avoid importing the training stack)
# ---------------------------------------------------------------------------

def _get_safe_dtype(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)
    if device_type == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    """Sinusoidal time embedding for Flow Matching timestep t ∈ [0, 1]."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    # Match FluxVLA PI0.5 exactly: the reference helper evaluates the
    # frequencies and sin/cos in FP32 before the BF16 autocast MLP boundary.
    dtype = _get_safe_dtype(torch.float32, device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Zero-pad the last dimension of a tensor to new_dim."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def build_attention_mask_and_position_ids(
    pad_masks: Tensor,
    att_masks: Tensor,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """Build 4D causal attention mask and position ids for the PI0 sequence.

    pad_masks: [B, L] bool — True for real tokens.
    att_masks: [B, L] float — nonzero values mark block boundaries.

    Returns:
        attention_mask: [B, 1, L, L] additive mask (0 = attend, -inf = block)
        position_ids:   [B, L] long
    """
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d = att_2d & pad_2d
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    mask_value = -2.3819763e38
    attention_mask = torch.where(
        att_2d,
        torch.zeros_like(att_2d, dtype=dtype),
        torch.full_like(att_2d, mask_value, dtype=dtype),
    ).unsqueeze(1)
    return attention_mask, position_ids


def build_shared_obs_attention_mask_and_position_ids(
    prefix_pad_masks: Tensor,
    prefix_att_masks: Tensor,
    suffix_pad_masks: Tensor,
    suffix_att_masks: Tensor,
    num_offsets: int,
    offset_mask: Tensor,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """Build attention mask for shared-observation multi-offset training.

    The shared-observation training mode encodes the prefix (images + language)
    once and processes multiple action-offset suffixes jointly. Each suffix
    branch can attend to the shared prefix and to earlier tokens within its own
    branch, but NOT to tokens in other branches.

    Returns:
        attention_mask: [B, 1, total_L, total_L]
        position_ids:   [B, total_L]
    """
    B = prefix_pad_masks.shape[0]
    prefix_L = prefix_pad_masks.shape[1]
    suffix_L = suffix_pad_masks.shape[1]
    total_L = prefix_L + suffix_L * num_offsets
    device = prefix_pad_masks.device
    mask_value = -2.3819763e38

    full_pad = torch.zeros(B, total_L, dtype=torch.bool, device=device)
    full_att = torch.zeros(B, total_L, dtype=prefix_att_masks.dtype, device=device)
    full_pad[:, :prefix_L] = prefix_pad_masks
    full_att[:, :prefix_L] = prefix_att_masks

    suffix_pad_tiled = (
        suffix_pad_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(B, -1)
    )
    suffix_att_tiled = (
        suffix_att_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(B, -1)
    )
    full_pad[:, prefix_L:] = suffix_pad_tiled
    full_att[:, prefix_L:] = suffix_att_tiled

    cumsum = torch.cumsum(full_att, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = full_pad[:, None, :] * full_pad[:, :, None]
    att_2d = att_2d & pad_2d

    # Block cross-offset attention in the suffix-suffix region
    suffix_pos = torch.arange(num_offsets * suffix_L, device=device)
    offset_ids = suffix_pos // suffix_L
    cross_ok = offset_ids.unsqueeze(1) == offset_ids.unsqueeze(0)
    att_2d[:, prefix_L:, prefix_L:] = att_2d[:, prefix_L:, prefix_L:] & cross_ok

    # Mask invalid offsets
    offset_valid = offset_mask.unsqueeze(2).expand(-1, -1, suffix_L).reshape(B, -1)
    att_2d[:, prefix_L:, :] = att_2d[:, prefix_L:, :] & offset_valid.unsqueeze(2)
    att_2d[:, :, prefix_L:] = att_2d[:, :, prefix_L:] & offset_valid.unsqueeze(1)

    position_ids = torch.zeros(B, total_L, dtype=torch.long, device=device)
    prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
    position_ids[:, :prefix_L] = prefix_pos
    last_pre = prefix_pos[:, -1]
    suffix_base = torch.cumsum(suffix_pad_masks.long(), dim=1)
    suffix_tiled = suffix_base.unsqueeze(1).expand(-1, num_offsets, -1).reshape(B, -1)
    position_ids[:, prefix_L:] = last_pre[:, None] + suffix_tiled

    attention_mask = torch.where(
        att_2d,
        torch.zeros_like(att_2d, dtype=dtype),
        torch.full_like(att_2d, mask_value, dtype=dtype),
    ).unsqueeze(1)
    return attention_mask, position_ids


# ---------------------------------------------------------------------------
# PI0PrefixEmbedder
# ---------------------------------------------------------------------------

class PI0PrefixEmbedder(nn.Module):
    """Embed images (via SigLIP) and language tokens (via Gemma embed_tokens).

    Wraps the PaliGemma model to extract:
      - Image features: SigLIP(img) → projected to VLM hidden_size (2048)
      - Language embeddings: Gemma embed_tokens(tokens) scaled by sqrt(hidden_size)

    This module is replicated (not tensor-parallel) since SigLIP is small and
    runs only on the first pipeline stage.

    Args:
        config: PI0TransformerConfig
        vlm: PaliGemmaForConditionalGeneration instance (from HuggingFace)
    """

    def __init__(self, config, vlm):
        super().__init__()
        self.config = config
        # Store vlm.model as a proper submodule so .to(device) includes SigLIP
        self._vlm_model = vlm.model
        self.img_embedder = self._vlm_model.get_image_features
        self.lang_embedder = self._vlm_model.language_model.embed_tokens

    def forward(
        self,
        images: List[Tensor],          # list of [B, C, H, W], one per camera
        img_masks: List[Tensor],        # list of [B] bool
        tokens: Tensor,                 # [B, L_text] long
        masks: Tensor,                  # [B, L_text] bool
        subtask_spans=None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            embs:      [B, L_prefix, D_vlm=2048]
            pad_masks: [B, L_prefix] bool
            att_masks: [B, L_prefix] float  (0 = within-block, 1 = block boundary)
        """
        embs_list = []
        pad_masks_list = []
        att_masks_flat = []
        vision_trace_dir = os.environ.get("PI05_VISION_ALIGNMENT_TRACE_DIR")
        vision_trace_enabled = bool(vision_trace_dir) and not getattr(
            self, "_vision_alignment_trace_done", False
        )
        if vision_trace_enabled:
            import torch.distributed as dist

            vision_trace_enabled = (
                not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
            )
            if vision_trace_enabled:
                Path(vision_trace_dir).mkdir(parents=True, exist_ok=True)
        layer_trace_dir = os.environ.get("PI05_VISION_LAYER_TRACE_DIR")
        full_batch_replay_dir = os.environ.get("PI05_VISION_FULL_BATCH_REPLAY_DIR")
        layer_trace_enabled = bool(layer_trace_dir) and not getattr(
            self, "_vision_layer_trace_done", False
        )
        if layer_trace_enabled:
            import torch.distributed as dist

            layer_trace_enabled = (
                not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
            )

        vision_features_list = []
        image_inputs = []
        reference_fp32 = os.environ.get("PI05_REFERENCE_FP32_VISION_FORWARD") == "1"
        # The FSDP reference runs every camera and the projector under one
        # outer autocast context. Keep that context alive across per-camera
        # calls so autocast reuses the same cached BF16 parameter casts.
        vision_autocast = ExitStack()
        vision_autocast.enter_context(
            torch.autocast(
                device_type=images[0].device.type,
                dtype=torch.bfloat16,
                enabled=images[0].is_cuda and not reference_fp32,
            )
        )
        for image_index, (img, img_mask) in enumerate(zip(images, img_masks)):
            if full_batch_replay_dir and layer_trace_enabled:
                replay = torch.load(
                    Path(full_batch_replay_dir) / f"reference_image_{image_index:02d}.pt",
                    map_location=img.device,
                    weights_only=True,
                )
                if replay.shape != img.shape:
                    raise RuntimeError(
                        f"Vision replay shape mismatch: reference={tuple(replay.shape)}, pmp={tuple(img.shape)}"
                    )
                img = replay.to(dtype=img.dtype)
            def embed_vision(image, image_index=image_index):
                # PI05se with use_amp=False evaluates its fp32_vision branch in
                # FP32.  Keep the historical BF16-autocast path as the default,
                # but provide a narrow diagnostic switch for cross-framework
                # forward alignment.
                image = image.to(dtype=torch.float32)
                trace_layers = (
                    layer_trace_dir
                    and layer_trace_enabled
                    and image_index == 0
                )
                handles = (
                    trace_siglip_layers(
                        self._vlm_model.vision_tower, layer_trace_dir, "pmp")
                    if trace_layers
                    else []
                )
                try:
                    with torch.autocast(
                        device_type=image.device.type,
                        dtype=torch.bfloat16,
                        enabled=image.is_cuda and not reference_fp32,
                    ):
                        output = self._vlm_model.vision_tower(image)
                        vision_features = output.last_hidden_state
                        _trace_vision_backward(
                            f"camera_{image_index:02d}_tower_output", vision_features
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                return vision_features

            with _profile_range(f"pi05.model.vision.{image_index}"):
                if self.training and self.config.recompute_granularity is not None:
                    vision_features = checkpoint(
                        embed_vision,
                        img,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    vision_features = embed_vision(img)
            vision_features_list.append(vision_features)
            image_inputs.append((img, img_mask, vision_features.shape[1]))

        # FluxVLA concatenates all camera tokens before its single projector
        # invocation. Matching that GEMM shape is required for bitwise BF16
        # input gradients into the vision tower.
        combined_vision_features = torch.cat(vision_features_list, dim=1)
        with _profile_range("pi05.model.vision_projector"), torch.autocast(
            device_type=combined_vision_features.device.type,
            dtype=torch.bfloat16,
            enabled=combined_vision_features.is_cuda and not reference_fp32,
        ):
            combined_img_emb = self._vlm_model.multi_modal_projector(
                combined_vision_features
            )
        vision_autocast.close()
        _trace_vision_backward("vision_projector_output", combined_img_emb)

        token_offset = 0
        for image_index, (img, img_mask, n_img) in enumerate(image_inputs):
            img_emb = combined_img_emb[:, token_offset : token_offset + n_img]
            token_offset += n_img
            img_emb = img_emb.to(dtype=img.dtype)
            if vision_trace_enabled:
                torch.save(
                    {"image": img[:1].detach().cpu(), "embedding": img_emb[:1].detach().cpu()},
                    Path(vision_trace_dir) / f"pmp_vision_{image_index:02d}.pt",
                )
            bsz = img_emb.shape[0]
            embs_list.append(img_emb)
            pad_masks_list.append(img_mask[:, None].expand(bsz, n_img))
            att_masks_flat += [0] * n_img

        with _profile_range("pi05.model.language_embedding"):
            lang_emb = self.lang_embedder(tokens)              # [B, L_text, D_vlm]
        # In the pinned 0811 stack, get_input_embeddings() is GemmaEmbedding and
        # already applies sqrt(hidden_size). The PMP container exposes a plain
        # nn.Embedding instead, so reproduce that reference contract explicitly.
        if not hasattr(self.lang_embedder, "embed_scale"):
            # GemmaEmbedding materializes the scale in the embedding dtype. A
            # Python float promotes this multiply differently on BF16 and leaves
            # one-ULP text-token differences which the first transformer layer
            # amplifies. Keep the scalar construction in the same dtype.
            embed_scale = torch.tensor(
                math.sqrt(lang_emb.shape[-1]), dtype=lang_emb.dtype, device=lang_emb.device
            )
            lang_emb = lang_emb * embed_scale
        if os.environ.get("PI05_EMBED_ALIGNMENT_TRACE") == "1":
            if int(os.environ.get("RANK", "0")) == 0:
                path = Path(
                    os.environ.get(
                        "PI05_EMBED_ALIGNMENT_TRACE_DIR", "work_dirs/embed_trace"
                    )
                )
                path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    lang_emb[:1].detach().cpu(),
                    path / "native_language_embedding.pt",
                )
        embs_list.append(lang_emb)
        pad_masks_list.append(masks)
        att_masks_flat += [0] * lang_emb.shape[1]

        embs = torch.cat(embs_list, dim=1)
        pad_masks = torch.cat(pad_masks_list, dim=1)
        att_masks = torch.tensor(att_masks_flat, dtype=embs.dtype, device=embs.device)
        if subtask_spans is not None and any(span is not None for span in subtask_spans):
            att_masks = att_masks[None, :].repeat(embs.shape[0], 1)
            n_img = att_masks.shape[1] - tokens.shape[1]
            for row, span in enumerate(subtask_spans):
                if span is not None:
                    att_masks[row, n_img + span[0] : n_img + span[1]] = 1
        else:
            att_masks = att_masks[None, :].expand(embs.shape[0], -1)
        if vision_trace_enabled:
            self._vision_alignment_trace_done = True
        if layer_trace_enabled:
            self._vision_layer_trace_done = True
        return embs, pad_masks, att_masks


# ---------------------------------------------------------------------------
# PI0SuffixEmbedder  (PI0 variant: state token + action tokens)
# ---------------------------------------------------------------------------

class PI0SuffixEmbedder(nn.Module):
    """PI0 suffix embedder: state token + noisy action tokens + time conditioning.

    PI0 mode (use_adarms=False):
      - State is prepended as the first explicit token in the suffix.
      - adarms_cond = None (no per-layer conditioning).
      - Suffix layout: [state_token, action_token_0, ..., action_token_{T-1}]
        total length = 1 + chunk_size

    Args:
        config: PI0TransformerConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.action_expert_hidden_size  # 1024

        self.state_proj = nn.Linear(config.max_state_dim, width)
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

    def forward(
        self,
        state: Tensor,         # [B, state_dim]
        noisy_actions: Tensor, # [B, T, action_dim]
        time: Tensor,          # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Returns:
            embs:       [B, 1+T, D_exp=1024]
            pad_masks:  [B, 1+T] bool
            att_masks:  [B, 1+T] float
            adarms_cond: None  (PI0 has no adaRMS conditioning)
        """
        state_emb = self.state_proj(state).to(dtype=torch.bfloat16)  # [B, D_exp]
        bsz = state_emb.shape[0]
        device = state_emb.device

        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.action_expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device,
        ).to(dtype=self.action_in_proj.weight.dtype)

        action_emb = self.action_in_proj(noisy_actions)        # [B, T, D_exp]
        time_expanded = time_emb[:, None, :].expand_as(action_emb)
        action_time = torch.cat([action_emb, time_expanded], dim=2)  # [B, T, 2*D]
        action_time = F.silu(self.action_time_mlp_in(action_time))
        action_time = self.action_time_mlp_out(action_time)           # [B, T, D_exp]

        embs = torch.cat([state_emb[:, None, :], action_time], dim=1)  # [B, 1+T, D]
        T_total = embs.shape[1]

        pad_masks = torch.ones(bsz, T_total, dtype=torch.bool, device=device)
        # Boundary: state token marks end-of-prefix / start-of-suffix block.
        # action tokens: first one is also a block boundary, rest are causal.
        att_masks_list = [1] + [1] + [0] * (self.config.chunk_size - 1)
        att_masks = torch.tensor(
            att_masks_list, dtype=embs.dtype, device=device
        )[None, :].expand(bsz, -1)

        return embs, pad_masks, att_masks, None


# ---------------------------------------------------------------------------
# PI05SuffixEmbedder  (PI0.5 variant: no state token, adaRMS conditioning)
# ---------------------------------------------------------------------------

class PI05SuffixEmbedder(nn.Module):
    """PI0.5 suffix embedder: noisy action tokens with adaRMS state+time conditioning.

    PI0.5 mode (use_adarms=True):
      - NO explicit state token in the sequence.
      - adarms_cond = time_mlp(sinusoidal(t)) + state_mlp(s)  [B, D_exp]
        This conditioning vector is applied to each layer's AdaRMS norms.
      - Suffix layout: [action_token_0, ..., action_token_{T-1}]
        total length = chunk_size

    Args:
        config: PI0TransformerConfig
        state_cond: whether to include state in the adaRMS conditioning
    """

    def __init__(self, config, state_cond: bool = True):
        super().__init__()
        self.config = config
        self.state_cond = state_cond
        width = config.action_expert_hidden_size  # 1024

        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)

        if state_cond:
            self.state_proj = nn.Linear(config.max_state_dim, width)
            self.state_mlp_in = nn.Linear(width, width)
            self.state_mlp_out = nn.Linear(width, width)
            nn.init.zeros_(self.state_mlp_out.weight)
            nn.init.zeros_(self.state_mlp_out.bias)

    def forward(
        self,
        state: Tensor,         # [B, state_dim]
        noisy_actions: Tensor, # [B, T, action_dim]
        time: Tensor,          # [B]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            embs:       [B, T, D_exp=1024]
            pad_masks:  [B, T] bool
            att_masks:  [B, T] float
            adarms_cond: [B, D_exp]  — conditioning signal for AdaRMS layers
        """
        device = time.device
        bsz = time.shape[0]
        T = noisy_actions.shape[1]

        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.action_expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device,
        ).to(dtype=self.time_mlp_in.weight.dtype)

        time_emb = F.silu(self.time_mlp_in(time_emb))
        time_emb = F.silu(self.time_mlp_out(time_emb))
        adarms_cond = time_emb  # [B, D_exp]

        if self.state_cond:
            state_f = state.to(dtype=self.state_proj.weight.dtype)
            state_emb = F.silu(self.state_mlp_in(self.state_proj(state_f)))
            state_emb = F.silu(self.state_mlp_out(state_emb))
            adarms_cond = adarms_cond + state_emb

        noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        embs = self.action_in_proj(noisy_actions)  # [B, T, D_exp]
        if os.environ.get("PI05_EMBED_ALIGNMENT_TRACE") == "1":
            if int(os.environ.get("RANK", "0")) == 0:
                path = Path(
                    os.environ.get(
                        "PI05_EMBED_ALIGNMENT_TRACE_DIR", "work_dirs/embed_trace"
                    )
                )
                path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    embs[:1].detach().cpu(), path / "native_action_embedding.pt"
                )
        pad_masks = torch.ones(bsz, T, dtype=torch.bool, device=device)
        # First action token is a block boundary; rest are causal within block.
        att_list = [1] + [0] * (T - 1)
        att_masks = torch.tensor(
            att_list, dtype=embs.dtype, device=device
        )[None, :].expand(bsz, -1)

        return embs, pad_masks, att_masks, adarms_cond


# ---------------------------------------------------------------------------
# AdaRMSNorm for PI0.5 per-layer conditioning
# ---------------------------------------------------------------------------

class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with DiT-style conditioning (scale + shift + gate).

    Used in PI0.5 to inject the state+time conditioning into each layer's
    input and post-attention norms. The conditioning comes from
    PI05SuffixEmbedder.adarms_cond.

    When cond is provided: cond_proj(cond) → [scale, shift, gate] (3×hidden),
    applies normed*(1+scale)+shift and returns gate for gated residual.
    Matches pi05_training PiGemmaRMSNorm exactly.

    When cond is None (VLM branch, cond_dim=0): standard GemmaRMSNorm via
    learnable norm_weight, returns gate=None.

    Args:
        hidden_size: dimension of the hidden states to normalize
        cond_dim:    dimension of the conditioning vector (D_exp = 1024); 0 disables
        eps:         RMSNorm epsilon
    """

    def __init__(self, hidden_size: int, cond_dim: int = 0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # cond_proj projects cond → [scale, shift, gate] concatenated (3×hidden_size)
        # Checkpoint dense.weight shape: (3*D_exp, D_cond) = (3072, 1024) — exact match
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, 3 * hidden_size, bias=True)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        else:
            # GemmaRMSNorm convention (1 + norm_weight), only for unconditioned norms.
            self.norm_weight = nn.Parameter(torch.zeros(hidden_size))
            self.cond_proj = None

    def forward(
        self,
        hidden_states: Tensor,         # [B, L, D]
        cond: Optional[Tensor] = None, # [B, D_cond] or None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            normed_hs: normalized and conditioned hidden states
            gate:      [B, 1, D] gate tensor for gated residual, or None
        """
        # Preserve the reference autograd graph, not only its forward value.
        # The input stays in its model dtype for the multiply while variance
        # and reciprocal RMS are evaluated in FP32.
        variance = torch.mean(
            torch.square(hidden_states.float()), dim=-1, keepdim=True
        )
        normed = hidden_states * torch.rsqrt(variance + self.eps)

        if cond is not None and self.cond_proj is not None:
            # DiT-style: project cond → scale, shift, gate; skip norm_weight
            modulation = self.cond_proj(cond.to(self.cond_proj.weight.dtype)).float()
            modulation = modulation[:, None, :]  # [B, 1, 3*D]
            scale, shift, gate = modulation.chunk(3, dim=-1)
            normed = (normed * (1.0 + scale) + shift).to(hidden_states.dtype)
            return normed, gate.to(hidden_states.dtype)
        else:
            # Standard GemmaRMSNorm: learnable scale (used by VLM branch)
            normed = (normed * (1.0 + self.norm_weight.float())).to(hidden_states.dtype)
            return normed, None


class GatedRMSNorm(nn.Module):
    """Standard RMSNorm that returns (normed, gate=None) for API compatibility.

    Used in PI0 mode (no adaRMS) to keep the same interface as AdaRMSNorm.
    The gate is always None — callers use _gated_residual(r, x, None) → r + x.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        hidden_states: Tensor,
        cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, None]:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        normed = (normed.float() * self.weight.float()).type_as(hidden_states)
        return normed, None
