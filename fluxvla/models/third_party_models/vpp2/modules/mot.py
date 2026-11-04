# Vendored from VPP2, revision 3365872910244f14df83003062f7c563f14c6d9e.
# See ../THIRD_PARTY_NOTICES.md for licenses and source notices.
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from logging import getLogger as get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        use_fixed_video_layers: Optional[int] = None,
        use_video_tokens: bool = False,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.use_fixed_video_layers = (
            None if use_fixed_video_layers is None else int(use_fixed_video_layers)
        )
        self.use_video_tokens = bool(use_video_tokens)
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads and not self.use_video_tokens:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        if self.use_fixed_video_layers is not None and not (
            0 <= self.use_fixed_video_layers < self.num_layers
        ):
            raise ValueError(
                "`use_fixed_video_layers` is a zero-based video layer index and must be in "
                f"[0, {self.num_layers - 1}], got {self.use_fixed_video_layers}."
            )
        action_uses_video_tokens = bool(getattr(self.mixtures["action"], "use_video_tokens", False))
        if action_uses_video_tokens != self.use_video_tokens:
            raise ValueError(
                "MoT and ActionDiT disagree on `use_video_tokens`: "
                f"{self.use_video_tokens} vs {action_uses_video_tokens}."
            )

        logger.info(
            f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}"
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B"
            )

    def _video_source_layer(self, action_layer_idx: int) -> int:
        if self.use_fixed_video_layers is not None:
            return self.use_fixed_video_layers
        return int(action_layer_idx)

    def materialize_action_video_kv_cache(
        self,
        video_cache: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        """Project raw video tokens once for reuse across action denoising steps."""
        if not self.use_video_tokens:
            return video_cache
        if len(video_cache) != self.num_layers:
            raise ValueError(
                f"`video_cache` must contain {self.num_layers} layers, got {len(video_cache)}."
            )

        expert = self.mixtures["action"]
        encoded_by_source: dict[int, torch.Tensor] = {}
        action_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            source_layer_idx = self._video_source_layer(layer_idx)
            source_cache = video_cache[source_layer_idx]
            if "tokens" not in source_cache or "freqs" not in source_cache:
                raise ValueError(
                    f"`video_cache[{source_layer_idx}]` must contain `tokens` and `freqs`."
                )
            if source_layer_idx not in encoded_by_source:
                encoded_by_source[source_layer_idx] = expert.encode_video_tokens(
                    source_cache["tokens"]
                )
            k, v = expert.project_video_tokens(
                layer_idx=layer_idx,
                video_tokens=encoded_by_source[source_layer_idx],
                video_freqs=source_cache["freqs"],
            )
            action_cache.append({"action_k": k, "action_v": v})
        return action_cache

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(
            6, dim=chunk_dim
        )
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: Optional[int] = None,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)
        attention_heads = self.num_heads if num_heads is None else int(num_heads)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=attention_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(
            block, t_mod
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """

        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        return_final_tokens: bool = False,
        max_layers: Optional[int] = None,
        video_feature_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> list[dict[str, torch.Tensor]] | tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """Prefill video branch once and cache K/V or raw tokens for action.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].
            video_feature_callback: Optional read-only collector called with
                (zero-based layer index, post-layer tokens), outside activation
                checkpointing. Must not modify the tokens in place.

        Returns:
            Layer-wise cache list with length `num_layers`, or `max_layers`
            when a bounded prefill is requested.
            In direct mode, required entries contain video `k`/`v`. With
            `use_video_tokens=true`, required entries contain pre-layer video
            `tokens` and their RoPE `freqs` instead.
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        layer_count = self.num_layers if max_layers is None else int(max_layers)
        if not 1 <= layer_count <= self.num_layers:
            raise ValueError(f"`max_layers` must be in [1, {self.num_layers}], got {layer_count}.")
        if self.use_fixed_video_layers is not None and self.use_fixed_video_layers >= layer_count:
            raise ValueError(
                "`max_layers` must include `use_fixed_video_layers`; got "
                f"max_layers={layer_count}, fixed={self.use_fixed_video_layers}."
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(layer_count):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
                num_heads=expert.num_heads,
            )
            cache_entry: dict[str, torch.Tensor] = {}
            if self.use_video_tokens:
                if self.use_fixed_video_layers is None or layer_idx == self.use_fixed_video_layers:
                    cache_entry.update({"tokens": x, "freqs": video_freqs})
            elif self.use_fixed_video_layers is None or layer_idx == self.use_fixed_video_layers:
                cache_entry.update({"k": k, "v": v})

            # Update video tokens for the next layer and persist its action input.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append(cache_entry)
            if video_feature_callback is not None:
                # Outside checkpointed blocks: retain the original graph, and
                # do not collect again during backward recomputation.
                video_feature_callback(layer_idx, x)
        if return_final_tokens:
            return kv_cache, x
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        clean_first_frame_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
        clean_first_frame_attention_mask: Optional[torch.Tensor] = None,
        clean_first_frame_video_seq_len: Optional[int] = None,
        clean_first_frame_start_layer: Optional[int] = None,
        action_video_conditioning_mask: Optional[torch.Tensor] = None,
        action_clean_first_frame_conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.
            clean_first_frame_*: Optional first-frame-only cache/mask/prefix
                used from `clean_first_frame_start_layer` through the final
                ActionDiT layer. The cache may come from an independent clean
                prefill or from frame 0 of the noisy cache; this mode is
                restricted to direct per-layer video K/V.
            action_video_conditioning_mask: Optional per-sample boolean vector
                [B]. False removes every cached video key from that sample's
                action attention while preserving action self-attention.
            action_clean_first_frame_conditioning_mask: Optional per-sample
                boolean vector [B]. True replaces the noisy-video prefix with
                the supplied first-frame-only cache for every ActionDiT layer.
                Mixed batches retain one common padded attention width.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        action_seq_len = int(action_tokens.shape[1])
        split_enabled = clean_first_frame_start_layer is not None
        timestep_clean_enabled = action_clean_first_frame_conditioning_mask is not None
        if split_enabled and timestep_clean_enabled:
            raise ValueError(
                "Layer-split and timestep-routed clean-first-frame conditioning "
                "cannot be enabled together."
            )
        if timestep_clean_enabled and action_video_conditioning_mask is not None:
            raise ValueError(
                "Clean-first-frame timestep routing and no-video timestep gating "
                "cannot be enabled together."
            )
        clean_cache_enabled = split_enabled or timestep_clean_enabled
        if clean_cache_enabled:
            if self.use_fixed_video_layers is not None or self.use_video_tokens:
                raise ValueError("Clean-first-frame routing requires direct per-layer video K/V.")
            if clean_first_frame_kv_cache is None:
                raise ValueError("Clean-first-frame routing requires `clean_first_frame_kv_cache`.")
            if clean_first_frame_attention_mask is None:
                raise ValueError(
                    "Clean-first-frame routing requires `clean_first_frame_attention_mask`."
                )
            if clean_first_frame_video_seq_len is None:
                raise ValueError(
                    "Clean-first-frame routing requires `clean_first_frame_video_seq_len`."
                )
            if len(clean_first_frame_kv_cache) != self.num_layers:
                raise ValueError(
                    "`clean_first_frame_kv_cache` must contain "
                    f"{self.num_layers} layers, got {len(clean_first_frame_kv_cache)}."
                )
        if split_enabled:
            split_layer = int(clean_first_frame_start_layer)
            if not 0 < split_layer < self.num_layers:
                raise ValueError(
                    "`clean_first_frame_start_layer` must be in "
                    f"[1, {self.num_layers - 1}], got {split_layer}."
                )
            if len(video_kv_cache) < split_layer:
                raise ValueError(
                    "`video_kv_cache` must cover every noisy-video ActionDiT layer; "
                    f"need at least {split_layer}, got {len(video_kv_cache)}."
                )
        else:
            split_layer = self.num_layers
            if len(video_kv_cache) != self.num_layers:
                raise ValueError(
                    f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
                )

        def _action_mask_rows(
            mask: torch.Tensor,
            prefix_seq_len: int,
            name: str,
        ) -> torch.Tensor:
            if mask.ndim != 2:
                raise ValueError(f"`{name}` must be 2D [S,S], got shape {tuple(mask.shape)}")
            if mask.shape[0] != mask.shape[1]:
                raise ValueError(f"`{name}` must be square, got shape {tuple(mask.shape)}")
            total = int(prefix_seq_len) + action_seq_len
            if mask.shape[0] != total:
                raise ValueError(
                    f"`{name}` seq length mismatch: mask={mask.shape[0]} vs expected_total={total}"
                )
            return mask[prefix_seq_len:total, :total]

        action_attention_mask = _action_mask_rows(
            attention_mask,
            int(video_seq_len),
            "attention_mask",
        )
        clean_action_attention_mask = None
        if clean_cache_enabled:
            clean_action_attention_mask = _action_mask_rows(
                clean_first_frame_attention_mask,
                int(clean_first_frame_video_seq_len),
                "clean_first_frame_attention_mask",
            )

        clean_conditioning = None
        any_clean_first_frame = False
        all_clean_first_frame = False
        if timestep_clean_enabled:
            clean_conditioning = action_clean_first_frame_conditioning_mask.to(
                device=action_tokens.device,
                dtype=torch.bool,
            )
            if clean_conditioning.ndim != 1:
                raise ValueError(
                    "`action_clean_first_frame_conditioning_mask` must be 1D "
                    f"[B], got shape {tuple(clean_conditioning.shape)}"
                )
            batch_size = int(action_tokens.shape[0])
            if clean_conditioning.shape[0] != batch_size:
                raise ValueError(
                    "`action_clean_first_frame_conditioning_mask` batch mismatch: "
                    f"mask={clean_conditioning.shape[0]} vs action={batch_size}"
                )
            if int(clean_first_frame_video_seq_len) > int(video_seq_len):
                raise ValueError(
                    "Clean-first-frame prefix cannot exceed noisy-video prefix: "
                    f"clean={clean_first_frame_video_seq_len}, noisy={video_seq_len}."
                )
            any_clean_first_frame = bool(clean_conditioning.any().item())
            all_clean_first_frame = bool(clean_conditioning.all().item())
            if all_clean_first_frame:
                action_attention_mask = clean_action_attention_mask
            elif any_clean_first_frame:
                clean_prefix_len = int(clean_first_frame_video_seq_len)
                noisy_prefix_len = int(video_seq_len)
                clean_prefix_mask = clean_action_attention_mask[:, :clean_prefix_len]
                clean_padding_mask = torch.zeros(
                    (
                        action_seq_len,
                        noisy_prefix_len - clean_prefix_len,
                    ),
                    device=action_attention_mask.device,
                    dtype=torch.bool,
                )
                action_self_mask = action_attention_mask[:, noisy_prefix_len:]
                padded_clean_mask = torch.cat(
                    [clean_prefix_mask, clean_padding_mask, action_self_mask],
                    dim=-1,
                )
                action_attention_mask = torch.where(
                    clean_conditioning[:, None, None],
                    padded_clean_mask.unsqueeze(0),
                    action_attention_mask.unsqueeze(0),
                )

        def _apply_video_conditioning_gate(
            mask: torch.Tensor,
            prefix_seq_len: int,
        ) -> torch.Tensor:
            if action_video_conditioning_mask is None:
                return mask
            conditioning = action_video_conditioning_mask.to(
                device=mask.device,
                dtype=torch.bool,
            )
            if conditioning.ndim != 1:
                raise ValueError(
                    "`action_video_conditioning_mask` must be 1D [B], got "
                    f"shape {tuple(conditioning.shape)}"
                )
            batch_size = int(action_tokens.shape[0])
            if conditioning.shape[0] != batch_size:
                raise ValueError(
                    "`action_video_conditioning_mask` batch mismatch: "
                    f"mask={conditioning.shape[0]} vs action={batch_size}"
                )
            gated = mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
            gated[:, :, :prefix_seq_len] &= conditioning[:, None, None]
            return gated

        action_attention_mask = _apply_video_conditioning_gate(
            action_attention_mask,
            int(video_seq_len),
        )
        if clean_action_attention_mask is not None:
            clean_action_attention_mask = _apply_video_conditioning_gate(
                clean_action_attention_mask,
                int(clean_first_frame_video_seq_len),
            )
        all_video_disabled = action_video_conditioning_mask is not None and not bool(
            action_video_conditioning_mask.to(dtype=torch.bool).any().item()
        )

        expert = self.mixtures["action"]
        x = action_tokens
        has_materialized_action_kv = self.use_video_tokens and all(
            "action_k" in layer_cache and "action_v" in layer_cache
            for layer_cache in video_kv_cache
        )
        encoded_video_tokens: dict[int, torch.Tensor] = {}
        if self.use_video_tokens and not has_materialized_action_kv and not all_video_disabled:
            for source_layer_idx in {
                self._video_source_layer(layer_idx) for layer_idx in range(self.num_layers)
            }:
                source_cache = video_kv_cache[source_layer_idx]
                if "tokens" not in source_cache:
                    raise ValueError(
                        f"`video_kv_cache[{source_layer_idx}]` must contain `tokens` "
                        "when `use_video_tokens=true`."
                    )
                encoded_video_tokens[source_layer_idx] = expert.encode_video_tokens(
                    source_cache["tokens"]
                )
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            use_clean_first_frame = (split_enabled and layer_idx >= split_layer) or (
                timestep_clean_enabled and all_clean_first_frame
            )
            if use_clean_first_frame:
                source_layer_idx = layer_idx
                layer_cache = clean_first_frame_kv_cache[source_layer_idx]
                selected_video_seq_len = int(clean_first_frame_video_seq_len)
                selected_action_attention_mask = clean_action_attention_mask
                cache_name = "clean_first_frame_kv_cache"
            else:
                source_layer_idx = self._video_source_layer(layer_idx)
                layer_cache = video_kv_cache[source_layer_idx]
                selected_video_seq_len = int(video_seq_len)
                selected_action_attention_mask = action_attention_mask
                cache_name = "video_kv_cache"
            if all_video_disabled:
                # The late denoising path has no video-visible samples. Avoid
                # projecting or concatenating masked video K/V, so inference
                # refinement also gets the intended attention-width reduction.
                k_cat = k_action
                v_cat = v_action
                selected_action_attention_mask = selected_action_attention_mask[
                    ..., selected_video_seq_len:
                ]
            elif self.use_video_tokens:
                if has_materialized_action_kv:
                    k_video = video_kv_cache[layer_idx]["action_k"]
                    v_video = video_kv_cache[layer_idx]["action_v"]
                else:
                    if "freqs" not in layer_cache:
                        raise ValueError(
                            f"`video_kv_cache[{source_layer_idx}]` must contain `freqs`."
                        )
                    k_video, v_video = expert.project_video_tokens(
                        layer_idx=layer_idx,
                        video_tokens=encoded_video_tokens[source_layer_idx],
                        video_freqs=layer_cache["freqs"],
                    )
            else:
                if "k" not in layer_cache or "v" not in layer_cache:
                    raise ValueError(
                        f"`{cache_name}[{source_layer_idx}]` must contain `k` and `v`."
                    )
                k_video = layer_cache["k"]
                v_video = layer_cache["v"]
            if timestep_clean_enabled and any_clean_first_frame and not all_clean_first_frame:
                clean_layer_cache = clean_first_frame_kv_cache[layer_idx]
                if "k" not in clean_layer_cache or "v" not in clean_layer_cache:
                    raise ValueError(
                        "`clean_first_frame_kv_cache` entries must contain `k` and `v`."
                    )
                clean_k = clean_layer_cache["k"]
                clean_v = clean_layer_cache["v"]
                clean_prefix_len = int(clean_first_frame_video_seq_len)
                if clean_k.shape[1] != clean_prefix_len or clean_v.shape[1] != clean_prefix_len:
                    raise ValueError(
                        "`clean_first_frame_kv_cache` seq len mismatch, expected "
                        f"{clean_prefix_len}."
                    )
                clean_k_padded = torch.cat(
                    [clean_k, k_video[:, clean_prefix_len:]],
                    dim=1,
                )
                clean_v_padded = torch.cat(
                    [clean_v, v_video[:, clean_prefix_len:]],
                    dim=1,
                )
                k_video = torch.where(
                    clean_conditioning[:, None, None],
                    clean_k_padded,
                    k_video,
                )
                v_video = torch.where(
                    clean_conditioning[:, None, None],
                    clean_v_padded,
                    v_video,
                )
            if not all_video_disabled:
                if (
                    k_video.shape[1] != selected_video_seq_len
                    or v_video.shape[1] != selected_video_seq_len
                ):
                    raise ValueError(
                        f"`{cache_name}[{source_layer_idx}]` seq len mismatch, "
                        f"expected {selected_video_seq_len}."
                    )

                # Mixed attention: action queries attend to cached video K/V
                # plus current action K/V.
                k_cat = torch.cat([k_video, k_action], dim=1)
                v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=selected_action_attention_mask,
                num_heads=expert.num_heads,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        video_feature_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        if self.use_fixed_video_layers is not None or self.use_video_tokens:
            video_seq_len = int(embeds_all["video"].shape[1])
            action_seq_len = int(embeds_all["action"].shape[1])
            if attention_mask.shape[0] != video_seq_len + action_seq_len:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={video_seq_len + action_seq_len}"
                )
            video_cache, final_video_tokens = self.prefill_video_cache(
                video_tokens=embeds_all["video"],
                video_freqs=freqs_all["video"],
                video_t_mod=t_mod_all["video"],
                video_context_payload=context_all.get("video"),
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                return_final_tokens=True,
                video_feature_callback=video_feature_callback,
            )
            final_action_tokens = self.forward_action_with_video_cache(
                action_tokens=embeds_all["action"],
                action_freqs=freqs_all["action"],
                action_t_mod=t_mod_all["action"],
                action_context_payload=context_all.get("action"),
                video_kv_cache=video_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            return {"video": final_video_tokens, "action": final_action_tokens}

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                if name == "video" and video_feature_callback is not None:
                    video_feature_callback(layer_idx, updated_tokens)
                start = end

        return tokens_all
