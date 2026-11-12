"""PI0/PI0.5 model wrapped for Pai-Megatron-Patch distributed training.

Supports:
  DP  – handled by Megatron's DDP wrapper (automatic via pretrain()).
  TP  – ColumnParallelLinear / RowParallelLinear in PI0ModelLayerTP.
  PP  – pre_process / post_process flags with a fixed-shape 3-D inter-stage tensor.
  SP  – can be enabled via config.sequence_parallel (experimental).

Pipeline Parallelism design
---------------------------
Megatron PP sends a fixed-shape (S, B, H) tensor between consecutive stages.
For PI0 we pack both VLM and Expert hidden states into this tensor:

    hidden_size (H) = D_vlm = 2048  (VLM hidden, larger of the two branches)
    batch_size  (B) = pp_max_batch_size
    seq_length  (S) = pp_max_prefix + pp_max_suffix + overhead_rows

Layout of the (S, B, H=2048) tensor, for each batch slot b:
    rows [0 : prefix_lens[b]]
        VLM prefix hidden states for sample b       → (D_vlm = 2048 channels)
    rows [max_prefix : max_prefix + suffix_lens[b]]
        Expert suffix hidden states for sample b    → first D_exp=1024 channels,
                                                       remaining 1024 channels are zero
    row  [max_prefix + max_suffix]
        adaRMS conditioning vector (PI0.5) or zeros (PI0)  → D_exp=1024 channels
    row  [max_prefix + max_suffix + 1]
        metadata (packed as float16/bf16 values in specific channels):
          col 0,1  : prefix_lens[b] hi, lo   (v = hi*128 + lo, exact in BF16)
          col 2,3  : suffix_lens[b] hi, lo
          col 4    : B_eff                   (only in slot b=0)
          col 5    : use_shared_obs          (0 or 1, only in slot b=0)
    rows [meta+2 : meta+7]
        position IDs packed as float32 (5 rows × H floats)
    rows [meta+7 : end]
        target velocity (dense-packed float; last stage uses these for MSE loss)

Matching Megatron arguments:
    --seq-length       = config.pp_seq_length
    --micro-batch-size = config.pp_max_batch_size
    --hidden-size      = config.hidden_size  (= 2048)
"""

import math
import os
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.module import MegatronModule
from megatron.core import parallel_state

from .transformer_config import PI0TransformerConfig
from .layers import PI0ModelLayerTP, GatedRMSNorm, AdaRMSNorm
from .embedders import (
    PI0PrefixEmbedder,
    PI0SuffixEmbedder,
    PI05SuffixEmbedder,
    build_attention_mask_and_position_ids,
    build_shared_obs_attention_mask_and_position_ids,
    pad_vector,
)
from .subtask import combine_pi05se_losses, compute_subtask_ce

import torch.distributed as dist


# Number of rows used for packed position IDs in the PP tensor
_N_IDS_ROWS = 5


def _profile_range(name: str):
    """Emit matching semantic ranges only for explicitly profiled runs."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if os.environ.get("PI05_PROFILE_MARKERS") == "1" and rank == 0:
        return torch.profiler.record_function(name)
    return nullcontext()


def _trace_backward_tensor(name: str, tensor: Optional[Tensor]) -> None:
    """Save one activation gradient for fixed-replay first-bad analysis."""
    trace_dir = os.environ.get("PI05_BACKWARD_ALIGNMENT_TRACE_DIR")
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if not trace_dir or rank != 0 or tensor is None or not tensor.requires_grad:
        return

    from pathlib import Path

    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)

    def _save(gradient: Tensor) -> Tensor:
        torch.save(gradient.detach().cpu(), path / f"native_{name}.pt")
        return gradient

    tensor.register_hook(_save)


def _try_get_pp_rank() -> int:
    try:
        return parallel_state.get_pipeline_model_parallel_rank()
    except Exception:
        return 0


def _try_get_pp_size() -> int:
    try:
        return parallel_state.get_pipeline_model_parallel_world_size()
    except Exception:
        return 1


class PI0Model(MegatronModule):
    """PI0/PI0.5 VLA model for Pai-Megatron-Patch distributed training.

    Parameters
    ----------
    config : PI0TransformerConfig
    pre_process : bool
        This stage owns the input embedding layers (PrefixEmbedder, SuffixEmbedder).
    post_process : bool
        This stage owns the output head (final LayerNorm + action_out_proj + loss).
    """

    def __init__(
        self,
        config: PI0TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__(config=config)
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process

        D_vlm = config.hidden_size                # 2048
        D_exp = config.action_expert_hidden_size  # 1024
        use_adarms = config.use_adarms

        # ------------------------------------------------------------------
        # Pre-process stage: embedders + possibly load PaliGemma weights
        # ------------------------------------------------------------------
        if self.pre_process:
            vlm = self._load_paligemma(config)
            self.prefix_embedder = PI0PrefixEmbedder(config, vlm)
            if use_adarms:
                # LeRobot PI0.5 conditions AdaRMS on time only; state is not consumed.
                self.suffix_embedder = PI05SuffixEmbedder(config, state_cond=False)
            else:
                self.suffix_embedder = PI0SuffixEmbedder(config)

        # ------------------------------------------------------------------
        # Determine which transformer layers belong to this PP stage
        # ------------------------------------------------------------------
        pp_size = _try_get_pp_size()
        pp_rank = _try_get_pp_rank()
        total_layers = config.num_layers  # 18

        layers_per_stage = (total_layers + pp_size - 1) // pp_size
        layer_start = pp_rank * layers_per_stage
        layer_end = min(layer_start + layers_per_stage, total_layers)

        self.layers = nn.ModuleList(
            [PI0ModelLayerTP(config) for _ in range(layer_end - layer_start)]
        )
        self.layer_offset = layer_start  # for debugging / checkpoint mapping
        for local_idx, layer in enumerate(self.layers):
            global_idx = layer_start + local_idx
            layer.alignment_layer_idx = global_idx
            layer.attn.alignment_layer_idx = global_idx

        # ------------------------------------------------------------------
        # Post-process stage: final LayerNorms + action output projection
        # ------------------------------------------------------------------
        if self.post_process:
            D_exp_cond = config.adarms_cond_dim if use_adarms else 0
            # VLM final norm: standard GemmaRMSNorm (1D weight, no conditioning)
            self.vlm_final_norm = AdaRMSNorm(D_vlm, 0) if use_adarms else GatedRMSNorm(D_vlm)
            # Expert final norm: AdaRMS conditioned on adarms_cond (same as per-layer expert norms)
            self.exp_final_norm = AdaRMSNorm(D_exp, D_exp_cond) if use_adarms else GatedRMSNorm(D_exp)
            # Projects action expert output → velocity prediction
            self.action_out_proj = nn.Linear(D_exp, config.max_action_dim, bias=True)

        # ------------------------------------------------------------------
        # PP communication tensor bookkeeping
        # ------------------------------------------------------------------
        self._pp_max_prefix = config.pp_max_prefix_per_sample
        self._pp_max_suffix = config.pp_max_suffix_per_sample
        self._pp_max_batch = config.pp_max_batch_size
        self._D_vlm = D_vlm
        self._D_exp = D_exp
        self._n_action_dim = config.max_action_dim

        # Row offsets in (S, max_B, D_vlm) tensor
        self._comm_cond_row = self._pp_max_prefix + self._pp_max_suffix
        self._comm_meta_row = self._comm_cond_row + 1
        self._comm_ids_start = self._comm_meta_row + 1
        target_rows = math.ceil(
            self._pp_max_suffix * config.max_action_dim / D_vlm
        )
        self._comm_target_start = self._comm_ids_start + _N_IDS_ROWS
        self._comm_S = self._comm_target_start + target_rows

        # Storage for targets received from PP tensor (set on post_process stage)
        self._received_targets: Optional[Tensor] = None
        self._received_target_lens: Optional[List[int]] = None

    def _apply(self, fn, recurse=True):
        """Apply dtype/device conversion with an explicit FP32 opt-in.

        The authoritative FluxVLA FSDP recipe keeps FP32 master parameters and
        uses BF16 autocast for transformer compute.  Megatron's ``--bf16``
        normally converts the model itself, so this opt-in preserves the
        source parameter-storage contract for precision-alignment runs.
        """
        if (not self.config.use_adarms
                or os.environ.get("PI05_PRESERVE_FP32_PARAMS") != "1"):
            return super()._apply(fn, recurse=recurse)

        preserved = {
            name: param.data.clone()
            for name, param in self.named_parameters()
            if param.dtype == torch.float32
        }
        result = super()._apply(fn, recurse=recurse)
        for name, param in self.named_parameters():
            if name in preserved:
                param.data = preserved[name].to(device=param.device, dtype=torch.float32)
        return result

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets=(),
        metadata=None,
    ):
        """Describe PI0.5 parameters to MCore distributed checkpointing.

        PI0.5 mixes ordinary PyTorch modules with MCore TP linear layers. A
        recursive default state dict can lose parameter identity at those
        module boundaries, which prevents MCore from associating Adam state
        with model shards. Building one keep-vars state dict preserves that
        identity while the TP metadata already attached by MCore determines
        each parameter's partition axis.
        """
        del metadata
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        state_dict = self.state_dict(prefix="", keep_vars=True)
        tp_axis_map = {
            name: int(getattr(parameter, "partition_dim", 0))
            for name, parameter in self.named_parameters()
            if getattr(parameter, "tensor_model_parallel", False)
        }
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            tp_axis_map,
            sharded_offsets,
        )

    # ------------------------------------------------------------------
    # PP: input tensor setter (called by Megatron PP schedule)
    # ------------------------------------------------------------------

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _trace_alignment_parameters(self):
        trace_dir = os.environ.get("PI05_PARAM_TENSOR_TRACE_DIR")
        if not trace_dir or int(os.environ.get("RANK", "0")) != 0:
            return
        step = getattr(self, "_alignment_param_forward", 0)
        if step >= int(os.environ.get("PI05_PARAM_TENSOR_TRACE_STEPS", "2")):
            return
        from pathlib import Path
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {
            "vision_q": self.prefix_embedder._vlm_model.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj.weight,
            "vlm_q": self.layers[0].attn.vlm_q.weight,
            "expert_cond": self.layers[0].exp_input_norm.cond_proj.weight,
            "action_out": self.action_out_proj.weight,
        }
        torch.save({name: value.detach().cpu() for name, value in tensors.items()},
                   path / f"native_params_before_forward_{step:02d}.pt")
        self._alignment_param_forward = step + 1

    def forward(
        self,
        # --- Supplied on pre_process stage ---
        images: Optional[List[Tensor]] = None,   # list of [B, C, H, W]
        img_masks: Optional[List[Tensor]] = None, # list of [B] bool
        tokens: Optional[Tensor] = None,          # [B, L_text] long
        text_masks: Optional[Tensor] = None,      # [B, L_text] bool
        states: Optional[Tensor] = None,          # [B, state_dim] or [B, N, state_dim]
        actions: Optional[Tensor] = None,         # [B, T, action_dim] or [B, N, T, action_dim]
        noise: Optional[Tensor] = None,           # same shape as actions
        time: Optional[Tensor] = None,            # [B] or [B, N]
        # --- Shared observation mode ---
        shared_obs: bool = False,
        offset_mask: Optional[Tensor] = None,     # [B, N] bool
        action_is_pad: Optional[Tensor] = None,   # [B, N, chunk_size] bool — True for padded steps
        # --- Loss-signal targets (packed through PP on pre_process) ---
        target_velocity: Optional[Tensor] = None, # [B, T, action_dim]  u_t = noise - actions
        target_lens: Optional[List[int]] = None,
        subtask_spans=None,
    ) -> Tensor:
        """Forward pass.

        On pre_process stage: performs full embedding + first N layers.
        On subsequent PP stages: receives packed tensor from previous stage.
        On post_process stage: applies output head and returns MSE loss.

        Returns:
            On post_process: scalar MSE loss.
            On other stages: packed PP communication tensor (S, B, D_vlm).
        """
        self._trace_alignment_parameters()
        # ---------------------------------------------------------------
        # PP: Unpack from previous stage OR compute initial embeddings
        # ---------------------------------------------------------------
        if self.pre_process:
            # ---- Flow Matching: add noise to actions ----
            assert actions is not None and states is not None
            if (self.config.subtask_ce_weight or self.config.knowledge_insulation) and _try_get_pp_size() != 1:
                raise NotImplementedError("PI05se P0 supports PP=1; PP metadata support is planned for P2")
            self._subtask_tokens = tokens
            self._subtask_spans = subtask_spans
            if shared_obs:
                with _profile_range("pi05.model.shared_observation_embedding"):
                    vlm_hs, exp_hs, attention_mask, position_ids, adarms_cond, \
                        target_vel, t_lens = self._embed_shared_obs(
                            images, img_masks, tokens, text_masks,
                            states, actions, noise, time, offset_mask
                        )
                # Store masking metadata for _compute_loss (valid for current forward only)
                self._loss_offset_mask = offset_mask  # [B, N] or None
                self._loss_action_is_pad = action_is_pad  # [B, N, chunk_size] or None
                self._loss_action_mask = None
                self._loss_N = states.shape[1]  # number of offsets
            else:
                with _profile_range("pi05.model.standard_embedding"):
                    vlm_hs, exp_hs, attention_mask, position_ids, adarms_cond, \
                        target_vel, t_lens = self._embed_standard(
                            images, img_masks, tokens, text_masks,
                            states, actions, noise, time
                        )
                self._loss_offset_mask = None
                self._loss_action_is_pad = None
                # Fixed FSDP replay stores its valid-action mask under the
                # legacy action_is_pad key. In standard mode it has the same
                # semantics as FluxVLA's action_masks (1 means valid).
                self._loss_action_mask = action_is_pad
                self._loss_N = None
            # Pack target into PP tensor for transmission to last stage
            target_velocity = target_vel
            target_lens = t_lens
        else:
            # Unpack from previous stage tensor
            (vlm_hs, exp_hs, attention_mask, position_ids, adarms_cond,
             target_velocity, target_lens) = self._unpack_pp_tensor(self.input_tensor)
            if self.post_process:
                self._received_targets = target_velocity
                self._received_target_lens = target_lens

        # ---------------------------------------------------------------
        # Run this stage's transformer layers
        # ---------------------------------------------------------------
        vlm_cond = None  # VLM branch never receives adaRMS cond
        # PI0.5 AE branch applies adaRMS conditioning (scale+shift+gate from
        # time_mlp(sinusoidal(t)), matching pi05_training PiGemmaRMSNorm behavior).
        # pi05_base dense.weight/bias are loaded for each layer's exp_input/post_attn_norm.
        exp_cond = adarms_cond if self.config.use_adarms else None

        trace_dir = os.environ.get("PI05_ALIGNMENT_TRACE_DIR")
        trace_enabled = trace_dir and torch.distributed.get_rank() == 0 and not getattr(
            self, "_alignment_trace_done", False
        )
        if trace_enabled:
            from pathlib import Path
            trace_path = Path(trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"prefix": vlm_hs[:1].detach().cpu(), "suffix": exp_hs[:1].detach().cpu()},
                trace_path / "pmp_embed.pt",
            )

        for layer_idx, layer in enumerate(self.layers):
            def run_layer(*args, _layer=layer):
                hidden_states = args[0]
                with torch.autocast(
                    device_type=hidden_states.device.type,
                    dtype=torch.bfloat16,
                    enabled=hidden_states.is_cuda,
                ):
                    return _layer(*args)

            recompute_layers = self.config.recompute_num_layers or len(self.layers)
            with _profile_range(f"pi05.model.transformer_layer.{layer_idx:02d}"):
                if (
                    self.training
                    and self.config.recompute_granularity is not None
                    and layer_idx < recompute_layers
                ):
                    vlm_hs, exp_hs = checkpoint(
                        run_layer,
                        vlm_hs,
                        exp_hs,
                        attention_mask,
                        position_ids,
                        vlm_cond,
                        exp_cond,
                        # Reentrant checkpointing releases the per-layer autograd
                        # graph between iterations. This is required at MBS=32 once
                        # AdamW state is materialized after the first optimizer step.
                        use_reentrant=True,
                        preserve_rng_state=False,
                    )
                else:
                    vlm_hs, exp_hs = run_layer(
                        vlm_hs,
                        exp_hs,
                        attention_mask,
                        position_ids,
                        vlm_cond,
                        exp_cond,
                    )
            _trace_backward_tensor(f"layer_{layer_idx:02d}_prefix_output", vlm_hs)
            _trace_backward_tensor(f"layer_{layer_idx:02d}_suffix_output", exp_hs)
            if trace_enabled:
                torch.save(
                    {"prefix": vlm_hs[:1].detach().cpu(), "suffix": exp_hs[:1].detach().cpu()},
                    trace_path / f"pmp_layer_{layer_idx:02d}.pt",
                )

        # ---------------------------------------------------------------
        # Post-process: final norm + output projection → loss
        # ---------------------------------------------------------------
        if self.post_process:
            with _profile_range("pi05.loss.total"):
                loss = self._compute_loss(vlm_hs, exp_hs, exp_cond, target_velocity, target_lens)
            ce_trace_dir = os.environ.get("PI05_CE_ALIGNMENT_TRACE_DIR")
            # CE tracing is diagnostic-only. Limit it to rank zero so the normal
            # DP8 training graph does not retain eight redundant trace payloads.
            ce_trace_enabled = (
                ce_trace_dir
                and torch.distributed.get_rank() == 0
                and not getattr(self, "_ce_alignment_trace_done", False)
            )
            if ce_trace_enabled:
                # Save every DP rank.  The summary is intentionally compact: it
                # contains the exact target logit/NLL for each caption token and
                # identifies the first divergent token before a full-vocabulary
                # trace is requested for that one location.
                from pathlib import Path

                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                spans = getattr(self, "_subtask_spans", None)
                tokens = getattr(self, "_subtask_tokens", None)
                # _compute_loss applies this norm before computing CE.  Trace
                # the same representation; the incoming vlm_hs is pre-norm.
                ce_vlm_hs, _ = self.vlm_final_norm(vlm_hs, None)
                records = []
                if spans is not None and tokens is not None:
                    n_img = ce_vlm_hs.shape[1] - tokens.shape[1]
                    head_weight = self.prefix_embedder.lang_embedder.weight
                    for row, span in enumerate(spans):
                        if span is None:
                            continue
                        lo, hi = span
                        # Keep only caption-target hidden states in the trace. This
                        # makes DP8 diagnosis affordable while distinguishing
                        # transformer-path drift from lm-head projection drift.
                        hidden = ce_vlm_hs[row, n_img + lo - 1:n_img + hi - 1]
                        logits = torch.nn.functional.linear(
                            hidden.to(head_weight.dtype),
                            head_weight,
                        ).float()
                        targets = tokens[row, lo:hi]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        records.append({
                            "row": row,
                            "span": (lo, hi),
                            "targets": targets.detach().cpu(),
                            "hidden": hidden.detach().cpu(),
                            "target_logits": logits.gather(1, targets[:, None]).squeeze(1).detach().cpu(),
                            "target_nll": (-log_probs.gather(1, targets[:, None]).squeeze(1)).detach().cpu(),
                            "argmax": logits.argmax(dim=-1).detach().cpu(),
                        })
                trace_path = Path(ce_trace_dir)
                trace_path.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema_version": 1,
                    "rank": rank,
                    "tokens": None if tokens is None else tokens.detach().cpu(),
                    "spans": spans,
                    "attention_mask": attention_mask.detach().cpu(),
                    "position_ids": position_ids.detach().cpu(),
                    "ce_tokens": records,
                }, trace_path / f"pmp_ce_step_000000_rank_{rank:02d}.pt")
                self._ce_alignment_trace_done = True
            if trace_enabled:
                torch.save(
                    {"prefix": vlm_hs[:1].detach().cpu(), "suffix": exp_hs[:1].detach().cpu()},
                    trace_path / "pmp_pre_final_norm.pt",
                )
                self._alignment_trace_done = True
            return loss

        # ---------------------------------------------------------------
        # Middle / pre-process (non-last) stage: pack and send
        # ---------------------------------------------------------------
        return self._pack_pp_tensor(
            vlm_hs, exp_hs, adarms_cond, attention_mask, position_ids,
            target_velocity, target_lens
        )

    # ------------------------------------------------------------------
    # Embedding helpers (pre_process only)
    # ------------------------------------------------------------------

    def _flow_matching_noise(self, actions, noise, time):
        """Compute x_t = t*noise + (1-t)*actions and target u_t = noise - actions."""
        if noise is None:
            noise = torch.randn_like(actions, dtype=torch.float32)
        if time is None:
            dist_beta = torch.distributions.Beta(
                self.config.time_sampling_beta_alpha,
                self.config.time_sampling_beta_beta,
            )
            time = dist_beta.sample((actions.shape[0],)).to(
                device=actions.device, dtype=torch.float32
            ) * 0.999 + 0.001

        # Expand time for broadcasting
        ndim_extra = actions.ndim - 1
        t_exp = time.view(*time.shape, *([1] * ndim_extra))
        x_t = t_exp * noise.float() + (1 - t_exp) * actions.float()
        u_t = noise.float() - actions.float()
        return x_t, u_t, time

    def _embed_standard(self, images, img_masks, tokens, text_masks,
                         states, actions, noise, time):
        """Standard (non-shared-obs) embedding."""
        B = actions.shape[0]
        with _profile_range("pi05.model.flow_target"):
            x_t, u_t, time = self._flow_matching_noise(actions, noise, time)

        with _profile_range("pi05.model.prefix_embedding"):
            prefix_embs, prefix_pad, prefix_att = self.prefix_embedder(
                images, img_masks, tokens, text_masks, getattr(self, "_subtask_spans", None)
            )
        with _profile_range("pi05.model.suffix_embedding"):
            suffix_inputs = (
                states,
                pad_vector(
                    x_t.to(dtype=self.suffix_embedder.action_in_proj.weight.dtype),
                    self.config.max_action_dim,
                ),
                time,
            )
            # The reference FSDP runner wraps the complete PI0.5 forward in
            # BF16 autocast, including action/time projections whose parameters
            # remain FP32 under the alignment contract.  Native MCore enters
            # its transformer autocast later, so make this boundary explicit.
            with torch.autocast(
                device_type=x_t.device.type,
                dtype=torch.bfloat16,
                enabled=x_t.is_cuda,
            ):
                suffix_embs, suffix_pad, suffix_att, adarms_cond = self.suffix_embedder(
                    *suffix_inputs
                )

        full_pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        full_att = torch.cat([prefix_att, suffix_att], dim=1)
        with _profile_range("pi05.model.attention_mask"):
            # FSDP keeps the language-promoted prefix in FP32 while action
            # embeddings and the additive attention mask remain BF16.  The
            # layer autocast below handles both branches without requiring a
            # lossy common input dtype.
            attention_mask, position_ids = build_attention_mask_and_position_ids(
                full_pad, full_att, suffix_embs.dtype
            )

        target_vel = pad_vector(u_t, self.config.max_action_dim)  # [B, T, max_action_dim]
        target_lens = [self.config.chunk_size] * B

        return prefix_embs, suffix_embs, attention_mask, position_ids, adarms_cond, target_vel, target_lens

    def _embed_shared_obs(self, images, img_masks, tokens, text_masks,
                           states, actions, noise, time, offset_mask):
        """Shared-observation multi-offset embedding."""
        B, N = states.shape[:2]

        # actions: [B, N, T, action_dim]
        x_t_all, u_t_all, time_all = self._flow_matching_noise(
            actions.reshape(B * N, *actions.shape[2:]),
            noise.reshape(B * N, *noise.shape[2:]) if noise is not None else None,
            time.reshape(B * N) if time is not None else None,
        )
        x_t_all = x_t_all.reshape(B, N, *actions.shape[2:])
        u_t_all = u_t_all.reshape(B, N, *actions.shape[2:])
        time_all = time_all.reshape(B, N)

        prefix_embs, prefix_pad, prefix_att = self.prefix_embedder(
            images, img_masks, tokens, text_masks, getattr(self, "_subtask_spans", None)
        )
        prefix_embs = prefix_embs.to(torch.bfloat16)

        # Embed all offsets flat: [B*N, T, action_dim]
        states_flat = states.reshape(B * N, -1)
        x_t_flat = pad_vector(
            x_t_all.reshape(B * N, self.config.chunk_size, -1).to(
                self.suffix_embedder.action_in_proj.weight.dtype
            ),
            self.config.max_action_dim
        )
        time_flat = time_all.reshape(B * N)
        suf_embs_flat, suf_pad_flat, suf_att_flat, ada_cond_flat = self.suffix_embedder(
            states_flat, x_t_flat, time_flat
        )

        suf_len = suf_embs_flat.shape[1]
        suffix_embs = suf_embs_flat.reshape(B, N * suf_len, -1)
        suf_pad_one = suf_pad_flat[:B]
        suf_att_one = suf_att_flat[:B]

        adarms_cond = ada_cond_flat.reshape(B, N, -1) if ada_cond_flat is not None else None
        # Use mean over valid offsets as the per-sample conditioning
        if adarms_cond is not None and offset_mask is not None:
            valid = offset_mask.float().unsqueeze(-1)  # [B, N, 1]
            adarms_cond = (adarms_cond * valid).sum(1) / valid.sum(1).clamp(min=1)

        backbone_dtype = prefix_embs.dtype
        suffix_embs = suffix_embs.to(backbone_dtype)

        attention_mask, position_ids = build_shared_obs_attention_mask_and_position_ids(
            prefix_pad, prefix_att, suf_pad_one, suf_att_one,
            N, offset_mask if offset_mask is not None else
            torch.ones(B, N, dtype=torch.bool, device=states.device),
            backbone_dtype,
        )

        target_vel = pad_vector(u_t_all.reshape(B, N * self.config.chunk_size, -1),
                                self.config.max_action_dim)
        target_lens = [N * self.config.chunk_size] * B

        return prefix_embs, suffix_embs, attention_mask, position_ids, adarms_cond, target_vel, target_lens

    # ------------------------------------------------------------------
    # Loss computation (post_process only)
    # ------------------------------------------------------------------

    def _compute_loss(self, vlm_hs, exp_hs, exp_cond,
                       target_velocity, target_lens) -> Tensor:
        """Apply final norm, project to action space, compute MSE loss."""
        _trace_backward_tensor("pre_final_norm", exp_hs)
        _trace_backward_tensor("final_norm_condition", exp_cond)
        compute_hs = exp_hs if exp_hs is not None else vlm_hs
        autocast = torch.autocast(
            device_type=compute_hs.device.type,
            dtype=torch.bfloat16,
            enabled=compute_hs.is_cuda,
        )
        # Final LayerNorm for each branch
        with _profile_range("pi05.model.final_norm"), autocast:
            if vlm_hs is not None:
                vlm_hs, _ = self.vlm_final_norm(vlm_hs, None)
            if exp_hs is not None:
                exp_hs, _ = self.exp_final_norm(exp_hs, exp_cond)

        # Extract action predictions from the Expert branch suffix
        # Suffix layout (PI0):   [state_tok, action_0, ..., action_{T-1}]  → last T toks
        # Suffix layout (PI0.5): [action_0, ..., action_{T-1}]             → last T toks
        T = target_lens[0] if target_lens else self.config.chunk_size
        action_out = exp_hs[:, -T:, :]                           # [B, T, D_exp]
        action_out = action_out.to(self.action_out_proj.weight.dtype)
        _trace_backward_tensor("action_head_input", action_out)
        with _profile_range("pi05.model.action_head"), torch.autocast(
            device_type=action_out.device.type,
            dtype=torch.bfloat16,
            enabled=action_out.is_cuda,
        ):
            v_t = self.action_out_proj(action_out)               # [B, T, max_action_dim]
        _trace_backward_tensor("prediction", v_t)

        # Flow Matching MSE loss — match vlash's masked aggregation formula
        u_t = target_velocity[:, :T, :]                          # [B, T, max_action_dim]

        # Retrieve masking metadata stored during _embed_shared_obs (or None for standard mode)
        offset_mask   = getattr(self, "_loss_offset_mask",   None)  # [B, N]
        action_is_pad = getattr(self, "_loss_action_is_pad", None)  # [B, N, chunk_size]
        action_mask   = getattr(self, "_loss_action_mask",   None)  # [B, chunk_size]
        N             = getattr(self, "_loss_N",             None)  # int

        # The reference runner wraps the complete model forward, including
        # loss construction, in BF16 autocast. MSE still returns FP32, but its
        # autograd kernel selection changes and therefore so does the BF16
        # prediction gradient. Keep this boundary explicit in native MCore.
        with _profile_range("pi05.loss.flow"), torch.autocast(
            device_type=v_t.device.type,
            dtype=torch.bfloat16,
            enabled=v_t.is_cuda,
        ):
            if offset_mask is not None or action_is_pad is not None:
            # Use F.mse_loss(reduction="none") as base — same kernel as vlash — to preserve
            # bf16 numerical path and keep step-1 bit-exactness.
                losses = F.mse_loss(u_t, v_t, reduction="none")  # [B, T, max_action_dim]
                B = losses.shape[0]
                chunk = self.config.chunk_size

            # Zero out padded action steps.  Cast mask to losses.dtype (bf16) — vlash uses
            # bool directly (bf16*bool=bf16); .to(dtype) is equivalent and avoids float32 upcast.
                if action_is_pad is not None:
                    in_ep = (~action_is_pad).to(dtype=losses.dtype)   # [B, N, chunk_size]
                    in_ep_flat = in_ep.reshape(B, T).unsqueeze(-1)    # [B, T, 1]
                    losses = losses * in_ep_flat

            # Zero out invalid offset groups and count valid offsets for denominator.
                if offset_mask is not None:
                    om = offset_mask.to(dtype=losses.dtype)            # [B, N]
                    om_flat = om.unsqueeze(-1).expand(-1, -1, chunk).reshape(B, T).unsqueeze(-1)
                    losses = losses * om_flat
                    num_valid_offsets = offset_mask.sum()
                else:
                    num_valid_offsets = torch.tensor(B * N, dtype=torch.int64, device=v_t.device)

                num_elements_per_offset = chunk * self.config.max_action_dim
                loss = losses.sum() / (num_valid_offsets * num_elements_per_offset).clamp(min=1)
            else:
                # Match FluxVLA's reduce_action_bc_loss operation order. Its
                # explicit masked sum has a different BF16 backward rounding
                # path from F.mse_loss(reduction="mean"), even for an all-one
                # mask and a bitwise-equal scalar loss.
                out_dim = self.config.output_action_dim or self.config.max_action_dim
                losses = F.mse_loss(
                    u_t[:, :, :out_dim], v_t[:, :, :out_dim], reduction="none"
                )
                if action_mask is None:
                    loss = losses.mean()
                else:
                    valid = action_mask.to(device=losses.device, dtype=losses.dtype)
                    while valid.ndim < losses.ndim:
                        valid = valid.unsqueeze(-1)
                    valid = valid.expand_as(losses)
                    loss = (losses * valid).sum() / (valid.sum() + 1e-8)
        subtask_ce = None
        spans = getattr(self, "_subtask_spans", None)
        if self.config.subtask_ce_weight and spans is not None:
            if _try_get_pp_size() != 1 or parallel_state.get_tensor_model_parallel_world_size() != 1:
                raise NotImplementedError("PI05se vocabulary CE P0 requires TP=1 and PP=1")
            tokens = self._subtask_tokens
            with _profile_range("pi05.loss.subtask_ce"):
                subtask_ce = compute_subtask_ce(
                    vlm_hs,
                    tokens,
                    spans,
                    vlm_hs.shape[1] - tokens.shape[1],
                    self.prefix_embedder.lang_embedder.weight,
                )
        self.last_flow_loss = loss.detach()
        self.last_subtask_ce = None if subtask_ce is None else subtask_ce.detach()
        trace_dir = os.environ.get("PI05_LOSS_ALIGNMENT_TRACE_DIR")
        if (
            trace_dir
            and int(os.environ.get("RANK", "0")) == 0
            and not getattr(self, "_alignment_trace_done", False)
        ):
            from pathlib import Path

            trace_path = Path(trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            trace_losses = F.mse_loss(u_t, v_t, reduction="none")
            torch.save(
                {
                    "suffix_final_norm": action_out[:1].detach().cpu(),
                    "prediction": v_t[:1].detach().cpu(),
                    "target": u_t[:1].detach().cpu(),
                    "elementwise_loss": trace_losses[:1].detach().cpu(),
                    "loss": loss.detach().cpu(),
                },
                trace_path / "pmp_loss.pt",
            )
        return combine_pi05se_losses(loss, subtask_ce, self.config.subtask_ce_weight)

    # ------------------------------------------------------------------
    # PP tensor packing / unpacking
    # ------------------------------------------------------------------

    def _pack_pp_tensor(
        self,
        vlm_hs: Tensor,           # [B, L_prefix, D_vlm]
        exp_hs: Tensor,           # [B, L_suffix, D_exp]
        adarms_cond: Optional[Tensor],  # [B, D_exp] or None
        attention_mask: Tensor,
        position_ids: Tensor,     # [B, L_total] long
        target_velocity: Tensor,  # [B, T, max_action_dim]
        target_lens: List[int],
    ) -> Tensor:
        """Pack VLM + Expert hidden states and metadata into (S, max_B, D_vlm) tensor."""
        D = self._D_vlm          # 2048
        max_pre = self._pp_max_prefix
        max_suf = self._pp_max_suffix
        max_B = self._pp_max_batch
        S = self._comm_S
        cond_row = self._comm_cond_row
        meta_row = self._comm_meta_row
        ids_start = self._comm_ids_start
        target_start = self._comm_target_start
        in_ch = self._n_action_dim

        B_eff = vlm_hs.shape[0]
        device, dtype = vlm_hs.device, vlm_hs.dtype

        out = torch.zeros(S, max_B, D, device=device, dtype=dtype)

        # ---- VLM prefix hidden states ----
        for b in range(B_eff):
            L_pre = vlm_hs.shape[1]  # could vary; for simplicity use fixed shape
            pre = vlm_hs[b]          # [L_pre, D_vlm]
            out[:L_pre, b, :] = pre

        # ---- Expert suffix hidden states (padded to D_vlm) ----
        for b in range(B_eff):
            L_suf = exp_hs.shape[1]
            suf = exp_hs[b]          # [L_suf, D_exp]
            out[max_pre: max_pre + L_suf, b, :self._D_exp] = suf

        # ---- adaRMS conditioning vector ----
        if adarms_cond is not None:
            out[cond_row, :B_eff, :self._D_exp] = adarms_cond.to(dtype)

        # ---- Metadata ----
        out[meta_row, 0, 4] = float(B_eff)
        for b in range(B_eff):
            L_pre_b = vlm_hs.shape[1]
            L_suf_b = exp_hs.shape[1]
            out[meta_row, b, 0] = float(L_pre_b // 128)
            out[meta_row, b, 1] = float(L_pre_b % 128)
            out[meta_row, b, 2] = float(L_suf_b // 128)
            out[meta_row, b, 3] = float(L_suf_b % 128)

        # ---- Position IDs (packed float32 across _N_IDS_ROWS rows) ----
        ids_per = (max_pre + max_suf) * 1  # 1D position ids per token
        for b in range(B_eff):
            L_total = position_ids.shape[1]
            flat = position_ids[b].float()[:ids_per]  # [ids_per]
            buf = out.new_zeros(_N_IDS_ROWS * D)
            buf[:flat.shape[0]] = flat
            out[ids_start: ids_start + _N_IDS_ROWS, b, :] = buf.reshape(_N_IDS_ROWS, D)

        # ---- Target velocity (dense-packed) ----
        if target_velocity is not None:
            t_off = 0
            for b in range(B_eff):
                tl = target_lens[b] if target_lens else target_velocity.shape[1]
                flat = target_velocity[b, :tl].to(dtype).reshape(-1)  # [tl * in_ch]
                rows_b = math.ceil(float(len(flat)) / D)
                padded = out.new_zeros(rows_b * D)
                padded[:flat.shape[0]] = flat
                out[target_start: target_start + rows_b, b, :] = padded.reshape(rows_b, D)

        return out

    def _unpack_pp_tensor(
        self, packed: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], List[int]]:
        """Unpack (S, max_B, D_vlm) tensor from previous PP stage.

        Returns:
            vlm_hs:          [B_eff, L_prefix, D_vlm]
            exp_hs:          [B_eff, L_suffix, D_exp]
            attention_mask:  [B_eff, 1, L_total, L_total] additive
            position_ids:    [B_eff, L_total] long
            adarms_cond:     [B_eff, D_exp] or None
            target_velocity: [total_T, max_action_dim] or None
            target_lens:     List[int]
        """
        D = packed.shape[-1]  # should be D_vlm = 2048
        max_pre = self._pp_max_prefix
        cond_row = self._comm_cond_row
        meta_row = self._comm_meta_row
        ids_start = self._comm_ids_start
        target_start = self._comm_target_start
        in_ch = self._n_action_dim

        # ---- Metadata ----
        B_eff = int(packed[meta_row, 0, 4].item())
        prefix_lens = [
            int(round(packed[meta_row, b, 0].item())) * 128
            + int(round(packed[meta_row, b, 1].item()))
            for b in range(B_eff)
        ]
        suffix_lens = [
            int(round(packed[meta_row, b, 2].item())) * 128
            + int(round(packed[meta_row, b, 3].item()))
            for b in range(B_eff)
        ]

        # ---- VLM hidden states ----
        L_pre_max = max(prefix_lens) if prefix_lens else 0
        vlm_list = [packed[:prefix_lens[b], b, :] for b in range(B_eff)]
        # Pad to L_pre_max for batched tensor
        vlm_hs = torch.stack([
            F.pad(v, (0, 0, 0, L_pre_max - v.shape[0]))
            for v in vlm_list
        ], dim=0)  # [B_eff, L_pre_max, D_vlm]

        # ---- Expert hidden states (recover D_exp slice) ----
        L_suf_max = max(suffix_lens) if suffix_lens else 0
        exp_list = [
            packed[max_pre: max_pre + suffix_lens[b], b, :self._D_exp]
            for b in range(B_eff)
        ]
        exp_hs = torch.stack([
            F.pad(e, (0, 0, 0, L_suf_max - e.shape[0]))
            for e in exp_list
        ], dim=0)  # [B_eff, L_suf_max, D_exp]

        # ---- adaRMS conditioning ----
        adarms_cond = None
        if self.config.use_adarms:
            adarms_cond = packed[cond_row, :B_eff, :self._D_exp]

        # ---- Position IDs ----
        L_total = L_pre_max + L_suf_max
        ids_per = L_total
        pos_list = []
        for b in range(B_eff):
            flat = (
                packed[ids_start: ids_start + _N_IDS_ROWS, b, :]
                .float().reshape(-1)[:ids_per]
            )
            pos_list.append(flat.long())
        position_ids = torch.stack(pos_list, dim=0)  # [B_eff, L_total]

        # ---- Build attention mask from position_ids (causal full-attention) ----
        # On non-pre-process stages we don't have the original pad/att masks,
        # so we reconstruct a simple causal mask: all tokens attend to all tokens
        # (the real masking was established on the pre-process stage).
        B, L = position_ids.shape
        dtype = vlm_hs.dtype
        attention_mask = torch.zeros(B, 1, L, L, device=vlm_hs.device, dtype=dtype)

        # ---- Target velocity ----
        target_velocity = None
        target_lens = []
        # Targets are stored packed; for now unpack assuming fixed chunk_size
        T_fixed = self.config.chunk_size
        if packed.shape[0] > target_start:
            t_parts = []
            for b in range(B_eff):
                rows_b = math.ceil(float(T_fixed * in_ch) / D)
                flat = (
                    packed[target_start: target_start + rows_b, b, :]
                    .reshape(-1)[:T_fixed * in_ch]
                )
                t_parts.append(flat.reshape(T_fixed, in_ch))
                target_lens.append(T_fixed)
            target_velocity = torch.stack(t_parts, dim=0)  # [B_eff, T, in_ch]

        return vlm_hs, exp_hs, attention_mask, position_ids, adarms_cond, target_velocity, target_lens

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _load_paligemma(self, config: PI0TransformerConfig):
        """Build PaliGemma architecture for PI0/PI0.5.

        Loads the config from pretrained_model_path (for text_config/vocab etc.) but
        overrides vision_config.intermediate_size=4304 to match the pi05_base checkpoint.
        Weights (SigLIP, embed_tokens, multi_modal_projector) are loaded later by
        _load_pi05_base via the OpenPI→Megatron key remapping.
        """
        try:
            from transformers.models.paligemma.modeling_paligemma import (
                PaliGemmaForConditionalGeneration,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for PaliGemma. Install via `pip install transformers`."
            ) from e

        if config.pretrained_model_path:
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(config.pretrained_model_path)
            # pi05_base SigLIP uses intermediate_size=4304 (not 4096 in e01163 config)
            if hasattr(hf_cfg, "vision_config"):
                hf_cfg.vision_config.intermediate_size = 4304
            vlm = PaliGemmaForConditionalGeneration(hf_cfg).to(dtype=torch.bfloat16)
        else:
            from .paligemma_config import build_pi0_paligemma_config

            vlm_config = build_pi0_paligemma_config()
            vlm = PaliGemmaForConditionalGeneration(vlm_config)

        # PI0.5's reference policy keeps the complete SigLIP tower and image
        # projector in FP32 when fp32_vision=True.  Constructing the PMP wrapper
        # in BF16 first is convenient for the language path, but would otherwise
        # make safetensors loading round these visual weights to BF16 permanently.
        # Restore their parameter storage before checkpoint loading; autocast in
        # PI0PrefixEmbedder still selects the same BF16 eligible kernels as the
        # reference training forward.
        if config.use_adarms and getattr(config, "fp32_vision", True):
            vlm.model.vision_tower.to(dtype=torch.float32)
            vlm.model.multi_modal_projector.to(dtype=torch.float32)

        # Apply freeze strategy.  Start with everything frozen, then selectively
        # unfreeze only the parts that PI0's forward pass actually calls.
        #
        # PMP architecture note: vlm.language_model.model.layers[0-17] are the
        # Gemma transformer layers embedded inside PaliGemmaForConditionalGeneration.
        # They are NEVER called by PI0PrefixEmbedder — the PI0ModelLayerTP layers
        # handle VLM computation.  When freeze_prefix_lm_layers=True (default) they
        # stay frozen; their grad is always None so AdamW never creates optimizer
        # states for them regardless.  Setting freeze_prefix_lm_layers=False makes
        # requires_grad=True for all params (vlash-alignment mode), with ZERO memory
        # cost because AdamW states are lazy and never fire for grad-free params.
        #
        # Parts actually used by PMP (always start frozen, then selectively thaw):
        #   - vlm.model.vision_tower         → SigLIP image encoder
        #   - vlm.model.multi_modal_projector → image projection
        #   - vlm.language_model.embed_tokens → language token embedding
        #
        # freeze_prefix_embedder=True  → keep ALL frozen (original PMP default)
        # freeze_vision_encoder=True   → keep vision_tower frozen; thaw projector + embed_tokens
        # both False                   → thaw vision_tower + projector + embed_tokens
        #                                (VLASH-matching: used parts trainable,
        #                                 unused Gemma LM layers stay frozen unless
        #                                 freeze_prefix_lm_layers=False)
        vlm.eval()
        for p in vlm.parameters():
            p.requires_grad_(False)

        # Transformers 4.x exposes the language model on the conditional-
        # generation wrapper; 5.x exposes it on PaliGemmaModel.
        language_model = getattr(vlm, "language_model", None)
        if language_model is None:
            language_model = vlm.model.language_model

        if not config.freeze_prefix_embedder:
            # Unfreeze embed_tokens and projector (used regardless of vision freeze)
            for p in language_model.embed_tokens.parameters():
                p.requires_grad_(True)
            projector = getattr(vlm.model, "multi_modal_projector", None)
            if projector is not None:
                for p in projector.parameters():
                    p.requires_grad_(True)

            if not config.freeze_vision_encoder:
                # Also unfreeze SigLIP vision tower
                vision_tower = getattr(vlm.model, "vision_tower", None)
                if vision_tower is not None:
                    vision_tower.train()
                    for p in vision_tower.parameters():
                        p.requires_grad_(True)

            if not getattr(config, "freeze_prefix_lm_layers", True):
                # vlash-alignment mode: unfreeze dead Gemma LM transformer layers too.
                # They never receive gradients so AdamW states are never allocated.
                lm_model = getattr(language_model, "model", None)
                if lm_model is not None:
                    lm_layers = getattr(lm_model, "layers", None)
                    if lm_layers is not None:
                        for p in lm_layers.parameters():
                            p.requires_grad_(True)
                # Also unfreeze any remaining norm / lm_head weights
                for name, p in language_model.named_parameters():
                    if not p.requires_grad:
                        p.requires_grad_(True)

        if config.use_adarms or getattr(config, "remove_prefix_lm_layers", False):
            # Delete dead Gemma LM transformer layers from vlm.model.language_model
            # (GemmaModel inside PaliGemmaModel).  PI0PrefixEmbedder never calls
            # these layers — it only uses embed_tokens and vision features.
            # Removing them reduces total params from ~5337M to ~3502M and saves
            # ~3503 MiB GPU memory per rank, matching vlash's parameter count.
            import torch.nn as _nn
            lm_backbone = getattr(vlm.model, "language_model", None)
            if lm_backbone is not None:
                if hasattr(lm_backbone, "layers"):
                    lm_backbone.layers = _nn.ModuleList([])
                if hasattr(lm_backbone, "norm"):
                    # norm is the final RMSNorm; never called after LM layers removed
                    del lm_backbone.norm
            del _nn

        return vlm
