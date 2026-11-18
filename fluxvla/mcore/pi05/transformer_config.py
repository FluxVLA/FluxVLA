"""PI0/PI0.5 Transformer configuration for Pai-Megatron-Patch.

PI0 is a dual-branch VLA (Vision-Language-Action) diffusion transformer for
robot control. It is NOT a language model. Key differences vs. standard GPT configs:

  - Two branches: VLM (PaliGemma, hidden=2048) + Action Expert (Gemma, hidden=1024)
  - Shared 18 transformer layers with joint cross-branch attention
  - No vocabulary output; predicts action velocity via Flow Matching (MSE loss)
  - Input: images (SigLIP-encoded) + language tokens + robot state + noisy actions
  - PI0 injects state as a token; PI0.5 uses adaRMS conditioning per layer

Architecture parameters:
  - VLM: hidden=2048, num_heads=8, head_dim=256, num_kv_heads=1, ffn=16384
  - Expert: hidden=1024, num_heads=8, head_dim=256, num_kv_heads=1, ffn=4096
  - Both branches share the same head_dim=256, allowing seq-dim concat in joint attention
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class PI0TransformerConfig(TransformerConfig):
    # -----------------------------------------------------------------------
    # VLM branch parameters (PaliGemma)
    # hidden_size, num_attention_heads, num_layers, ffn_hidden_size are
    # inherited from TransformerConfig and correspond to the VLM branch.
    # hidden_size = 2048 (also used as the PP communication tensor hidden dim)
    # num_layers = 18
    # num_attention_heads = 8
    # ffn_hidden_size = 16384 (VLM MLP intermediate size)
    # kv_channels = head_dim = 256
    # -----------------------------------------------------------------------

    # Number of key-value heads in the VLM branch (GQA: 1 KV head, 8 Q heads)
    vlm_num_kv_heads: int = 1

    # -----------------------------------------------------------------------
    # Action Expert branch parameters (Gemma ~300M)
    # Smaller hidden size than VLM, but same head_dim=256 for joint attention.
    # -----------------------------------------------------------------------

    # Hidden size of the action expert Gemma model
    action_expert_hidden_size: int = 1024

    # FFN intermediate size of the action expert
    action_expert_ffn_hidden_size: int = 4096

    # Number of attention heads in the action expert (same as VLM for joint attn)
    action_expert_num_attention_heads: int = 8

    # Number of KV heads in the action expert (GQA: 1 KV head)
    action_expert_num_kv_heads: int = 1

    # -----------------------------------------------------------------------
    # Joint attention parameters
    # Both branches share head_dim=256 so their Q/K/V can be concatenated
    # along the sequence dimension for joint attention.
    # -----------------------------------------------------------------------

    # Attention head dimension (shared across both branches)
    head_dim: int = 256

    # Keep the attention kernel contract explicit for backend alignment.
    attention_implementation: str = "sdpa"

    # -----------------------------------------------------------------------
    # VLA-specific architecture parameters
    # -----------------------------------------------------------------------

    # Maximum action vector dimension (robot DOF) — used for architecture sizing and padding
    max_action_dim: int = 32

    # Actual output action dimension for the current dataset (≤ max_action_dim).
    # When set, loss is truncated to only these dims (matching pi05_training behavior).
    # None = use max_action_dim (backward-compatible default).
    output_action_dim: Optional[int] = None

    # Maximum robot state vector dimension
    max_state_dim: int = 32

    # Number of actions predicted per chunk (= inference output length)
    chunk_size: int = 50

    # Number of image tokens produced by SigLIP per image
    num_image_tokens: int = 256

    # Maximum number of input images per sample
    max_images: int = 1

    # Maximum number of language tokens per sample
    tokenizer_max_length: int = 200

    # PI05se subtask-recovery. Disabled defaults preserve PI0.5 behavior.
    subtask_ce_weight: float = 0.0
    knowledge_insulation: bool = False

    # -----------------------------------------------------------------------
    # PI0 vs PI0.5 selection
    # PI0:   use_adarms=False — state is injected as the first suffix token
    # PI0.5: use_adarms=True  — state conditions each layer via AdaRMSNorm
    # -----------------------------------------------------------------------

    # Whether to use adaptive RMSNorm conditioning (True → PI0.5 mode)
    use_adarms: bool = False

    # Conditioning dimension for adaRMS (hidden_size of the adaRMS MLP output)
    # Only used when use_adarms=True. Typically = action_expert_hidden_size.
    adarms_cond_dim: int = 1024

    # -----------------------------------------------------------------------
    # Trainable-parameter freeze controls  (mirrors lerobot/vlash PI05Config)
    # -----------------------------------------------------------------------

    # Freeze only the SigLIP vision encoder + multi_modal_projector.
    # embed_tokens and all Transformer layers remain trainable.
    # Default False = full training (matches VLASH default).
    # Pass --pi0-freeze-vision-encoder to enable.
    freeze_vision_encoder: bool = False

    # Freeze the entire PaliGemma prefix embedder (SigLIP + embed_tokens +
    # multi_modal_projector).  When True, only the Transformer layers,
    # suffix embedder, and action head are updated.
    # Default False = full training (matches VLASH default).
    # Pass --pi0-freeze-prefix-embedder to enable (restores original PMP behaviour).
    freeze_prefix_embedder: bool = False

    # Freeze the unused Gemma language-model transformer layers that are loaded
    # inside PaliGemmaForConditionalGeneration but never called by PI0PrefixEmbedder.
    # (vlm.language_model.model.layers[0-17])
    #
    # Default True = keep original PMP behaviour: these dead layers remain frozen
    #   (they receive no gradients, so AdamW creates no optimizer states for them).
    # Set False = vlash-alignment mode: requires_grad=True for ALL params including
    #   the dead Gemma LM layers, giving frozen_params=0 like vlash.
    #   Memory impact is ZERO: AdamW is lazy — optimizer states are only created on
    #   the first .step() after a non-None gradient, which never fires for dead layers.
    freeze_prefix_lm_layers: bool = True

    # When True, physically delete the unused Gemma LM transformer layers from
    # vlm.model.language_model (the GemmaModel embedded inside PaliGemmaModel)
    # after checkpoint loading.  These 18 decoder layers are never called by
    # PI0PrefixEmbedder — it only uses vision_tower, multi_modal_projector, and
    # embed_tokens.  Deleting them reduces total params by ~1835M:
    #   5337.5M → ~3502.4M  (matches vlash total)
    # and saves ~3503 MiB of GPU memory per rank.
    #
    # Default False: keep dead layers (backward-compat, zero training cost).
    # True = vlash-alignment mode: delete layers[0-17] and norm from
    #   vlm.model.language_model, giving total params ≈ vlash.
    #   Implies frozen_params=0 (removed layers can no longer be frozen).
    remove_prefix_lm_layers: bool = False

    # -----------------------------------------------------------------------
    # Flow Matching training parameters
    # -----------------------------------------------------------------------

    # Minimum sinusoidal period for time embedding
    min_period: float = 4e-3

    # Maximum sinusoidal period for time embedding
    max_period: float = 4.0

    # Beta distribution alpha parameter for time step sampling
    time_sampling_beta_alpha: float = 1.5

    # Beta distribution beta parameter for time step sampling
    time_sampling_beta_beta: float = 1.0

    # -----------------------------------------------------------------------
    # Pipeline-parallel communication tensor dimensions
    #
    # Megatron PP pre-allocates: (seq_length, micro_batch_size, hidden_size)
    # For PI0 we pack both VLM and Expert hidden states into this tensor:
    #
    #   hidden_size = 2048 (VLM hidden, larger of the two branches)
    #
    #   seq_length = pp_max_prefix + pp_max_suffix + overhead_rows
    #     pp_max_prefix: max VLM prefix tokens per sample
    #       = max_images * num_image_tokens + tokenizer_max_length
    #     pp_max_suffix: max Expert suffix tokens per sample
    #       = chunk_size + 1 (state token for PI0) or chunk_size (PI0.5)
    #     overhead_rows: adaRMS vec (1) + metadata (1) + pos_ids (5) + targets (N)
    #
    #   micro_batch_size = pp_max_batch_size (= actual_mbs + 1 for TP padding)
    #
    # Layout of (S, max_B, 2048) per batch slot b:
    #   rows [0 : prefix_lens[b]]         VLM hidden states (D=2048)
    #   rows [max_prefix : +suffix_lens[b]] Expert hidden states (D=1024, padded to 2048)
    #   row  [max_prefix+max_suffix]       adaRMS conditioning vector (PI0.5) or zeros
    #   row  [max_prefix+max_suffix+1]     metadata
    #   rows [+2 : +7]                     position IDs (packed float32, 5 rows)
    #   rows [+7 : end]                    velocity targets (dense-packed, for loss)
    # -----------------------------------------------------------------------

    # Maximum VLM prefix tokens per sample (image tokens + language tokens)
    pp_max_prefix_per_sample: int = 512

    # Maximum Expert suffix tokens per sample (state + action tokens)
    pp_max_suffix_per_sample: int = 64

    # Maximum batch slots in PP comm tensor = actual_micro_batch_size + 1
    pp_max_batch_size: int = 19

    # -----------------------------------------------------------------------
    # External model path
    # -----------------------------------------------------------------------

    # Path to the pretrained HuggingFace model checkpoint
    # (directory containing model.safetensors, config.json, etc.)
    pretrained_model_path: str = ""

    # -----------------------------------------------------------------------
    # Derived / overridden fields
    # -----------------------------------------------------------------------

    def __post_init__(self):
        # Map VLM architecture onto the inherited TransformerConfig fields:
        #   hidden_size         = 2048  (must be set externally when constructing)
        #   num_layers          = 18    (must be set externally)
        #   num_attention_heads = 8     (must be set externally)
        #   ffn_hidden_size     = 16384 (VLM MLP intermediate)
        #   kv_channels         = head_dim = 256

        if self.ffn_hidden_size == 0 or self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 8  # 2048*8=16384 for VLM

        if self.kv_channels is None:
            self.kv_channels = self.head_dim  # 256

        # adarms_cond_dim defaults to action_expert_hidden_size when not set
        if self.adarms_cond_dim == 0:
            self.adarms_cond_dim = self.action_expert_hidden_size

        super().__post_init__()

    @property
    def pp_seq_length(self) -> int:
        """Total rows in PP communication tensor (= --seq-length Megatron arg)."""
        # overhead: adaRMS vec (1) + metadata (1) + pos_ids (5) + target rows
        # target rows = ceil(max_suffix * max_action_dim / hidden_size)
        target_rows = math.ceil(
            self.pp_max_suffix_per_sample * self.max_action_dim / self.hidden_size
        )
        return self.pp_max_prefix_per_sample + self.pp_max_suffix_per_sample + 7 + target_rows

    @property
    def vlm_num_attention_heads(self) -> int:
        """Number of Q heads in VLM branch (= num_attention_heads)."""
        return self.num_attention_heads

    @property
    def vlm_hidden_size(self) -> int:
        """Hidden size of VLM branch."""
        return self.hidden_size

    @property
    def vlm_ffn_hidden_size(self) -> int:
        """FFN intermediate size of VLM branch."""
        return self.ffn_hidden_size
