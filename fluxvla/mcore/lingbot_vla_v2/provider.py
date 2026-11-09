"""Native MCore model provider for LingBot-VLA-v2."""

from __future__ import annotations

import math
import os
import re
import argparse
from pathlib import Path

import torch

from ..diagnostics import register_alignment_parameters
from ..tensor_parallel import MCoreTopology, validate_mcore_topology
from .model import LingBotVLA2MCoreModel
from .optimizer import register_lingbot_fsdp2_reduction_units
from .tensor_parallel import convert_lingbot_dense_modules, shard_lingbot_experts


def _model_config_overrides(args, *, requested_sp: bool, requested_ep: bool) -> dict:
    """Build the explicit Native/MCore forward-math contract."""
    overrides = {
        "attention_implementation": args.lingbot_attention_implementation,
        "vit_attn_implementation": args.lingbot_vit_attn_implementation,
    }
    if (
        getattr(args, "lingbot_data_mode", "replay") == "online"
        and getattr(args, "lingbot_online_teacher_mode", "disabled") == "disabled"
    ):
        # Teacher inference belongs in a GPU batch processor, never in loader
        # workers. Disable auxiliary heads explicitly until that processor is
        # selected instead of silently omitting required targets.
        overrides["align_params"] = {}
    moe_implementation = getattr(args, "lingbot_moe_implementation", "auto")
    if (requested_sp or requested_ep) and moe_implementation == "fused":
        raise ValueError("LingBot MCore SP/EP requires eager MoE")
    if requested_sp or requested_ep or moe_implementation == "eager":
        overrides["moe_implementation"] = "eager"
    elif moe_implementation == "fused":
        overrides["moe_implementation"] = "fused"
    return overrides


def _adapter_data_parallel_size(*, expert_parallel_size: int) -> int:
    """Return the independent DP factor used by the adapter topology check.

    MCore's dense data-parallel group overlaps the expert-parallel group.  For
    an EP topology, the independent ``DP x TP x EP`` factor is therefore the
    expert-data-parallel size, not the dense data-parallel world size.
    """
    from megatron.core import parallel_state

    if expert_parallel_size > 1:
        return parallel_state.get_expert_data_parallel_world_size()
    return parallel_state.get_data_parallel_world_size()


def _install_alignment_gradient_trace(model: torch.nn.Module) -> None:
    """Capture selected parameter gradients before MCore's DDP reduction."""
    root_value = os.environ.get("FLUXVLA_AUTOGRAD_TENSOR_TRACE_DIR")
    if not root_value:
        return
    pattern = re.compile(
        os.environ.get(
            "FLUXVLA_GRAD_TRACE_PATTERN",
            r"(state_proj|action_out_proj)\.(weight|bias)$",
        )
    )
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    steps: dict[str, int] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not pattern.search(name):
            continue

        def _save(grad, *, parameter_name=name):
            steps[parameter_name] = steps.get(parameter_name, 0) + 1
            step = steps[parameter_name]
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", parameter_name)
            torch.save(
                {
                    "backend": "mcore",
                    "phase": "autograd",
                    "step": step,
                    "rank": rank,
                    "name": parameter_name,
                    "grad": grad.detach().float().cpu().clone(),
                },
                root
                / f"mcore_autograd_step_{step:06d}_rank_{rank:02d}_{safe_name}.pt",
            )
            return grad

        parameter.register_hook(_save)


def _configure_train_scope(model: torch.nn.Module, scope: str) -> None:
    """Apply the same staged-freeze contract used by the Native FSDP oracle."""
    if scope == "all":
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    patterns = {
        "action-head": ("action_out_proj.",),
        "action-expert": ("action_out_proj.", "qwen_expert."),
        "alignment": ("action_out_proj.", "qwen_expert.", "align_head", "shared_task_proj", "video_align", "depth_align"),
        "qwen": ("action_out_proj.", "qwen_expert.", "align_head", "shared_task_proj", "video_align", "depth_align", "qwenvl."),
    }
    try:
        selected = patterns[scope]
    except KeyError as exc:
        raise ValueError(f"Unsupported LingBot train scope: {scope}") from exc
    for name, parameter in model.named_parameters():
        if any(token in name for token in selected):
            parameter.requires_grad_(True)


def model_provider(pre_process: bool = True, post_process: bool = True):
    from megatron.core import parallel_state
    from megatron.training import get_args, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args

    args = get_args()
    if (
        getattr(args, "lingbot_data_mode", "replay") == "online"
        and getattr(args, "lingbot_online_teacher_mode", "disabled") == "required"
    ):
        raise NotImplementedError(
            "LingBot online teacher mode requires the GPU MoRGBD/DINO batch processor; "
            "use --lingbot-online-teacher-mode disabled only for flow-loss training."
        )
    topology = {
        "TP": parallel_state.get_tensor_model_parallel_world_size(),
        "PP": parallel_state.get_pipeline_model_parallel_world_size(),
        "CP": parallel_state.get_context_parallel_world_size(),
        "EP": parallel_state.get_expert_model_parallel_world_size(),
    }
    # LingBot keeps its native batch-first block layout.  SP is enabled only at
    # the MoE MLP boundary, where sequence shards are restored before residuals.
    validate_mcore_topology(
        MCoreTopology(
            tp=topology["TP"],
            pp=topology["PP"],
            cp=topology["CP"],
            ep=topology["EP"],
            sp=bool(getattr(args, "sequence_parallel", False)),
        ),
        allow_tp=True,
        allow_sp=True,
        allow_ep=True,
    )
    if not pre_process or not post_process:
        raise NotImplementedError(
            "LingBot MCore requires a single pre_process/post_process stage; "
            f"received pre_process={pre_process}, post_process={post_process}"
        )

    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    # Native LingBot checkpoints use a fused 3-D expert tensor.  MCore SP/EP
    # needs an ordinary expert module boundary so token ownership is explicit;
    # the adapter maps the fused checkpoint tensors to eager experts while
    # preserving the exact weights.  EP is intentionally replicated here until
    # a dispatcher-backed local-expert checkpoint contract is available.
    requested_sp = bool(getattr(args, "sequence_parallel", False))
    requested_ep = topology["EP"] > 1
    model_config = _model_config_overrides(
        args,
        requested_sp=requested_sp,
        requested_ep=requested_ep,
    )
    adapter = LingBotVLAV2(
        native_config_path=args.lingbot_native_config,
        pretrained_name_or_path=args.lingbot_checkpoint,
        tokenizer_path=args.lingbot_tokenizer_path,
        model_config=model_config,
        action_fp32=args.lingbot_action_fp32,
        low_cpu_mem_usage=True,
    )
    if requested_ep:
        print_rank_0("LingBot MCore EP: enabling local expert sharding with differentiable EP collectives.")
    adapter.from_pretrained()
    native_policy = getattr(adapter.model, "model", adapter.model)
    native_policy.qwenvl_with_expert._fluxvla_action_hidden_fp32 = True
    if requested_ep and os.environ.get("FLUXVLA_EP_REPLICATED", "0") != "1":
        expert_count = shard_lingbot_experts(adapter.model)
        print_rank_0(f"LingBot MCore EP local expert blocks: {expert_count}")
    elif requested_ep:
        print_rank_0("LingBot MCore EP: replicated eager experts diagnostic mode")
    # EP token collectives are part of the autograd graph. Native activation
    # checkpoint replay would execute those collectives a second time in a
    # different order, so disable checkpointing for the correctness path.
    if args.lingbot_gradient_checkpointing and not requested_ep:
        adapter.gradient_checkpointing_enable()
    adapter.initialize_distributed_runtime(
        dp_size=_adapter_data_parallel_size(expert_parallel_size=topology["EP"]),
        tp_size=parallel_state.get_tensor_model_parallel_world_size(),
        ep_size=topology["EP"],
        dp_mode="fsdp1",
    )

    config = core_transformer_config_from_args(args)
    tp_counts = convert_lingbot_dense_modules(
        adapter.model,
        config,
        sequence_parallel=bool(getattr(args, "sequence_parallel", False)),
        convert_attention=bool(getattr(args, "lingbot_tp_attention", True)),
        convert_mlp=bool(getattr(args, "lingbot_tp_mlp", True)),
        convert_routed_experts=bool(
            getattr(args, "lingbot_tp_routed_experts", True)
        ),
        convert_shared_expert=bool(
            getattr(args, "lingbot_tp_shared_expert", True)
        ),
        convert_mlp_column=bool(getattr(args, "lingbot_tp_mlp_column", True)),
        convert_mlp_row=bool(getattr(args, "lingbot_tp_mlp_row", True)),
        parity_reference=bool(
            getattr(args, "lingbot_tp_parity_reference", False)
        ),
    )
    print_rank_0(f"LingBot MCore TP conversion: {tp_counts}")
    if getattr(args, "sequence_parallel", False) and tp_counts.get("sp_skipped_routed_moe", 0):
        print_rank_0(
            "LingBot MCore SP capability: routed MoE boundaries remain native; "
            f"skipped {tp_counts['sp_skipped_routed_moe']} MoE blocks. "
            "This is not full MoE activation-SP."
        )
    model = LingBotVLA2MCoreModel(
        config,
        adapter,
        preserve_fsdp2_fp32_modules=bool(
            args.lingbot_preserve_fsdp2_fp32_modules
        ),
    )
    _configure_train_scope(model, args.lingbot_train_scope)
    if args.lingbot_fsdp2_strict_dp_reduction:
        register_lingbot_fsdp2_reduction_units(adapter)
    _install_alignment_gradient_trace(model)
    register_alignment_parameters(model)
    print_rank_0("Built native FluxVLA LingBot-VLA-v2 MCore model")
    return model


def get_lingbot_args(parser):
    group = parser.add_argument_group("FluxVLA native LingBot-VLA-v2")
    group.add_argument("--lingbot-native-config", type=str, required=True)
    group.add_argument("--lingbot-checkpoint", type=str, required=True)
    group.add_argument("--lingbot-tokenizer-path", type=str, required=True)
    group.add_argument(
        "--lingbot-data-mode",
        choices=("online", "replay"),
        default="online",
        help="Use raw LeRobot data in production; replay is diagnostic only.",
    )
    group.add_argument("--lingbot-data-path", type=str)
    group.add_argument("--lingbot-norm-stats", type=str)
    group.add_argument("--lingbot-manifest", type=str)
    group.add_argument("--lingbot-num-workers", type=int, default=0)
    group.add_argument("--lingbot-action-horizon", type=int, default=50)
    group.add_argument("--lingbot-image-size", type=int, default=256)
    group.add_argument("--lingbot-tokenizer-max-length", type=int, default=72)
    group.add_argument(
        "--lingbot-use-future-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    group.add_argument(
        "--lingbot-online-teacher-mode",
        choices=("disabled", "required"),
        default="disabled",
        help=(
            "Auxiliary teacher contract for online data. 'required' fails until "
            "the GPU teacher batch processor is configured."
        ),
    )
    group.add_argument(
        "--lingbot-train-scope",
        choices=("action-head", "action-expert", "alignment", "qwen", "all"),
        default="all",
        help="Staged-freeze scope for parity experiments; production default is all.",
    )
    group.add_argument("--lingbot-fixed-time", type=float, default=0.5)
    group.add_argument("--lingbot-sampler-seed", type=int, default=42)
    group.add_argument(
        "--lingbot-flow-rng-seed",
        type=int,
        default=42,
        help="Per-sample flow target seed; keep aligned with Native online training (default: 42).",
    )
    group.add_argument(
        "--lingbot-replay-order",
        choices=("manifest", "official_step_rank"),
        default="official_step_rank",
        help="Order captured replay entries exactly as the Native production oracle.",
    )
    group.add_argument(
        "--lingbot-attention-implementation",
        choices=("eager", "flex", "flex_cached"),
        default="eager",
        help="Action-transformer attention backend; production parity uses eager.",
    )
    group.add_argument(
        "--lingbot-vit-attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        # The Native LingBot production parity recipe uses eager vision
        # attention.  FlashAttention is an opt-in performance variant: its
        # reduction/order differs from Native and can move the first loss
        # beyond the parity gate even with identical inputs and weights.
        default="eager",
        help=(
            "Vision attention backend; Native parity defaults to eager. "
            "Use flash_attention_2 only for an explicitly non-parity run."
        ),
    )
    group.add_argument(
        "--lingbot-moe-implementation",
        choices=("auto", "fused", "eager"),
        default="auto",
        help=(
            "MoE operator contract. Auto preserves fused DP and selects eager "
            "for SP/EP; eager is the deterministic cross-backend parity path."
        ),
    )
    group.add_argument(
        "--lingbot-expert-lr-multiplier",
        type=float,
        default=math.sqrt(8.0),
    )
    group.add_argument(
        "--lingbot-fsdp2-parameterwise-grad-norm",
        action="store_true",
        default=True,
        help="Use FSDP2-compatible parameter-wise FP32 grad norm for DP parity.",
    )
    group.add_argument(
        "--no-lingbot-fsdp2-parameterwise-grad-norm",
        action="store_false",
        dest="lingbot_fsdp2_parameterwise_grad_norm",
    )
    group.add_argument(
        "--lingbot-tp-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard Q/K/V/O projections in TP; disable for MLP-only diagnostics.",
    )
    group.add_argument(
        "--lingbot-tp-mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard routed/shared MLP projections in TP; disable for attention-only TP.",
    )
    group.add_argument(
        "--lingbot-tp-routed-experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard routed expert MLPs; disable only for TP parity diagnostics.",
    )
    group.add_argument(
        "--lingbot-tp-shared-expert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard shared expert MLPs; disable only for TP parity diagnostics.",
    )
    group.add_argument(
        "--lingbot-tp-mlp-column",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard MLP gate/up projections; disable only for TP parity diagnostics.",
    )
    group.add_argument(
        "--lingbot-tp-mlp-row",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard MLP down projections; disable only for TP parity diagnostics.",
    )
    group.add_argument(
        "--lingbot-tp-parity-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "All-gather RowParallel weights for strict Native operator parity; "
            "this is a correctness oracle, not the performance path."
        ),
    )
    group.add_argument(
        "--lingbot-action-fp32",
        action="store_true",
        default=False,
        help="Run the action branch in FP32; disabled by the production parity contract.",
    )
    group.add_argument(
        "--no-lingbot-action-fp32",
        action="store_false",
        dest="lingbot_action_fp32",
    )
    group.add_argument(
        "--lingbot-preserve-fsdp2-fp32-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep Native FSDP2 basic-module parameters in FP32 under MCore "
            "BF16 wrapping. Production Native parity requires this independently "
            "of the model action-fp32 input switch."
        ),
    )
    group.add_argument(
        "--lingbot-gradient-checkpointing",
        action="store_true",
        default=True,
    )
    group.add_argument(
        "--no-lingbot-gradient-checkpointing",
        action="store_false",
        dest="lingbot_gradient_checkpointing",
    )
    group.add_argument(
        "--lingbot-low-memory-adam",
        action="store_true",
        default=True,
        help="Chunk native foreach AdamW groups to bound temporary memory.",
    )
    group.add_argument(
        "--no-lingbot-low-memory-adam",
        action="store_false",
        dest="lingbot_low_memory_adam",
    )
    group.add_argument(
        "--lingbot-adam-foreach-chunk-size",
        type=int,
        default=1,
        help="Number of parameters per low-memory AdamW foreach group.",
    )
    group.add_argument(
        "--lingbot-fsdp2-reduction-order",
        action="store_true",
        default=True,
        help="Apply FSDP2-compatible SUM-then-divide DistOpt gradient reduction.",
    )
    group.add_argument(
        "--no-lingbot-fsdp2-reduction-order",
        action="store_false",
        dest="lingbot_fsdp2_reduction_order",
    )
    group.add_argument(
        "--lingbot-fsdp2-strict-dp-reduction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Accuracy-certification path that reduces gradients in Native FSDP2 "
            "unit packing order before mapping them back to MCore DistOpt shards. "
            "TP shards are reconstructed before DP reduction; EP remains unsupported."
        ),
    )
    return parser


__all__ = ["get_lingbot_args", "model_provider"]
