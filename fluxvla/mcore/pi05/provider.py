"""Megatron-Core-native PI0.5 model provider.

Unlike the compatibility provider, this creates a ``MegatronModule`` made of
MCore TP layers before training.  The HF checkpoint is only an initialization
source; it is not the runtime model or optimizer contract.
"""

from __future__ import annotations

import os

import torch

from ..diagnostics import register_alignment_parameters
from .alignment import (
    FSDP_GRAD_NORM_ZERO_KEYS,
    FSDP_GRAD_REDUCE_ZERO_SEGMENTS,
    fsdp_grad_norm_order,
)
from .model import PI0Model
from .transformer_config import PI0TransformerConfig


def _install_alignment_gradient_trace(model) -> None:
    """Install opt-in native gradient summaries before Megatron wraps the model."""
    import json
    import os
    import re
    from pathlib import Path

    tensor_root_value = os.environ.get("FLUXVLA_AUTOGRAD_TENSOR_TRACE_DIR")
    if os.environ.get("FLUXVLA_GRAD_TRACE") != "1" and not tensor_root_value:
        return
    pattern = re.compile(
        os.environ.get(
            "FLUXVLA_GRAD_TRACE_PATTERN",
            r"(q_proj|k_proj|v_proj|o_proj|vlm_q|vlm_k|vlm_v|vlm_o|"
            r"exp_q|exp_k|exp_v|exp_o|gate_proj|up_proj|down_proj|"
            r"action_out_proj|action_in_proj|time_mlp|cond_proj|final_norm|norm)"
            r"(\.layer)?\.(weight|bias|dense\.weight|dense\.bias)$",
        )
    )
    root = Path(os.environ.get("FLUXVLA_GRAD_TRACE_DIR", "work_dirs/grad_trace"))
    rank = int(os.environ.get("RANK", "0"))
    path = root / f"native_rank{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_root = Path(tensor_root_value) if tensor_root_value else None
    if tensor_root is not None:
        tensor_root.mkdir(parents=True, exist_ok=True)
    selected = [
        name for name, parameter in model.named_parameters() if pattern.search(name)
    ]
    state = {"backward": 0, "seen": set(), "total": len(selected)}
    tensor_steps: dict[str, int] = {}

    for name, parameter in model.named_parameters():
        if not pattern.search(name):
            continue

        def _hook(grad, _name=name):
            # A parameter hook runs once per backward.  The first hook starts a
            # new step; subsequent hooks append to the same JSONL artifact.
            # Autograd invokes a registered parameter hook once per backward.
            # The first repeated parameter marks the next backward even when
            # frozen parameters make the selected hook count smaller than the
            # number of registered hooks.
            if _name in state["seen"]:
                state["backward"] += 1
                state["seen"].clear()
            elif not state["seen"]:
                state["backward"] += 1
            state["seen"].add(_name)
            value = grad.detach().float()
            if tensor_root is not None:
                tensor_steps[_name] = tensor_steps.get(_name, 0) + 1
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", _name)
                torch.save(
                    {
                        "backend": "mcore",
                        "phase": "autograd",
                        "step": tensor_steps[_name],
                        "rank": rank,
                        "name": _name,
                        "grad": value.cpu().clone(),
                    },
                    tensor_root
                    / f"mcore_autograd_step_{tensor_steps[_name]:06d}_rank_{rank:02d}_{safe_name}.pt",
                )
            if os.environ.get("FLUXVLA_GRAD_TRACE") != "1":
                return grad
            row = {
                "step": state["backward"],
                "name": _name,
                "shape": list(value.shape),
                "grad_norm": float(value.norm().item()),
                "grad_mean": float(value.mean().item()),
                "grad_abs_max": float(value.abs().max().item()),
                "grad_sum": float(value.sum().item()),
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            # Megatron's zero_grad starts the next backward with no hooks; use
            # the last observed parameter as the boundary marker.
            return grad

        parameter.register_hook(_hook)


def model_provider(pre_process: bool = True, post_process: bool = True) -> PI0Model:
    from megatron.training import get_args, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args

    args = get_args()
    if getattr(args, "pi0_allow_legacy_v21", False):
        os.environ["PI05_ALLOW_LEGACY_V21"] = "1"
    if getattr(args, "pi0_fluxvla_parquet", False):
        os.environ["PI05_FLUXVLA_PARQUET"] = "1"
    config = core_transformer_config_from_args(args, PI0TransformerConfig)
    if os.environ.get("PI05_PRESERVE_FP32_PARAMS") == "1":
        # Match FluxVLA FSDP's FP32 parameter-storage contract. BF16 remains
        # the compute dtype through the model's explicit autocast regions.
        config.params_dtype = torch.float32
    config.use_adarms = True
    config.action_expert_hidden_size = getattr(args, "pi0_expert_hidden_size", 1024)
    config.action_expert_ffn_hidden_size = getattr(
        args, "pi0_expert_ffn_hidden_size", 4096
    )
    config.action_expert_num_attention_heads = getattr(args, "pi0_expert_heads", 8)
    config.max_action_dim = getattr(args, "pi0_action_dim", 32)
    config.output_action_dim = getattr(args, "pi0_output_action_dim", None)
    config.max_state_dim = getattr(args, "pi0_state_dim", 32)
    config.chunk_size = getattr(args, "pi0_chunk_size", 50)
    config.num_image_tokens = getattr(args, "pi0_num_image_tokens", 256)
    config.tokenizer_max_length = getattr(args, "pi0_tokenizer_max_length", 200)
    config.pp_max_prefix_per_sample = getattr(args, "pi0_pp_max_prefix", 512)
    config.pp_max_suffix_per_sample = getattr(args, "pi0_pp_max_suffix", 64)
    config.pp_max_batch_size = (
        getattr(args, "pi0_pp_max_batch_size", None) or args.micro_batch_size
    )
    config.pretrained_model_path = getattr(args, "pi0_pretrained_path", "") or ""
    config.adarms_cond_dim = config.action_expert_hidden_size
    config.attention_implementation = getattr(
        args, "pi0_attention_implementation", "sdpa"
    )
    if (
        os.environ.get("PI05_MCORE_TP_ACCURACY_CONTRACT") == "1"
        and config.attention_implementation != "eager"
    ):
        raise ValueError(
            "PI05_MCORE_TP_ACCURACY_CONTRACT requires "
            "--pi0-attention-implementation eager to match the FSDP golden path"
        )
    config.freeze_vision_encoder = bool(
        getattr(args, "pi0_freeze_vision_encoder", False)
    )
    config.freeze_prefix_embedder = bool(
        getattr(args, "pi0_freeze_prefix_embedder", False)
    )
    config.__post_init__()
    model = PI0Model(config, pre_process=pre_process, post_process=post_process)
    _install_alignment_gradient_trace(model)
    register_alignment_parameters(
        model,
        grad_norm_order=fsdp_grad_norm_order,
        grad_norm_zero_keys=FSDP_GRAD_NORM_ZERO_KEYS,
        grad_reduce_zero_segments=FSDP_GRAD_REDUCE_ZERO_SEGMENTS,
    )
    init_ckpt = getattr(args, "pi0_init_ckpt", None)
    if init_ckpt:
        from .checkpoint import load_pi05_checkpoint

        report = load_pi05_checkpoint(model, init_ckpt)
        print_rank_0(
            "Native PI0.5 checkpoint init: "
            f"loaded={report['loaded']} skipped={report['skipped']} "
            f"source_tensors={report['source_tensors']}"
        )
        if report["skipped_keys"]:
            print_rank_0(
                "Native PI0.5 checkpoint skipped keys: "
                + ", ".join(report["skipped_keys"])
            )
    print_rank_0("Built native FluxVLA PI0.5 MCore model")
    return model


def get_pi05_args(parser):
    group = parser.add_argument_group("FluxVLA native PI0.5")
    specs = {
        "--pi0-pretrained-path": (str, "", None),
        "--pi0-init-ckpt": (str, None, None),
        "--pi0-dataset-repo-id": (str, None, None),
        "--pi0-dataset-root": (str, None, None),
        "--pi0-tokenizer-path": (str, None, None),
        "--pi0-action-dim": (int, 32, None),
        "--pi0-output-action-dim": (int, None, None),
        "--pi0-state-dim": (int, 32, None),
        "--pi0-chunk-size": (int, 50, None),
        "--pi0-num-image-tokens": (int, 256, None),
        "--pi0-tokenizer-max-length": (int, 200, None),
        "--pi0-attention-implementation": (str, "sdpa", ("sdpa", "eager")),
        "--pi0-pp-max-prefix": (int, 512, None),
        "--pi0-pp-max-suffix": (int, 64, None),
        "--pi0-pp-max-batch-size": (int, None, None),
        "--pi0-expert-hidden-size": (int, 1024, None),
        "--pi0-expert-ffn-hidden-size": (int, 4096, None),
        "--pi0-expert-heads": (int, 8, None),
        "--pi0-fixed-batch-dir": (str, None, None),
        "--pi0-fixed-batch-start-step": (int, None, None),
        "--pi0-image-keys": (str, None, None),
        "--pi0-max-delay-steps": (int, 0, None),
    }
    for flag, (kind, default, choices) in specs.items():
        kwargs = {"type": kind, "default": default}
        if choices:
            kwargs["choices"] = choices
        group.add_argument(flag, **kwargs)
    group.add_argument("--pi0-freeze-vision-encoder", action="store_true")
    group.add_argument("--pi0-freeze-prefix-embedder", action="store_true")
    group.add_argument("--pi0-shared-observation", action="store_true")
    group.add_argument("--pi0-reference-dataloader", action="store_true")
    group.add_argument("--pi0-fixed-batch-repeat", action="store_true")
    group.add_argument("--pi0-allow-legacy-v21", action="store_true")
    group.add_argument("--pi0-fluxvla-parquet", action="store_true")
    group.add_argument(
        "--pi0-lr-scheduler-contract",
        choices=("fluxvla", "megatron"),
        default="fluxvla",
        help=(
            "Use FluxVLA's zero-LR first optimizer update or Megatron's native "
            "step-zero LR initialization."
        ),
    )
    return parser
