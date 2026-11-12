"""Replay native LingBot micro-batches through FluxVLA and print loss traces."""

from pathlib import Path
import os
import random
import torch
import torch.distributed as dist

from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

ROOT = Path("/ceph/teamGolden/andyzeng/VLA")
CHECKPOINT = Path(
    __import__("os").environ.get(
        "LINGBOT_REPLAY_CHECKPOINT",
        str(ROOT / "models/lingbot-vla-v2-6b"),
    )
)
TOKENIZER = Path(
    __import__("os").environ.get(
        "LINGBOT_REPLAY_TOKENIZER", str(ROOT / "models/Qwen3-VL-4B-Instruct")
    )
)
REPLAY = Path(
    os.environ.get(
        "LINGBOT_REPLAY_DIR",
        str(ROOT / "FluxVLA/work_dirs/native_micro_replay_20261013"),
    )
)


def main() -> None:
    seed = int(os.environ.get("LINGBOT_REPLAY_SEED", "42"))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    from lingbotvla.distributed.parallel_state import init_parallel_state

    init_parallel_state(dp_size=dist.get_world_size(), dp_mode="fsdp2")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    model_config = {}
    if os.environ.get("LINGBOT_REPLAY_MOE_IMPL"):
        model_config["moe_implementation"] = os.environ["LINGBOT_REPLAY_MOE_IMPL"]
    if os.environ.get("LINGBOT_REPLAY_ATTN_IMPL"):
        model_config["attention_implementation"] = os.environ["LINGBOT_REPLAY_ATTN_IMPL"]
    if os.environ.get("LINGBOT_REPLAY_ALIGN_PARAMS") == "none":
        model_config["align_params"] = {}
    if os.environ.get("LINGBOT_REPLAY_ACTION_FP32") == "1":
        model_config["action_fp32"] = True
    if os.environ.get("LINGBOT_REPLAY_POST_TRAINING") == "0":
        model_config["post_training"] = False
    model = LingBotVLAV2(
        tokenizer_path=str(TOKENIZER),
        pretrained_name_or_path=str(CHECKPOINT),
        native_config_path=str(ROOT / "FluxVLA/work_dirs/robotwin_reduced_v2.yaml"),
        model_config=model_config,
    )
    # The adapter's checkpoint loader casts incoming shards to the model's
    # current parameter dtype.  Set that dtype before loading so FP32 parity
    # replays do not silently quantize basic/action projections to BF16.
    if os.environ.get("LINGBOT_REPLAY_PARAM_DTYPE") == "fp32":
        model.model.to(dtype=torch.float32)
    model.from_pretrained()
    if os.environ.get("LINGBOT_REPLAY_DEBUG_CONFIG") == "1" and dist.get_rank() == 0:
        config = model.model.config
        print(
            "FLUX_CONFIG",
            {name: getattr(config, name, None) for name in (
                "post_training", "action_fp32", "loss_type", "chunk_size",
                "max_action_dim", "max_state_dim", "tokenizer_max_length",
                "sequence_wise_loss_coeff", "router_z_loss_coeff",
                "routed_scaling_factor", "action_dim",
            )},
            flush=True,
        )
    if os.environ.get("LINGBOT_REPLAY_ACTION_ATTN_IMPL"):
        impl = os.environ["LINGBOT_REPLAY_ACTION_ATTN_IMPL"]
        for layer in model.model.model.qwenvl_with_expert.qwen_expert.model.layers:
            layer.self_attn.config._attn_implementation = impl
        print("FLUX_ACTION_ATTN_IMPL", impl, flush=True)
    print("FLUX_ACTION_FP32", getattr(model.model.config, "action_fp32", None), flush=True)
    # Native training evaluates the policy in train mode even during the
    # forward portion of gradient accumulation; preserve that contract so
    # router/dropout and normalization behavior are identical.
    input_dtype = torch.float32 if os.environ.get("LINGBOT_REPLAY_DTYPE") == "fp32" else torch.bfloat16
    parameter_dtype = (
        torch.float32
        if os.environ.get("LINGBOT_REPLAY_PARAM_DTYPE", "") == "fp32"
        else input_dtype
    )
    if os.environ.get("LINGBOT_REPLAY_ENABLE_FP32") == "1":
        # Preserve native basic-module FP32 master parameters loaded by the
        # adapter.  A blanket ``model.to(dtype=bf16)`` would quantize them
        # before the per-module FP32 policy below is applied.
        model.to(device=f"cuda:{local_rank}").train()
    else:
        model.to(device=f"cuda:{local_rank}", dtype=parameter_dtype).train()
    if os.environ.get("LINGBOT_REPLAY_ACTION_ONLY_FP32") == "1":
        action_names = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")
        for name, module in model.named_modules():
            if any(name.endswith(token) for token in action_names):
                module.to(device=f"cuda:{local_rank}", dtype=torch.float32)
    if os.environ.get("LINGBOT_REPLAY_ENABLE_FP32") == "1":
        def cast_fp32(value):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                return value.float()
            if isinstance(value, tuple):
                return tuple(cast_fp32(item) for item in value)
            if isinstance(value, list):
                return [cast_fp32(item) for item in value]
            if isinstance(value, dict):
                return {key: cast_fp32(item) for key, item in value.items()}
            return value

        def cast_module_inputs(_module, args, kwargs):
            return cast_fp32(args), cast_fp32(kwargs)

        for module in model.get_fsdp2_full_precision_modules():
            module.to(device=f"cuda:{local_rank}", dtype=torch.float32)
            module.register_forward_pre_hook(cast_module_inputs, with_kwargs=True)
    rows = []
    # Native training evaluates each micro-batch with autograd enabled before
    # calling backward.  Keep grad mode enabled here so kernels selected by
    # ``torch.is_grad_enabled()`` follow the same path during parity replay.
    with torch.enable_grad():
        paths = sorted(REPLAY.glob("batch_rank0_step*.pt"), key=lambda p: int(p.stem.split("step")[-1]))
        limit = int(os.environ.get("LINGBOT_REPLAY_LIMIT", "0"))
        for path in paths[:limit or None]:
            batch = torch.load(path, map_location=f"cuda:{local_rank}", weights_only=False)
            batch = {k: v.to(dtype=input_dtype) if v.is_floating_point() else v for k, v in batch.items()}
            batch["noise"] = torch.zeros_like(batch["actions"])
            batch["time"] = torch.full((batch["actions"].shape[0],), 0.5, device=batch["actions"].device, dtype=batch["actions"].dtype)
            output = model(**batch)
            row = {k: float(output[k]) for k in ("loss", "loss_vla", "loss_depth", "loss_future_depth", "loss_future_video") if k in output}
            rows.append(row)
            if dist.get_rank() == 0:
                print(path.name, row, flush=True)
    if dist.get_rank() == 0:
        torch.save(rows, REPLAY / "flux_rank0_loss_trace.pt")
    dist.barrier()


if __name__ == "__main__":
    main()
