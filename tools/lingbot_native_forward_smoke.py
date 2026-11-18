"""Pure native LingBot forward for parity tracing, without teachers or optimizer."""

from __future__ import annotations

import os
import torch

from lingbotvla.models import build_foundation_model
from lingbotvla.models.config_registry import get_config_registry


ROOT = "/ceph/teamGolden/andyzeng/VLA"
CHECKPOINT = os.environ.get(
    "LINGBOT_SMOKE_CHECKPOINT",
    "/ceph/teamGolden/andyzeng/VLA/models/lingbot-vla-v2-6b",
)
TOKENIZER = os.environ.get(
    "LINGBOT_SMOKE_TOKENIZER",
    "/ceph/teamGolden/andyzeng/VLA/models/Qwen3-VL-4B-Instruct",
)


def main() -> None:
    batch = torch.load(
        os.environ.get("LINGBOT_SMOKE_BATCH", "/tmp/lingbot_real_batch.pt"),
        map_location="cpu",
        weights_only=False,
    )
    if batch["images"].ndim == 3 and os.environ.get("LINGBOT_SMOKE_KEEP_SQUARE") != "1":
        batch["images"] = torch.nn.functional.interpolate(
            batch["images"].float().unsqueeze(0), size=(192, 1536), mode="bilinear", align_corners=False
        )
        batch["image_grid_thw"] = torch.tensor([[[1, 12, 16]] * 3], dtype=torch.long)
    batch.setdefault("img_masks", torch.ones((1, 3), dtype=torch.bool))
    batch.setdefault("joint_mask", torch.ones((1, 50, 55), dtype=torch.bool))
    batch.setdefault("action_is_pad", torch.zeros((1, 50), dtype=torch.bool))
    flow_path = os.environ.get("LINGBOT_SMOKE_FLOW")
    if flow_path:
        flow = torch.load(flow_path, map_location="cpu", weights_only=False)
        batch["noise"] = torch.as_tensor(flow["noise"])
        batch["time"] = torch.as_tensor(flow["time"])
    else:
        batch["noise"] = torch.zeros_like(batch["actions"])
        batch["time"] = torch.full((1,), 0.5)
    registry = get_config_registry()
    config_cls = registry.get_config_cls_from_config_key("LingbotVLAV2Config")
    config = config_cls(
        model_path=CHECKPOINT,
        tokenizer_path=TOKENIZER,
        config_key="LingbotVLAV2Config",
        # The checkpoint includes optional video-alignment heads.  Pure VLA
        # forward parity does not exercise those heads, so allow the native
        # loader to ignore their keys instead of constructing teacher modules.
        post_training=False,
        adanorm_time=True,
        max_action_dim=55,
        max_state_dim=55,
        chunk_size=50,
        action_dim=55,
        action_fp32=os.environ.get("LINGBOT_SMOKE_ACTION_FP32", "0") == "1",
        vlm_causal=True,
        tokenizer_max_length=72,
        precompute_grid_thw=True,
        loss_type="L1_fm",
        sequence_wise_loss_coeff=1e-3,
        sequence_wise_mode="per_sequence",
        router_z_loss_coeff=1e-4,
        attn_implementation="eager",
        attention_implementation="eager",
        use_moe=True,
        moe_implementation=os.environ.get("LINGBOT_SMOKE_MOE_IMPL", "fused"),
        token_moe_layers=list(range(36)),
        token_num_experts=32,
        token_top_k=4,
        token_moe_intermediate_size=512,
        token_shared_intermediate_size=704,
        router_activation="sigmoid",
        routed_scaling_factor=4.0,
        use_shared_expert_gate=False,
    )
    model = build_foundation_model(
        config_path=CHECKPOINT,
        config_cls=config,
        weights_path=CHECKPOINT,
        torch_dtype="bfloat16",
        init_device="cuda",
        config_kwargs=vars(config),
        moe_implementation=os.environ.get("LINGBOT_SMOKE_MOE_IMPL", "fused"),
    )
    if os.environ.get("LINGBOT_SMOKE_PRINT_MODULES") == "1":
        print(
            "MODULES",
            [name for name, _ in model.named_modules() if "action" in name or "qwen_expert.model.layers.0" in name][:200],
            flush=True,
        )
        print(
            "DTYPES",
            {str(dtype): sum(1 for name, p in model.named_parameters() if ("qwen_expert" in name or "action_" in name) and p.dtype == dtype) for dtype in {p.dtype for p in model.parameters()}},
            flush=True,
        )
    trace_dir = os.environ.get("LINGBOT_SMOKE_TRACE_DIR")
    trace_handles = []
    trace_values = {}
    if trace_dir:
        from pathlib import Path

        selected = {
            "model.state_proj",
            "model.action_in_proj",
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
            "model.action_out_proj",
            "qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.q_proj",
            "qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.k_proj",
            "qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.v_proj",
            "qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.o_proj",
            "qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm",
            "qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm.gamma",
            "qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm.beta",
            "qwenvl_with_expert.qwen_expert.model.layers.0.post_attention_layernorm",
            "qwenvl_with_expert.qwen_expert.model.layers.0.mlp",
            "qwenvl_with_expert.qwen_expert.model.layers.0",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.q_proj",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.k_proj",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.v_proj",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0",
        }
        selected.update(
            f"model.qwenvl_with_expert.qwen_expert.model.layers.{index}"
            for index in range(36)
        )

        def first_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (tuple, list)):
                return next((item for item in value if isinstance(item, torch.Tensor)), None)
            return None

        for name, module in model.named_modules():
            if name not in selected and not any(name.endswith(token) for token in selected):
                continue

            def save(_module, inputs, output, *, module_name=name):
                trace_values[f"{module_name}.input"] = [
                    item.detach().float().cpu()
                    for item in inputs
                    if isinstance(item, torch.Tensor)
                ]
                # Capture the object actually executing this call.  This is
                # intentionally collected in the forward hook rather than at
                # model load time: AdaRMSNorm and MCore wrappers can replace
                # parameter storage or dtype during initialization.
                runtime = {
                    "module_id": id(_module),
                    "class": f"{type(_module).__module__}.{type(_module).__qualname__}",
                    "dtype": str(next(_module.parameters(), torch.empty(0)).dtype),
                }
                for parameter_name, parameter in _module.named_parameters(recurse=False):
                    trace_values[f"{module_name}.runtime_parameter.{parameter_name}"] = (
                        parameter.detach().float().cpu()
                    )
                for attribute in ("variance_epsilon", "eps"):
                    if hasattr(_module, attribute):
                        runtime[attribute] = float(getattr(_module, attribute))
                trace_values[f"{module_name}.runtime"] = runtime
                tensor = first_tensor(output)
                if tensor is not None:
                    trace_values[f"{module_name}.output"] = tensor.detach().float().cpu()

            trace_handles.append(module.register_forward_hook(save))
    train_step = os.environ.get("LINGBOT_TRAIN_STEP", "0") == "1"
    if train_step:
        model.train()
    else:
        model.eval()
    for key in ("images", "state", "actions", "noise", "time"):
        batch[key] = batch[key].cuda()
        if not (
            os.environ.get("LINGBOT_SMOKE_ACTION_FP32") == "1"
            and key in ("state", "actions", "noise", "time")
        ):
            batch[key] = batch[key].bfloat16()
    for key in ("img_masks", "joint_mask", "action_is_pad", "image_grid_thw"):
        if key in batch and batch[key].ndim > 0 and key != "image_grid_thw":
            if key == "img_masks" and batch[key].ndim == 1:
                batch[key] = batch[key].unsqueeze(0)
            elif key == "joint_mask" and batch[key].ndim == 2:
                batch[key] = batch[key].unsqueeze(0)
            elif key == "action_is_pad" and batch[key].ndim == 1:
                batch[key] = batch[key].unsqueeze(0)
        if key == "image_grid_thw" and batch[key].ndim == 2:
            batch[key] = batch[key].unsqueeze(0)
    for key in ("depth_targets", "future_depth_targets", "future_video_targets", "future_video_current_patch"):
        if key in batch:
            batch[key] = batch[key].cuda().bfloat16()
    for key in ("image_grid_thw", "img_masks", "joint_mask", "action_is_pad", "lang_tokens", "lang_masks"):
        if key in batch:
            batch[key] = batch[key].cuda()
    steps = int(os.environ.get("LINGBOT_SMOKE_STEPS", "1"))
    profile_dir = os.environ.get("LINGBOT_PROFILE_DIR")
    profiler = None
    if profile_dir:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.__enter__()
    if train_step:
        train_names = {"state_proj", "action_in_proj", "action_out_proj"}
        train_params = [
            parameter for name, parameter in model.named_parameters()
            if any(f".{module_name}." in f".{name}." for module_name in train_names)
        ]
        if not train_params:
            raise RuntimeError("No trainable action/state projection parameters found")
        print("TRAIN_PARAM_COUNT", len(train_params), flush=True)
        optimizer = torch.optim.AdamW(
            train_params,
            lr=float(os.environ.get("LINGBOT_SMOKE_LR", "1e-6")),
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("LingBot/forward") if profiler else torch.autograd.profiler.record_function("LingBot/forward"):
                output = model(**batch)
            if profiler:
                profiler.step()
            values = [float(output[0]), float(output[1])] if not isinstance(output, dict) else [float(output["loss"]), float(output.get("loss_vla", output["loss"]))]
            output[0].backward() if not isinstance(output, dict) else output["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            clip_coef = min(1.0, 1.0 / (float(grad_norm) + 1e-6))
            before = [parameter.detach().float().clone() for parameter in train_params]
            optimizer.step()
            param_delta = torch.sqrt(
                sum((parameter.detach().float() - previous).pow(2).sum() for parameter, previous in zip(train_params, before))
            ).item()
            state_norms = []
            for parameter in train_params:
                state = optimizer.state[parameter]
                state_norms.extend(value.detach().float().norm().item() for key, value in state.items() if key in ("exp_avg", "exp_avg_sq"))
            print("STEP", step + 1, "LOSS", values[0], "LOSS_VLA", values[1], "GRAD_NORM", float(grad_norm), "CLIP_COEF", clip_coef, "PARAM_DELTA", param_delta, "ADAM_STATE_NORM", sum(x * x for x in state_norms) ** 0.5, flush=True)
    else:
        with torch.no_grad():
            for step in range(steps):
                with torch.profiler.record_function("LingBot/forward") if profiler else torch.autograd.profiler.record_function("LingBot/forward"):
                    output = model(**batch)
                if profiler:
                    profiler.step()
                values = [float(output[0]), float(output[1])] if not isinstance(output, dict) else [float(output["loss"]), float(output.get("loss_vla", output["loss"]))]
                print("STEP", step + 1, "LOSS", values[0], "LOSS_VLA", values[1], flush=True)
    if profiler:
        from pathlib import Path

        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(str(Path(profile_dir) / "official_forward_trace.json"))
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40), flush=True)
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        torch.save(trace_values, Path(trace_dir) / "native.pt")
        for handle in trace_handles:
            handle.remove()


if __name__ == "__main__":
    main()
