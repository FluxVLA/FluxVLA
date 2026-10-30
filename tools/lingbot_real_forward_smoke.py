"""Run a low-memory LingBot real-batch forward smoke in a detached process."""

from __future__ import annotations

import torch
import os
from pathlib import Path

from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2


ROOT = "/ceph/teamGolden/andyzeng/VLA"
CHECKPOINT = os.environ.get(
    "LINGBOT_SMOKE_CHECKPOINT",
    f"{ROOT}/models/lingbot-vla-v2-6b",
)
TOKENIZER = os.environ.get("LINGBOT_SMOKE_TOKENIZER", f"{ROOT}/models/Qwen3-VL-4B-Instruct")


def main() -> None:
    batch = torch.load(
        os.environ.get("LINGBOT_SMOKE_BATCH", "/tmp/lingbot_real_batch.pt"),
        map_location="cpu",
        weights_only=False,
    )
    # The native model accepts [num_views, patches, width]. Reduce the raw
    # real-image patches to the 224-level Qwen grid before the smoke forward.
    if batch["images"].ndim == 3 and os.environ.get("LINGBOT_SMOKE_KEEP_SQUARE") != "1":
        batch["images"] = torch.nn.functional.interpolate(
            batch["images"].float().unsqueeze(0), size=(192, 1536), mode="bilinear", align_corners=False
        )
        batch["image_grid_thw"] = torch.tensor([[[1, 12, 16]] * 3], dtype=torch.long)
    # Captured production replays already contain the exact masks and flow
    # inputs used by the reference run.  Do not overwrite them with smoke
    # defaults: doing so silently changes the loss while the model weights
    # remain identical.  Defaults are only for minimal synthetic batches.
    batch.setdefault("img_masks", torch.ones((1, 3), dtype=torch.bool))
    batch.setdefault("joint_mask", torch.ones((1, 50, 55), dtype=torch.bool))
    batch.setdefault("action_is_pad", torch.zeros((1, 50), dtype=torch.bool))
    batch.setdefault("noise", torch.zeros_like(batch["actions"]))
    batch.setdefault("time", torch.full((1,), 0.5))
    flow_path = os.environ.get("LINGBOT_SMOKE_FLOW")
    if flow_path:
        flow = torch.load(flow_path, map_location="cpu", weights_only=False)
        batch["noise"] = torch.as_tensor(flow["noise"])
        batch["time"] = torch.as_tensor(flow["time"])

    model_config = {}
    if os.environ.get("LINGBOT_SMOKE_EAGER") == "1":
        model_config = {"attention_implementation": "eager", "vit_attn_implementation": "eager"}
    # Match the native parity smoke: action/VLA loss only, without optional
    # post-training depth/video alignment heads.
    post_training = os.environ.get("LINGBOT_SMOKE_POST_TRAINING", "0") == "1"
    # Keep the production precision boundary from the native YAML by default;
    # smoke callers can explicitly set LINGBOT_SMOKE_ENABLE_FP32=0 when they
    # need the reduced-precision diagnostic path.
    model_config.update({
        "post_training": post_training,
        "enable_fp32": os.environ.get("LINGBOT_SMOKE_ENABLE_FP32", "1") != "0",
    })
    if not post_training:
        model_config["align_params"] = {}
    if os.environ.get("LINGBOT_SMOKE_DISABLE_DROPOUT") == "1":
        model_config["attention_dropout"] = 0.0
        model_config["hidden_dropout"] = 0.0
    moe_implementation = os.environ.get("LINGBOT_SMOKE_MOE_IMPLEMENTATION")
    if moe_implementation:
        model_config["moe_implementation"] = moe_implementation
    model = LingBotVLAV2(
        source_root=f"{ROOT}/lingbot-vla-v2",
        tokenizer_path=TOKENIZER,
        pretrained_name_or_path=CHECKPOINT,
        native_config_path=os.environ.get(
            "LINGBOT_NATIVE_CONFIG", f"{ROOT}/lingbot-vla-v2/golden_artifacts/robotwin_reduced.yaml"
        ),
        model_config=model_config,
    )
    model.from_pretrained()
    trace_dir = os.environ.get("LINGBOT_REAL_TRACE_DIR")
    trace_values = {}
    trace_handles = []
    if trace_dir:
        selected = {
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.q_proj",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.k_proj",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.v_proj",
            "model.qwenvl_with_expert.qwenvl.model.language_model.layers.0",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.q_proj",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.k_proj",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.v_proj",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.o_proj",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.post_attention_layernorm",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp",
            "model.qwenvl_with_expert.qwen_expert.model.layers.0",
        }

        def first_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (tuple, list)):
                return next((item for item in value if isinstance(item, torch.Tensor)), None)
            return None

        root = model.model.model
        for name, module in root.named_modules():
            if name not in selected and f"model.{name}" not in selected:
                continue
            trace_name = name if name.startswith("model.") else f"model.{name}"

            def save(_module, inputs, output, *, module_name=trace_name):
                trace_values[f"{module_name}.input"] = [
                    item.detach().float().cpu() for item in inputs if isinstance(item, torch.Tensor)
                ]
                runtime = {
                    "class": f"{type(_module).__module__}.{type(_module).__qualname__}",
                    "dtype": str(next(_module.parameters(), torch.empty(0)).dtype),
                }
                for parameter_name, parameter in _module.named_parameters(recurse=False):
                    trace_values[f"{module_name}.runtime_parameter.{parameter_name}"] = (
                        parameter.detach().float().cpu()
                    )
                trace_values[f"{module_name}.runtime"] = runtime
                tensor = first_tensor(output)
                if tensor is not None:
                    trace_values[f"{module_name}.output"] = tensor.detach().float().cpu()

            trace_handles.append(module.register_forward_hook(save))
    train_step = os.environ.get("LINGBOT_TRAIN_STEP", "0") == "1"
    model_train_mode = train_step or os.environ.get("LINGBOT_SMOKE_MODEL_TRAIN", "0") == "1"
    model.to(device="cuda", dtype=torch.bfloat16)
    if os.environ.get("LINGBOT_SMOKE_PRINT_BUFFERS") == "1":
        print(
            "ROTARY_BUFFERS",
            {
                name: str(buffer.dtype)
                for name, buffer in model.named_buffers()
                if "rotary" in name or "inv_freq" in name
            },
            flush=True,
        )
    if model_train_mode:
        model.train()
        model.model.gradient_checkpointing_enable()
    else:
        model.eval()
    if train_step:
        model.train()
    for key in ("images", "state", "actions", "noise", "time"):
        batch[key] = batch[key].cuda().bfloat16()
    for key in ("depth_targets", "future_depth_targets", "future_video_targets", "future_video_current_patch"):
        if key in batch:
            batch[key] = batch[key].cuda().bfloat16()
    for key in ("image_grid_thw", "img_masks", "joint_mask", "action_is_pad", "lang_tokens", "lang_masks"):
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
            parameter
            for name, parameter in model.model.model.named_parameters()
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
            output["loss"].backward()
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
                state_norms.extend(
                    value.detach().float().norm().item()
                    for key, value in state.items()
                    if key in ("exp_avg", "exp_avg_sq")
                )
            print(
                "STEP", step + 1,
                "LOSS", float(output["loss"]),
                "LOSS_VLA", float(output["loss_vla"]),
                "GRAD_NORM", float(grad_norm),
                "CLIP_COEF", clip_coef,
                "PARAM_DELTA", param_delta,
                "ADAM_STATE_NORM", sum(x * x for x in state_norms) ** 0.5,
                flush=True,
            )
    else:
        with torch.no_grad():
            for step in range(steps):
                with torch.profiler.record_function("LingBot/forward") if profiler else torch.autograd.profiler.record_function("LingBot/forward"):
                    output = model(**batch)
                if profiler:
                    profiler.step()
            print("STEP", step + 1, "LOSS", float(output["loss"]), "LOSS_VLA", float(output["loss_vla"]), flush=True)
    if profiler:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(str(Path(profile_dir) / "flux_forward_trace.json"))
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40), flush=True)
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        torch.save(trace_values, Path(trace_dir) / "native_flux.pt")
        for handle in trace_handles:
            handle.remove()


if __name__ == "__main__":
    main()
