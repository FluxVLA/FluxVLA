from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]


def test_native_entry_uses_megatron_pretrain_contract():
    source = (ROOT / "scripts/train_mcore_pi05.py").read_text()
    assert "from fluxvla.mcore.launcher import run_mcore" in source
    assert "PI05_SPEC" in source
    assert "DDPTrainRunner" not in source


def test_native_model_is_megatron_module_with_tp_layers():
    model = (ROOT / "fluxvla/mcore/pi05/model.py").read_text()
    layers = (ROOT / "fluxvla/mcore/pi05/layers.py").read_text()
    assert "class PI0Model(MegatronModule)" in model
    assert "ColumnParallelLinear" in layers
    assert "RowParallelLinear" in layers
    assert "PI0ModelLayerTP" in model
    assert 'os.environ.get("PI05_PRESERVE_FP32_PARAMS")' in model
    assert "def sharded_state_dict(" in model
    assert 'self.state_dict(prefix="", keep_vars=True)' in model
    assert 'getattr(parameter, "tensor_model_parallel", False)' in model
    assert "make_sharded_tensors_for_checkpoint" in model


def test_native_provider_does_not_use_compatibility_conversion():
    provider = (ROOT / "fluxvla/mcore/pi05/provider.py").read_text()
    assert "from .checkpoint import load_pi05_checkpoint" in provider
    assert "PI0Model(config" in provider
    assert "load_pi05_checkpoint" in provider


def test_native_checkpoint_maps_pi05_action_and_adarms_tensors():
    from fluxvla.mcore.pi05.checkpoint import (
        _match_source_parameter_load_dtype,
        _source_key,
    )

    assert (
        _source_key("suffix_embedder.action_in_proj.weight") == "action_in_proj.weight"
    )
    assert _source_key("suffix_embedder.time_mlp_in.bias") == "time_mlp_in.bias"
    assert _source_key("layers.0.exp_input_norm.cond_proj.weight") == (
        "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight"
    )
    assert _source_key("layers.17.exp_post_attn_norm.cond_proj.bias") == (
        "paligemma_with_expert.gemma_expert.model.layers.17.post_attention_layernorm.dense.bias"
    )
    assert _source_key("vlm_final_norm.norm_weight") == (
        "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    )
    source = torch.tensor([1.003, 2.007], dtype=torch.float32)
    quantized = _match_source_parameter_load_dtype(
        "layers.0.vlm_input_norm.norm_weight", source
    )
    assert quantized.dtype == source.dtype
    assert torch.equal(quantized, source.to(torch.bfloat16).float())
    projector = _match_source_parameter_load_dtype("action_out_proj.weight", source)
    assert torch.equal(projector, source)


def test_native_time_embedding_matches_fluxvla_fp32_contract():
    from fluxvla.engines.utils.model_utils import (
        create_sinusoidal_pos_embedding as reference_embedding,
    )
    from fluxvla.mcore.pi05.embedders import (
        create_sinusoidal_pos_embedding as native_embedding,
    )

    time = torch.tensor([0.1234567, 0.9876543], dtype=torch.float32)
    reference = reference_embedding(time, 1024, 4e-3, 4.0, device=time.device)
    native = native_embedding(time, 1024, 4e-3, 4.0, device=time.device)

    assert native.dtype == torch.float32
    assert torch.equal(native, reference)


def test_vision_backward_trace_has_matching_fsdp_and_native_boundaries():
    native = (ROOT / "fluxvla/mcore/pi05/embedders.py").read_text()
    fsdp = (ROOT / "fluxvla/models/vlas/pi0_flowmatching.py").read_text()

    for boundary in ("tower_output", "projector_output"):
        native_name = (
            f"camera_{{image_index:02d}}_{boundary}"
            if boundary == "tower_output"
            else '"vision_projector_output"'
        )
        assert native_name in native
        assert f'"vision_{boundary}"' in fsdp

    assert "combined_vision_features = torch.cat(vision_features_list, dim=1)" in native
    assert "multi_modal_projector(\n                combined_vision_features" in native
    assert "vision_autocast = ExitStack()" in native
    assert native.index("vision_autocast.enter_context") < native.index(
        "for image_index, (img, img_mask)"
    )
    assert native.index("vision_autocast.close()") > native.index(
        "multi_modal_projector(\n                combined_vision_features"
    )


def test_native_has_opt_in_fsdp_grad_norm_contract():
    diagnostics = (ROOT / "fluxvla/mcore/diagnostics.py").read_text()
    recipe = (ROOT / "fluxvla/mcore/pi05/recipe.py").read_text()

    assert 'os.environ.get("PI05_FSDP_GRAD_NORM_CONTRACT") != "1"' in diagnostics
    assert "torch.linalg.vector_norm(" in diagnostics
    assert "torch._foreach_mul_(gradients, coefficient)" in diagnostics
    assert "torch.as_tensor(total_norm, dtype=torch.float32" in diagnostics
    assert "_REGISTERED_MODEL_PARAMETERS: list[torch.nn.Parameter]" in diagnostics
    assert 'getattr(parameter, "main_grad", parameter.grad)' in diagnostics
    assert "install_fsdp_grad_norm_contract" in recipe


def test_native_fsdp_reduce_contract_uses_bf16_bucket_collectives():
    diagnostics = (ROOT / "fluxvla/mcore/diagnostics.py").read_text()

    assert "_install_fsdp_bf16_grad_communication()" in diagnostics
    assert "DistributedDataParallel.finish_grad_sync" in diagnostics
    assert "_ordered_grad_reduce_entries" in diagnostics
    assert "_REGISTERED_GRAD_REDUCE_ENTRIES" in diagnostics
    assert "grad_reduce_zero_segments" in diagnostics
    assert "compact = torch.zeros(" in diagnostics
    assert "_fsdp_gradient_divide_factors(world_size)" in diagnostics
    assert "compact.div_(predivide)" in diagnostics
    assert "compact.div_(postdivide)" in diagnostics
    assert "torch.distributed.all_reduce(" in diagnostics
    assert 'os.environ.get("PI05_FSDP_MICROBATCH_GRAD_CONTRACT") == "1"' in diagnostics
    assert "_install_fsdp_microbatch_grad_roundtrip()" in diagnostics
    assert "num_microbatches != 2" in diagnostics
    assert "original = schedules.backward_step" in diagnostics
    assert "schedules.backward_step = fsdp_microbatch_backward_step" in diagnostics
    assert "_apply_bf16_grad_roundtrip(_REGISTERED_MODEL_PARAMETERS)" in diagnostics
    assert 'os.environ.get("FLUXVLA_MICROBATCH_GRAD_TRACE")' in diagnostics


def test_native_pi05_sequence_parallel_restores_batch_first_layout():
    layers = (ROOT / "fluxvla/mcore/pi05/layers.py").read_text()
    provider = (ROOT / "fluxvla/mcore/pi05/provider.py").read_text()

    assert "gather_from_sequence_parallel_region" in layers
    assert "sequence_first = hidden_states.transpose(0, 1).contiguous()" in layers
    assert "output = output.transpose(0, 1).contiguous()[:, :L]" in layers
    assert "_ActivationDTypeBridge.apply(output, hidden_states.dtype)" in layers
    assert "tensor_parallel_output_grad=False" in layers
    assert "_linear_full_sequence(self.vlm_o, slice_vlm)" in layers
    assert "_linear_full_sequence(self.down_proj, hidden)" in layers
    assert "q.reshape(B, L_total, local_h * D)" in layers
    assert "attn_out = scatter_to_tensor_model_parallel_region(" in layers
    assert "if not accuracy_contract:" in layers
    assert "if accuracy_contract and not sequence_parallel:" in layers
    assert "PI05_MCORE_TP_ACCURACY_CONTRACT requires" in provider
    assert "and config.sequence_parallel" not in provider
    assert 'config.attention_implementation != "eager"' in provider


def test_native_grad_norm_reproduces_fsdp_no_shard_foreach_path():
    from fluxvla.mcore.diagnostics import _fsdp_no_shard_total_norm

    gradients = [torch.tensor([3.0]), torch.tensor([0.0]), torch.tensor([4.0])]
    total_norm = _fsdp_no_shard_total_norm(gradients)

    parameters = [
        torch.nn.Parameter(torch.zeros_like(gradient)) for gradient in gradients
    ]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    fsdp_no_shard_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)

    assert torch.equal(total_norm, fsdp_no_shard_norm)


def test_optimizer_trace_skips_unowned_distributed_optimizer_parameters():
    diagnostics = (ROOT / "fluxvla/mcore/diagnostics.py").read_text()

    assert "model_parameter not in group_map" in diagnostics
    assert (
        'param_range = child._get_model_param_range_map(model_parameter)["param"]'
        in diagnostics
    )
    assert "return None" in diagnostics


def test_optimizer_trace_continues_from_checkpoint_iteration():
    diagnostics = (ROOT / "fluxvla/mcore/diagnostics.py").read_text()
    reduce_contract = diagnostics.split(
        "def install_fsdp_grad_reduce_contract", 1
    )[1].split("def install_fsdp_grad_norm_contract", 1)[0]
    optimizer_trace = diagnostics.split(
        "def install_optimizer_alignment_trace", 1
    )[1]

    assert 'state = {"step": 0}' in reduce_contract
    assert 'state = {"step": None}' in optimizer_trace
    assert (
        'state["step"] = int(getattr(get_args(), "iteration", 0))'
        in optimizer_trace
    )


def test_grad_norm_input_trace_captures_real_backend_orders():
    diagnostics = (ROOT / "fluxvla/mcore/diagnostics.py").read_text()
    fsdp_runner = (ROOT / "fluxvla/engines/runners/fsdp_train_runner.py").read_text()

    for source in (diagnostics, fsdp_runner):
        assert "FLUXVLA_GRAD_NORM_INPUT_TRACE_DIR" in source
        assert "_write_grad_norm_input_trace" in source
    assert 'backend="mcore"' in diagnostics
    assert "for handle in self.vla._all_handles" in fsdp_runner
    assert "handle.flat_param._params" in fsdp_runner
    assert "if all_no_shard:" in fsdp_runner
    assert (
        "ordered_parameters = sharded_parameters + nonsharded_parameters" in fsdp_runner
    )
    assert 'backend="fsdp"' in fsdp_runner


def test_fsdp_can_report_dp_mean_loss_for_backend_alignment():
    fsdp_runner = (ROOT / "fluxvla/engines/runners/fsdp_train_runner.py").read_text()

    assert 'os.environ.get("FLUXVLA_REPORT_DP_MEAN_LOSS") != "1"' in fsdp_runner
    assert "dist.all_reduce(reduced, op=dist.ReduceOp.SUM)" in fsdp_runner
    assert "reduced /= dist.get_world_size()" in fsdp_runner


def test_pi05_fsdp_grad_norm_order_separates_interleaved_branches():
    from fluxvla.mcore.pi05.alignment import (
        FSDP_GRAD_NORM_ZERO_KEYS,
        FSDP_GRAD_REDUCE_ZERO_SEGMENTS,
        fsdp_grad_norm_order,
    )

    names = [
        "layers.0.exp_input_norm.cond_proj.weight",
        "suffix_embedder.time_mlp_out.weight",
        "layers.0.vlm_input_norm.norm_weight",
        "action_out_proj.weight",
        "layers.0.attn.exp_q.weight",
        "layers.0.attn.vlm_q.weight",
    ]
    ordered = sorted(names, key=fsdp_grad_norm_order)
    assert ordered == [
        "layers.0.attn.vlm_q.weight",
        "layers.0.vlm_input_norm.norm_weight",
        "action_out_proj.weight",
        "suffix_embedder.time_mlp_out.weight",
        "layers.0.attn.exp_q.weight",
        "layers.0.exp_input_norm.cond_proj.weight",
    ]
    assert FSDP_GRAD_NORM_ZERO_KEYS == ((9,),)
    assert FSDP_GRAD_REDUCE_ZERO_SEGMENTS == (((9,), 257152 * 1024),)


def test_native_has_opt_in_fsdp_bf16_grad_reduce_contract():
    from fluxvla.mcore.diagnostics import _apply_bf16_grad_roundtrip

    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.main_grad = torch.tensor([0.1234567, -0.3333333, 1.0001])
    expected = parameter.main_grad.to(torch.bfloat16).float()

    _apply_bf16_grad_roundtrip([parameter])

    assert torch.equal(parameter.main_grad, expected)
    diagnostics = (ROOT / "fluxvla/mcore/diagnostics.py").read_text()
    recipe = (ROOT / "fluxvla/mcore/pi05/recipe.py").read_text()
    assert "PI05_FSDP_GRAD_REDUCE_CONTRACT" in diagnostics
    assert "FLUXVLA_POST_REDUCE_GRAD_TRACE_DIR" in diagnostics
    assert "_write_post_reduce_gradient_trace" in diagnostics
    assert "install_fsdp_grad_reduce_contract" in recipe
    assert recipe.index("install_optimizer_alignment_trace,") < recipe.rindex(
        "install_fsdp_grad_reduce_contract,"
    )


def test_optimizer_tensor_trace_is_opt_in_and_has_matching_backend_fields():
    diagnostics = (ROOT / "fluxvla/mcore/diagnostics.py").read_text()
    runner = (ROOT / "fluxvla/engines/runners/base_train_runner.py").read_text()

    for source in (diagnostics, runner):
        assert "FLUXVLA_OPTIMIZER_TENSOR_TRACE_DIR" in source
        for field in (
            "parameter_before",
            "raw_grad",
            "post_clip_grad",
            "parameter_after",
            "exp_avg",
            "exp_avg_sq",
        ):
            assert f'"{field}"' in source


def test_autograd_tensor_trace_is_installed_before_distributed_wrapping():
    provider = (ROOT / "fluxvla/mcore/pi05/provider.py").read_text()
    fsdp_runner = (ROOT / "fluxvla/engines/runners/fsdp_train_runner.py").read_text()

    assert "FLUXVLA_AUTOGRAD_TENSOR_TRACE_DIR" in provider
    assert provider.index("_install_alignment_gradient_trace(model)") < provider.index(
        "return model"
    )
    assert "FLUXVLA_AUTOGRAD_TENSOR_TRACE_DIR" in fsdp_runner
    assert "flat_parameter.register_hook(_save_flat)" in fsdp_runner
    assert "'_param_infos'" in fsdp_runner
    assert fsdp_runner.index("self._install_alignment_autograd_tensor_trace()") > (
        fsdp_runner.index("self.vla = FSDP(")
    )
    for source, backend in ((provider, "mcore"), (fsdp_runner, "fsdp")):
        assert '"phase": "autograd"' in source
        assert f'"backend": "{backend}"' in source


def test_fsdp_alignment_runner_can_consume_frozen_model_ready_replay():
    runner = (ROOT / "fluxvla/engines/runners/base_train_runner.py").read_text()

    assert "PI05_ALIGNMENT_REPLAY_DIR" in runner
    assert "source_rank = rank + micro_step * world_size" in runner
    assert 'f"step_{step:06d}_rank_{source_rank:02d}.pt.gz"' in runner
    assert "micro_step >= self.grad_accumulation_steps" in runner
    assert '"lang_tokens": artifact.get("tokens")' in runner
    assert '"action_masks": artifact.get("action_is_pad")' in runner
    assert runner.index(
        "batch = self._maybe_load_pi05_alignment_batch(batch)"
    ) < runner.index("batch = self._prepare_batch(batch, self.device_id, batch_dtype)")


def test_native_loss_trace_defines_elementwise_loss_in_its_own_block():
    source = (ROOT / "fluxvla/mcore/pi05/model.py").read_text()
    loss_trace = source.split(
        'trace_dir = os.environ.get("PI05_LOSS_ALIGNMENT_TRACE_DIR")', 1
    )[1]

    assert loss_trace.index("trace_losses = F.mse_loss") < loss_trace.index(
        '"elementwise_loss": trace_losses'
    )


def test_native_mse_argument_order_matches_fluxvla_mixed_dtype_backward():
    source = (ROOT / "fluxvla/mcore/pi05/model.py").read_text()
    compute_loss = source.split("def _compute_loss", 1)[1].split(
        "def _pack_pp_tensor", 1
    )[0]

    assert "F.mse_loss(v_t, u_t" not in compute_loss
    assert compute_loss.count("F.mse_loss(u_t, v_t") == 2
    assert "u_t[:, :, :out_dim], v_t[:, :, :out_dim]" in compute_loss
    assert "(losses * valid).sum() / (valid.sum() + 1e-8)" in compute_loss
    assert 'with _profile_range("pi05.loss.flow"), torch.autocast(' in compute_loss


def test_native_loss_tail_uses_bf16_autocast_with_fp32_master_parameters():
    source = (ROOT / "fluxvla/mcore/pi05/model.py").read_text()
    compute_loss = source.split("def _compute_loss", 1)[1].split(
        "def _pack_pp_tensor", 1
    )[0]

    assert 'with _profile_range("pi05.model.final_norm"), autocast:' in compute_loss
    assert (
        'with _profile_range("pi05.model.action_head"), torch.autocast(' in compute_loss
    )
    assert "dtype=torch.bfloat16" in compute_loss
    assert "u_t = u_t.to(v_t.dtype)" not in compute_loss


def test_native_fp32_alignment_mode_preserves_all_parameter_storage():
    model = (ROOT / "fluxvla/mcore/pi05/model.py").read_text()
    provider = (ROOT / "fluxvla/mcore/pi05/provider.py").read_text()

    apply_body = model.split("def _apply", 1)[1].split("def set_input_tensor", 1)[0]
    assert "if param.dtype == torch.float32" in apply_body
    assert "fp32_selectors" not in apply_body
    assert "config.params_dtype = torch.float32" in provider


def test_native_adarms_matches_fluxvla_forward_and_backward():
    from fluxvla.mcore.pi05.embedders import AdaRMSNorm
    from fluxvla.models.backbones.llms.condition_gemma import GemmaRMSNorm

    torch.manual_seed(123)
    reference = GemmaRMSNorm(16, cond_dim=8)
    native = AdaRMSNorm(16, cond_dim=8)
    native.cond_proj.load_state_dict(reference.dense.state_dict())
    reference = reference.to(torch.bfloat16)
    native = native.to(torch.bfloat16)
    reference_input = torch.randn(2, 3, 16, dtype=torch.bfloat16, requires_grad=True)
    native_input = reference_input.detach().clone().requires_grad_(True)
    reference_cond = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    native_cond = reference_cond.detach().clone().requires_grad_(True)

    reference_output, reference_gate = reference(reference_input, reference_cond)
    native_output, native_gate = native(native_input, native_cond)
    reference_loss = (
        reference_output.float().square().sum() + reference_gate.float().square().sum()
    )
    native_loss = (
        native_output.float().square().sum() + native_gate.float().square().sum()
    )
    reference_loss.backward()
    native_loss.backward()

    assert torch.equal(native_output, reference_output)
    assert torch.equal(native_gate, reference_gate)
    assert torch.equal(native_input.grad, reference_input.grad)
    assert torch.equal(native_cond.grad, reference_cond.grad)
    assert torch.equal(native.cond_proj.weight.grad, reference.dense.weight.grad)
    assert torch.equal(native.cond_proj.bias.grad, reference.dense.bias.grad)


def test_native_post_attention_residual_matches_fluxvla_clone_contract():
    source = (ROOT / "fluxvla/mcore/pi05/layers.py").read_text()
    layer_forward = source.split("class PI0ModelLayerTP", 1)[1].split(
        "return vlm_hs, exp_hs", 1
    )[0]

    assert "vlm_res = vlm_hs.clone()" in layer_forward
    assert "exp_res = exp_hs.clone()" in layer_forward


def test_native_joint_attention_matches_fluxvla_gqa_expansion_contract():
    from types import SimpleNamespace

    from fluxvla.engines.utils.model_utils import sdpa_attention_forward
    from fluxvla.mcore.pi05.subtask import joint_attention

    torch.manual_seed(123)
    reference_q = torch.randn(1, 4, 7, 8, requires_grad=True)
    reference_k = torch.randn(1, 1, 7, 8, requires_grad=True)
    reference_v = torch.randn(1, 1, 7, 8, requires_grad=True)
    native_q = reference_q.detach().clone().requires_grad_(True)
    native_k = reference_k.detach().clone().requires_grad_(True)
    native_v = reference_v.detach().clone().requires_grad_(True)
    mask = torch.zeros(1, 1, 7, 7)
    upstream = torch.randn(1, 4, 7, 8)

    reference, _ = sdpa_attention_forward(
        SimpleNamespace(num_key_value_groups=4, training=True),
        reference_q,
        reference_k,
        reference_v,
        mask,
        8**-0.5,
    )
    native = joint_attention(
        native_q,
        native_k,
        native_v,
        mask,
        8**-0.5,
        prefix_length=5,
        knowledge_insulation=False,
    )
    reference.transpose(1, 2).backward(upstream)
    native.backward(upstream)

    assert torch.equal(native, reference.transpose(1, 2))
    assert torch.equal(native_q.grad, reference_q.grad)
    assert torch.equal(native_k.grad, reference_k.grad)
    assert torch.equal(native_v.grad, reference_v.grad)
