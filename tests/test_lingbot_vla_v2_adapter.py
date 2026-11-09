from pathlib import Path

import pytest
import torch

from fluxvla.models.vlas.lingbot_vla_v2 import (
    _install_chunked_dense_mlp_oracle,
    _native_output_to_dict,
    _resolve_decoder_layers,
)
from fluxvla.transforms.lingbot_vla_v2 import (
    LingBotPad55D,
    LingBotQwen3VLImageTransform,
    LingBotRobotSchemaPack,
)


def test_native_output_adapter_preserves_autograd_and_metrics():
    loss = torch.tensor(1.0, requires_grad=True)
    loss_moe_aux = torch.tensor(0.1, requires_grad=True)
    output = (loss, torch.tensor(0.5), 0, 0, 0, loss_moe_aux, {"router_z_loss": torch.tensor(0.2)}, None, None, None, None)
    converted = _native_output_to_dict(output)
    assert converted["loss"] is loss
    assert converted["loss_vla"].item() == pytest.approx(0.5)
    assert converted["router_z_loss"].item() == pytest.approx(0.2)
    assert converted["loss_moe_aux"].item() == pytest.approx(0.1)
    converted["loss"].backward()
    assert loss.grad.item() == pytest.approx(1.0)
    converted["loss_moe_aux"].backward()
    assert loss_moe_aux.grad.item() == pytest.approx(1.0)


def test_schema_pack_builds_55d_masks_without_preexisting_actions_key():
    sample = {
        "arm_state": torch.tensor([1.0, 2.0]),
        "arm_action": torch.ones(4, 2),
        "gripper_state": torch.tensor([3.0]),
        "gripper_action": torch.ones(4, 1),
    }
    transform = LingBotRobotSchemaPack(
        feature_groups=[
            {"name": "arm", "state_key": "arm_state", "action_key": "arm_action", "offset": 0, "dim": 2},
            {"name": "gripper", "state_key": "gripper_state", "action_key": "gripper_action", "offset": 4, "dim": 2},
        ]
    )
    output = transform(sample)
    assert output["state"].shape == (55,)
    assert output["actions"].shape == (4, 55)
    assert output["state_joint_mask"].sum().item() == 3
    assert output["joint_mask"].sum().item() == 12


def test_pad55d_rejects_oversized_features():
    transform = LingBotPad55D()
    with pytest.raises(ValueError, match="exceed 55"):
        transform({"state": torch.zeros(56), "actions": torch.zeros(2, 55)})


def test_qwen3vl_transform_restores_per_image_patch_axis():
    processor_path = "/ceph/teamGolden/andyzeng/VLA/models/Qwen3-VL-4B-Instruct"
    transform = LingBotQwen3VLImageTransform(processor_path)
    images = torch.randint(0, 255, (3, 3, 224, 224), dtype=torch.uint8).numpy()
    output = transform({"images": images})
    assert output["images"].ndim == 3
    assert output["images"].shape[0] == 3
    assert output["image_grid_thw"].shape == (3, 3)
    assert int(output["image_grid_thw"].prod(dim=-1).max()) <= 16 * 16


def test_resolve_decoder_layers_supports_text_and_multimodal_qwen_layouts():
    layer = torch.nn.Linear(2, 2)
    text = torch.nn.Module()
    text.model = torch.nn.Module()
    text.model.layers = torch.nn.ModuleList([layer])
    assert _resolve_decoder_layers(text) is text.model.layers

    multimodal = torch.nn.Module()
    multimodal.model = torch.nn.Module()
    multimodal.model.language_model = torch.nn.Module()
    multimodal.model.language_model.layers = torch.nn.ModuleList([layer])
    assert _resolve_decoder_layers(multimodal) is multimodal.model.language_model.layers


def test_chunked_dense_mlp_oracle_preserves_module_state_contract():
    class DenseLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(4, 8, bias=False),
                torch.nn.SiLU(),
                torch.nn.Linear(8, 4, bias=False),
            )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.qwenvl_with_expert = torch.nn.Module()
    qwen = torch.nn.Module()
    qwen.model = torch.nn.Module()
    qwen.model.language_model = torch.nn.Module()
    qwen.model.language_model.layers = torch.nn.ModuleList(
        [DenseLayer(), DenseLayer()]
    )
    model.model.qwenvl_with_expert.qwenvl = qwen
    keys_before = tuple(model.state_dict())

    assert _install_chunked_dense_mlp_oracle(model, chunks=2) == 2
    assert _install_chunked_dense_mlp_oracle(model, chunks=2) == 0
    assert tuple(model.state_dict()) == keys_before

    hidden = torch.randn(2, 5, 4, requires_grad=True)
    output = qwen.model.language_model.layers[0].mlp(hidden)
    assert output.shape == hidden.shape
    output.sum().backward()
    assert hidden.grad is not None


def test_fsdp_policy_resolves_native_policy_model_wrapper():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    adapter = LingBotVLAV2.__new__(LingBotVLAV2)
    torch.nn.Module.__init__(adapter)
    adapter.model = torch.nn.Module()
    adapter.model.model = torch.nn.Module()
    adapter.model.model.qwenvl_with_expert = torch.nn.Module()
    qwen = torch.nn.Module()
    qwen.model = torch.nn.Module()
    qwen.model.language_model = torch.nn.Module()
    qwen.model.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    adapter.model.model.qwenvl_with_expert.qwenvl = qwen
    expert = torch.nn.Module()
    expert.model = torch.nn.Module()
    expert.model.layers = torch.nn.ModuleList([torch.nn.ReLU()])
    adapter.model.model.qwenvl_with_expert.qwen_expert = expert

    assert callable(adapter.get_fsdp_wrapping_policy())
    assert adapter._fsdp_vlm_decoder_layer_classes() == {torch.nn.Linear}
    assert adapter.get_fsdp2_wrapping_modules() == [
        qwen.model.language_model.layers[0]
    ]


def test_fsdp2_full_precision_modules_follow_native_child_first_order():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    Qwen2DecoderLayer = type("Qwen2DecoderLayer", (torch.nn.Module,), {})
    FixAdaRMSNorm = type("FixAdaRMSNorm", (torch.nn.Module,), {})
    layer = Qwen2DecoderLayer()
    layer.norm = FixAdaRMSNorm()
    adapter = LingBotVLAV2.__new__(LingBotVLAV2)
    torch.nn.Module.__init__(adapter)
    adapter.model = torch.nn.Module()
    adapter.model._no_split_modules = ["Qwen2DecoderLayer", "FixAdaRMSNorm"]
    adapter.model.layer = layer

    assert adapter.get_fsdp2_full_precision_modules() == [layer.norm, layer]


def test_lingbot_adapter_exposes_generic_train_runner_contract():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    adapter = LingBotVLAV2.__new__(LingBotVLAV2)
    torch.nn.Module.__init__(adapter)
    adapter.all_module_keys = ["model"]
    adapter.llm_backbone = None
    adapter.vlm_backbone = None
    adapter.llm_expert = None

    assert adapter.all_module_keys == ["model"]
    assert adapter.llm_backbone is None
    assert adapter.vlm_backbone is None
    assert adapter.llm_expert is None


def test_lingbot_training_batch_preserves_native_field_dtypes():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    adapter = LingBotVLAV2.__new__(LingBotVLAV2)
    torch.nn.Module.__init__(adapter)
    batch = {
        "images": torch.ones(1, dtype=torch.float32),
        "depth_targets": torch.ones(1, dtype=torch.float32),
        "future_depth_targets": torch.ones(1, dtype=torch.float32),
        "future_video_targets": torch.ones(1, dtype=torch.float32),
        "future_video_current_patch": torch.ones(1, dtype=torch.float32),
        "state": torch.ones(1, dtype=torch.float32),
        "actions": torch.ones(1, dtype=torch.float32),
        "noise": torch.ones(1, dtype=torch.float32),
        "time": torch.ones(1, dtype=torch.float32),
    }

    prepared = adapter.prepare_training_batch(batch)

    for key in (
        "images",
        "depth_targets",
        "future_depth_targets",
        "future_video_targets",
        "future_video_current_patch",
    ):
        assert prepared[key].dtype == torch.bfloat16
    for key in ("state", "actions", "noise", "time"):
        assert prepared[key].dtype == torch.float32
    assert batch["images"].dtype == torch.float32


def test_lingbot_action_fp32_restores_flow_inputs_after_fsdp_cast():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    adapter = LingBotVLAV2.__new__(LingBotVLAV2)
    torch.nn.Module.__init__(adapter)
    adapter.config = type("Config", (), {"action_fp32": True})()
    batch = {
        key: torch.ones(2, dtype=torch.bfloat16)
        for key in ("state", "actions", "noise", "time")
    }
    batch["images"] = torch.ones(2, dtype=torch.bfloat16)

    prepared = adapter._promote_action_flow_inputs(batch)

    for key in ("state", "actions"):
        assert prepared[key].dtype == torch.float32
    for key in ("noise", "time"):
        assert prepared[key].dtype == torch.bfloat16
    assert prepared["images"].dtype == torch.bfloat16
    assert batch["time"].dtype == torch.bfloat16


def test_lingbot_distributed_runtime_is_noop_without_process_group():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    adapter = LingBotVLAV2.__new__(LingBotVLAV2)
    torch.nn.Module.__init__(adapter)

    adapter.initialize_distributed_runtime()


@pytest.mark.parametrize("shell_dtype", [torch.float32, torch.bfloat16])
def test_low_memory_checkpoint_materialization_preserves_fp32_master_values(
    monkeypatch, tmp_path, shell_dtype
):
    safetensors = pytest.importorskip("safetensors.torch")
    import fluxvla.models.vlas.lingbot_vla_v2 as lingbot_adapter

    class FakeConfig:
        def __init__(self, **kwargs):
            self.max_action_dim = 55
            self.max_state_dim = 55
            self.n_action_steps = 1
            self.chunk_size = 1
            self.action_fp32 = kwargs.get("action_fp32", False)

    class FakePolicy(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(3, dtype=shell_dtype))

        def sample_actions(self):
            return None

    monkeypatch.setattr(
        lingbot_adapter,
        "_load_native_symbols",
        lambda: (FakeConfig, FakePolicy),
    )
    checkpoint_value = torch.tensor(
        [1.0001, -0.3333, 123.4567], dtype=torch.float32
    )
    safetensors.save_file({"weight": checkpoint_value}, tmp_path / "model.safetensors")

    adapter = lingbot_adapter.LingBotVLAV2(
        pretrained_name_or_path=str(tmp_path),
        low_cpu_mem_usage=True,
    )
    assert adapter.model.weight.is_meta
    assert adapter.model.weight.dtype == shell_dtype

    adapter.from_pretrained()

    assert not adapter.model.weight.is_meta
    assert adapter.model.weight.dtype == torch.float32
    torch.testing.assert_close(adapter.model.weight, checkpoint_value, rtol=0, atol=0)


def test_fsdp2_root_output_dtype_matches_native_mixed_precision_boundary():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    assert LingBotVLAV2.get_fsdp2_root_output_dtype(torch.bfloat16) == torch.bfloat16
    assert LingBotVLAV2.get_fsdp2_root_output_dtype(torch.float32) == torch.float32


def test_activation_trace_can_target_a_later_training_step():
    source = Path("fluxvla/models/vlas/lingbot_vla_v2.py").read_text()

    assert "FLUXVLA_LINGBOT_ACTIVATION_TRACE_STEP" in source
    assert "LINGBOT_ACTIVATION_TRACE_LAYER" in source
    assert "if not 0 <= trace_layer < 36" in source
    assert "_activation_trace_call_index" in source
    assert "FLUXVLA_LINGBOT_ACTIVATION_GRAD_TRACE" in source
    assert "register_multi_grad_hook" in source
    assert "_activation_grads.pt" in source
