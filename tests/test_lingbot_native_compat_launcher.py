import inspect

import torch

from tools.lingbot_native_compat_launcher import (
    _expand_fused_checkpoint_item,
    _install_checkpoint_save_skip,
    _install_lerobot_reader_compat,
    _install_native_fused_moe_backward_trace,
    _install_native_batch_dump,
    _install_native_optimizer_trace,
    _install_null_cli_compat,
    _resolve_native_flow_inputs,
)


class _Reader:
    def __init__(self):
        self.delta_indices = {"action": [0, 1]}
        self._absolute_to_relative_idx = {7: 0}
        self._video_backend = "pyav"

    def _get_query_indices(self, index, episode_index):
        return index, episode_index

    def _get_query_timestamps(self, timestamp, query_indices=None):
        return timestamp, query_indices


def test_lerobot_reader_compat_forwards_properties_and_methods():
    class Dataset:
        def __init__(self):
            self.reader = _Reader()

    _install_lerobot_reader_compat(Dataset)
    dataset = Dataset()

    assert dataset.delta_indices == {"action": [0, 1]}
    assert dataset._absolute_to_relative_idx == {7: 0}
    assert dataset.video_backend == "pyav"
    assert dataset._get_query_indices(4, 2) == (4, 2)
    assert dataset._get_query_timestamps(0.5, {"action": [4]}) == (
        0.5,
        {"action": [4]},
    )

    dataset.delta_indices = {"action": [3]}
    dataset.video_backend = "torchcodec"
    assert dataset.reader.delta_indices == {"action": [3]}
    assert dataset.reader._video_backend == "torchcodec"


def test_lerobot_reader_compat_preserves_existing_dataset_api():
    class Dataset:
        delta_indices = "native"

        def _get_query_indices(self, *_args):
            return "native"

    _install_lerobot_reader_compat(Dataset)
    dataset = Dataset()

    assert dataset.delta_indices == "native"
    assert dataset._get_query_indices(1, 2) == "native"


def test_lerobot_reader_compat_property_supports_pre_reader_assignment():
    class Dataset:
        pass

    _install_lerobot_reader_compat(Dataset)
    dataset = Dataset()
    dataset.delta_indices = {"action": [0]}

    assert dataset.delta_indices == {"action": [0]}


def test_fused_checkpoint_item_expands_per_expert_without_value_change():
    fused = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    items = list(
        _expand_fused_checkpoint_item(
            "model.layers.0.mlp.experts.down_proj", fused
        )
    )
    assert [key for key, _ in items] == [
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    assert torch.equal(items[0][1], fused[0])
    assert torch.equal(items[1][1], fused[1])


def test_fused_checkpoint_item_supports_all_mlp_projections():
    fused = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    for projection in ("gate_proj", "up_proj", "down_proj"):
        items = list(
            _expand_fused_checkpoint_item(
                f"model.layers.0.mlp.experts.{projection}", fused
            )
        )
        assert len(items) == 2
        assert items[0][0].endswith(f"experts.0.{projection}.weight")
        assert torch.equal(items[1][1], fused[1])


def test_fused_checkpoint_item_keeps_non_expert_entries_unchanged():
    tensor = torch.ones(2, 3)
    items = list(_expand_fused_checkpoint_item("model.norm.weight", tensor))
    assert items[0][0] == "model.norm.weight"
    assert torch.equal(items[0][1], tensor)


def test_flow_input_takes_precedence_over_fixed_zero_mode(monkeypatch, tmp_path):
    flow_path = tmp_path / "flow_rank1_step3.pt"
    torch.save(
        {
            "noise": torch.full((1, 2, 3), 7.0, dtype=torch.float32),
            "time": torch.tensor([0.75], dtype=torch.float32),
        },
        flow_path,
    )
    monkeypatch.setenv(
        "LINGBOT_FLOW_INPUT",
        str(tmp_path / "flow_rank{rank}_step{step}.pt"),
    )
    monkeypatch.setenv("RANK", "1")
    actions = torch.zeros((1, 2, 3), dtype=torch.bfloat16)

    noise, time = _resolve_native_flow_inputs(
        actions=actions,
        sample_time=lambda *_args: torch.tensor([-1.0]),
        call_index=3,
        capture_flow=False,
    )

    assert noise.dtype == actions.dtype
    assert time.dtype == actions.dtype
    assert torch.equal(noise, torch.full_like(actions, 7.0))
    assert torch.equal(time, torch.tensor([0.75], dtype=torch.bfloat16))


def test_fixed_flow_without_input_uses_zero_noise(monkeypatch):
    monkeypatch.delenv("LINGBOT_FLOW_INPUT", raising=False)
    actions = torch.ones((2, 3, 4), dtype=torch.float32)

    noise, time = _resolve_native_flow_inputs(
        actions=actions,
        sample_time=lambda *_args: torch.tensor([-1.0]),
        call_index=0,
        capture_flow=False,
    )

    assert torch.count_nonzero(noise) == 0
    assert torch.equal(time, torch.full((2,), 0.5))


def test_optimizer_trace_does_not_materialize_dtensors_before_grad_norm():
    source = inspect.getsource(_install_native_optimizer_trace)

    assert "detached = detached.to_local()" in source
    assert "def clip_grad_norm_with_trace" in source
    assert "return original_clip_grad_norm(" in source
    assert "clip_grad_module.clip_grad_norm_ =" not in source
    assert "clip_grads_with_norm_ =" not in source
    assert 'LINGBOT_OPTIMIZER_TRACE_STEP' in source
    assert 'LINGBOT_GRAD_NORM_INPUT_TRACE_DIR' in source
    assert '"grad_norms"' in source
    assert '"exp_avg_before"' in source


def test_native_fused_moe_backward_trace_is_opt_in_and_read_only():
    source = inspect.getsource(_install_native_fused_moe_backward_trace)

    assert 'LINGBOT_FUSED_MOE_BACKWARD_TRACE_DIR' in source
    assert 'LINGBOT_FUSED_MOE_BACKWARD_TRACE_STEP' in source
    assert "function_cls.backward = staticmethod(backward_with_trace)" in source
    assert "result = original_backward(ctx, grad_output)" in source
    assert "LINGBOT_ROUTER_GRAD_TRACE_DIR" in source
    assert "logits.register_hook(save_gradient)" in source
    assert "return result" in source


def test_native_activation_trace_can_target_a_later_training_step():
    source = inspect.getsource(_install_native_batch_dump)

    assert 'LINGBOT_ACTIVATION_TRACE_STEP' in source
    assert 'LINGBOT_ACTIVATION_TRACE_LAYER' in source
    assert "if not 0 <= trace_layer < 36" in source
    assert 'f"native_step{call_index:06d}_rank{rank}.pt"' in source
    assert 'LINGBOT_ACTIVATION_GRAD_TRACE' in source
    assert "register_multi_grad_hook" in source
    assert "_activation_grads.pt" in source


def test_native_cli_compat_restores_yaml_null_sentinels():
    source = inspect.getsource(_install_null_cli_compat)

    assert 'value in ("null", "None")' in source
    assert "setattr(group, name, None)" in source
    assert '("model", "basic_modules")' in source
    assert '("train", "muon_exclude_name_patterns")' in source
    assert "group.get(field_name) == []" in source
    assert "yaml_module.safe_load = original_safe_load" in source


def test_checkpoint_save_skip_installs_the_proxy_builder():
    source = inspect.getsource(_install_checkpoint_save_skip)

    assert 'sys.argv.append("--train.save_hf_weights=false")' in source
    assert "checkpoint_module.build_checkpointer = " \
        "build_checkpointer_without_save" in source
    assert source.index("checkpoint_module.build_checkpointer =") \
        > source.index("def build_checkpointer_without_save")
