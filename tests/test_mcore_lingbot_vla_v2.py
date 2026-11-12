from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch


def test_lingbot_decoder_gradient_fork_has_fixed_accumulation_order():
    from fluxvla.models.vlas.lingbot_native_core.modeling_lingbot_vla_v2 import (
        _deterministic_gradient_fork,
    )

    value = torch.tensor([1.0, -2.0], requires_grad=True)
    qkv_value, residual_value = _deterministic_gradient_fork(value)
    qkv_grad = torch.tensor([1.0e8, 1.0e-8])
    residual_grad = torch.tensor([-1.0e8 + 8.0, 2.0e-8])

    torch.autograd.backward(
        (qkv_value, residual_value),
        (qkv_grad, residual_grad),
    )

    assert torch.equal(value.grad, qkv_grad + residual_grad)




def test_lingbot_recipe_uses_native_mcore_contract():
    from fluxvla.mcore.lingbot_vla_v2.recipe import LINGBOT_VLA_V2_SPEC

    assert LINGBOT_VLA_V2_SPEC.name == "lingbot-vla-v2"
    assert LINGBOT_VLA_V2_SPEC.model_provider.__module__.endswith(
        "lingbot_vla_v2.provider"
    )
    assert LINGBOT_VLA_V2_SPEC.dataset_provider.__module__.endswith(
        "lingbot_vla_v2.recipe"
    )
    assert LINGBOT_VLA_V2_SPEC.forward_step.__module__.endswith(
        "lingbot_vla_v2.forward_step"
    )


def test_lingbot_entry_uses_common_mcore_launcher():
    source = (
        Path(__file__).parents[1] / "scripts" / "train_mcore_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")

    assert "run_mcore(LINGBOT_VLA_V2_SPEC)" in source
    assert "FSDPTrainRunner" not in source


def test_lingbot_mcore_defaults_to_the_native_production_assets():
    source = (
        Path(__file__).parents[1] / "scripts" / "train_mcore_lingbot_vla_v2.sh"
    ).read_text(encoding="utf-8")

    assert "${VLA_ROOT}/models/lingbot-vla-v2-6b" in source
    assert "${VLA_ROOT}/models/Qwen3-VL-4B-Instruct" in source
    assert "LINGBOT_DATA_MODE:-online" in source
    assert "datasets/huggingface/lerobot/lingbot_beat_block_hammer_reduced" in source
    assert "lingbot_runs/reduced_beat_block_hammer/lingbotvla_cli.yaml" in source
    assert "robotwin_norm_stats.json" in source
    assert "official_captured_batch_manifest_20261020.json" not in source
    assert '--lr-decay-iters "${LR_DECAY_ITERS:-100}"' in source
    assert '${LR_DECAY_ITERS:-${STEPS:-2}}' not in source
    assert '--eval-iters "${EVAL_ITERS:-0}"' in source
    assert '--master_port="${MASTER_PORT:-29500}"' in source
    assert '--lingbot-moe-implementation "${LINGBOT_MOE_IMPLEMENTATION:-auto}"' in source
    assert "--ddp-average-in-collective" in source


def test_lingbot_online_parity_config_pins_compute_and_stable_data_contract():
    source = (
        Path(__file__).parents[1]
        / "configs"
        / "lingbot_vla_v2"
        / "lingbot_vla_v2_online_flow_only_100.py"
    ).read_text(encoding="utf-8")

    assert 'enable_forward_autocast=True' in source
    assert 'runner.max_steps=100' not in source
    assert 'max_steps=100' in source
    assert "datasets/huggingface/lerobot/lingbot_beat_block_hammer_reduced" in source
    assert "lingbot_runs/reduced_beat_block_hammer/lingbotvla_cli.yaml" in source
    assert "FluxVLA/work_dirs/lingbot_beat_block_hammer_reduced_v2" not in source


def test_lingbot_recipe_uses_distopt_owned_shard_grad_norm():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/recipe.py"
    ).read_text(encoding="utf-8")

    assert "install_fsdp_grad_norm_contract" not in source
    assert "install_lingbot_fsdp2_reduction_order" in source


def test_lingbot_parameterwise_grad_norm_supports_tp_and_ep_owned_shards():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/optimizer.py"
    ).read_text(encoding="utf-8")

    assert "args.tensor_model_parallel_size != 1" not in source
    assert "args.expert_model_parallel_size != 1" not in source
    assert "param_is_not_tensor_parallel_duplicate(parameter)" in source
    assert "args.pipeline_model_parallel_size != 1" in source
    assert "if owned_gradients:" in source
    assert "device=trace_device" in source
    assert "local_squared.sum().sqrt()" in source


def test_lingbot_fsdp2_clip_coefficient_uses_partial_norm_denominator():
    from fluxvla.mcore.lingbot_vla_v2 import optimizer as optimizer_module

    local_norms = torch.tensor(
        [1.206250786781311, 0.8654600977897644,
         1.0807801485061646, 0.972142219543457],
        dtype=torch.float32,
    )
    global_norm = torch.linalg.vector_norm(local_norms, 2.0)
    optimizer_module._FSDP2_PARTIAL_CLIP_DENOMINATOR = (
        torch.linalg.vector_norm(local_norms + 1.0e-6, 2.0)
    )
    try:
        coefficient = optimizer_module._fsdp2_clip_coefficient(
            1.0, global_norm, global_norm.device
        )
    finally:
        optimizer_module._FSDP2_PARTIAL_CLIP_DENOMINATOR = None

    assert coefficient.item() == 0.48128026723861694
    assert coefficient.item() != (1.0 / (global_norm + 2.0e-6)).item()


def test_lingbot_clip_coefficient_matches_native_epsilon_without_partial_norm():
    from fluxvla.mcore.lingbot_vla_v2 import optimizer as optimizer_module

    optimizer_module._FSDP2_PARTIAL_CLIP_DENOMINATOR = None
    total_norm = torch.tensor(1.9115468263626099, dtype=torch.float32)
    coefficient = optimizer_module._fsdp2_clip_coefficient(
        1.0, total_norm, total_norm.device
    )
    expected = 1.0 / (total_norm + 1.0e-6)
    assert coefficient.item() == expected.item()


def test_lingbot_activation_trace_can_be_limited_to_action_head():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/models/vlas/lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")

    assert "FLUXVLA_LINGBOT_TRACE_ACTION_HEAD_ONLY" in source
    assert 'selected = {"model.action_out_proj"}' in source


def test_lingbot_activation_trace_includes_final_norm_boundaries():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/models/vlas/lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")

    assert '"model.qwenvl_with_expert.qwenvl.model.language_model.norm"' in source
    assert '"model.qwenvl_with_expert.qwen_expert.model.norm"' in source


def test_lingbot_local_loss_trace_is_shared_by_native_and_mcore(tmp_path, monkeypatch):
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    monkeypatch.setenv("FLUXVLA_LINGBOT_LOCAL_LOSS_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    LingBotVLAV2._write_local_loss_trace(
        {
            "loss": torch.tensor(0.25, dtype=torch.bfloat16),
            "loss_vla": torch.tensor(0.125),
            "ignored": torch.ones(2),
        },
        call_index=7,
    )

    row = json.loads((tmp_path / "rank3.jsonl").read_text(encoding="utf-8"))
    assert row == {
        "call_index": 7,
        "loss": 0.25,
        "loss_vla": 0.125,
        "rank": 3,
    }


def test_mcore_model_preserves_only_native_fp32_units(monkeypatch):
    from fluxvla.mcore.lingbot_vla_v2.model import LingBotVLA2MCoreModel

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = torch.nn.Linear(2, 2, bias=False)
            self.action = torch.nn.Linear(2, 2, bias=False)
            self.register_buffer("scale", torch.ones(1, dtype=torch.float32))

        def get_fsdp2_full_precision_modules(self):
            return [self.action]

        def forward(self, **batch):
            return {"loss": self.action(batch["value"]).sum()}

    adapter = Adapter()
    adapter.vlm.weight.allreduce = False
    vlm_before = adapter.vlm.weight.detach().clone()
    action_before = adapter.action.weight.detach().clone()
    model = LingBotVLA2MCoreModel(
        SimpleNamespace(expert_model_parallel_size=1), adapter
    )
    model.bfloat16()

    assert model.adapter.vlm.weight.dtype == torch.bfloat16
    assert model.adapter.action.weight.dtype == torch.float32
    assert torch.equal(model.adapter.vlm.weight.float(), vlm_before.bfloat16().float())
    assert torch.equal(model.adapter.action.weight, action_before)
    assert all(parameter.allreduce for parameter in model.parameters())
    assert model.adapter.scale.dtype == torch.float32


def test_mcore_production_precision_is_independent_of_action_input_precision():
    from fluxvla.mcore.lingbot_vla_v2.model import LingBotVLA2MCoreModel

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(action_fp32=False)
            self.vlm = torch.nn.Linear(2, 2, bias=False)
            self.action = torch.nn.Linear(2, 2, bias=False)
            self.register_buffer("rotary", torch.ones(1, dtype=torch.float32))

        def get_fsdp2_full_precision_modules(self):
            return [self.action]

        def forward(self, **batch):
            return {"loss": self.action(batch["value"]).sum()}

    model = LingBotVLA2MCoreModel(
        SimpleNamespace(expert_model_parallel_size=1),
        Adapter(),
        preserve_fsdp2_fp32_modules=True,
    )
    model.bfloat16()

    assert model.adapter.vlm.weight.dtype == torch.bfloat16
    assert model.adapter.action.weight.dtype == torch.float32
    assert model.adapter.rotary.dtype == torch.float32


def test_mcore_provider_reproduces_fsdp2_action_hidden_precision_boundary():
    root = Path(__file__).parents[1]
    provider = (
        root / "fluxvla/mcore/lingbot_vla_v2/provider.py"
    ).read_text(encoding="utf-8")
    model_core = (
        root
        / "fluxvla/models/vlas/lingbot_native_core/modeling_lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root / "scripts/train_mcore_lingbot_vla_v2.sh"
    ).read_text(encoding="utf-8")

    assert "_fluxvla_action_hidden_fp32 = True" in provider
    assert 'self, "_fluxvla_action_hidden_fp32", False' in model_core
    assert "hidden_states = hidden_states.float()" in model_core
    assert "out_emb = out_emb.float()" in model_core
    assert "router_logits = router_logits.float()" in model_core
    assert 'FLUXVLA_MCORE_PRESERVE_FP32_PARAMS:-0' in launcher


def test_mcore_fp32_master_mode_keeps_float16_wrapper_from_rounding(monkeypatch):
    from fluxvla.mcore.lingbot_vla_v2.model import LingBotVLA2MCoreModel

    monkeypatch.setenv("FLUXVLA_MCORE_PRESERVE_FP32_PARAMS", "1")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))

        def get_fsdp2_full_precision_modules(self):
            return []

    model = LingBotVLA2MCoreModel(
        SimpleNamespace(expert_model_parallel_size=1), Adapter()
    )
    assert model.bfloat16() is model
    assert model.adapter.weight.dtype == torch.float32


def test_mcore_model_preserves_local_expert_marker_when_ep_enabled():
    from fluxvla.mcore.lingbot_vla_v2.model import LingBotVLA2MCoreModel

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = torch.nn.Linear(2, 2, bias=False)
            self.expert = torch.nn.Linear(2, 2, bias=False)
            self.expert.weight.allreduce = False

        def get_fsdp2_full_precision_modules(self):
            return []

        def forward(self, **batch):
            return {"loss": self.dense(batch["value"]).sum()}

    model = LingBotVLA2MCoreModel(
        SimpleNamespace(expert_model_parallel_size=2), Adapter()
    )

    assert getattr(model.adapter.dense.weight, "allreduce", True)
    assert model.adapter.expert.weight.allreduce is False


def test_replay_dataset_matches_distributed_sampler_order(tmp_path, monkeypatch):
    import fluxvla.datasets.lingbot_teacher_cache as dataset_module
    from fluxvla.mcore.lingbot_vla_v2.dataset import LingBotReplayIterableDataset

    class FakeCache:
        def __init__(self, **_kwargs):
            self.values = list(range(8))

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return {"value": torch.tensor(self.values[index])}

    monkeypatch.setattr(dataset_module, "LingBotTeacherCacheDataset", FakeCache)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    datasets = [
        LingBotReplayIterableDataset(
            str(manifest),
            dp_rank=rank,
            dp_size=2,
            micro_batch_size=1,
            sampler_seed=0,
            repeat=False,
        )
        for rank in range(2)
    ]
    actual = [
        [int(batch["value"].item()) for batch in dataset]
        for dataset in datasets
    ]
    expected = []
    for rank in range(2):
        sampler = torch.utils.data.DistributedSampler(
            list(range(8)),
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False,
        )
        expected.append(list(iter(sampler)))

    assert actual == expected


def test_replay_dataset_implements_external_dataloader_iterator(tmp_path, monkeypatch):
    import fluxvla.datasets.lingbot_teacher_cache as dataset_module
    from fluxvla.mcore.lingbot_vla_v2.dataset import LingBotReplayIterableDataset

    class FakeCache:
        def __init__(self, **_kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"value": torch.tensor(index)}

    monkeypatch.setattr(dataset_module, "LingBotTeacherCacheDataset", FakeCache)
    dataset = LingBotReplayIterableDataset(
        str(tmp_path / "manifest.json"),
        dp_rank=0,
        dp_size=1,
        micro_batch_size=1,
        repeat=False,
    )

    assert iter(dataset) is dataset
    assert int(next(dataset)["value"].item()) == 0


def test_replay_dataset_preserves_official_step_rank_order(tmp_path, monkeypatch):
    import fluxvla.datasets.lingbot_teacher_cache as dataset_module
    from fluxvla.mcore.lingbot_vla_v2.dataset import LingBotReplayIterableDataset

    class FakeCache:
        def __init__(self, *, replay_order, **_kwargs):
            assert replay_order == "official_step_rank"
            self.values = list(range(8))
            self.deterministic_order = True

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return {"value": torch.tensor(self.values[index])}

    monkeypatch.setattr(dataset_module, "LingBotTeacherCacheDataset", FakeCache)
    datasets = [
        LingBotReplayIterableDataset(
            str(tmp_path / "manifest.json"),
            dp_rank=rank,
            dp_size=2,
            micro_batch_size=1,
            sampler_seed=42,
            repeat=False,
        )
        for rank in range(2)
    ]

    assert [int(batch["value"]) for batch in datasets[0]] == [0, 2, 4, 6]
    assert [int(batch["value"]) for batch in datasets[1]] == [1, 3, 5, 7]


def test_routed_expert_lr_match_is_scoped():
    from fluxvla.mcore.lingbot_vla_v2.optimizer import (
        _chunk_parameter_groups,
        is_routed_expert_parameter,
    )

    parameter = torch.nn.Parameter(torch.ones(1))
    assert is_routed_expert_parameter(
        "adapter.model.model.qwenvl_with_expert.qwen_expert.model.layers.0."
        "mlp.experts.3.gate_proj.weight",
        parameter,
    )
    assert is_routed_expert_parameter(
        "adapter.model.model.qwenvl_with_expert.qwen_expert.model.layers.0."
        "mlp.module.experts.3.gate_proj.weight",
        parameter,
    )
    assert not is_routed_expert_parameter(
        "adapter.model.model.qwenvl_with_expert.qwen_expert.model.layers.0."
        "mlp.shared_expert.gate_proj.weight",
        parameter,
    )
    groups = _chunk_parameter_groups(
        [{"params": [parameter, parameter], "lr_mult": 2.0}],
        chunk_size=1,
    )
    assert [len(group["params"]) for group in groups] == [1, 1]
    assert [group["lr_mult"] for group in groups] == [2.0, 2.0]


def test_lingbot_args_enable_low_memory_adam_by_default():
    import argparse

    from fluxvla.mcore.lingbot_vla_v2.provider import get_lingbot_args

    parser = argparse.ArgumentParser()
    get_lingbot_args(parser)
    args = parser.parse_args(
        [
            "--lingbot-native-config",
            "config",
            "--lingbot-checkpoint",
            "checkpoint",
            "--lingbot-tokenizer-path",
            "tokenizer",
            "--lingbot-manifest",
            "manifest",
        ]
    )

    assert args.lingbot_low_memory_adam is True
    assert args.lingbot_fsdp2_reduction_order is True
    assert args.lingbot_action_fp32 is False
    assert args.lingbot_sampler_seed == 42
    assert args.lingbot_replay_order == "official_step_rank"
    assert args.lingbot_attention_implementation == "eager"
    assert args.lingbot_vit_attn_implementation == "eager"
    assert args.lingbot_tp_routed_experts is True
    assert args.lingbot_tp_shared_expert is True
    assert args.lingbot_tp_mlp_column is True
    assert args.lingbot_tp_mlp_row is True
    assert args.lingbot_tp_parity_reference is False
    assert not hasattr(args, "lingbot_source_root")

    explicit_args = parser.parse_args(
        [
            "--lingbot-native-config",
            "config",
            "--lingbot-checkpoint",
            "checkpoint",
            "--lingbot-tokenizer-path",
            "tokenizer",
            "--lingbot-manifest",
            "manifest",
            "--lingbot-action-fp32",
        ]
    )
    assert explicit_args.lingbot_action_fp32 is True


def test_lingbot_mcore_model_overrides_match_native_production_math():
    from fluxvla.mcore.lingbot_vla_v2.provider import _model_config_overrides

    args = SimpleNamespace(
        lingbot_attention_implementation="eager",
        lingbot_vit_attn_implementation="flash_attention_2",
        lingbot_moe_implementation="auto",
    )
    assert _model_config_overrides(
        args, requested_sp=False, requested_ep=False
    ) == {
        "attention_implementation": "eager",
        "vit_attn_implementation": "flash_attention_2",
    }
    assert _model_config_overrides(
        args, requested_sp=True, requested_ep=False
    ) == {
        "attention_implementation": "eager",
        "vit_attn_implementation": "flash_attention_2",
        "moe_implementation": "eager",
    }
    args.lingbot_moe_implementation = "eager"
    assert _model_config_overrides(
        args, requested_sp=False, requested_ep=False
    )["moe_implementation"] == "eager"


def test_lingbot_ep_topology_uses_expert_data_parallel_factor(monkeypatch):
    from fluxvla.mcore.lingbot_vla_v2 import provider
    from megatron.core import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_expert_data_parallel_world_size",
        lambda: 2,
    )

    assert provider._adapter_data_parallel_size(expert_parallel_size=1) == 4
    assert provider._adapter_data_parallel_size(expert_parallel_size=2) == 2


def test_mcore_tensor_parallel_primitives_are_shared():
    from fluxvla.mcore.tensor_parallel import (
        ColumnParallelLinear,
        MCoreTopology,
        RowParallelLinear,
        validate_mcore_topology,
    )

    assert ColumnParallelLinear.__module__.startswith("megatron.core")
    assert RowParallelLinear.__module__.startswith("megatron.core")
    validate_mcore_topology(MCoreTopology(tp=2, dp=1), allow_tp=True)
    try:
        validate_mcore_topology(MCoreTopology(pp=2), allow_tp=True)
    except NotImplementedError as error:
        assert "PP" in str(error)
    else:
        raise AssertionError("unsupported PP must be rejected")

    linear_source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/tensor_parallel/linear.py"
    ).read_text(encoding="utf-8")
    assert "set_tensor_model_parallel_attributes(" in linear_source
    assert "dim=0 if column else 1" in linear_source


def test_lingbot_provider_enables_tp_and_native_mlp_sp():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/provider.py"
    ).read_text(encoding="utf-8")
    assert "convert_lingbot_dense_modules(" in source
    assert "allow_tp=True" in source
    assert "allow_sp=True" in source
    assert "sequence_parallel=bool" in source
    assert "tp_size=parallel_state.get_tensor_model_parallel_world_size()" in source
    assert 'ep_size=topology["EP"]' in source


def test_lingbot_tp_selector_uses_shared_adapter():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/tensor_parallel.py"
    ).read_text(encoding="utf-8")
    assert "from ..tensor_parallel import MCoreTPLinear" in source
    assert "ColumnParallelLinear" not in source
    assert "RowParallelLinear" not in source
    assert '("action_time_mlp_in", True)' not in source
    assert "SequenceParallelMLP" in source
    assert "_replace_fused_experts" in source
    assert "expert_modules = getattr(mlp, \"experts\", None)" in source
    assert "shared_expert = getattr(mlp, \"shared_expert\", None)" in source
    assert "convert_routed_experts" in source
    assert "convert_shared_expert" in source
    assert "convert_mlp_column" in source
    assert "convert_mlp_row" in source
    assert "use_native_linear=True" in source
    assert "grouped-GEMM adapter" in source
    assert "use eager experts" in source
    assert '"sp_wrapped"' in source
    assert '"sp_skipped_routed_moe"' in source
    assert (
        '"model.qwenvl_with_expert.qwenvl.model.language_model.layers."'
        in source
    )
    assert '"model.qwenvl_with_expert.qwen_expert.model.layers."' in source


def test_mcore_optimizer_trace_reports_bf16_compute_projection():
    source = (
        Path(__file__).parents[1] / "fluxvla/mcore/diagnostics.py"
    ).read_text(encoding="utf-8")

    assert '"parameter_bf16_norm"' in source
    assert '"parameter_bf16_sum"' in source
    assert '"parameter_bf16_bits_sum"' in source


def test_lingbot_mcore_uses_native_bf16_autocast_for_all_storage_modes():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/model.py"
    ).read_text(encoding="utf-8")

    # Native FSDP autocasts both FP32-master and BF16-storage paths.  The
    # MCore wrapper must not make autocast conditional on parameter storage.
    assert 'FLUXVLA_MCORE_DISABLE_AUTOCAST' in source
    assert 'dtype=torch.bfloat16' in source
    assert 'if self._preserve_all_fp32_params and torch.cuda.is_available()' not in source


def test_mcore_optimizer_tensor_trace_persists_pre_update_boundary():
    source = (
        Path(__file__).parents[1] / "fluxvla/mcore/diagnostics.py"
    ).read_text(encoding="utf-8")

    assert 'phase="pre_update"' in source
    assert 'phase="post_update"' in source
    assert 'FLUXVLA_PARITY_STOP_BEFORE_OPTIMIZER' in source


def test_lingbot_ep_is_owned_by_the_mcore_backend():
    adapter_source = (
        Path(__file__).parents[1]
        / "fluxvla/models/vlas/lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")
    provider_source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/provider.py"
    ).read_text(encoding="utf-8")
    ep_source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/tensor_parallel.py"
    ).read_text(encoding="utf-8")

    assert "initialize_distributed_runtime" in adapter_source
    assert "init_parallel_state" not in adapter_source
    assert "shard_lingbot_experts(adapter.model)" in provider_source
    assert "class LocalExpertTokenMoE" in ep_source
    assert "parameter.allreduce = False" in ep_source
    assert "from megatron.core.tensor_parallel import all_to_all" in ep_source


def test_tp_wrapper_preserves_ep_gradient_ownership_metadata():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/tensor_parallel/linear.py"
    ).read_text(encoding="utf-8")

    # LocalExpertTokenMoE marks routed expert gradients as already reduced
    # before projection replacement. The wrapper must carry that contract to
    # the new MCore parameter or DistributedOptimizer will reduce it twice.
    assert '"_fluxvla_grad_already_reduced"' in source
    assert '"allreduce"' in source
    assert "self.linear.weight" in source


def test_sequence_parallel_boundary_pads_uneven_sequence():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/tensor_parallel/activation.py"
    ).read_text(encoding="utf-8")
    assert "pad_length = (-original_length) % tp_size" in source
    assert "item[:original_length]" in source
    assert "sp_meta = (batch_size, local_length)" in source
    assert "gathered_length = item.shape[0]" in source
    assert "dist.all_reduce(gradient, group=group)" in source
    assert 'getattr(parameter, "tensor_model_parallel", False)' in source
    assert "tensor_parallel_output_grad=False" in source
    shared = (
        Path(__file__).parents[1] / "fluxvla/mcore/tensor_parallel/linear.py"
    ).read_text(encoding="utf-8")
    assert "return output.clone()" in shared


def test_lingbot_sp_uses_single_token_partition_boundary():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/tensor_parallel.py"
    ).read_text(encoding="utf-8")
    assert "shard the token dimension twice" in source
    assert "sequence_parallel=False" in source
    assert "all_to_all" in source
    assert "allreduce = False" in source
    assert "for source_rank in range(self.ep_size)" in source
    assert "[expert(source_hidden) for expert in self.experts]" in source
    assert 'torch.einsum("etd,te->td"' in source
    assert "returned_outputs" in source
    assert "_gather_expert_weights" in source
    assert "class _NativeGradientExpertAllGather" in source
    assert "_fsdp2_ring_average" in source
    assert "_fluxvla_grad_already_reduced" in source
    assert "routed = reference_routed" in source
    assert "FLUXVLA_LINGBOT_EP_WGRAD_TRACE_DIR" in source
    assert '"phase": "pre_expert_wgrad_collective"' in source


def test_lingbot_ep_reference_wgrad_matches_fsdp2_ring_order():
    from fluxvla.mcore.lingbot_vla_v2.tensor_parallel import _fsdp2_ring_average

    gradients = [
        torch.tensor([[1.0e20, 1.0, -1.0e20, 2.0] * 4], dtype=torch.float32),
        torch.tensor([[-1.0e20, 3.0, 1.0e20, 4.0] * 4], dtype=torch.float32),
        torch.tensor([[5.0, -1.0e20, 6.0, 1.0e20] * 4], dtype=torch.float32),
        torch.tensor([[7.0, 1.0e20, 8.0, -1.0e20] * 4], dtype=torch.float32),
    ]

    reduced = _fsdp2_ring_average(gradients)
    expected_shards = []
    shard_numel = gradients[0][0].numel() // len(gradients)
    for owner in range(len(gradients)):
        start = owner * shard_numel
        order = [(owner + offset) % len(gradients) for offset in range(1, 5)]
        shard = gradients[order[0]][0, start : start + shard_numel] / 2.0
        for source in order[1:]:
            shard = shard + gradients[source][0, start : start + shard_numel] / 2.0
        expected_shards.append(shard / 2.0)

    assert torch.equal(reduced, torch.cat(expected_shards).unsqueeze(0))

    uneven = [torch.arange(5, dtype=torch.float32).unsqueeze(0) + rank for rank in range(4)]
    uneven_reduced = _fsdp2_ring_average(uneven)
    assert uneven_reduced.shape == (1, 5)
    assert torch.equal(uneven_reduced, torch.arange(5, dtype=torch.float32).unsqueeze(0) + 1.5)


def test_lingbot_strict_ep_reduction_skips_already_reduced_experts():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/optimizer.py"
    ).read_text(encoding="utf-8")
    provider = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/provider.py"
    ).read_text(encoding="utf-8")

    assert "_strict_fsdp2_average_ep_dense_parameters" in source
    assert 'not getattr(parameter, "_fluxvla_grad_already_reduced", False)' in source
    assert "FLUXVLA_STRICT_EP_GATHER_CHUNK_NUMEL" in source
    assert 'if topology["EP"] != 1' not in provider


def test_fused_expert_checkpoint_expands_without_value_change():
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    fused = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    expected = {
        "layer.mlp.experts.0.gate_proj.weight": torch.empty(3, 4),
        "layer.mlp.experts.1.gate_proj.weight": torch.empty(3, 4),
    }
    expanded = LingBotVLAV2._expand_fused_expert_tensor(
        "layer.mlp.experts.gate_proj", fused, expected
    )
    assert expanded is not None
    assert torch.equal(expanded["layer.mlp.experts.0.gate_proj.weight"], fused[0])
    assert torch.equal(expanded["layer.mlp.experts.1.gate_proj.weight"], fused[1])


def test_lingbot_autograd_trace_is_opt_in(tmp_path, monkeypatch):
    from fluxvla.mcore.lingbot_vla_v2.provider import (
        _install_alignment_gradient_trace,
    )

    model = torch.nn.Linear(3, 2, bias=False)
    monkeypatch.setenv("FLUXVLA_AUTOGRAD_TENSOR_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("FLUXVLA_GRAD_TRACE_PATTERN", r"weight$")
    monkeypatch.setenv("RANK", "0")
    _install_alignment_gradient_trace(model)
    model(torch.ones(1, 3)).sum().backward()

    trace = torch.load(next(tmp_path.glob("*.pt")), weights_only=False)
    assert trace["backend"] == "mcore"
    assert trace["phase"] == "autograd"
    assert trace["name"] == "weight"
    assert torch.equal(trace["grad"], torch.ones(2, 3))


def test_lingbot_activation_trace_can_filter_ranks(tmp_path, monkeypatch):
    from fluxvla.models.vlas.lingbot_vla_v2 import LingBotVLAV2

    adapter = object.__new__(LingBotVLAV2)
    monkeypatch.setenv("FLUXVLA_LINGBOT_ACTIVATION_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("FLUXVLA_LINGBOT_TRACE_RANKS", "1,3")
    monkeypatch.setenv("RANK", "0")

    assert adapter._install_activation_trace_hooks() == ([], {}, [])


def test_lingbot_activation_trace_only_dumps_experts_when_requested():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/models/vlas/lingbot_vla_v2.py"
    ).read_text(encoding="utf-8")

    assert "trace_local_experts and is_local_expert" in source
    assert "LINGBOT_REPLAY_TRACE_ROUTER_ONLY" in source
    assert "LINGBOT_REPLAY_TRACE_MOE_ACTIVATIONS_ONLY" in source
    assert "moe_activations_only and is_moe_component" in source
    assert "trace_fused_router" in source
    assert "parent.register_forward_pre_hook(trace_fused_router)" in source


def test_lingbot_distopt_reduction_matches_fsdp2_predivide():
    from fluxvla.mcore.lingbot_vla_v2.optimizer import (
        _sum_then_postdivide_grad_sync,
    )

    bucket = SimpleNamespace(
        grad_data=torch.tensor([2.0, 4.0, 6.0, 8.0]),
        gradient_scaling_factor=0.5,
    )
    group = SimpleNamespace(
        buckets=[bucket],
        ddp_config=SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            num_distributed_optimizer_instances=1,
            average_in_collective=False,
        ),
        intra_distributed_optimizer_instance_size=2,
        intra_distributed_optimizer_instance_rank=1,
        grad_reduce_handle=None,
    )

    def fake_reduce_scatter(bucket_group):
        assert bucket_group.buckets[0].gradient_scaling_factor == 0.5
        bucket_group.buckets[0].grad_data.mul_(0.5)
        # Stand in for a SUM collective without overwriting the unowned shard.
        bucket_group.buckets[0].grad_data[2:].mul_(2.0)

    _sum_then_postdivide_grad_sync(group, fake_reduce_scatter)

    assert torch.equal(bucket.grad_data, torch.tensor([1.0, 2.0, 6.0, 8.0]))
    assert bucket.gradient_scaling_factor == 0.5


def test_lingbot_distopt_expert_reduction_keeps_dense_dp_scaling():
    from fluxvla.mcore.lingbot_vla_v2.optimizer import (
        _sum_then_postdivide_grad_sync,
    )

    bucket = SimpleNamespace(
        grad_data=torch.tensor([2.0, 4.0, 6.0, 8.0]),
        gradient_scaling_factor=0.25,
    )
    group = SimpleNamespace(
        buckets=[bucket],
        ddp_config=SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            num_distributed_optimizer_instances=1,
            average_in_collective=False,
        ),
        intra_distributed_optimizer_instance_size=2,
        intra_distributed_optimizer_instance_rank=1,
        grad_reduce_handle=None,
    )

    def fake_reduce_scatter(bucket_group):
        assert bucket_group.buckets[0].gradient_scaling_factor == 0.5
        bucket_group.buckets[0].grad_data.mul_(0.5)
        bucket_group.buckets[0].grad_data[2:].mul_(2.0)

    _sum_then_postdivide_grad_sync(group, fake_reduce_scatter)

    assert torch.equal(bucket.grad_data, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert bucket.gradient_scaling_factor == 0.25


def test_lingbot_distopt_skips_already_reduced_expert_bucket():
    from fluxvla.mcore.lingbot_vla_v2.optimizer import (
        _sum_then_postdivide_grad_sync,
    )

    parameter = torch.nn.Parameter(torch.ones(1))
    parameter._fluxvla_grad_already_reduced = True
    bucket = SimpleNamespace(
        grad_data=torch.tensor([1.0, 2.0]),
        gradient_scaling_factor=0.25,
        params_list=[parameter],
    )
    group = SimpleNamespace(
        buckets=[bucket],
        ddp_config=SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            num_distributed_optimizer_instances=1,
            average_in_collective=False,
        ),
        grad_reduce_handle=None,
    )

    def unexpected_collective(_bucket_group):
        raise AssertionError("already-reduced expert gradients must not be reduced twice")

    _sum_then_postdivide_grad_sync(group, unexpected_collective)

    assert torch.equal(bucket.grad_data, torch.tensor([1.0, 2.0]))
    assert group.grad_reduce_handle is None


def test_lingbot_distopt_uses_native_average_collective_when_enabled():
    from fluxvla.mcore.lingbot_vla_v2.optimizer import (
        _sum_then_postdivide_grad_sync,
    )

    bucket = SimpleNamespace(
        grad_data=torch.tensor([2.0, 4.0]),
        gradient_scaling_factor=1.0,
        params_list=[torch.nn.Parameter(torch.ones(1))],
    )
    group = SimpleNamespace(
        buckets=[bucket],
        ddp_config=SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_grad_reduce=False,
            num_distributed_optimizer_instances=1,
            average_in_collective=True,
        ),
        grad_reduce_handle=None,
    )
    called = []

    def average_collective(bucket_group):
        assert bucket_group is group
        assert bucket_group.buckets[0].gradient_scaling_factor == 1.0
        called.append(True)

    _sum_then_postdivide_grad_sync(group, average_collective)

    assert called == [True]
    assert torch.equal(bucket.grad_data, torch.tensor([2.0, 4.0]))


def test_lingbot_strict_fsdp2_units_claim_children_before_parents():
    from fluxvla.mcore.lingbot_vla_v2 import optimizer as optimizer_module

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.root_weight = torch.nn.Parameter(torch.ones(1))
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(2, 2, bias=False),
                torch.nn.LayerNorm(2),
            )

        def get_fsdp2_full_precision_modules(self):
            return [self.layer[1]]

        def get_fsdp2_wrapping_modules(self):
            return [self.layer]

        def forward(self, value):
            return self.layer(value)

    adapter = Adapter()
    optimizer_module.register_lingbot_fsdp2_reduction_units(adapter)
    units = optimizer_module._FSDP2_REDUCTION_UNITS

    assert [unit.name.split(":", 1)[0] for unit in units] == [
        "fp32_basic_0",
        "vlm_layer_0",
        "root",
    ]
    assert units[0].parameters == tuple(adapter.layer[1].parameters())
    assert units[1].parameters == (adapter.layer[0].weight,)
    assert units[2].parameters == (adapter.root_weight,)
    assert len({id(parameter) for unit in units for parameter in unit.parameters}) == 4
    assert all(
        parameter._fluxvla_autograd_grad_seen is False
        for unit in units
        for parameter in unit.parameters
    )
    adapter(torch.ones(1, 2)).sum().backward()
    assert adapter.layer[0].weight._fluxvla_autograd_grad_seen is True
    assert adapter.layer[1].weight._fluxvla_autograd_grad_seen is True
    assert adapter.layer[1].bias._fluxvla_autograd_grad_seen is True
    assert adapter.root_weight._fluxvla_autograd_grad_seen is False


def test_lingbot_strict_fsdp2_units_use_child_first_parameter_order():
    from fluxvla.mcore.lingbot_vla_v2 import optimizer as optimizer_module

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.direct = torch.nn.Parameter(torch.ones(1))
            self.child = torch.nn.Linear(2, 2)

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.root = torch.nn.Parameter(torch.ones(1))
            self.nested = Nested()

        def get_fsdp2_full_precision_modules(self):
            return [self.nested]

        def get_fsdp2_wrapping_modules(self):
            return []

    adapter = Adapter()
    optimizer_module.register_lingbot_fsdp2_reduction_units(adapter)
    units = optimizer_module._FSDP2_REDUCTION_UNITS

    assert units[0].parameters == (
        adapter.nested.child.weight,
        adapter.nested.child.bias,
        adapter.nested.direct,
    )
    assert units[1].parameters == (adapter.root,)


def test_lingbot_strict_fsdp2_reduction_defaults_off():
    from fluxvla.mcore.lingbot_vla_v2.provider import get_lingbot_args

    import argparse

    parser = argparse.ArgumentParser()
    get_lingbot_args(parser)
    args = parser.parse_args(
        [
            "--lingbot-native-config",
            "config.yaml",
            "--lingbot-checkpoint",
            "checkpoint",
            "--lingbot-tokenizer-path",
            "tokenizer",
            "--lingbot-manifest",
            "manifest.json",
        ]
    )

    assert args.lingbot_fsdp2_strict_dp_reduction is False


def test_lingbot_logging_metrics_use_one_packed_dp_collective(monkeypatch):
    from fluxvla.mcore.lingbot_vla_v2 import forward_step as forward_step_module

    calls = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        forward_step_module.parallel_state,
        "get_data_parallel_group",
        lambda: "dp-group",
    )
    monkeypatch.setattr(
        forward_step_module.parallel_state,
        "get_data_parallel_world_size",
        lambda: 2,
    )

    def all_reduce(tensor, group):
        calls.append((tensor.shape, group))
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    metrics = forward_step_module._dp_mean_metrics(
        {
            "loss": torch.tensor(1.25, dtype=torch.bfloat16),
            "aux": torch.tensor(2.5, dtype=torch.float32),
        }
    )

    assert calls == [(torch.Size([2]), "dp-group")]
    assert metrics.keys() == {"loss", "aux"}
    assert metrics["loss"].dtype == torch.float32
    assert metrics["loss"].item() == 1.25
    assert metrics["aux"].item() == 2.5


def test_lingbot_strict_fsdp2_reduces_before_microbatch_accumulation():
    source = (
        Path(__file__).parents[1]
        / "fluxvla/mcore/lingbot_vla_v2/optimizer.py"
    ).read_text(encoding="utf-8")

    assert "_install_strict_fsdp2_microbatch_reduction" in source
    assert "reduce_current_microbatch()" in source
    assert "parameter.main_grad.zero_()" in source
    assert "_refresh_strict_fsdp2_local_grad_norms()" in source
    assert "torch._foreach_norm(local_shards, 2.0)" in source
    assert "_FSDP2_MICROBATCH_REDUCTION_COMPLETE = True" in source
    assert "requires GA=2" not in source


def test_lingbot_strict_fsdp2_accumulates_arbitrary_microbatches():
    from fluxvla.mcore.lingbot_vla_v2.optimizer import (
        _accumulate_strict_microbatch_gradients,
    )

    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(2))
    for parameter in (first, second):
        parameter.main_grad = torch.zeros(2)
        parameter._fluxvla_autograd_grad_seen = True
        parameter._fluxvla_fsdp2_local_grad_norm = torch.tensor(1.0)
    accumulator = {}

    first.main_grad.copy_(torch.tensor([1.0, 2.0]))
    _accumulate_strict_microbatch_gradients(
        [first], [first, second], accumulator, final_microbatch=False
    )
    first.main_grad.copy_(torch.tensor([3.0, 4.0]))
    second.main_grad.copy_(torch.tensor([10.0, 20.0]))
    _accumulate_strict_microbatch_gradients(
        [first, second], [first, second], accumulator, final_microbatch=False
    )
    # The final microbatch activates only the first parameter. The second must
    # still be restored from its prior accumulated gradient.
    first.main_grad.copy_(torch.tensor([5.0, 6.0]))
    finalized = _accumulate_strict_microbatch_gradients(
        [first], [first, second], accumulator, final_microbatch=True
    )

    assert finalized == [first, second]
    assert accumulator == {}
    assert torch.equal(first.main_grad, torch.tensor([9.0, 12.0]))
    assert torch.equal(second.main_grad, torch.tensor([10.0, 20.0]))


def test_lingbot_distopt_reduction_rejects_overlap():
    from fluxvla.mcore.lingbot_vla_v2.optimizer import (
        _sum_then_postdivide_grad_sync,
    )

    group = SimpleNamespace(
        buckets=[],
        ddp_config=SimpleNamespace(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            num_distributed_optimizer_instances=1,
            average_in_collective=False,
        ),
    )

    try:
        _sum_then_postdivide_grad_sync(group, lambda _: None)
    except RuntimeError as error:
        assert "synchronous" in str(error)
    else:
        raise AssertionError("overlapped reduction must be rejected")
