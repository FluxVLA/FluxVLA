from types import SimpleNamespace

import pytest
import torch

from fluxvla.mcore.checkpointing import _optimizer_id_to_model_parameter


def _parameter(value: float) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.tensor([value]))


def test_optimizer_id_mapping_preserves_mixed_group_order():
    bf16_model = _parameter(1.0)
    bf16_main = _parameter(1.0)
    fp32_model = _parameter(2.0)
    optimizer = SimpleNamespace(
        float16_groups=[[bf16_model]],
        fp32_from_float16_groups=[[bf16_main]],
        fp32_from_fp32_groups=[[fp32_model]],
        optimizer=SimpleNamespace(param_groups=[{"params": [fp32_model, bf16_main]}]),
    )
    state_dict = {"optimizer": {"param_groups": [{"params": [41, 73]}]}}

    id_to_model, runtime_to_id = _optimizer_id_to_model_parameter(optimizer, state_dict)

    assert id_to_model[41] is fp32_model
    assert id_to_model[73] is bf16_model
    assert runtime_to_id[id(fp32_model)] == 41
    assert runtime_to_id[id(bf16_main)] == 73


def test_optimizer_id_mapping_rejects_group_size_mismatch():
    optimizer = SimpleNamespace(
        float16_groups=[[]],
        fp32_from_float16_groups=[[]],
        optimizer=SimpleNamespace(param_groups=[{"params": [_parameter(1.0)]}]),
    )
    state_dict = {"optimizer": {"param_groups": [{"params": []}]}}

    with pytest.raises(ValueError, match="parameter count differs"):
        _optimizer_id_to_model_parameter(optimizer, state_dict)


def test_pi05_recipe_installs_checkpoint_adapter_before_training_hooks():
    from fluxvla.mcore.pi05.recipe import PI05_SPEC

    names = [hook.__name__ for hook in PI05_SPEC.setup_hooks]
    assert names[0] == "install_mixed_precision_optimizer_checkpoint_adapter"


def test_checkpoint_adapter_restores_numpy_product_alias(monkeypatch):
    import numpy as np

    from fluxvla.mcore.checkpointing import (
        install_mixed_precision_optimizer_checkpoint_adapter,
    )

    monkeypatch.delattr(np, "product", raising=False)
    install_mixed_precision_optimizer_checkpoint_adapter()

    assert np.product is np.prod
