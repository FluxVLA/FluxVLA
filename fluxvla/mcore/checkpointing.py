"""FluxVLA adapters for Megatron distributed checkpointing."""

from __future__ import annotations

from typing import Any


def _optimizer_id_to_model_parameter(
    optimizer: Any, optimizer_state_dict: dict
) -> tuple[dict[int, Any], dict[int, int]]:
    """Map serialized optimizer IDs to their original model parameters."""
    main_to_model = {
        id(main_parameter): model_parameter
        for model_group, main_group in zip(
            optimizer.float16_groups,
            optimizer.fp32_from_float16_groups,
        )
        for model_parameter, main_parameter in zip(model_group, main_group)
    }
    state_groups = optimizer_state_dict["optimizer"]["param_groups"]
    runtime_groups = optimizer.optimizer.param_groups
    if len(state_groups) != len(runtime_groups):
        raise ValueError(
            "optimizer runtime/state parameter group count differs: "
            f"{len(runtime_groups)} != {len(state_groups)}"
        )

    id_to_model_parameter = {}
    runtime_parameter_to_id = {}
    for runtime_group, state_group in zip(runtime_groups, state_groups):
        runtime_parameters = runtime_group["params"]
        state_ids = state_group["params"]
        if len(runtime_parameters) != len(state_ids):
            raise ValueError(
                "optimizer runtime/state parameter count differs within a group: "
                f"{len(runtime_parameters)} != {len(state_ids)}"
            )
        for runtime_parameter, parameter_id in zip(runtime_parameters, state_ids):
            model_parameter = main_to_model.get(
                id(runtime_parameter), runtime_parameter
            )
            id_to_model_parameter[parameter_id] = model_parameter
            runtime_parameter_to_id[id(runtime_parameter)] = parameter_id
    return id_to_model_parameter, runtime_parameter_to_id


def _mixed_precision_sharded_state_dict(
    optimizer: Any,
    model_sharded_state_dict: dict,
    *,
    is_loading: bool = False,
) -> dict:
    """Build a sharded state dict for mixed BF16-master and native FP32 params."""
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.optimizer import (
        make_sharded_optimizer_tensor,
        optim_state_to_sharding_state,
    )
    from megatron.core.dist_checkpointing.utils import (
        extract_sharded_tensors_and_factories,
    )
    from megatron.core.utils import to_local_if_dtensor

    if is_loading:
        optimizer.init_state_fn(optimizer.optimizer, optimizer.config)

    state_dict = optimizer.state_dict()
    id_to_model_parameter, runtime_parameter_to_id = _optimizer_id_to_model_parameter(
        optimizer, state_dict
    )
    model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(
        model_sharded_state_dict
    )
    sharded_by_parameter = {
        id(to_local_if_dtensor(sharded.data)): sharded
        for sharded in nested_values(model_sharded_state_dict)
    }
    id_to_sharded_parameter = {
        parameter_id: sharded_by_parameter[id(to_local_if_dtensor(model_parameter))]
        for parameter_id, model_parameter in id_to_model_parameter.items()
        if id(to_local_if_dtensor(model_parameter)) in sharded_by_parameter
    }
    missing_ids = sorted(
        parameter_id
        for parameter_id in state_dict["optimizer"]["state"]
        if parameter_id not in id_to_sharded_parameter
    )
    if missing_ids:
        raise ValueError(
            "optimizer parameters do not match model sharded parameters; "
            f"first missing IDs: {missing_ids[:8]}"
        )

    state_dict["fp32_from_fp16_params"] = [
        [
            make_sharded_optimizer_tensor(
                id_to_sharded_parameter[runtime_parameter_to_id[id(main_parameter)]],
                main_parameter,
                prefix="optimizer.state.fp32_param",
            )
            for main_parameter in main_group
        ]
        for main_group in optimizer.fp32_from_float16_groups
    ]

    step = optimizer._extract_common_per_param_step(state_dict["optimizer"])
    optim_state_to_sharding_state(
        state_dict["optimizer"],
        id_to_sharded_parameter,
        exclude_keys=("step",),
    )
    if step is not None:
        state_dict["optimizer"]["state"]["common_step"] = step
    return state_dict


def install_mixed_precision_optimizer_checkpoint_adapter() -> None:
    """Patch the MCore mixed optimizer's torch_dist state mapping when needed."""
    import numpy as np
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    # Megatron releases predating NumPy 2 use the removed ``np.product`` alias
    # in torch_dist validation and fully-parallel exchange code.
    if getattr(np, "product", None) is None:
        np.product = np.prod

    original = Float16OptimizerWithFloat16Params.sharded_state_dict
    if getattr(original, "_fluxvla_mixed_checkpoint", False):
        return

    def sharded_state_dict(
        optimizer,
        model_sharded_state_dict,
        is_loading=False,
        metadata=None,
    ):
        del metadata
        if not any(optimizer.fp32_from_fp32_groups):
            return original(
                optimizer,
                model_sharded_state_dict,
                is_loading=is_loading,
            )
        return _mixed_precision_sharded_state_dict(
            optimizer,
            model_sharded_state_dict,
            is_loading=is_loading,
        )

    sharded_state_dict._fluxvla_mixed_checkpoint = True
    Float16OptimizerWithFloat16Params.sharded_state_dict = sharded_state_dict


__all__ = ["install_mixed_precision_optimizer_checkpoint_adapter"]
