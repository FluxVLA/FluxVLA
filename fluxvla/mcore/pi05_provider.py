"""PI0.5 model provider backed by Megatron-Core tensor-parallel layers.

Megatron-LM is imported lazily and is never modified.  The provider accepts an
already constructed FluxVLA PI0.5 model, so checkpoint loading, tokenizer
construction and the PI0.5 flow-matching loss remain owned by FluxVLA.  Linear
projections are converted to native MCore ``ColumnParallelLinear`` modules
after loading the full checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class MCoreModelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    params_dtype: torch.dtype = torch.float32


def _mcore_linear(source: nn.Linear, config: MCoreModelConfig, group):
    """Create and populate one native MCore column-parallel projection."""
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    mcore_config = ModelParallelConfig(
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        sequence_parallel=config.sequence_parallel,
        context_parallel_size=config.context_parallel_size,
        expert_model_parallel_size=config.expert_model_parallel_size,
        params_dtype=source.weight.dtype,
        perform_initialization=False,
    )
    layer = ColumnParallelLinear(
        source.in_features,
        source.out_features,
        config=mcore_config,
        init_method=lambda tensor: nn.init.zeros_(tensor),
        bias=source.bias is not None,
        gather_output=True,
        tp_group=group,
    )
    rank = torch.distributed.get_rank(group)
    world = torch.distributed.get_world_size(group)
    rows = source.out_features // world
    start, end = rank * rows, (rank + 1) * rows
    with torch.no_grad():
        layer.weight.copy_(source.weight.detach()[start:end])
        if source.bias is not None:
            layer.bias.copy_(source.bias.detach()[start:end])
    layer.weight.requires_grad_(source.weight.requires_grad)
    if layer.bias is not None:
        layer.bias.requires_grad_(source.bias.requires_grad)
    return MCoreLinearAdapter(layer)


class MCoreLinearAdapter(nn.Module):
    """Restore the ``nn.Linear`` tensor-only return contract."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        return output[0] if isinstance(output, tuple) else output


def _convert_linears(module: nn.Module, config: MCoreModelConfig, group,
                     minimum_out_features: int) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if child.out_features >= minimum_out_features:
                if child.out_features % config.tensor_model_parallel_size:
                    raise ValueError(
                        f"{name} output dimension {child.out_features} is not "
                        f"divisible by TP={config.tensor_model_parallel_size}")
                setattr(module, name, _mcore_linear(child, config, group))
                count += 1
        else:
            count += _convert_linears(child, config, group,
                                      minimum_out_features)
    return count


class MCorePI05ModelProvider(nn.Module):
    """Provider exposing a native-MCore PI0.5 module contract.

    ``sequence_parallel`` is recorded in the MCore layer configuration and
    exposed to the training entry.  PI0.5's multimodal prefix/suffix attention
    requires a model-specific sequence boundary implementation; until that
    implementation is enabled, the provider deliberately keeps the full
    sequence on each TP rank and raises when a caller requests unsafe context
    parallelism.
    """

    def __init__(self, model: nn.Module, config: MCoreModelConfig,
                 tp_group=None, minimum_out_features: int = 128):
        super().__init__()
        if config.pipeline_model_parallel_size != 1:
            raise NotImplementedError("PI0.5 PP provider is planned for phase 2")
        if config.context_parallel_size != 1:
            raise NotImplementedError("PI0.5 CP provider is planned for phase 2")
        self.model = model
        self.config = config
        self.tp_group = tp_group
        self.converted_linear_count = 0
        if config.tensor_model_parallel_size > 1:
            if tp_group is None:
                raise RuntimeError("TP provider requires an initialized TP group")
            self.converted_linear_count = _convert_linears(
                self.model, config, tp_group, minimum_out_features)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def freeze_backbones(self):
        return self.model.freeze_backbones()

    def from_pretrained(self):
        return self.model.from_pretrained()

    def __getattr__(self, name):
        if name in {"model", "config", "tp_group", "converted_linear_count"}:
            return super().__getattr__(name)
        return getattr(self.model, name)


def build_pi05_mcore_provider(model: nn.Module, runtime, *,
                              sequence_parallel: bool = False,
                              minimum_out_features: int = 128):
    """Build a provider from an initialized :class:`MegatronRuntime`."""
    config = MCoreModelConfig(
        tensor_model_parallel_size=runtime.config.tensor_model_parallel_size,
        pipeline_model_parallel_size=runtime.config.pipeline_model_parallel_size,
        sequence_parallel=sequence_parallel,
        context_parallel_size=runtime.config.context_parallel_size,
        expert_model_parallel_size=runtime.config.expert_model_parallel_size,
    )
    return MCorePI05ModelProvider(
        model,
        config,
        runtime.tensor_model_parallel_group(),
        minimum_out_features=minimum_out_features,
    )
