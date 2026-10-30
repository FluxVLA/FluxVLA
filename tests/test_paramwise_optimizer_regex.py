from types import SimpleNamespace

import torch

from fluxvla.optimizers.lr_scheduler_policies import BaseLRSchedulerPolicy


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].mlp = torch.nn.Module()
        self.layers[0].mlp.experts = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False)]
        )
        self.layers[0].mlp.shared_expert = torch.nn.Linear(2, 2, bias=False)


def test_paramwise_learning_rate_regex_only_scales_routed_experts():
    model = _Model()
    runner = SimpleNamespace(
        vla=model,
        optimizer_cfg={
            "lr": 1e-4,
            "weight_decay": 0.0,
            "paramwise_learning_rate": {},
            "paramwise_learning_rate_regex": {
                r"(?:^|\.)layers\.\d+\.mlp\.experts\.": 2.8284271247461903e-4,
            },
        },
    )

    groups = BaseLRSchedulerPolicy().build_param_groups(runner)
    parameter_lrs = {
        id(parameter): group["lr"]
        for group in groups
        for parameter in group["params"]
    }

    assert parameter_lrs[id(model.layers[0].mlp.experts[0].weight)] == 2.8284271247461903e-4
    assert parameter_lrs[id(model.layers[0].mlp.shared_expert.weight)] == 1e-4
