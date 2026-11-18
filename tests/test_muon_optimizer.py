import torch

from fluxvla.optimizers.muon import CombinedOptimizer, build_muon_optimizer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.ones(4, 4))
        self.bias = torch.nn.Parameter(torch.ones(4))
        self.embed_tokens = torch.nn.Parameter(torch.ones(8, 4))
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].mlp = torch.nn.Module()
        self.layers[0].mlp.experts = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(4, 4))])


def test_muon_splits_matrix_vectors_embeddings_and_experts():
    model = TinyModel()
    optimizer = build_muon_optimizer(model, {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "token_num_experts": 32,
        "token_top_k": 4,
    })
    assert isinstance(optimizer, CombinedOptimizer)
    assert len(optimizer.optimizers) == 2
    assert len(optimizer.optimizers[0].param_groups) == 2
    assert len(optimizer.optimizers[1].param_groups) == 1


def test_muon_step_updates_matrix_and_adamw_updates_vector():
    model = TinyModel()
    optimizer = build_muon_optimizer(model, {"lr": 1e-4, "weight_decay": 0.0})
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer.step()
    assert not torch.equal(model.matrix, before["matrix"])
    assert not torch.equal(model.bias, before["bias"])
    assert "momentum_buffer" in optimizer.optimizers[0].state[model.matrix]
    assert "exp_avg" in optimizer.optimizers[1].state[model.bias]
