# Copyright 2026 Limx Dynamics
"""Small, real checkpoint round trips in place of pretrained GPU snapshots."""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from fluxvla.models.vlas.base_vla import BaseVLA


class _TinyVLA(BaseVLA):

    def __init__(self, **kwargs):
        super().__init__(strict_mapping=True, **kwargs)
        self.projector = nn.Linear(3, 2)
        self.register_buffer('scale', torch.tensor(0.5))

    def forward(self, inputs):
        return self.projector(inputs) * self.scale


def _weights():
    return {
        'projector.weight': torch.arange(6, dtype=torch.float32).reshape(2, 3),
        'projector.bias': torch.tensor([-1.0, 1.0]),
        'scale': torch.tensor(2.0),
    }


@pytest.mark.parametrize('format',
                         ['pt', 'wrapped_pt', 'safetensors', 'shards'])
def test_checkpoint_restores_parameters_and_buffers(tmp_path, format):
    weights = _weights()
    if format == 'shards':
        path = tmp_path
        save_file({'scale': weights['scale']},
                  str(path / 'part-1.safetensors'))
        save_file({k: v
                   for k, v in weights.items() if k != 'scale'},
                  str(path / 'part-2.safetensors'))
    elif format == 'safetensors':
        path = tmp_path / 'model.safetensors'
        save_file(weights, str(path))
    else:
        path = tmp_path / 'model.pt'
        torch.save({'model': weights} if format == 'wrapped_pt' else weights,
                   path)

    model = _TinyVLA(pretrained_name_or_path=str(path))
    model.from_pretrained()
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, weights[name], rtol=0, atol=0)
        assert actual.device.type == 'cpu'
    torch.testing.assert_close(
        model(torch.ones(1, 3)), torch.tensor([[4.0, 26.0]]), rtol=0, atol=0)


def test_checkpoint_name_mapping_and_strict_missing_key(tmp_path):
    path = tmp_path / 'mapped.pt'
    weights = _weights()
    torch.save(
        {k.replace('projector', 'source'): v
         for k, v in weights.items()}, path)
    model = _TinyVLA(
        pretrained_name_or_path=str(path),
        name_mapping={'projector': 'source'})
    model.from_pretrained()
    torch.testing.assert_close(
        model.projector.weight, weights['projector.weight'], rtol=0, atol=0)
    torch.testing.assert_close(
        model.projector.bias, weights['projector.bias'], rtol=0, atol=0)

    torch.save({'projector.weight': weights['projector.weight']}, path)
    model.name_mapping = None
    with pytest.raises(RuntimeError, match='Missing key'):
        model.from_pretrained()
