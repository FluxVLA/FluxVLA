"""Regression tests for shape-dependent PI0 flow-conditioning paths."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching
from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching


def _make_pi0_suffix_model():
    model = PI0FlowMatching.__new__(PI0FlowMatching)
    nn.Module.__init__(model)
    model.state_proj = nn.Linear(3, 4)
    model.action_in_proj = nn.Linear(2, 4)
    model.action_time_mlp_in = nn.Linear(8, 4)
    model.action_time_mlp_out = nn.Linear(4, 4)
    model.proj_width = 4
    model.n_action_steps = 8
    return model


def _make_pi05_suffix_model():
    model = PI05FlowMatching.__new__(PI05FlowMatching)
    nn.Module.__init__(model)
    model.state_proj = None
    model.action_in_proj = nn.Linear(2, 4)
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model.proj_width = 4
    model.n_action_steps = 8
    model.openpi_fp32_flow = False
    return model


@pytest.mark.parametrize(
    ('factory', 'expected_attention'),
    [
        (_make_pi0_suffix_model, [True, True, False, False]),
        (_make_pi05_suffix_model, [True, False, False]),
    ],
)
def test_suffix_attention_mask_matches_runtime_horizon(factory,
                                                       expected_attention):
    """A truncated action window must keep the flow attention mask aligned."""
    model = factory()
    states = torch.randn(2, 3)
    actions = torch.randn(2, 3, 2)
    timestep = torch.full((2, ), 0.5)

    embs, pad_masks, att_masks, _ = model.embed_suffix(states, actions,
                                                       timestep)

    assert embs.shape == (2, len(expected_attention), 4)
    assert pad_masks.shape == embs.shape[:2]
    assert att_masks.shape == embs.shape[:2]
    assert att_masks[0].tolist() == expected_attention
    assert torch.all(pad_masks)


@pytest.mark.parametrize('factory',
                         [_make_pi0_suffix_model, _make_pi05_suffix_model])
def test_suffix_rejects_empty_action_window(factory):
    model = factory()
    states = torch.randn(1, 3)
    actions = torch.empty(1, 0, 2)
    timestep = torch.full((1, ), 0.5)

    with pytest.raises(ValueError, match='at least one step'):
        model.embed_suffix(states, actions, timestep)


class _TinyNorm(nn.Module):

    def forward(self, hidden_states, cond=None):
        return hidden_states, None


class _TinyMLP(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.up_proj(hidden_states)


class _TinyAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        projection_size = num_heads * head_dim
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)


class _TinyLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.input_layernorm = _TinyNorm()
        self.self_attn = _TinyAttention(hidden_size, num_heads, head_dim)
        self.post_attention_layernorm = _TinyNorm()
        self.mlp = _TinyMLP(hidden_size)


class _TinyRotary:

    def __call__(self, dummy_tensor, position_ids):
        shape = (*position_ids.shape, dummy_tensor.shape[-1])
        return (torch.ones(
            shape, device=dummy_tensor.device, dtype=dummy_tensor.dtype),
                torch.zeros(
                    shape,
                    device=dummy_tensor.device,
                    dtype=dummy_tensor.dtype))


def test_joint_flow_forward_uses_actual_attention_width():
    """Joint PI0 execution must not assume PaliGemma's eight heads."""
    model = PI0FlowMatching.__new__(PI0FlowMatching)
    nn.Module.__init__(model)

    # Two heads are sufficient to expose the old hard-coded eight-head view.
    hidden_size, num_heads, head_dim = 4, 2, 2
    vlm_layer = _TinyLayer(hidden_size, num_heads, head_dim)
    expert_layer = _TinyLayer(hidden_size, num_heads, head_dim)
    model.llm_backbone = SimpleNamespace(
        layers=[vlm_layer], rotary_emb=_TinyRotary(), norm=_TinyNorm())
    model.llm_expert = SimpleNamespace(layers=[expert_layer], norm=_TinyNorm())

    def fake_attention(module, query, key, value, mask, scaling):
        del module, key, value, mask, scaling
        return torch.zeros(query.shape[0], query.shape[2], query.shape[1],
                           query.shape[3]), None

    model.attention_interface = fake_attention
    prefix = torch.randn(1, 2, hidden_size)
    suffix = torch.randn(1, 1, hidden_size)
    position_ids = torch.arange(3, dtype=torch.long)[None]

    outputs = model._forward_transformer_layers(
        inputs_embeds=[prefix, suffix],
        attention_masks=None,
        position_ids=position_ids,
        models=[model.llm_backbone, model.llm_expert],
        num_layers=1,
        adarms_cond=[None, None],
    )

    assert outputs[0].shape == prefix.shape
    assert outputs[1].shape == suffix.shape
    assert torch.isfinite(outputs[0]).all()
    assert torch.isfinite(outputs[1]).all()
