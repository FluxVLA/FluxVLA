# Copyright 2026 Limx Dynamics
"""Real CPU action transformers: loss arithmetic, gradients and inference."""

import pytest
import torch

from fluxvla.engines import build_head_from_cfg
from fluxvla.models.blocks.cross_attention_dit import DiT


def _assert_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(grad.abs().sum() > 0 for grad in gradients)


@pytest.mark.parametrize(('state_dim', 'add_pos_embed'), [(0, False),
                                                          (4, True)])
def test_tiny_dit4dit_head_loss_backward_and_predict(state_dim, add_pos_embed):
    config = dict(
        type='DiT4DiTActionHead',
        action_dim=4,
        ori_action_dim=3,
        hidden_size=16,
        state_dim=state_dim,
        action_horizon=3,
        num_inference_timesteps=2,
        add_pos_embed=add_pos_embed,
        max_seq_len=8,
        diffusion_model_cfg=dict(
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=2,
            cross_attention_dim=16,
            interleave_self_attention=True,
            output_dim=16,
            dropout=0.0,
            final_dropout=False))
    head = build_head_from_cfg(config).eval()
    assert isinstance(head.model, DiT)
    features = torch.linspace(-1, 1, 128).reshape(2, 4, 16).requires_grad_()
    states = (
        torch.linspace(-0.5, 0.5, 8).reshape(2, 4).requires_grad_()
        if state_dim else None)
    actions = torch.linspace(-0.5, 0.5, 24).reshape(2, 3, 4)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[..., -1] = False
    mask[0, -1] = False

    # Reproduce only the RNG input, not the DiT or its predictions. A read-only
    # hook exposes the real decoder output for an independent loss reduction.
    with torch.random.fork_rng(devices=[]):
        noise = torch.randn(actions.shape)
    predictions = []

    def record_prediction(module, inputs, output):
        predictions.append(output)

    handle = head.action_decoder.register_forward_hook(record_prediction)
    try:
        loss = head(
            input_features=features,
            states=states,
            actions=actions,
            action_masks=mask)
    finally:
        handle.remove()
    expected = (predictions[0][:, -3:] -
                (noise - actions)).square()[mask].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    for module in (head.model, head.action_encoder, head.action_decoder):
        _assert_gradients(module)
    if states is not None:
        assert torch.isfinite(
            states.grad).all() and states.grad.abs().sum() > 0

    restored = build_head_from_cfg(config).eval()
    restored.load_state_dict(head.state_dict(), strict=True)
    results = []
    for model in (head, restored):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            results.append(
                model.predict_action(
                    input_features=features.detach(),
                    states=None if states is None else states.detach()))
    assert results[0].shape == (2, 3, 3)
    assert torch.isfinite(results[0]).all()
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@pytest.mark.parametrize('weighted', [False, True])
def test_tiny_llava_action_head_masked_loss_backward_and_predict(weighted):
    head = build_head_from_cfg(
        dict(
            type='LlavaActionHead',
            hidden_size=16,
            state_dim=4,
            num_layers=1,
            num_heads=2,
            traj_length=3,
            action_dim=3,
            act_decoder_dim=16,
            max_seq_len=8)).eval()
    features = torch.linspace(-1, 1, 128).reshape(2, 4, 16).requires_grad_()
    states = torch.linspace(-0.5, 0.5, 8).reshape(2, 4).requires_grad_()
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    actions = torch.linspace(-0.5, 0.5, 18).reshape(2, 3, 3)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    weights = torch.tensor([0.5, 2.0]) if weighted else None
    result = head(
        input_features=features,
        states=states,
        attention_mask=attention_mask,
        actions=actions,
        action_masks=mask,
        sample_weight=weights)
    errors = (result['pred_actions'] - actions).square()
    valid = mask[..., None].expand_as(errors).float()
    if weights is not None:
        valid = valid * weights[:, None, None]
    expected = (errors * valid).sum() / valid.sum()
    torch.testing.assert_close(result['loss'], expected)
    result['loss'].backward()
    _assert_gradients(head.decode_action)
    _assert_gradients(head.state_encoder)
    assert features.grad.abs().sum() > 0 and torch.isfinite(
        features.grad).all()
    assert states.grad.abs().sum() > 0 and torch.isfinite(states.grad).all()
    with torch.no_grad():
        actual = head.predict_action(features, states, attention_mask)
        # Padded features must not become the last valid language token.
        changed = features.detach().clone()
        changed[0, 2:] = 1000
        padded = head.predict_action(changed, states, attention_mask)
    torch.testing.assert_close(actual, result['pred_actions'], rtol=0, atol=0)
    torch.testing.assert_close(padded, actual, rtol=0, atol=0)


@pytest.mark.parametrize('with_prefix', [False, True])
def test_tiny_flow_head_seeded_inference_and_prefix(with_prefix):
    head = build_head_from_cfg(
        dict(
            type='FlowMatchingHead',
            state_dim=4,
            hidden_size=16,
            input_embedding_dim=16,
            backbone_embedding_dim=16,
            num_inference_timesteps=2,
            num_steps=3,
            action_dim=4,
            ori_action_dim=3,
            max_num_embodiments=1,
            num_target_vision_tokens=2,
            max_seq_len=8,
            vl_self_attention_cfg=dict(
                attention_head_dim=8,
                num_attention_heads=2,
                num_layers=1,
                dropout=0.0,
                final_dropout=False),
            diffusion_model_cfg=dict(
                attention_head_dim=8,
                num_attention_heads=2,
                num_layers=2,
                cross_attention_dim=16,
                output_dim=16,
                dropout=0.0,
                final_dropout=False))).eval()
    assert isinstance(head.model, DiT)
    inputs = dict(
        input_features=torch.linspace(-1, 1, 128).reshape(2, 4, 16),
        states=torch.linspace(-0.5, 0.5, 8).reshape(2, 4),
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
        embodiment_ids=torch.zeros(2, dtype=torch.long),
        seed=23)
    previous = torch.linspace(-0.5, 0.5, 18).reshape(2, 3, 3)
    if with_prefix:
        inputs.update(
            prev_actions=previous,
            prefix_len=1,
            rtc_config=dict(method='prefix'))
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = head.predict_action(**inputs)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        torch.rand(
            11)  # An unrelated random draw must not affect explicit seed.
        second = head.predict_action(**inputs)
    assert first.shape == (2, 3, 3) and torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    if with_prefix:
        torch.testing.assert_close(
            first[:, :1], previous[:, :1], rtol=0, atol=0)
