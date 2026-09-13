# Copyright 2026 Limx Dynamics
"""DreamZero loss/scheduler/cache contracts with a deterministic CPU DiT stub.

Only the expensive Wan network is replaced; the head and schedulers are real.
Every test builds a fresh head, so causal history cannot leak between tests.
"""

import pytest
import torch
from torch import nn

from fluxvla.models.heads import dreamzero_head as module
from fluxvla.models.third_party_models.dreamzero.modules import \
    flow_match_scheduler


class _TinyWan(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.dim, self.num_heads, self.num_layers = 4, 2, 1
        self.local_attn_size = -1
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, video, action=None, kv_cache=None, **kwargs):
        video_prediction = torch.zeros_like(video) + self.weight
        action_prediction = (None if action is None else
                             torch.zeros_like(action) + self.weight)
        if kv_cache is None:
            return video_prediction, action_prediction
        updated = [
            torch.cat([
                cache,
                cache.new_ones(2, video.shape[0], video.shape[2],
                               self.num_heads, 2)
            ],
                      dim=2) for cache in kv_cache
        ]
        return video_prediction, action_prediction, updated


@pytest.fixture
def head(monkeypatch):
    monkeypatch.setattr(
        module, '_import_dreamzero_modules', lambda:
        (_TinyWan, flow_match_scheduler.FlowMatchScheduler))
    return module.DreamZeroHead(
        action_dim=2,
        max_action_dim=4,
        action_horizon=4,
        max_state_dim=4,
        num_action_per_block=4,
        use_gradient_checkpointing=False)


@pytest.fixture
def inputs():
    return dict(
        prompt_embs=torch.ones(1, 2, 4),
        latents=torch.zeros(1, 3, 2, 2, 2),  # B, T, C, H, W
        clip_feas=torch.ones(1, 4),
        ys=torch.zeros(1, 4, 5, 2, 2),
        states=torch.zeros(1, 1, 4),
        embodiment_ids=torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize('valid_dims', [0, 2])
def test_training_loss_masks_and_backward(head, inputs, monkeypatch,
                                          valid_dims):
    monkeypatch.setattr(torch, 'randn_like',
                        lambda value: torch.full_like(value, 0.75))
    monkeypatch.setattr(
        torch, 'randint',
        lambda lo, hi, shape: torch.full(shape, 500, dtype=torch.long))
    inputs['latents'] = inputs['latents'].transpose(1, 2)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[:, :, :valid_dims] = True
    output = head(**inputs, actions=torch.zeros(1, 4, 4), action_masks=mask)
    weight = head.scheduler.training_weight(
        head.scheduler.timesteps[500:501])[0]
    torch.testing.assert_close(output['action_loss'],
                               weight * 0.25 * valid_dims / 4)
    assert torch.isfinite(output['loss'])
    torch.testing.assert_close(output['loss'],
                               output['dynamics_loss'] + output['action_loss'])
    output['loss'].backward()
    assert torch.isfinite(head.model.weight.grad)
    assert head.model.weight.grad.abs() > 0


def test_stateless_prediction_is_repeatable_and_keeps_no_history(head, inputs):
    head.eval()
    with torch.no_grad():
        torch.manual_seed(17)
        first = head.predict_action(num_inference_steps=2, **inputs)
        torch.manual_seed(17)
        second = head.predict_action(num_inference_steps=2, **inputs)
    assert first.shape == (1, 4, 4)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert head.inference_kv_cache is None
    assert head.current_start_frame == 0


def test_cache_reuses_history_and_explicit_reset_restarts_it(head, inputs):
    head.use_cache = True
    with torch.no_grad():
        head.predict_action(num_inference_steps=2, **inputs)
        cache = head.inference_kv_cache
        assert head.current_start_frame == 3
        assert cache[0].shape[2] == 1

        head.predict_action(num_inference_steps=2, **inputs)
        assert head.inference_kv_cache is cache
        assert cache[0].shape[2] == 3
        assert head.current_start_frame == 5

        head.predict_action(
            num_inference_steps=2,
            reset_history=True,
            observed_latent_frames=1,
            **inputs)
        assert head.inference_kv_cache is not cache
        assert head.inference_kv_cache[0].shape[2] == 1
        assert head.current_start_frame == 3


def test_changed_prompt_invalidates_cache(head, inputs):
    head.use_cache = True
    with torch.no_grad():
        head.predict_action(num_inference_steps=2, **inputs)
        cache = head.inference_kv_cache
        inputs['prompt_embs'] = inputs['prompt_embs'] + 1
        head.predict_action(num_inference_steps=2, **inputs)
    assert head.inference_kv_cache is not cache
    assert head.current_start_frame == 3
