# Copyright 2026 Limx Dynamics

from types import SimpleNamespace

import torch
import torch.nn as nn

from fluxvla.models.backbones.vlms.cosmos25 import Cosmos25Backbone
from fluxvla.models.vlas.dit4dit_vla import DiT4DiTVLA


class _RecordingBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.last_lang_tokens = None
        self.last_lang_masks = None
        self.last_prompts = None

    def forward(self,
                images,
                prompts=None,
                lang_tokens=None,
                lang_masks=None,
                **kwargs):
        self.last_prompts = prompts
        self.last_lang_tokens = lang_tokens
        self.last_lang_masks = lang_masks
        hidden = images.new_ones(images.shape[0], 3, 4)
        return SimpleNamespace(hidden_states=[hidden], future_video_loss=None)


class _FakeActionHead(nn.Module):
    action_dim = 2
    action_horizon = 2
    state_dim = 0

    def forward(self, features, actions, action_mask, state=None):
        return features.sum() * 0 + actions.sum() * 0 + 1

    def predict_action(self, features, state=None):
        return features.new_zeros(features.shape[0], self.action_horizon,
                                  self.action_dim)


class _FakeTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.last_input_ids = None

    def forward(self, input_ids, output_hidden_states):
        assert output_hidden_states is True
        self.last_input_ids = input_ids
        values = input_ids.float()
        hidden_one = torch.stack((values, values + 1), dim=-1)
        hidden_two = torch.stack((values * 2, values * 2 + 1), dim=-1)
        return SimpleNamespace(hidden_states=(values, hidden_one, hidden_two))


def _build_lightweight_vla():
    vla = DiT4DiTVLA.__new__(DiT4DiTVLA)
    nn.Module.__init__(vla)
    vla.vlm_backbone = _RecordingBackbone()
    vla.vla_head = _FakeActionHead()
    vla.repeated_diffusion_steps = 1
    vla.image_layout = 'auto'
    vla.multiview_strategy = 'tile'
    return vla


def test_dit4dit_forward_and_predict_consume_transform_tokens():
    vla = _build_lightweight_vla()
    images = torch.zeros(2, 3, 8, 8)
    lang_tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    lang_masks = lang_tokens.ne(0)
    actions = torch.zeros(2, 2, 2)

    output = vla(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        actions=actions,
    )
    predicted = vla.predict_action(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
    )

    assert output['loss'].item() == 1
    assert tuple(predicted.shape) == (2, 2, 2)
    assert vla.vlm_backbone.last_prompts is None
    assert torch.equal(vla.vlm_backbone.last_lang_tokens, lang_tokens)
    assert torch.equal(vla.vlm_backbone.last_lang_masks, lang_masks)


def test_cosmos_backbone_encodes_precomputed_token_ids():
    backbone = Cosmos25Backbone.__new__(Cosmos25Backbone)
    nn.Module.__init__(backbone)
    backbone.text_encoder = _FakeTextEncoder()
    lang_tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])

    prompt_embeds = backbone._get_prompt_embeds(
        prompts=None,
        device=torch.device('cpu'),
        dtype=torch.float32,
        input_ids=lang_tokens,
    )

    assert tuple(prompt_embeds.shape) == (2, 3, 4)
    assert torch.equal(backbone.text_encoder.last_input_ids, lang_tokens)
