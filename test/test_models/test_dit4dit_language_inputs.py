# Copyright 2026 Limx Dynamics

from types import SimpleNamespace

import torch
import torch.nn as nn

from fluxvla.engines import HEADS, VLM_BACKBONES, build_vla_from_cfg
from fluxvla.models.backbones.vlms.cosmos25 import Cosmos25Backbone
from fluxvla.models.backbones.vlms.outputs import VLMBackboneOutput
from fluxvla.models.heads.dit4dit_action_head import DiT4DiTActionHead
from fluxvla.models.vlas.dit4dit_vla import DiT4DiTVLA
from fluxvla.models.vlas.llava_vla import LlavaVLA


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
        return VLMBackboneOutput(last_hidden_state=hidden)


class _FakeActionHead(nn.Module):
    action_dim = 2
    action_horizon = 2
    state_dim = 0

    def forward(self,
                input_features,
                actions,
                action_masks,
                states=None,
                attention_mask=None,
                **kwargs):
        return input_features.sum() * 0 + actions.sum() * 0 + 1

    def predict_action(self, input_features, states=None, **kwargs):
        return input_features.new_zeros(input_features.shape[0],
                                        self.action_horizon, self.action_dim)


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


@VLM_BACKBONES.register_module(name='ContractTestBackbone', force=True)
class _ContractTestBackbone(nn.Module):

    def __init__(self, output_style='standard'):
        super().__init__()
        self.output_style = output_style
        self.dummy_weight = nn.Parameter(torch.ones(()))

    def forward(self, images, lang_tokens, lang_masks, **kwargs):
        hidden = images.new_ones(images.shape[0], 3, 4)
        attention_mask = lang_masks[:, :3]
        if self.output_style == 'legacy':
            return hidden, attention_mask, {'source': 'legacy'}
        return VLMBackboneOutput(
            last_hidden_state=hidden,
            attention_mask=attention_mask,
            auxiliary_losses={
                'future_video_loss': hidden.sum() * 0 + 0.25,
            },
        )


@HEADS.register_module(name='ContractTestActionHead', force=True)
class _ContractTestActionHead(nn.Module):
    action_dim = 2
    action_horizon = 2
    state_dim = 0

    def __init__(self, output_style='tensor'):
        super().__init__()
        self.output_style = output_style
        self.dummy_weight = nn.Parameter(torch.ones(()))
        self.last_batch_size = None
        self.last_attention_mask = None
        self.last_output = None

    def forward(self,
                input_features,
                actions,
                action_masks,
                attention_mask=None,
                **kwargs):
        self.last_batch_size = input_features.shape[0]
        self.last_attention_mask = attention_mask
        loss = input_features.sum() * 0 + actions.sum() * 0 + 1
        if self.output_style == 'mapping':
            self.last_output = {'loss': loss, 'source': 'legacy'}
            return self.last_output
        return loss

    def predict_action(self, input_features, **kwargs):
        return input_features.new_zeros(input_features.shape[0],
                                        self.action_horizon, self.action_dim)


def _build_lightweight_vla():
    vla = DiT4DiTVLA.__new__(DiT4DiTVLA)
    nn.Module.__init__(vla)
    vla.vlm_backbone = _RecordingBackbone()
    vla.vla_head = _FakeActionHead()
    vla.repeated_diffusion_steps = 1
    vla.auxiliary_loss_weights = {}
    vla.image_layout = 'auto'
    vla.multiview_strategy = 'tile'
    return vla


def _build_lightweight_llava_vla():
    vla = LlavaVLA.__new__(LlavaVLA)
    nn.Module.__init__(vla)
    vla.vlm_backbone = _RecordingBackbone()
    vla.vla_head = _FakeActionHead()
    vla.llm_backbone = None
    return vla


def test_dit4dit_forward_and_predict_consume_transform_tokens():
    vla = _build_lightweight_vla()
    images = torch.zeros(2, 3, 1, 8, 8)
    lang_tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    lang_masks = lang_tokens.ne(0)
    actions = torch.zeros(2, 2, 2)
    action_masks = torch.ones_like(actions)

    output = vla(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        actions=actions,
        action_masks=action_masks,
    )
    predicted = vla.predict_action(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        states=None,
    )

    assert output['loss'].item() == 1
    assert tuple(predicted.shape) == (2, 2, 2)
    assert vla.vlm_backbone.last_prompts is None
    assert torch.equal(vla.vlm_backbone.last_lang_tokens, lang_tokens)
    assert torch.equal(vla.vlm_backbone.last_lang_masks, lang_masks)


def test_llava_vla_accepts_the_same_standard_backbone_and_head_contract():
    vla = _build_lightweight_llava_vla()
    images = torch.zeros(2, 3, 1, 8, 8, dtype=torch.bfloat16)
    lang_tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    lang_masks = lang_tokens.ne(0)
    actions = torch.zeros(2, 2, 2)
    action_masks = torch.ones_like(actions)

    output = vla(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        actions=actions,
        action_masks=action_masks,
    )
    predicted = vla.predict_action(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        states=None,
    )

    assert output['loss'].item() == 1
    assert output['action_loss'].item() == 1
    assert tuple(predicted.shape) == (2, 2, 2)
    assert predicted.dtype == torch.float32


def test_llava_vla_can_swap_to_standard_backbone_entirely_from_config():
    vla = build_vla_from_cfg(
        dict(
            type='LlavaVLA',
            vlm_backbone=dict(type='ContractTestBackbone'),
            vla_head=dict(type='ContractTestActionHead'),
        ))
    images = torch.zeros(2, 3, 1, 8, 8)
    lang_tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    lang_masks = lang_tokens.ne(0)
    actions = torch.zeros(2, 2, 2)
    action_masks = torch.ones_like(actions)

    output = vla(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        actions=actions,
        action_masks=action_masks,
    )

    assert output['action_loss'].item() == 1
    assert output['future_video_loss'].item() == 0.25
    assert output['loss'].item() == 1.25
    assert vla.vla_head.last_batch_size == 2
    assert tuple(vla.vla_head.last_attention_mask.shape) == (2, 3)


def test_llava_vla_keeps_dit4dit_checkpoint_keys():
    component_cfg = dict(
        vlm_backbone=dict(type='ContractTestBackbone'),
        vla_head=dict(type='ContractTestActionHead'),
    )
    generic = build_vla_from_cfg(dict(type='LlavaVLA', **component_cfg))
    legacy = build_vla_from_cfg(dict(type='DiT4DiTVLA', **component_cfg))
    assert list(generic.state_dict()) == list(legacy.state_dict())


def test_llava_vla_keeps_legacy_backbone_tuple_compatibility():
    vla = build_vla_from_cfg(
        dict(
            type='LlavaVLA',
            vlm_backbone=dict(
                type='ContractTestBackbone', output_style='legacy'),
            vla_head=dict(
                type='ContractTestActionHead', output_style='mapping'),
        ))
    images = torch.zeros(2, 3, 1, 8, 8)
    lang_tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    lang_masks = lang_tokens.ne(0)

    output = vla(
        images=images,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        actions=torch.zeros(2, 2, 2),
        action_masks=torch.ones(2, 2, 2),
    )

    assert output['loss'].item() == 1
    assert output is vla.vla_head.last_output
    assert torch.equal(vla.vla_head.last_attention_mask, lang_masks[:, :3])


def _build_tiny_dit4dit_head():
    return DiT4DiTActionHead(
        action_dim=3,
        hidden_size=8,
        state_dim=4,
        action_horizon=2,
        action_model_type='DiT-B',
        num_inference_timesteps=2,
        add_pos_embed=True,
        max_seq_len=8,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=100,
        ori_action_dim=3,
        diffusion_model_cfg=dict(
            attention_head_dim=4,
            num_attention_heads=2,
            cross_attention_dim=6,
            dropout=0.0,
            final_dropout=False,
            interleave_self_attention=False,
            norm_type='ada_norm',
            num_layers=1,
            output_dim=8,
            positional_embeddings=None,
        ),
    )


def test_dit4dit_head_legacy_and_standard_keyword_contracts_are_exact():
    torch.manual_seed(1234)
    head = _build_tiny_dit4dit_head()
    features = torch.randn(2, 1, 6)
    states = torch.randn(2, 1, 4)
    actions = torch.randn(2, 2, 3)
    action_masks = torch.ones_like(actions)

    torch.manual_seed(4321)
    legacy_loss = head(
        features,
        actions=actions,
        action_mask=action_masks,
        state=states,
    )
    torch.manual_seed(4321)
    standard_loss = head(
        input_features=features,
        actions=actions,
        action_masks=action_masks,
        states=states,
    )

    torch.testing.assert_close(standard_loss, legacy_loss, rtol=0, atol=0)
    torch.manual_seed(9876)
    legacy_actions = head.predict_action(features, state=states)
    torch.manual_seed(9876)
    standard_actions = head.predict_action(
        input_features=features, states=states)
    torch.testing.assert_close(
        standard_actions, legacy_actions, rtol=0, atol=0)


def test_cosmos_backbone_encodes_precomputed_token_ids():
    backbone = Cosmos25Backbone.__new__(Cosmos25Backbone)
    nn.Module.__init__(backbone)
    backbone.text_encoder = _FakeTextEncoder()
    lang_tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])

    prompt_embeds = backbone._encode_lang_tokens(
        lang_tokens,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )

    assert tuple(prompt_embeds.shape) == (2, 3, 4)
    assert torch.equal(backbone.text_encoder.last_input_ids, lang_tokens)
