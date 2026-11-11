# Copyright 2026 Limx Dynamics
"""Language loss and real, small Pi05 training without external checkpoints."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from mmengine import Config

from fluxvla.collators.vlm_sft_collator import VLMSFTCollator
from fluxvla.engines import build_vla_from_cfg
from fluxvla.engines.losses.causal_lm import causal_lm_loss
from fluxvla.engines.runners.mixed_training import MixedTraining
from fluxvla.models.backbones.vlms.causal_vlm import CausalVLMBackbone
from fluxvla.models.vlas.vlm_wrapper import VLMWrapper


@pytest.fixture()
def pi05():
    path = (
        Path(__file__).resolve().parents[2] / 'configs/pi05/'
        'pi05_paligemma_llava665k_full_finetune.py')
    cfg = Config.fromfile(path).model.to_dict()
    cfg.update(
        pretrained_name_or_path=None,
        freeze_llm_expert=False,
        proj_width=16,
        n_action_steps=4,
        num_steps=2)
    for name in ('llm_backbone', 'llm_expert'):
        cfg[name].update(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8)
    cfg['llm_expert']['adarms_cond_dim'] = 16
    cfg['vision_backbone']['vision_config'].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4)
    for name, in_dim, out_dim in (
        ('projector', 16, 16),
        ('time_mlp_in', 16, 16),
        ('time_mlp_out', 16, 16),
        ('action_in_proj', 32, 16),
        ('action_out_proj', 16, 32),
    ):
        cfg[name].update(in_dim=in_dim, out_dim=out_dim)
    return build_vla_from_cfg(cfg).float().eval()


def test_causal_loss_shifts_once_and_excludes_prompt_padding():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, -100],
                           [-100, 4, -100, -100, -100]])
    result = causal_lm_loss(logits, labels)
    expected = F.cross_entropy(
        torch.stack([logits[0, 1], logits[0, 2], logits[1, 0]]),
        torch.tensor([2, 3, 4]),
        reduction='sum')
    torch.testing.assert_close(result['loss_sum'], expected)
    torch.testing.assert_close(result['loss'], expected / 3)
    assert result['num_tokens'] == 3
    result['loss_sum'].backward()
    assert logits.grad[0, 0].count_nonzero() == 0
    assert logits.grad[:, -1].count_nonzero() == 0


def test_empty_local_language_targets_still_participate_in_backward():
    logits = torch.randn(1, 4, 7, requires_grad=True)
    result = causal_lm_loss(logits, torch.full((1, 4), -100))
    assert result['num_tokens'] == result['loss'] == 0
    result['loss_sum'].backward()
    assert logits.grad is not None and logits.grad.count_nonzero() == 0


@pytest.mark.parametrize('with_images', [False, True])
def test_pi05_vlm_forward_backward_and_freezing(pi05, with_images):
    keys = set(pi05.state_dict())
    pi05.prepare_training_task('vlm')
    inputs = dict(
        input_ids=torch.tensor([[2, 5, 6, 1], [2, 7, 1, 0]]),
        labels=torch.tensor([[-100, -100, 6, 1], [-100, 7, 1, 0]]),
        attention_mask=torch.tensor([[True] * 4, [True, True, True, False]]))
    if with_images:
        inputs.update(
            images=torch.randn(2, 6, 8, 8),
            img_masks=torch.tensor([[True, True], [False, False]]))
    output = pi05(task='vlm', **inputs)
    assert output['num_tokens'] == 4
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    assert pi05.llm_backbone.embed_tokens.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in pi05.llm_expert.parameters())
    assert all(not p.requires_grad for p in pi05.action_in_proj.parameters())
    assert set(pi05.state_dict()) == keys
    if with_images:
        assert pi05.projector.projector.weight.grad.abs().sum() > 0


def test_pi05_future_text_cannot_leak_through_image_prefix(pi05):
    inputs = dict(
        input_ids=torch.tensor([[2, 5, 6, 7, 1]]),
        labels=torch.tensor([[-100, -100, 6, -100, -100]]),
        images=torch.randn(1, 3, 8, 8))
    first = pi05(task='vlm', **inputs)['loss']
    inputs['input_ids'] = torch.tensor([[2, 5, 6, 23, 24]])
    second = pi05(task='vlm', **inputs)['loss']
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_pi05_mixed_vla_statistics_preserve_default_loss(pi05):
    pi05.prepare_training_task('mixed')
    inputs = dict(
        images=torch.randn(2, 3, 8, 8),
        img_masks=torch.ones(2, 1, dtype=torch.bool),
        lang_tokens=torch.tensor([[2, 5, 1], [2, 6, 1]]),
        lang_masks=torch.ones(2, 3, dtype=torch.bool),
        states=torch.randn(2, 32),
        actions=torch.randn(2, 4, 32),
        noise=torch.randn(2, 4, 32),
        time=torch.tensor([0.25, 0.75]))
    ordinary = pi05(**inputs)
    mixed = pi05(task='vla', return_loss_statistics=True, **inputs)
    torch.testing.assert_close(ordinary['loss'], mixed['loss'])
    assert mixed['num_action_targets'] == 2 * 4 * 32
    mixed['loss'].backward()
    assert pi05.action_in_proj.projector.weight.grad.abs().sum() > 0
    assert not pi05.llm_expert.embed_tokens.weight.requires_grad
    with pytest.raises(ValueError, match='one vla or vlm'):
        pi05(task='mixed', **inputs)


@pytest.mark.parametrize('with_images', [False, True])
def test_pi05_mixed_zero_slices_match_autograd_unused_parameters(
        pi05, with_images):
    """Real Pi05 + Adam: emulate FSDP's materialized unused flat slices.

    The oracle uses autograd None, not inactive_training_parameters(), so an
    omission in that list cannot make both implementations incorrectly pass.
    The standalone two-GPU regression also exercises actual FSDP/DDP.
    """
    pi05.prepare_training_task('mixed')
    reference = copy.deepcopy(pi05)
    optimizer = torch.optim.AdamW(pi05.parameters(), lr=1e-3)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    runner = SimpleNamespace(
        vla=pi05,
        optimizer=optimizer,
        metric=SimpleNamespace(commit=lambda **kwargs: None),
        _unwrap_vla=lambda: pi05)
    # Only backward() is under test; no data stream or scheduling is needed.
    mixed = MixedTraining(
        Config(dict(train_dataloader={}, runner={})), task_schedule=('vla', ))
    action_inputs = dict(
        images=torch.randn(1, 3, 8, 8),
        img_masks=torch.ones(1, 1, dtype=torch.bool),
        lang_tokens=torch.tensor([[2, 5, 1]]),
        lang_masks=torch.ones(1, 3, dtype=torch.bool),
        states=torch.randn(1, 32),
        actions=torch.randn(1, 4, 32),
        noise=torch.randn(1, 4, 32),
        time=torch.tensor([0.25]),
        return_loss_statistics=True)
    language_inputs = dict(
        input_ids=torch.tensor([[2, 5, 6, 1]]),
        labels=torch.tensor([[-100, 5, 6, 1]]),
        images=torch.randn(1, 3, 8, 8))
    if not with_images:
        language_inputs.pop('images')
    # Populate language-tail momentum before the first action-only step.
    for task in ('vlm', 'vla', 'vlm', 'vla'):
        inputs = language_inputs if task == 'vlm' else action_inputs
        ref_optimizer.zero_grad(set_to_none=True)
        expected = reference(task=task, **inputs)
        expected['loss_sum'].backward()
        count = expected['num_tokens' if task ==
                         'vlm' else 'num_action_targets']
        optimizer.zero_grad(set_to_none=True)
        for parameter in pi05.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        mixed.active_task = task
        mixed.backward(runner, pi05(task=task, **inputs), should_step=True)
        for (name, actual), (_, ref) in zip(pi05.named_parameters(),
                                            reference.named_parameters()):
            assert (actual.grad is None) == (ref.grad is None), name
            if ref.grad is not None:
                ref.grad.mul_((1.0 / count.double()).to(ref.grad.dtype))
                torch.testing.assert_close(actual.grad, ref.grad)
        optimizer.step()
        ref_optimizer.step()
        for actual, ref in zip(pi05.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, ref)
            actual_state, ref_state = (optimizer.state.get(actual, {}),
                                       ref_optimizer.state.get(ref, {}))
            assert actual_state.keys() == ref_state.keys()
            for key in ref_state:
                torch.testing.assert_close(actual_state[key], ref_state[key])
    tail = pi05.llm_backbone.layers[-1]
    assert optimizer.state[tail.self_attn.o_proj.weight]['step'] == 2
    assert optimizer.state[pi05.action_in_proj.projector.weight]['step'] == 2
    assert optimizer.state[pi05.projector.projector.weight]['step'] == (
        4 if with_images else 2)
    with pytest.raises(ValueError, match='optimizer step'):
        pi05.inactive_training_parameters('mixed')


def test_pi05_collator_pads_views_and_keeps_robot_fields_out():
    samples = [
        dict(
            input_ids=torch.tensor([2, 7, 1]),
            labels=torch.tensor([-100, 7, 1]),
            images=torch.ones(6, 8, 8),
            data_root='/unused'),
        dict(
            input_ids=torch.tensor([2, 1]),
            labels=torch.tensor([-100, 1]),
            images=torch.empty(0, 8, 8)),
    ]
    batch = VLMSFTCollator(pad_token_id=0, image_layout='views')(samples)
    assert set(batch) == {
        'input_ids', 'attention_mask', 'labels', 'images', 'img_masks'
    }
    assert batch['images'].shape == (2, 6, 8, 8)
    assert batch['img_masks'].tolist() == [[True, True], [False, False]]
    assert batch['input_ids'][1, -1] == 0
    assert batch['labels'][1, -1] == -100
    assert not batch['attention_mask'][1, -1]


@pytest.mark.parametrize('with_images', [False, True])
def test_native_qwen_vlm_forward_backward_without_pretrained_weights(
        with_images):
    native_config = dict(
        model_type='qwen3_vl',
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters=dict(rope_type='default', mrope_section=[2, 1,
                                                                     1])),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            deepstack_visual_indexes=[]),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63)
    model = VLMWrapper(
        vlm_backbone=dict(
            type='CausalVLMBackbone', model_config=native_config))
    assert isinstance(model.vlm_backbone, CausalVLMBackbone)
    model.prepare_training_task('vlm')
    ids = [2, 62, 60, 63, 5, 1] if with_images else [2, 5, 1]
    inputs = dict(
        input_ids=torch.tensor([ids]),
        labels=torch.tensor([[-100] * (len(ids) - 2) + [5, 1]]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long))
    if with_images:
        inputs.update(
            pixel_values=torch.randn(4, 12),
            image_grid_thw=torch.tensor([[1, 2, 2]]))
    output = model(**inputs)
    assert output['num_tokens'] == 2
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    language = model.vlm_backbone.vlm.model.language_model
    assert language.embed_tokens.weight.grad.abs().sum() > 0
    if with_images:
        grads = [
            p.grad for p in model.vlm_backbone.vlm.model.visual.parameters()
            if p.grad is not None
        ]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    with pytest.raises(ValueError, match='train_task'):
        model.validate_training_task('mixed')

    # Generation must use the same native multimodal path, including grids,
    # without labels or the SFT loss dispatch. Compare actual greedy tokens.
    model.eval()
    generation_inputs = {
        key: value
        for key, value in inputs.items() if key != 'labels'
    }
    generation_inputs.update(
        max_new_tokens=2,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        suppress_tokens=[60, 61, 62, 63])
    expected = model.vlm_backbone.vlm.generate(**generation_inputs)
    actual = model.generate_text(**generation_inputs)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, len(ids) + 2)
