# Copyright 2026 Limx Dynamics
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from mmengine import Config
from torch.nn.attention import SDPBackend, sdpa_kernel

from fluxvla.engines import build_vla_from_cfg
from fluxvla.engines.utils.model_utils import sdpa_math_fp32_attention_forward
from fluxvla.models.backbones.llms.condition_gemma import ConditionGemmaModel
from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('boolean_mask', [False, True])
def test_large_logit_attention_backward_matches_double(device, boolean_mask):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator(device=device).manual_seed(7)
    shape = (1, 2, 32, 64)
    tensors = [
        torch.randn(shape, generator=generator, device=device) * scale
        for scale in (1000, 1, 1)
    ]
    # Compare identical representable inputs; input quantization is excluded.
    tensors = [value.bfloat16().float() for value in tensors]
    dy = torch.randn(shape, generator=generator, device=device)
    mask = torch.ones(1, 1, 32, 32, dtype=torch.bool, device=device)
    mask[..., -4:] = False
    if not boolean_mask:
        mask = torch.zeros_like(
            mask, dtype=torch.float32).masked_fill(~mask, float('-inf'))
    results = []
    for reference in (True, False):
        dtype = torch.float64 if reference else torch.float32
        inputs = [value.to(dtype).requires_grad_() for value in tensors]
        if reference:
            ref_mask = mask if boolean_mask else mask.double()
            with sdpa_kernel(SDPBackend.MATH):
                output = F.scaled_dot_product_attention(
                    *inputs, attn_mask=ref_mask, scale=0.125)
        else:
            with torch.autocast(device, dtype=torch.bfloat16):
                output, _ = sdpa_math_fp32_attention_forward(
                    SimpleNamespace(num_key_value_groups=1, training=True),
                    *inputs, mask, 0.125)
            assert output.dtype == torch.float32
            output = output.transpose(1, 2)
        gradients = torch.autograd.grad(output, inputs, dy.to(dtype))
        results.append((output.detach(), gradients))
    for actual, expected in zip(results[1][1], results[0][1]):
        relative_error = ((actual.double() - expected).norm() /
                          expected.norm().clamp_min(1e-12))
        assert relative_error < 0.002
    torch.testing.assert_close(
        results[1][0].double(), results[0][0], rtol=0.002, atol=0.0002)


def test_selected_blocks_keep_fp32_compute_and_checkpoint_layout():
    kwargs = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        adarms_cond_dim=12,
        adarms_fp32=True,
        use_cache=False)
    torch.manual_seed(920)
    legacy = ConditionGemmaModel(**kwargs).float().eval()
    protected = ConditionGemmaModel(
        **kwargs, attention_math_fp32=True, fp32_layers=(0, )).float().eval()
    protected.load_state_dict(legacy.state_dict(), strict=True)
    assert list(legacy.state_dict()) == list(protected.state_dict())
    dtypes = []
    handles = []
    for layer in protected.layers:
        for module in (layer.self_attn.q_proj, layer.mlp.up_proj):
            handles.append(
                module.register_forward_hook(lambda module, inputs, output:
                                             dtypes.append(output.dtype)))
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = protected(
                inputs_embeds=torch.randn(2, 3, 16),
                adarms_cond=torch.randn(2, 3, 12))
        result.last_hidden_state.square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert dtypes == [torch.float32] * 2 + [torch.bfloat16] * 2


@pytest.mark.parametrize('indices', [(-1, ), (2, ), (True, ), (0.5, )])
def test_invalid_fp32_layers_rejected(indices):
    with pytest.raises(ValueError, match='fp32_layers'):
        ConditionGemmaModel(num_hidden_layers=2, fp32_layers=indices)


@pytest.mark.parametrize('amp', [False, True])
def test_protected_attention_cache_matches_joint_block_mask(amp):
    torch.manual_seed(54)
    model = ConditionGemmaModel(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        adarms_cond_dim=12,
        adarms_fp32=True,
        attention_math_fp32=True,
        fp32_layers=(0, )).float().eval()
    for module in model.modules():
        if getattr(module, 'dense', None) is not None:
            torch.nn.init.normal_(module.dense.weight, std=0.1)
    hidden = torch.randn(2, 7, 16)
    cond = torch.randn(2, 7, 12)
    mask = torch.zeros(2, 1, 7, 7)
    mask[:, :, :3, 3:] = float('-inf')
    positions = torch.arange(7)[None].expand(2, -1)
    autocast = torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp)
    with torch.no_grad(), autocast:
        joint = model(
            inputs_embeds=hidden,
            adarms_cond=cond,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False).last_hidden_state[:, 3:]
        prefix = model(
            inputs_embeds=hidden[:, :3],
            adarms_cond=cond[:, :3],
            attention_mask=mask[:, :, :3, :3],
            position_ids=positions[:, :3],
            use_cache=True)
        suffix = model(
            inputs_embeds=hidden[:, 3:],
            adarms_cond=cond[:, 3:],
            attention_mask=mask[:, :, 3:],
            position_ids=positions[:, 3:],
            past_key_values=prefix.past_key_values,
            use_cache=False).last_hidden_state
    torch.testing.assert_close(suffix, joint, rtol=0.002, atol=0.0002)


def test_math_attention_preserves_implicit_causal_mask():
    torch.manual_seed(55)
    model = ConditionGemmaModel(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        use_adarms=False,
        attention_math_fp32=True).float().eval()
    hidden = torch.randn(2, 4, 16)
    mask = torch.full((2, 1, 4, 4), float('-inf')).triu(1)
    with torch.no_grad():
        implicit = model(inputs_embeds=hidden).last_hidden_state
        explicit = model(
            inputs_embeds=hidden, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(implicit, explicit, rtol=1e-6, atol=1e-6)


def test_new_recipe_keeps_native_actions_and_legacy_precision_defaults():
    root = Path(__file__).resolve().parents[2]
    path = root / 'configs/pi05'
    old = Config.fromfile(
        str(path /
            'pi05_paligemma_basket_all_rtc_bf16_adarms_fp32_full_finetune.py'))
    new = Config.fromfile(
        str(path /
            'pi05_paligemma_basket_all_rtc_bf16_expert6_fp32_full_finetune.py')
    )
    assert dict(new.runner) == dict(old.runner)
    assert new.runner.max_keep_ckpts == 2
    assert new.model.max_action_dim == new.model.loss_action_dim == 42
    for section in ('model', 'inference_model'):
        expected = copy.deepcopy(dict(old[section]))
        expected['llm_backbone']['attention_math_fp32'] = True
        expected['llm_expert'].update(
            attention_math_fp32=True, fp32_layers=(0, 1, 2, 3, 4, 5))
        assert dict(new[section]) == expected
        with torch.device('meta'):
            model = build_vla_from_cfg(new[section])
        assert model.attention_interface is sdpa_math_fp32_attention_forward
        assert [layer.mlp.compute_fp32 for layer in model.llm_expert.layers
                ] == [True] * 6 + [False] * 12


@pytest.mark.parametrize('variant', ['full_finetune', 'rtc_inference'])
def test_aloha_precision_recipe_training_and_cached_inference(variant):
    root = Path(__file__).resolve().parents[2] / 'configs/pi05'
    legacy = Config.fromfile(root / 'pi05_paligemma_aloha_full_finetune.py')
    cfg = Config.fromfile(
        root / f'pi05_paligemma_aloha_bf16_expert6_fp32_{variant}.py')
    assert cfg.train_dataloader == legacy.train_dataloader
    assert cfg.runner == legacy.runner
    assert cfg.inference.dataset == legacy.inference.dataset
    assert cfg.inference.denormalize_action == legacy.inference.denormalize_action  # noqa: E501
    assert cfg.inference.keep_params_fp32
    assert cfg.inference.enable_mixed_precision
    assert cfg.model == cfg.inference_model
    assert cfg.model.ori_action_dim == 14
    assert cfg.model.max_action_dim == cfg.model.loss_action_dim == 32
    assert cfg.model.n_action_steps == cfg.inference.action_chunk == 50
    assert not legacy.model.llm_expert.get('adarms_fp32', False)
    assert not legacy.model.llm_expert.get('fp32_layers', ())
    assert not legacy.model.llm_backbone.get('attention_math_fp32', False)

    # Use the real recipe, reducing only backbone widths/depth and image size.
    # Retain all six protected blocks, one BF16 block and Aloha action shapes.
    model_cfg = copy.deepcopy(dict(cfg.inference_model))
    for branch in ('llm_backbone', 'llm_expert'):
        model_cfg[branch].update(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=7,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8)
    model_cfg['llm_expert']['adarms_cond_dim'] = 16
    model_cfg['vision_backbone']['vision_config'].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4)
    for key, in_dim, out_dim in (
        ('projector', 16, 16),
        ('time_mlp_in', 16, 16),
        ('time_mlp_out', 16, 16),
        ('action_in_proj', 32, 16),
        ('action_out_proj', 16, 32),
    ):
        model_cfg[key].update(in_dim=in_dim, out_dim=out_dim)
    model_cfg.update(proj_width=16, num_steps=2)
    policy = build_vla_from_cfg(model_cfg).float().eval()
    # A kernel policy would bypass Gemma's precision switches at inference.
    assert policy.predict_action.__func__ is PI0FlowMatching.predict_action
    assert policy.attention_interface is sdpa_math_fp32_attention_forward
    assert [layer.mlp.compute_fp32
            for layer in policy.llm_expert.layers] == [True] * 6 + [False]
    inputs = dict(
        images=torch.randn(2, 9, 8, 8),
        img_masks=torch.tensor([[True] * 3, [True, True, False]]),
        lang_tokens=torch.tensor([[1, 5, 2, 0], [1, 7, 3, 2]]),
        lang_masks=torch.tensor([[True, True, True, False], [True] * 4]),
        states=torch.randn(2, 32),
        noise=torch.randn(2, 50, 32))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = policy(
            **inputs,
            actions=torch.randn(2, 50, 32),
            time=torch.tensor([0.25, 0.75]))
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    for module in (policy.vision_backbone, policy.llm_backbone,
                   policy.llm_expert, policy.time_mlp_in):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert any(grad.abs().sum() > 0 for grad in grads)

    rtc = {}
    if variant == 'rtc_inference':
        assert cfg.inference.type == 'AlohaRTCInferenceRunner'
        rtc = dict(
            prev_actions=torch.randn(2, 50, 32),
            prefix_len=cfg.inference.rtc_config.prefix_len,
            rtc_config=dict(cfg.inference.rtc_config))
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        actions = policy.predict_action(**inputs, **rtc)
    assert actions.shape == (2, 50, 32)
    assert torch.isfinite(actions).all()
    if rtc:
        length = rtc['prefix_len']
        torch.testing.assert_close(
            actions[:, :length],
            rtc['prev_actions'][:, :length],
            rtol=0,
            atol=0)
