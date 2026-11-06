# Copyright 2026 Limx Dynamics
"""Real PI training/RTC execution through the shared policy computations.

Run with the existing --cpu-model-tests import bootstrap. The separate original
revision probe owns pre-extraction numerical evidence; these tests use no model
stubs, patched computations, or fabricated baseline tensors.
"""

import inspect
import sys
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from fluxvla.engines import VLAS, build_vla_from_cfg
from fluxvla.engines.losses import reduce_action_bc_loss
from fluxvla.engines.utils.rtc_training import (apply_rtc_time_conditioning,
                                                sample_training_delay)
from fluxvla.models.vlas.base_vla import BaseVLA
from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching
from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching
from fluxvla_export.components.pi_policy import PI0Computation, PI05Computation
from fluxvla_export.components.tensor_utils import (
    create_sinusoidal_pos_embedding, make_att_2d_masks)
from test.export_components_probe import fixture_inputs, tiny_policy_config

POLICIES = (PI0FlowMatching, PI05FlowMatching)
SHARED_METHODS = ('_disable_autocast', '_cast_gemma_input',
                  '_project_action_output', 'get_attention_interface',
                  'embed_prefix', 'forward_model',
                  '_prepare_attention_masks_4d', '_predict_action_plain',
                  'predict_action', 'denoise_step')
TRAINING_METHODS = ('__init__', 'forward', '_forward_transformer_layers',
                    'sample_noise', 'sample_time', 'to_bfloat16',
                    '_select_flow_loss_dimensions',
                    '_predict_action_prefix_rtc',
                    '_predict_action_guidance_rtc', 'get_fsdp_wrapping_policy',
                    'get_fsdp_execution_block_wrapping_policy')


@contextmanager
def _count_real_calls(*functions):
    """Observe Python calls without replacing or wrapping model operations."""
    counts = {function.__code__: 0 for function in functions}
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == 'call' and frame.f_code in counts:
            counts[frame.f_code] += 1
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def _model(policy, amp=False, **kwargs):
    config = tiny_policy_config(policy.__name__, protected=True)
    config.update(
        enable_mixed_precision_training=amp,
        openpi_fp32_flow=True,
        preserve_fp32_residuals=True)
    config.update(kwargs)
    return build_vla_from_cfg(config).float()


def _inference_inputs():
    return {
        name: value
        for name, value in fixture_inputs('policy').items()
        if name not in ('actions', 'action_masks', 'time')
    }


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('name', SHARED_METHODS)
def test_training_uses_exact_shared_policy_methods(policy, name):
    assert getattr(policy, name) is getattr(PI0Computation, name)
    assert name not in PI0FlowMatching.__dict__
    assert name not in PI05FlowMatching.__dict__


def test_suffix_override_and_training_extension_ownership():
    assert PI0FlowMatching.embed_suffix is PI0Computation.embed_suffix
    assert PI05FlowMatching.embed_suffix is PI05Computation.embed_suffix
    assert PI05Computation.__bases__ == (PI0Computation, )
    for shared in (PI0Computation, PI05Computation):
        assert '__init__' not in shared.__dict__
        assert not issubclass(shared, torch.nn.Module)
        assert shared.__init__ is object.__init__
        for name in TRAINING_METHODS[1:]:
            assert not hasattr(shared, name)
    for name in TRAINING_METHODS:
        assert name in PI0FlowMatching.__dict__
        function = getattr(PI0FlowMatching, name)
        assert function.__module__ == PI0FlowMatching.__module__
    assert PI05FlowMatching.forward is PI0FlowMatching.forward
    assert set(name for name, value in PI05FlowMatching.__dict__.items()
               if inspect.isfunction(value)) == {'__init__'}


@pytest.mark.parametrize('policy', POLICIES)
def test_real_constructor_runs_once_with_cooperative_mro(policy):
    constructors = (PI0FlowMatching.__init__, BaseVLA.__init__)
    with _count_real_calls(*constructors) as calls:
        model = _model(policy)
    assert all(calls[function.__code__] == 1 for function in constructors)
    assert VLAS.get(policy.__name__) is policy
    if policy is PI05FlowMatching:
        assert policy.__mro__[:5] == (PI05FlowMatching, PI05Computation,
                                      PI0FlowMatching, PI0Computation, BaseVLA)
    else:
        assert policy.__mro__[:3] == (PI0FlowMatching, PI0Computation, BaseVLA)
    assert list(model._modules) == (
        ['vision_backbone', 'llm_backbone', 'projector'] +
        (['state_proj'] if policy is PI0FlowMatching else []) +
        ['action_in_proj', 'action_out_proj'] +
        (['action_time_mlp_in', 'action_time_mlp_out']
         if policy is PI0FlowMatching else ['time_mlp_in', 'time_mlp_out']) +
        ['llm_expert'])
    assert not any(
        key.startswith(('core.', 'shared.')) for key in model.state_dict())
    assert callable(model.get_fsdp_wrapping_policy())
    assert callable(model.get_fsdp_execution_block_wrapping_policy())


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('amp', [False, True])
@pytest.mark.parametrize('rtc', [False, True])
def test_real_training_loss_gradients_and_per_position_rtc(policy, amp, rtc):
    rtc_config = (
        dict(enabled=True, max_delay=2, distribution='uniform')
        if rtc else None)
    model = _model(
        policy, amp, loss_action_dim=2,
        rtc_training_config=rtc_config).train()
    inputs = fixture_inputs('policy')
    inputs['action_masks'] = torch.tensor([[True, False, True],
                                           [True, True, False]])
    originals = {name: tensor.clone() for name, tensor in inputs.items()}
    sample_weight = torch.tensor([0.25, 1.5])
    time, masks = inputs['time'], inputs['action_masks']
    if rtc:
        # Sample actual delays, then restore RNG so forward samples the same
        # values itself. Nothing in the training implementation is patched.
        torch.manual_seed(0)
        before_delay = torch.random.get_rng_state()
        delays = sample_training_delay(2, 2, distribution='uniform')
        assert torch.any(delays > 0)
        time, masks = apply_rtc_time_conditioning(
            time, masks, delays, 3, clean_time=0.0)
        torch.random.set_rng_state(before_delay)
    expanded_time = time.unsqueeze(-1) if rtc else time[:, None, None]
    expected_noisy = (
        expanded_time * inputs['noise'] +
        (1 - expanded_time) * inputs['actions'])
    expected_time = create_sinusoidal_pos_embedding(
        time, model.proj_width, min_period=4e-3, max_period=4.0)
    captured_actions, captured_times = [], []

    def record_actions(module, args):
        captured_actions.append(args[0].detach().clone())

    def record_times(module, args):
        value = args[0]
        if policy is PI0FlowMatching:
            value = value[..., model.proj_width:]
        captured_times.append(value.detach().clone())

    time_module = (
        model.action_time_mlp_in
        if policy is PI0FlowMatching else model.time_mlp_in)
    hooks = [
        model.action_in_proj.register_forward_pre_hook(record_actions),
        time_module.register_forward_pre_hook(record_times)
    ]
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            with _count_real_calls(model._forward_transformer_layers) as calls:
                result = model(**inputs, sample_weight=sample_weight)
    finally:
        for hook in hooks:
            hook.remove()
    assert calls[model._forward_transformer_layers.__code__] == 1
    torch.testing.assert_close(
        captured_actions[0], expected_noisy, rtol=0, atol=0)
    if policy is PI0FlowMatching and not rtc:
        expected_time = expected_time[:, None].expand(2, 3, model.proj_width)
    torch.testing.assert_close(
        captured_times[0], expected_time, rtol=0, atol=0)
    predictions = result['predictions']
    assert predictions.shape == (2, 3, 2)
    assert predictions.dtype == torch.float32
    target = (inputs['noise'] - inputs['actions'])[..., :2]
    expected_loss = reduce_action_bc_loss(
        F.mse_loss(target, predictions, reduction='none'),
        action_mask=masks,
        sample_weight=sample_weight)
    torch.testing.assert_close(result['loss'], expected_loss, rtol=0, atol=0)
    result['loss'].backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    for module in (model.vision_backbone, model.llm_backbone, model.llm_expert,
                   model.action_in_proj, model.action_out_proj, time_module):
        assert any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad) > 0
            for parameter in module.parameters())
    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor, originals[name], rtol=0, atol=0)


@pytest.mark.parametrize('policy', POLICIES)
def test_training_random_sampling_matches_explicit_real_samples(policy):
    model = _model(policy).train()
    inputs = fixture_inputs('policy')
    inputs.pop('noise')
    inputs.pop('time')
    initial_rng = torch.random.get_rng_state()
    noise = model.sample_noise(inputs['actions'].shape, 'cpu')
    time = model.sample_time(2, 'cpu')
    expected_rng = torch.random.get_rng_state()
    explicit = model(**inputs, noise=noise, time=time)
    torch.random.set_rng_state(initial_rng)
    sampled = model(**inputs)
    torch.testing.assert_close(
        torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    for name in ('loss', 'predictions'):
        torch.testing.assert_close(
            sampled[name], explicit[name], rtol=0, atol=0)


@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('amp', [False, True])
@pytest.mark.parametrize('method',
                         ['plain', 'prefix', 'guidance', 'guidance-vjp'])
def test_cached_prediction_dispatches_real_euler_and_rtc(policy, amp, method):
    model = _model(policy, amp).eval()
    inputs = _inference_inputs()
    state = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
    }
    fixture = {name: tensor.clone() for name, tensor in inputs.items()}
    # Exercise the existing ori_action_dim padding path as well as RTC itself.
    previous = torch.linspace(-0.3, 0.5, 18).reshape(2, 3, 3)
    rtc = {} if method == 'plain' else dict(
        prev_actions=previous,
        prefix_len=1,
        rtc_config=dict(
            method=method.split('-')[0],
            decay_end=2,
            use_vjp=method == 'guidance-vjp'))
    handler = getattr(
        model,
        '_predict_action_plain' if method == 'plain' else '_predict_action_' +
        method.split('-')[0] + '_rtc')
    captured_noisy = []

    def record_actions(module, args):
        captured_noisy.append(args[0].detach().clone())

    hook = model.action_in_proj.register_forward_pre_hook(record_actions)
    before_rng = torch.random.get_rng_state()
    noise = inputs['noise'].clone()
    inputs['noise'] = noise
    try:
        with torch.no_grad(), torch.autocast(
                'cpu', dtype=torch.bfloat16, enabled=amp):
            with _count_real_calls(handler, model.denoise_step) as calls:
                output = model.predict_action(**inputs, **rtc)
            repeated = model.predict_action(
                **{
                    name: tensor.clone()
                    for name, tensor in fixture.items()
                }, **rtc)
    finally:
        hook.remove()
    assert calls[handler.__code__] == 1
    assert calls[model.denoise_step.__code__] == model.num_steps
    assert output.data_ptr() == noise.data_ptr()  # Original in-place Euler.
    assert not torch.equal(noise, fixture['noise'])
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, repeated, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.random.get_rng_state(), before_rng, rtol=0, atol=0)
    if method == 'prefix':
        padded = F.pad(previous, (0, 1))
        torch.testing.assert_close(
            output[:, :1], padded[:, :1], rtol=0, atol=0)
        for noisy in captured_noisy:
            torch.testing.assert_close(
                noisy[:, :1], padded[:, :1], rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    for name, tensor in inputs.items():
        if name != 'noise':
            torch.testing.assert_close(tensor, fixture[name], rtol=0, atol=0)


@pytest.mark.parametrize('policy', POLICIES)
def test_plain_euler_matches_explicit_steps_with_immutable_prefix_cache(
        policy):
    model = _model(policy).eval()
    inputs = _inference_inputs()
    with torch.no_grad():
        prefix, padding, blocks = model.embed_prefix(**inputs)
        prefix = model._cast_gemma_input(prefix)
        masks = model._prepare_attention_masks_4d(
            make_att_2d_masks(padding, blocks))
        _, cache = model.forward_model([prefix, None],
                                       masks,
                                       torch.cumsum(padding, dim=1) - 1,
                                       use_cache=True)
        before_cache = [(layer.keys.clone(), layer.values.clone())
                        for layer in cache.layers]
        expected = inputs['noise'].clone()
        time = torch.tensor(1.0)
        dt = torch.tensor(-1.0 / model.num_steps)
        while time >= -dt / 2:
            expected += dt * model.denoise_step(
                inputs['states'], padding, cache, expected, time.expand(2))
            time += dt
        actual = model.predict_action(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for layer, (keys, values) in zip(cache.layers, before_cache):
        torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, values, rtol=0, atol=0)
