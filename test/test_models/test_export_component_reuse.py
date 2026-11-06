# Copyright 2026 Limx Dynamics
"""Training adapters must execute shared components, without copied math.

These are real CPU model tests under the existing --cpu-model-tests bootstrap.
Original-revision numerical evidence is separately captured by the explicit
export_components_probe.py CLI; these tests never manufacture baseline goldens.
"""

import inspect

import pytest
import torch

from fluxvla.engines import (LLM_BACKBONES, PROJECTORS, VISION_BACKBONES,
                             build_vla_from_cfg)
from fluxvla.engines.utils import model_utils
from fluxvla.models.backbones.llms import condition_gemma as training_gemma
from fluxvla.models.backbones.visions.siglip_vit import (SigLIPViTBackbone as
                                                         TrainingSigLIP)
from fluxvla.models.projectors.linear_projector import (LinearProjector as
                                                        TrainingProjector)
from fluxvla_export.components import condition_gemma as shared_gemma
from fluxvla_export.components import tensor_utils
from fluxvla_export.components.projector import LinearProjector
from fluxvla_export.components.siglip import SigLIPViTBackbone
from test.export_components_probe import (SEED, build_case_model, case_specs,
                                          execute_case, fixture_inputs,
                                          tiny_policy_config)


@pytest.mark.parametrize('name', [
    'GemmaRMSNorm',
    'GemmaMLP',
    'GemmaRotaryEmbedding',
    'GemmaAttention',
    'GemmaDecoderLayer',
    'GemmaPreTrainedModel',
    'repeat_kv',
    'rotate_half',
    'apply_rotary_pos_emb',
    'eager_attention_forward',
    '_gated_residual',
])
def test_training_gemma_reexports_exact_shared_computation(name):
    assert getattr(training_gemma, name) is getattr(shared_gemma, name)


def test_training_gemma_adapter_inherits_shared_constructor_and_forward():
    training = training_gemma.ConditionGemmaModel
    shared = shared_gemma.ConditionGemmaModel
    assert issubclass(training, shared)
    assert training.__init__ is shared.__init__
    assert training.forward is shared.forward
    assert training_gemma.GemmaModel is training
    assert LLM_BACKBONES.get('ConditionGemmaModel') is training
    assert 'get_fsdp_wrapping_policy' in training.__dict__
    assert not hasattr(shared, 'get_fsdp_wrapping_policy')
    assert inspect.signature(training) == inspect.signature(shared)


def test_siglip_adapter_and_projector_share_real_implementations():
    assert issubclass(TrainingSigLIP, SigLIPViTBackbone)
    assert TrainingSigLIP.forward is SigLIPViTBackbone.forward
    assert TrainingSigLIP._forward_openpi_stem is SigLIPViTBackbone._forward_openpi_stem
    assert VISION_BACKBONES.get('SigLIPViTBackbone') is TrainingSigLIP
    assert TrainingProjector is LinearProjector
    assert PROJECTORS.get('LinearProjector') is LinearProjector
    assert not hasattr(SigLIPViTBackbone, 'get_fsdp_wrapping_policy')


@pytest.mark.parametrize('name', [
    'create_sinusoidal_pos_embedding',
    'make_att_2d_masks',
    'gated_residual',
    'repeat_kv',
    'eager_attention_forward',
    'sdpa_attention_forward',
    'sdpa_math_fp32_attention_forward',
    'apply_rope',
    'rotate_half',
    'apply_rotary_pos_emb',
])
def test_training_tensor_helpers_are_shared_aliases(name):
    assert getattr(model_utils, name) is getattr(tensor_utils, name)


@pytest.mark.parametrize(
    'spec', [case for case in case_specs() if case['kind'] != 'policy'],
    ids=lambda case: case['name'])
def test_training_and_shared_full_state_outputs_and_gradients_match(spec):
    torch.manual_seed(SEED)
    training = build_case_model(spec, shared=False)
    state = {
        name: value.detach().clone()
        for name, value in training.state_dict().items()
    }
    inputs = fixture_inputs(spec['kind'])
    training_results, training_details = execute_case(training, spec, inputs)

    torch.manual_seed(SEED)
    shared = build_case_model(spec, shared=True)
    assert list(shared.state_dict()) == list(state)
    # Initialization order is part of the training compatibility contract.
    for name, value in shared.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    shared.load_state_dict(state, strict=True)
    shared_results, shared_details = execute_case(
        shared, spec, inputs, shared=True)
    assert shared_details == training_details
    assert set(shared_results) == set(training_results)
    for name, value in shared_results.items():
        torch.testing.assert_close(
            value,
            training_results[name],
            rtol=0,
            atol=0,
            msg=lambda msg: f'{spec["name"]}/{name}: {msg}')


@pytest.mark.parametrize('policy_type',
                         ['PI0FlowMatching', 'PI05FlowMatching'])
@pytest.mark.parametrize('protected', [False, True])
def test_real_pi_policy_uses_shared_modules_without_extra_state_prefixes(
        policy_type, protected):
    policy = build_vla_from_cfg(tiny_policy_config(policy_type,
                                                   protected)).float()
    assert isinstance(policy.llm_backbone, shared_gemma.ConditionGemmaModel)
    assert isinstance(policy.llm_expert, shared_gemma.ConditionGemmaModel)
    assert isinstance(policy.vision_backbone, SigLIPViTBackbone)
    assert type(policy.projector) is LinearProjector
    assert type(policy.action_in_proj) is LinearProjector
    assert type(policy.action_out_proj) is LinearProjector
    assert type(
        policy.llm_backbone.layers[0]) is shared_gemma.GemmaDecoderLayer
    assert type(policy.llm_expert.layers[0].mlp) is shared_gemma.GemmaMLP
    for model in (policy.llm_backbone, policy.llm_expert):
        assert model.forward.__func__ is shared_gemma.ConditionGemmaModel.forward
        assert model.half_precision_dtype is torch.bfloat16
        assert callable(model.get_fsdp_wrapping_policy())
    keys = policy.state_dict()
    for key in (
            'llm_backbone.embed_tokens.weight',
            'llm_expert.layers.0.self_attn.q_proj.weight',
            'vision_backbone.vision.vision_model.embeddings.patch_embedding.weight',
            'projector.projector.weight', 'action_out_proj.projector.weight'):
        assert key in keys
    assert not any('.core.' in key or '.shared.' in key for key in keys)
    expected_attention = (
        tensor_utils.sdpa_math_fp32_attention_forward
        if protected else tensor_utils.sdpa_attention_forward)
    assert policy.attention_interface is expected_attention
