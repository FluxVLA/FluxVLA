"""Keep main's selective initialization and RL's numerical opt-in together."""

import copy

import pytest
import torch

from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching
from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching


@pytest.mark.parametrize('model_cls', [PI0FlowMatching, PI05FlowMatching])
@pytest.mark.parametrize('rl_numerics', [False, True])
def test_skip_prefixes_and_rl_numerics_coexist(tmp_path, model_cls,
                                               rl_numerics):
    gemma = dict(
        type='ConditionGemmaModel',
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        adarms_cond_dim=8)
    path = tmp_path / 'policy.pt'
    model = model_cls(
        action_in_proj=dict(type='LinearProjector', in_dim=2, out_dim=8),
        action_out_proj=dict(type='LinearProjector', in_dim=8, out_dim=2),
        llm_backbone=copy.deepcopy(gemma),
        llm_expert=copy.deepcopy(gemma),
        proj_width=8,
        pretrained_name_or_path=str(path),
        name_mapping=None,
        strict_mapping=True,
        pretrained_skip_prefixes=['action_in_proj.'],
        openpi_fp32_flow=True,
        openpi_rl_numerics=rl_numerics).float()
    before = model.action_in_proj.projector.weight.detach().clone()
    weights = {
        name: value.clone()
        for name, value in model.state_dict().items()
    }
    weights['action_in_proj.projector.weight'].fill_(19)
    weights['action_out_proj.projector.weight'].fill_(7)
    torch.save(weights, path)
    model.from_pretrained()
    assert model.pretrained_skip_prefixes == ('action_in_proj.', )
    assert model.openpi_rl_numerics is rl_numerics
    torch.testing.assert_close(
        model.action_in_proj.projector.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(
        model.action_out_proj.projector.weight,
        weights['action_out_proj.projector.weight'],
        rtol=0,
        atol=0)
    if rl_numerics:
        for branch in (model.llm_backbone, model.llm_expert):
            assert torch.equal(branch.rotary_emb.inv_freq,
                               branch.rotary_emb.inv_freq.bfloat16().float())
            assert all(module.adarms_fp32 for module in branch.modules()
                       if hasattr(module, 'adarms_fp32'))
