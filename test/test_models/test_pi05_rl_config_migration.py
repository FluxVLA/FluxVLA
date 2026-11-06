"""Exercise old-config migration at the actual RL builder boundary."""

import copy

import pytest
from mmengine import Config
from omegaconf import OmegaConf

from fluxvla.models.vlas import pi05_rl_builder as builder


@pytest.mark.parametrize('old_flags', [False, True])
@pytest.mark.parametrize('explicit', [None, False, True])
def test_builder_migrates_missing_flags_before_construction(
        tmp_path, monkeypatch, old_flags, explicit):
    model_cfg = dict(
        type='PI05FlowMatching',
        openpi_fp32_flow=old_flags,
        vision_backbone=dict(openpi_stem_fp32=old_flags))
    original = copy.deepcopy(model_cfg)
    overrides = {}
    if explicit is not None:
        overrides = dict(
            openpi_rl_numerics=explicit,
            vision_backbone=dict(openpi_rl_numerics=explicit))
    cfg = OmegaConf.create(
        dict(
            model_path=str(tmp_path),
            num_steps=5,
            add_value_head=True,
            fluxvla=dict(
                config_path=str(tmp_path / 'config.py'),
                model_overrides=overrides)))
    captured = []

    class BeforeWeightLoading(Exception):
        pass

    def construct(model):
        captured.append(model)
        raise BeforeWeightLoading

    monkeypatch.setattr(builder.Config, 'fromfile',
                        lambda path: Config(dict(model=model_cfg)))
    monkeypatch.setattr(builder, 'build_vla_from_cfg', construct)
    with pytest.raises(BeforeWeightLoading):
        builder.build_pi05_policy(cfg)
    expected = old_flags if explicit is None else explicit
    assert captured[0].openpi_rl_numerics is expected
    assert captured[0].vision_backbone.openpi_rl_numerics is expected
    assert model_cfg == original
