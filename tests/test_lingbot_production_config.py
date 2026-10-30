from pathlib import Path

from mmengine.config import Config


def test_official_trace_attention_contract():
    config_path = (
        Path(__file__).parents[1]
        / "configs/lingbot_vla_v2/lingbot_vla_v2_official_production_100_20261016.py"
    )
    config = Config.fromfile(str(config_path))

    # These values mirror the actual official production trace, not only the
    # defaults in robotwin_reduced.yaml.
    assert config.model.attention_implementation == "eager"
    assert config.model.vit_attn_implementation == "flash_attention_2"
    assert config.model.attention.implementation == "eager"
    assert config.model.attention.vit_implementation == "flash_attention_2"
    # This model-level input contract is independent from the official
    # runner's train.enable_fp32/FSDP parameter policy.
    assert config.model.action_fp32 is False
    assert config.runner.max_steps == 100
    assert config.runner.enable_mixed_precision_training is True
    assert config.runner.enable_forward_autocast is False
    assert config.runner.batch_dtype == "bf16"
    assert config.runner.reduce_in_full_precision is True
    assert config.runner.keep_params_fp32 is False
