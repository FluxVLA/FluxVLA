"""Regression checks for the FluxVLA-owned LingBot model core."""

from __future__ import annotations

import sys


def test_native_core_import_does_not_load_reference_lingbot_package():
    from fluxvla.models.vlas.lingbot_native_core.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from fluxvla.models.vlas.lingbot_native_core.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )

    assert LingbotVLAV2Config.model_type == "lingbotvla"
    assert LingbotVlaV2Policy.__module__.startswith(
        "fluxvla.models.vlas.lingbot_native_core"
    )
    assert not any(
        name == "lingbotvla" or name.startswith("lingbotvla.")
        for name in sys.modules
    )


def test_native_mcore_launcher_has_no_reference_source_root_argument():
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts/train_mcore_lingbot_vla_v2.sh"
    text = script.read_text(encoding="utf-8")
    assert "--lingbot-source-root" not in text
    assert "lingbot-vla-v2/golden_artifacts" not in text


def test_lingbot_forward_smoke_checkpoint_is_environment_overridable():
    from pathlib import Path

    script = (Path(__file__).parents[1] / "tools" / "lingbot_real_forward_smoke.py").read_text(
        encoding="utf-8"
    )
    assert "LINGBOT_SMOKE_CHECKPOINT" in script
    assert "LINGBOT_SMOKE_TOKENIZER" in script
    assert "os.environ.get" in script


def test_fused_moe_fallback_uses_explicit_routing_contract():
    from pathlib import Path

    script = (Path(__file__).parents[1] / "fluxvla" / "ops" / "robby_moe.py").read_text(
        encoding="utf-8"
    )
    assert "routing_weights: torch.Tensor | None" in script
    assert "selected_experts: torch.Tensor | None" in script
    assert "top_k=2" not in script
