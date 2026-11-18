from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_single_native_mcore_layout():
    assert (ROOT / "fluxvla/mcore/pi05/model.py").exists()
    assert (ROOT / "fluxvla/mcore/pi05/tensor_parallel.py").exists()
    assert (ROOT / "fluxvla/mcore/pi05/checkpoint.py").exists()
    assert (ROOT / "fluxvla/mcore/pi05/forward_step.py").exists()
    assert not (ROOT / "fluxvla/mcore/checkpoint.py").exists()
    assert not (ROOT / "fluxvla/mcore/schedule.py").exists()
    legacy_dir = ROOT / "fluxvla/mcore_native"
    assert not any(legacy_dir.glob("*.py"))
    assert not any(legacy_dir.glob("**/*.py"))
    assert not (ROOT / "fluxvla/mcore/pi05_provider.py").exists()


def test_native_entrypoints_use_single_mcore_namespace():
    source = (ROOT / "scripts/train_mcore_pi05.py").read_text()
    assert "fluxvla.mcore.pi05.recipe" in source
    assert "mcore_native" not in source

    recipe = (ROOT / "fluxvla/mcore/pi05/recipe.py").read_text()
    assert "from .forward_step import forward_step" in recipe
    assert "MCoreModelSpec" in recipe


def test_mcore_root_does_not_bind_a_concrete_model():
    source = (ROOT / "fluxvla/mcore/__init__.py").read_text()
    assert "pi05.provider" not in source
    assert "model_provider" not in source
