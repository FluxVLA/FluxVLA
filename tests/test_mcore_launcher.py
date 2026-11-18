from types import SimpleNamespace

import pytest

from fluxvla.mcore.contracts import MCoreModelSpec


def test_model_spec_rejects_missing_callback():
    with pytest.raises(TypeError, match="model_provider"):
        MCoreModelSpec(
            name="toy",
            model_type=SimpleNamespace(),
            model_provider=None,
            dataset_provider=lambda _: None,
            forward_step=lambda *_: None,
            args_provider=lambda parser: parser,
        )


def test_launcher_reuses_contract_for_a_second_model(monkeypatch):
    import fluxvla.mcore.launcher as launcher

    calls = []

    def hook():
        calls.append("hook")

    def dataset_provider(num_samples):
        calls.append(("dataset", num_samples))
        return ("train", "valid", "test")

    spec = MCoreModelSpec(
        name="toy-vla",
        model_type=SimpleNamespace(name="toy"),
        model_provider=lambda: calls.append("model"),
        dataset_provider=dataset_provider,
        forward_step=lambda *_: calls.append("forward"),
        args_provider=lambda parser: parser,
        setup_hooks=(hook,),
        args_defaults={"dataloader_type": "external"},
    )

    captured = {}

    def fake_pretrain(dataset, model, model_type, forward, **kwargs):
        captured.update(
            dataset=dataset,
            model=model,
            model_type=model_type,
            forward=forward,
            kwargs=kwargs,
        )
        assert dataset(7) == ("train", "valid", "test")
        assert dataset.is_distributed is True

    monkeypatch.setattr("megatron.training.pretrain", fake_pretrain)
    launcher.run_mcore(spec)

    assert calls == ["hook", ("dataset", 7)]
    assert captured["model"] is spec.model_provider
    assert captured["forward"] is spec.forward_step
    assert captured["kwargs"]["args_defaults"] == {"dataloader_type": "external"}


def test_pi05_recipe_declares_all_native_callbacks():
    from fluxvla.mcore.pi05.recipe import PI05_SPEC

    assert PI05_SPEC.name == "pi05"
    assert PI05_SPEC.model_provider.__module__.endswith("pi05.provider")
    assert PI05_SPEC.dataset_provider.__module__.endswith("pi05.recipe")
    assert PI05_SPEC.forward_step.__module__.endswith("pi05.forward_step")
    assert PI05_SPEC.args_provider.__module__.endswith("pi05.provider")
