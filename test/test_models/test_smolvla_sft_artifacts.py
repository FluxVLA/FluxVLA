"""SmolVLA PPO must resolve SFT artifacts instead of borrowing PI0.5 stats."""

from pathlib import Path

import pytest
from mmengine import Config

from fluxvla.models.vlas.smolvla_rl_builder import resolve_sft_artifacts


@pytest.mark.parametrize('in_checkpoints_dir', [False, True])
def test_resolves_published_and_flat_sft_layouts(tmp_path, in_checkpoints_dir):
    parent = tmp_path / 'checkpoints' if in_checkpoints_dir else tmp_path
    parent.mkdir(exist_ok=True)
    checkpoint = parent / 'model.safetensors'
    checkpoint.touch()
    stats = tmp_path / 'dataset_statistics.json'
    stats.write_text('{}')
    tokenizer = tmp_path / 'tokenizer'
    tokenizer.mkdir()

    assert resolve_sft_artifacts(checkpoint) == (stats, tokenizer)


def test_missing_stats_fails_instead_of_using_pi05_defaults(tmp_path):
    (tmp_path / 'tokenizer').mkdir()
    with pytest.raises(FileNotFoundError, match='simulator.norm_stats_path'):
        resolve_sft_artifacts(tmp_path / 'model.pt')


def test_missing_tokenizer_reports_override(tmp_path):
    (tmp_path / 'dataset_statistics.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='simulator.tokenizer_path'):
        resolve_sft_artifacts(tmp_path / 'model.pt')


def test_explicit_artifacts_override_checkpoint_sidecars(tmp_path):
    (tmp_path / 'dataset_statistics.json').write_text('{}')
    (tmp_path / 'tokenizer').mkdir()
    stats = tmp_path / 'custom_stats.json'
    stats.write_text('{}')
    tokenizer = tmp_path / 'custom_tokenizer'
    tokenizer.mkdir()
    assert resolve_sft_artifacts(
        tmp_path / 'model.pt', norm_stats_path=stats,
        tokenizer_path=tokenizer) == (stats, tokenizer)


def test_missing_explicit_stats_never_falls_back(tmp_path):
    (tmp_path / 'dataset_statistics.json').write_text('{}')
    (tmp_path / 'tokenizer').mkdir()
    with pytest.raises(FileNotFoundError, match='simulator.norm_stats_path'):
        resolve_sft_artifacts(tmp_path / 'model.pt',
                              norm_stats_path=tmp_path / 'missing.json')


def test_default_model_uses_sft_config_and_matching_libero_reward_scale():
    root = Path(__file__).resolve().parents[2]
    model = Config.fromfile(root / 'configs/rl/model/smolvla.py').model
    simulator = Config.fromfile(root / 'configs/rl/simulator/libero10.py')
    assert model.fluxvla.config_path.endswith(
        '/configs/smolvla/smolvla_libero_10_finetune.py')
    assert simulator.env.train.reward_coef == simulator.env.eval.reward_coef


def test_ppo_native_eval_preserves_master_weight_precision():
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(
        root / 'configs/smolvla/smolvla_libero_10_ppo_eval.py')
    assert cfg.model.vlm_backbone.torch_dtype == 'float32'
    assert cfg.model.llm_expert.torch_dtype == 'float32'
    assert cfg.model.pretrained_name_or_path is None
    assert cfg.model.name_mapping is None
    assert cfg.model.strict_mapping is True
    assert cfg.eval.mixed_precision_dtype == 'bf16'
    assert cfg.eval.model_build_dtype == 'float32'
