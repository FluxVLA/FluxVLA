"""Portable PPO inference bundles must retain weights and run settings."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from mmengine import Config
from omegaconf import OmegaConf
from safetensors.torch import load_file

from fluxvla.engines.utils.rl_export import export_native_bundle


@pytest.fixture
def bundle_inputs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    sft = tmp_path / 'sft'
    (sft / 'tokenizer').mkdir(parents=True)
    (sft / 'tokenizer/tokenizer_config.json').write_text('{}')
    (sft / 'tokenizer/tokenizer.json').write_text('{}')
    (sft / 'dataset_statistics.json').write_text(
        json.dumps({'libero_10_no_noops': {
            'action': {
                'mean': [0] * 7
            }
        }}))
    (sft / 'model.pt').touch()
    step = tmp_path / 'global_step_1'
    checkpoint = step / 'actor/model_state_dict/full_weights.pt'
    checkpoint.parent.mkdir(parents=True)
    state = {
        'policy.weight': torch.tensor([1.0001, 2.0001]),
        'policy.buffer': torch.tensor([3], dtype=torch.int64),
        'value_head.weight': torch.ones(2)
    }
    torch.save(state, checkpoint)
    cfg = OmegaConf.create(
        dict(
            actor=dict(
                model=dict(
                    model_type='fluxvla_smolvla',
                    model_path=str(sft / 'model.pt'),
                    num_steps=10,
                    num_action_chunks=7,
                    action_dim=7,
                    fluxvla=dict(
                        config_path=str(
                            root /
                            'configs/smolvla/smolvla_libero_10_finetune.py'),
                        norm_stats_path=None,
                        tokenizer_path=None,
                        compute_dtype='bf16',
                        observation_adapter=dict(type='LiberoRLAdapter'),
                        action_horizon=20,
                        model_overrides=dict(num_steps=3)))),
            runner=dict(
                logger=dict(log_path=str(tmp_path), experiment_name='run'),
                export_native_policy=True)))
    return step, state, cfg


@pytest.mark.parametrize('compute_dtype', ['bf16', 'fp32'])
def test_bundle_can_be_moved_without_sft_or_checkpoint(bundle_inputs, tmp_path,
                                                       compute_dtype):
    step, state, cfg = bundle_inputs
    cfg.actor.model.fluxvla.compute_dtype = compute_dtype
    output = tmp_path / 'export'
    manifest = export_native_bundle(step, output, cfg)
    assert manifest['policy_tensor_count'] == 2
    assert manifest['removed_value_head_tensors'] == 1
    assert manifest['resumable'] is False
    exported = load_file(str(output / 'model.safetensors'))
    assert set(exported) == {'policy.weight', 'policy.buffer'}
    for key, tensor in exported.items():
        assert torch.equal(tensor,
                           state[key]) and tensor.dtype == state[key].dtype
    original = torch.load(
        step / 'actor/model_state_dict/full_weights.pt', weights_only=True)
    assert 'value_head.weight' in original
    moved = tmp_path / 'moved bundle'
    shutil.move(output, moved)
    shutil.rmtree(step)
    shutil.rmtree(tmp_path / 'sft')
    evaluation = Config.fromfile(moved / 'eval_config.py')
    assert evaluation.model.num_steps == 3  # Overrides match RL builder order.
    assert evaluation.model.chunk_size == 20
    assert evaluation.model.name_mapping is None
    assert evaluation.model.vlm_backbone.torch_dtype == 'float32'
    assert evaluation.eval.model_build_dtype == 'float32'
    assert evaluation.eval.eval_chunk_size == 7
    assert evaluation.eval.enable_mixed_precision_training == (
        compute_dtype == 'bf16')
    assert evaluation.eval.dataset.state_dtype == ('bfloat16' if compute_dtype
                                                   == 'bf16' else 'float32')
    assert Path(
        evaluation.eval.norm_stats_path) == moved / 'dataset_statistics.json'
    tokenizer_path = evaluation.eval.dataset.transforms[2].tokenizer.model_path
    assert Path(tokenizer_path) == moved / 'tokenizer'
    assert Path(tokenizer_path, 'tokenizer.json').exists()
    assert (moved / 'rl_config.yaml').exists()


def test_failed_bundle_leaves_no_partial_export(bundle_inputs, tmp_path,
                                                monkeypatch):
    step, _, cfg = bundle_inputs
    output = tmp_path / 'export'
    import fluxvla.engines.utils.rl_export as module

    def fail_copy(*args, **kwargs):
        raise OSError('copy failure')

    monkeypatch.setattr(module.shutil, 'copytree', fail_copy)
    with pytest.raises(OSError, match='copy failure'):
        export_native_bundle(step, output, cfg)
    assert not output.exists()
    assert not list(tmp_path.glob('.native_export-*'))
    assert (step / 'actor/model_state_dict/full_weights.pt').exists()


def test_existing_bundle_is_not_overwritten(bundle_inputs, tmp_path):
    step, _, cfg = bundle_inputs
    output = tmp_path / 'export'
    output.mkdir()
    (output / 'keep').write_text('original')
    with pytest.raises(FileExistsError):
        export_native_bundle(step, output, cfg)
    assert (output / 'keep').read_text() == 'original'


def test_unsupported_model_is_not_mislabelled(bundle_inputs, tmp_path):
    step, _, cfg = bundle_inputs
    cfg.actor.model.model_type = 'fluxvla_pi05'
    with pytest.raises(ValueError, match='SmolVLA LIBERO'):
        export_native_bundle(step, tmp_path / 'export', cfg)


def test_checkpoint_hook_waits_then_exports_and_preserves_resume_on_failure(
        bundle_inputs, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import fluxvla.engines.runners.rlinf_checkpoint_runner as module
    step, _, cfg = bundle_inputs
    calls = []
    cfg.runner.logger.log_path = str(tmp_path)
    cfg.runner.logger.experiment_name = 'run'
    runner = module.FluxEmbodiedRunner.__new__(module.FluxEmbodiedRunner)
    runner.cfg, runner.global_step = cfg, 1
    runner.logger = SimpleNamespace(
        info=lambda text: calls.append('logged'),
        exception=lambda text: calls.append('export_error'))
    expected = tmp_path / 'run/checkpoints/global_step_1'

    def save(self):
        calls.append('saved')
        shutil.copytree(step, expected)

    def export(checkpoint, output, run_cfg):
        assert checkpoint == expected
        assert (checkpoint / 'actor/model_state_dict/full_weights.pt').exists()
        calls.append('exported')
        raise OSError('no space')

    monkeypatch.setattr(module.EmbodiedRunner, '_save_checkpoint', save)
    monkeypatch.setattr(module, 'export_native_bundle', export)
    runner._save_checkpoint()
    assert calls == ['saved', 'exported', 'export_error']
    assert (expected / 'actor/model_state_dict/full_weights.pt').exists()


def test_checkpoint_hook_can_disable_export(bundle_inputs, monkeypatch):
    import fluxvla.engines.runners.rlinf_checkpoint_runner as module
    step, _, cfg = bundle_inputs
    cfg.runner.export_native_policy = False
    runner = module.FluxEmbodiedRunner.__new__(module.FluxEmbodiedRunner)
    runner.cfg = cfg
    calls = []
    monkeypatch.setattr(module.EmbodiedRunner, '_save_checkpoint',
                        lambda self: calls.append('saved'))
    monkeypatch.setattr(module, 'export_native_bundle',
                        lambda *args: pytest.fail('export should be disabled'))
    runner._save_checkpoint()
    assert calls == ['saved']


def test_checkpoint_hook_publishes_bundle_after_save(bundle_inputs, tmp_path,
                                                     monkeypatch):
    from types import SimpleNamespace

    import fluxvla.engines.runners.rlinf_checkpoint_runner as module
    step, _, cfg = bundle_inputs
    expected = tmp_path / 'run/checkpoints/global_step_1'
    runner = module.FluxEmbodiedRunner.__new__(module.FluxEmbodiedRunner)
    runner.cfg, runner.global_step = cfg, 1
    runner.logger = SimpleNamespace(
        info=lambda text: None, exception=lambda text: pytest.fail(text))
    monkeypatch.setattr(module.EmbodiedRunner, '_save_checkpoint',
                        lambda self: shutil.copytree(step, expected))
    runner._save_checkpoint()
    assert (expected / 'actor/model_state_dict/full_weights.pt').exists()
    assert (expected / 'rl_config.yaml').exists()
    assert (expected / 'native_policy/model.safetensors').exists()


@pytest.mark.parametrize('state_dtype', ['float32', 'bfloat16'])
def test_eval_dataset_emits_configured_state_dtype(state_dtype, monkeypatch):
    import numpy as np

    from fluxvla.datasets.parquet_dataset import LiberoParquetEvalDataset
    dataset = LiberoParquetEvalDataset(
        norm_stats=None,
        task_suite_name='libero_10',
        transforms=[],
        norm_stats_key='libero_10_no_noops',
        state_dtype=state_dtype)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda tensor: tensor)
    original = np.array([1.0001, 2.0001], dtype=np.float32)
    batch, _ = dataset(
        dict(
            states=original,
            lang_tokens=[1],
            lang_masks=[True],
            pixel_values=torch.zeros(6, 4, 4)))
    assert batch['states'].dtype == getattr(torch, state_dtype)
    assert torch.equal(
        batch['states'][0],
        torch.from_numpy(original).to(getattr(torch, state_dtype)))
