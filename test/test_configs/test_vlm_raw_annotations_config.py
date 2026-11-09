# Copyright 2026 Limx Dynamics
"""Raw LLaVA examples use the portable directory layout in the READMEs."""

import re
from collections.abc import Mapping
from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
LLAVA_ROOT = Path('datasets/llava665k')
INDEX_ROOT = Path('datasets/.cache/vlm')

RAW_CONFIGS = [
    ('vlm/qwen3vl_2b_llava665k_full_finetune.py', 'train_dataloader'),
    ('vlm/qwen_llava665k_full_finetune.py', 'train_dataloader'),
    ('pi05/pi05_paligemma_llava665k_full_finetune.py', 'train_dataloader'),
    ('pi05/pi05_paligemma_robotwin_llava665k_mixed_full_finetune.py',
     'vlm_dataloader'),
    ('pi05/pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py',
     'vlm_dataloader'),
]
RELATED_CONFIGS = [config for config, _ in RAW_CONFIGS] + [
    'pi05/pi05_paligemma_robotwin_all_data_full_finetune.py',
]


@pytest.fixture(autouse=True)
def portable_config_environment(monkeypatch, tmp_path):
    # Test defaults on a machine with no datasets, not a developer's mounts.
    for key in ('FLUXVLA_VLM_PATH', 'FLUXVLA_VLM_PROCESSOR',
                'FLUXVLA_VLM_TRAIN', 'FLUXVLA_VLM_VAL', 'FLUXVLA_VLM_MEDIA',
                'ROBOCASA_DATA_ROOT', 'PI05_CHECKPOINT', 'PI05_TOKENIZER'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize('config,loader', RAW_CONFIGS)
def test_llava_examples_read_raw_json(config, loader):
    cfg = Config.fromfile(ROOT / 'configs' / config)
    dataset = cfg[loader].dataset
    annotation = Path(dataset.anno_json_path)
    assert dataset.type == 'LLaVAFormatDataset'
    assert annotation.name == 'llava_v1_5_mix665k.json'
    assert annotation.parent == Path(dataset.data_root_path) == LLAVA_ROOT
    assert dataset.transforms[0].type == 'BuildVLMMessages'
    assert Path(dataset.index_cache_dir) == INDEX_ROOT
    # A symlinked raw-data root may be read-only; cache outside that root.
    assert not INDEX_ROOT.is_relative_to(LLAVA_ROOT)


def assert_relative_paths(value, key='config'):
    if isinstance(value, Mapping):
        for name, item in value.items():
            assert_relative_paths(item, f'{key}.{name}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_relative_paths(item, f'{key}[{index}]')
    elif isinstance(value, str):
        assert not Path(value).is_absolute(), f'{key}: {value}'
        assert not value.startswith('~/'), f'{key}: {value}'


@pytest.mark.parametrize('config', RELATED_CONFIGS)
def test_all_inherited_runtime_config_paths_are_relative(config):
    cfg = Config.fromfile(ROOT / 'configs' / config)
    # Legacy base configs may retain private shared-mount fallback constants.
    # Only paths in the new recipe's effective runtime settings must be local.
    assert_relative_paths({
        key: value
        for key, value in cfg.to_dict().items() if not key.startswith('_')
    })


def test_robotwin_mixture_keeps_documented_robot_paths():
    cfg = Config.fromfile(
        ROOT / 'configs/pi05/'
        'pi05_paligemma_robotwin_llava665k_mixed_full_finetune.py')
    roots = cfg.train_dataloader.dataset.datasets[0].data_root_path
    assert [Path(root) for root in roots] == [
        Path('datasets/robotwin_lerobotv2.1/robotwin_clean_lerobotv2.1'),
        Path('datasets/robotwin_lerobotv2.1/robotwin_randomized_lerobotv2.1'),
    ]


def test_robocasa_mixed_defaults_do_not_fall_back_to_shared_mounts():
    cfg = Config.fromfile(
        ROOT / 'configs/pi05/'
        'pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py')
    roots = cfg.train_dataloader.dataset.datasets.data_root_path
    assert len(roots) == 24
    assert all(
        Path(root).parent == Path('datasets/robocasa_lerobot_V2.1')
        for root in roots)


def test_qwen_environment_overrides_still_work(monkeypatch):
    monkeypatch.setenv('FLUXVLA_VLM_PATH', './checkpoints/custom-vlm')
    monkeypatch.setenv('FLUXVLA_VLM_MEDIA', './datasets/custom')
    monkeypatch.setenv('FLUXVLA_VLM_TRAIN', './datasets/custom/train.jsonl')
    monkeypatch.setenv('FLUXVLA_VLM_VAL', './datasets/custom/val.jsonl')
    monkeypatch.setenv('FLUXVLA_VLM_PROCESSOR', 'qwen3_5')
    cfg = Config.fromfile(ROOT / 'configs/vlm/qwen_llava665k_full_finetune.py')
    assert cfg.model.vlm_backbone.model_path == './checkpoints/custom-vlm'
    assert cfg.train_dataloader.dataset.data_root_path == './datasets/custom'
    assert cfg.train_dataloader.dataset.anno_json_path.endswith('train.jsonl')
    assert cfg.eval_dataset.anno_json_path.endswith('val.jsonl')
    assert cfg.eval_dataset.index_cache_dir == './datasets/.cache/vlm'
    assert cfg.eval.type == 'TextVQAEvalRunner'
    assert cfg.eval.preprocessing.processor_path == './checkpoints/custom-vlm'
    assert cfg.eval.preprocessing.processor_type == 'qwen3_5'
    assert cfg.eval.image_root == './datasets/custom/textvqa/train_images'


@pytest.mark.parametrize('config', [name for name, _ in RAW_CONFIGS[:2]])
def test_qwen_training_budget_matches_pi05_llava(config):
    cfg = Config.fromfile(ROOT / 'configs' / config)
    baseline = Config.fromfile(
        ROOT / 'configs/pi05/pi05_paligemma_llava665k_full_finetune.py')
    assert cfg.runner.max_steps is baseline.runner.max_steps is None
    assert cfg.runner.max_epochs == baseline.runner.max_epochs == 1
    assert cfg.runner.grad_accumulation_steps == (
        baseline.runner.grad_accumulation_steps) == 2
    assert cfg.runner.save_epoch_interval == (
        baseline.runner.save_epoch_interval) == 1
    for key in ('max_keep_ckpts', 'optimizer', 'lr_scheduler'):
        assert cfg.runner[key] == baseline.runner[key]
    assert cfg.train_dataloader.per_device_batch_size == (
        baseline.train_dataloader.per_device_batch_size) == 4
    assert (2 * cfg.train_dataloader.per_device_batch_size *
            cfg.runner.grad_accumulation_steps) == 16
    assert Path(cfg.runner.metric.run_dir).name == Path(config).stem
    assert Path(cfg.eval.checkpoint_dir).parent == Path(
        cfg.runner.metric.run_dir)
    assert Path(cfg.eval.run_dir).parent == Path(cfg.runner.metric.run_dir)


@pytest.mark.parametrize('config', [name for name, _ in RAW_CONFIGS[3:]])
def test_mixed_training_keeps_explicit_step_budget(config):
    cfg = Config.fromfile(ROOT / 'configs' / config)
    assert cfg.runner.train_task == 'mixed'
    assert cfg.runner.max_epochs is None
    assert cfg.runner.max_steps > 0


@pytest.mark.parametrize('config', [name for name, _ in RAW_CONFIGS[:2]])
def test_qwen_textvqa_is_separate_from_sft_and_matches_its_processor(config):
    cfg = Config.fromfile(ROOT / 'configs' / config)
    assert cfg.eval.type == 'TextVQAEvalRunner'
    assert cfg.eval.compare_base is True
    assert cfg.eval.max_samples is None
    assert cfg.eval.max_new_tokens == 32
    assert Path(cfg.eval.annotation_path) == Path(
        'datasets/textvqa/TextVQA_0.5.1_val.json')
    assert Path(cfg.eval.image_root) == LLAVA_ROOT / 'textvqa/train_images'
    assert cfg.eval.annotation_path != (
        cfg.train_dataloader.dataset.anno_json_path)
    assert 'eval_dataset' not in cfg
    train_processor = cfg.train_dataloader.dataset.transforms[1]
    for key, value in cfg.eval.preprocessing.items():
        assert train_processor[key] == value


def test_vlm_configs_follow_model_dataset_training_mode_names():
    assert {path.name
            for path in (ROOT / 'configs/vlm').glob('*.py')} == {
                'qwen3vl_2b_llava665k_full_finetune.py',
                'qwen_llava665k_full_finetune.py',
            }


def test_robocasa_explicit_data_override_still_works(monkeypatch):
    monkeypatch.setenv('ROBOCASA_DATA_ROOT', './datasets/custom-robocasa')
    cfg = Config.fromfile(
        ROOT / 'configs/pi05/'
        'pi05_paligemma_robocasa_llava665k_mixed_full_finetune.py')
    roots = cfg.train_dataloader.dataset.datasets.data_root_path
    assert all(
        Path(root).parent == Path('datasets/custom-robocasa')
        for root in roots)


@pytest.mark.parametrize('readme',
                         ['README.md', 'README_zh-CN.md', 'README_ja.md'])
def test_readmes_document_the_same_raw_layout(readme):
    text = (ROOT / readme).read_text()
    section = next(
        block for block in text.split('<details>')
        if block.lstrip().startswith('<summary><b>LLaVA-665K'))
    section = section.split('</details>')[0]
    assert 'llava_v1_5_mix665k.json' in section
    assert './datasets/llava665k' in section
    assert './datasets/.cache/vlm/' in section
    for media in ('coco/train2017', 'gqa/images', 'ocr_vqa/images',
                  'textvqa/train_images', 'VG_100K', 'VG_100K_2'):
        assert media in section
    for link in re.findall(r'\]\(([^)]+)\)', section):
        if not link.startswith(('https://', 'http://')):
            assert (ROOT / link).is_file(), link
