# Copyright 2026 Limx Dynamics
"""TextVQA generation, checkpoint loading and native/PaliGemma contracts."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from mmengine import Config
from PIL import Image
from safetensors.torch import save_file

from fluxvla.engines.runners import textvqa_eval_runner as evaluation


class LocalTokenizer:
    eos_token_id = 1
    backend_tokenizer = SimpleNamespace(to_str=lambda: 'local-test-tokenizer')

    def bos_id(self):
        return 2

    def eos_id(self):
        return self.eos_token_id

    def pad_id(self):
        return 0

    def encode(self, text):
        assert 'base' not in text and 'trained' not in text
        return [7]

    def decode(self, tokens, skip_special_tokens=False):
        return ' '.join({
            4: 'base',
            5: 'trained'
        }.get(t, '') for t in tokens).strip()


class LocalNativeProcessor:
    tokenizer = LocalTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        content = messages[0]['content']
        assert [item['type'] for item in content] == ['image', 'text']
        assert 'base' not in content[1]['text']
        assert 'trained' not in content[1]['text']
        assert kwargs['add_generation_prompt'] is True
        assert kwargs['enable_thinking'] is False
        assert kwargs['truncation'] is False
        return dict(
            input_ids=torch.tensor([[2, 7]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            pixel_values=torch.ones(4, 12),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.tensor([[0, 1]]))


class LocalGenerationModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Module()
        self.head.weight = torch.nn.Parameter(torch.tensor([4.0]))
        self.pretrained_loads = 0

    def validate_training_task(self, task):
        assert task == 'vlm'

    def from_pretrained(self):
        self.pretrained_loads += 1

    def generate_text(self, input_ids, attention_mask, max_new_tokens,
                      eos_token_id, do_sample, **kwargs):
        assert not do_sample
        assert not self.training
        assert not torch.is_grad_enabled()
        assert input_ids.shape == attention_mask.shape
        assert input_ids.ndim == 2
        assert 'labels' not in kwargs
        if 'pixel_values' in kwargs:
            assert kwargs['pixel_values'].shape == (4, 12)
            assert kwargs['image_grid_thw'].shape == (1, 3)
            assert kwargs['mm_token_type_ids'].shape == input_ids.shape
        else:
            assert kwargs['images'].shape == (1, 3, 8, 8)
        answer = [int(self.head.weight.item()), eos_token_id][:max_new_tokens]
        return torch.cat((input_ids, torch.tensor([answer])), dim=1)


@pytest.fixture(params=['native', 'paligemma'])
def runner(request, tmp_path, monkeypatch):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    Image.new('RGB', (8, 8)).save(image_root / 'image.jpg')
    rows = [
        dict(
            question_id=index,
            image_id='image',
            set_name='val',
            question='What word is shown?',
            answers=['base'] * 3 + ['trained'] * 7) for index in range(3)
    ]
    annotation = tmp_path / 'val.json'
    annotation.write_text(json.dumps(dict(data=rows)))
    tokenizer = tmp_path / 'tokenizer.model'
    tokenizer.write_bytes(b'local-test-tokenizer')
    if request.param == 'native':
        preprocessing = dict(
            type='ProcessVLMSFTInputs',
            processor_path='local-native-model',
            image_max_pixels=16)
        model = dict(
            type='VLMWrapper',
            vlm_backbone=dict(
                type='CausalVLMBackbone', model_path='local-native-model'))
    else:
        # Existing Pi05 configs omit preprocessing.type; keep this working.
        preprocessing = dict(tokenizer_path=str(tokenizer), image_size=8)
        model = dict(
            type='PI05FlowMatching',
            pretrained_name_or_path='local-pi05-base.safetensors')
    cfg = Config(dict(model=model))
    runner = evaluation.TextVQAEvalRunner(
        cfg=cfg,
        annotation_path=annotation,
        image_root=image_root,
        preprocessing=preprocessing,
        ckpt_path='base',
        max_samples=2,
        max_new_tokens=2,
        run_dir=tmp_path / 'report',
        enable_mixed_precision_training=False,
        device='cpu')
    if request.param == 'native':
        runner.processor._processor = LocalNativeProcessor()
        runner.processor._processor_pid = os.getpid()
    else:
        runner.processor._tokenizer = LocalTokenizer()
        runner.processor._pid = os.getpid()
    monkeypatch.setattr(evaluation, 'build_vla_from_cfg',
                        lambda cfg: LocalGenerationModel())
    return runner


@pytest.mark.parametrize('format', ['safetensors', 'flat_pt', 'grouped_pt'])
def test_base_and_checkpoint_generation_report(runner, tmp_path, format):
    state = {'head.weight': torch.tensor([5.0])}
    if format == 'safetensors':
        checkpoint = tmp_path / 'step-2-model.safetensors'
        save_file(state, str(checkpoint))
    else:
        checkpoint = tmp_path / 'step-2-model.pt'
        if format == 'grouped_pt':
            state = {'head': {'weight': state['head.weight']}}
        torch.save(dict(model=state), checkpoint)
    runner.checkpoint = checkpoint
    runner.run_setup()
    report = runner.run()
    assert runner.vla.pretrained_loads == 1
    assert report['base_checkpoint'] == ('local-native-model'
                                         if runner.native_inputs else
                                         'local-pi05-base.safetensors')
    assert report['dataset']['total_questions'] == 3
    assert report['dataset']['evaluated_questions'] == 2
    assert report['metrics']['base']['accuracy'] == pytest.approx(0.9)
    assert report['metrics']['checkpoint']['accuracy'] == 1
    assert report['delta_accuracy_percentage_points'] == pytest.approx(10)
    assert report['metrics']['checkpoint']['token_limit_count'] == 0
    assert len(report['protocol']['tokenizer_sha256']) == 64
    if runner.native_inputs:
        assert report['protocol']['image_max_pixels'] == 16
    else:
        assert report['protocol']['image_size'] == 8
    assert json.loads((runner.run_dir / 'summary.json').read_text()) == report
    assert (runner.run_dir / 'summary.md').is_file()
    assert (runner.run_dir / 'config.py').is_file()
    predictions = [
        json.loads(line)
        for line in (runner.run_dir /
                     'checkpoint_predictions.jsonl').read_text().splitlines()
    ]
    assert [row['question_id'] for row in predictions] == [0, 1]
    assert all(row['prediction'] == 'trained' for row in predictions)


def test_base_only_and_token_limit(runner):
    runner.max_new_tokens = 1
    runner.run_setup()
    report = runner.run()
    assert set(report['metrics']) == {'base'}
    assert report['metrics']['base']['token_limit_count'] == 2


def test_checkpoint_only_does_not_require_base_weights(runner, tmp_path):
    checkpoint = tmp_path / 'step-2-model.safetensors'
    save_file({'head.weight': torch.tensor([5.0])}, str(checkpoint))
    runner.cfg.model.pop('pretrained_name_or_path', None)
    runner.cfg.model.pop('vlm_backbone', None)
    runner.compare_base = False
    runner.checkpoint = checkpoint
    runner.run_setup()
    report = runner.run()
    assert runner.vla.pretrained_loads == 0
    assert set(report['metrics']) == {'checkpoint'}


def test_base_comparison_rejects_random_initialization(runner):
    runner.cfg.model.pop('pretrained_name_or_path', None)
    runner.cfg.model.pop('vlm_backbone', None)
    with pytest.raises(ValueError, match='pretrained weights'):
        runner.run_setup()


def test_missing_images_fail_before_model_construction(runner, monkeypatch):
    (runner.image_root / 'image.jpg').unlink()
    monkeypatch.setattr(evaluation, 'build_vla_from_cfg',
                        lambda cfg: pytest.fail('Must validate media first.'))
    with pytest.raises(FileNotFoundError, match='Missing 1 TextVQA images'):
        runner.run_setup()


def test_checkpoint_loading_is_strict(runner, tmp_path):
    checkpoint = tmp_path / 'bad.safetensors'
    save_file({'wrong_key': torch.tensor([5.0])}, str(checkpoint))
    runner.checkpoint = checkpoint
    runner.run_setup()
    with pytest.raises(RuntimeError, match='Missing key'):
        runner._load_checkpoint()


def test_checkpoint_resolution_prefers_published_alias_then_latest_step(
        tmp_path):
    first = tmp_path / 'step-2-model.safetensors'
    second = tmp_path / 'step-10-model.pt'
    first.touch()
    second.touch()
    assert evaluation.resolve_vlm_checkpoint('base') is None
    assert evaluation.resolve_vlm_checkpoint('latest', tmp_path) == second
    alias = tmp_path / 'latest-checkpoint.safetensors'
    alias.symlink_to(first)
    assert evaluation.resolve_vlm_checkpoint(None, tmp_path) == first


def test_unknown_processor_fails_explicitly():
    with pytest.raises(ValueError, match='Unsupported TextVQA processor'):
        evaluation.TextVQAEvalRunner(
            cfg=Config(),
            annotation_path='unused',
            image_root='unused',
            preprocessing=dict(type='NotAProcessor'),
            ckpt_path='base')
