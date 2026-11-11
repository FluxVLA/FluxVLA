# Copyright 2026 Limx Dynamics
"""Generic LeRobot subtask sampling and native VLM processor/model units."""

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from mmengine import Config
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (PreTrainedTokenizerFast, Qwen2VLImageProcessor,
                          Qwen3VLProcessor, Qwen3VLVideoProcessor)

from fluxvla.collators.vlm_sft_collator import VLMSFTCollator
from fluxvla.datasets.lerobot_subtask_dataset import LeRobotSubtaskDataset
from fluxvla.engines import build_dataset_from_cfg
from fluxvla.models.backbones.vlms.causal_vlm import \
    CausalVLMBackbone  # noqa: F401
from fluxvla.models.vlas.vlm_wrapper import VLMWrapper
from fluxvla.transforms import subtask_vlm_transforms as subtask_transforms
from fluxvla.transforms.vlm_transforms import ProcessVLMSFTInputs

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/vlm/qwen3_5_4b_robodojo_subtask_full_finetune.py'
CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
VIDEO_KEYS = ['observation.images.' + name for name in CAMERAS]
TARGETS = [
    'The right arm grasped the purple block.',
    'The left arm placed the wooden block on the purple block.',
    'The right arm released the block.'
]


def write_metadata(root, records):
    path = root / 'meta/episodes/chunk-000/file-000.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / 'data'
    records = [
        dict(
            episode_index=index,
            length=8,
            tasks=[f'Whole task {index // 4}'],
            dense_subtask_names=['  ' + s + '  ' for s in TARGETS],
            dense_subtask_start_frames=[0, 1, 5],
            dense_subtask_end_frames=[1, 5, 8]) for index in range(8)
    ]
    write_metadata(root, records)
    (root / 'meta/info.json').write_text(
        json.dumps(
            dict(
                fps=25,
                chunks_size=1000,
                video_path='videos/{video_key}/'
                'episode_{episode_index:06d}.mp4',
                features={key: dict(dtype='video')
                          for key in VIDEO_KEYS})))
    return root


@pytest.fixture
def processor_path(tmp_path):
    words = [
        '[UNK]', '[PAD]', '<|im_start|>', '<|im_end|>', '<|image_pad|>',
        '<|video_pad|>', '<|vision_start|>', '<|vision_end|>', 'assistant',
        'user'
    ]
    words += sorted(set(' '.join(TARGETS).split()) - set(words))
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)))), unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token='[UNK]',
        pad_token='[PAD]',
        eos_token='<|im_end|>',
        additional_special_tokens=words[2:8])
    template = ('{% for message in messages %}'
                "{{ '<|im_start|>' + message['role'] + '\n' }}"
                "{% for content in message['content'] %}"
                "{% if content['type'] == 'image' %}"
                "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
                "{% else %}{{ content['text'] }}{% endif %}"
                "{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}"
                '{% if add_generation_prompt %}'
                "{{ '<|im_start|>assistant\n' }}{% endif %}")
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Qwen2VLImageProcessor(
            patch_size=2,
            temporal_patch_size=1,
            merge_size=2,
            min_pixels=16,
            max_pixels=16),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            size=dict(shortest_edge=16, longest_edge=16)))
    path = tmp_path / 'processor'
    processor.save_pretrained(path)
    return path


def raw_sample(data_root):
    return LeRobotSubtaskDataset(data_root, split='all')[0]


def mock_decoder(monkeypatch):
    calls = []

    def decode(path, timestamps, **kwargs):
        calls.append((path, timestamps, kwargs))
        return torch.full((1, 3, 4, 4), len(calls), dtype=torch.uint8)

    monkeypatch.setattr(subtask_transforms, 'decode_video_frames', decode)
    return calls


def test_split_is_episode_disjoint_per_task_and_order_independent(data_root):
    train = LeRobotSubtaskDataset(data_root, validation_fraction=0.25)
    val = LeRobotSubtaskDataset(
        data_root, validation_fraction=0.25, split='val')
    assert set(train.episode_ids).isdisjoint(val.episode_ids)
    assert set(train.episode_ids) | set(val.episode_ids) == set(range(8))
    assert set(train.task_episode_counts.values()) == {3}
    assert set(val.task_episode_counts.values()) == {1}
    assert len(train) == 6 * 7
    assert len(val) == 2 * 7
    path = data_root / 'meta/episodes/chunk-000/file-000.parquet'
    write_metadata(data_root, list(reversed(pq.read_table(path).to_pylist())))
    repeat = LeRobotSubtaskDataset(data_root, validation_fraction=0.25)
    assert repeat.samples == train.samples


def test_frames_stay_in_current_segment_and_never_use_future_labels(data_root):
    dataset = LeRobotSubtaskDataset(data_root, split='all')
    assert not hasattr(dataset, 'dataset_statistics')
    samples = sorted([
        dataset[i] for i in range(len(dataset)) if dataset.samples[i][0] == 0
    ],
                     key=lambda s: s['frame_index'])
    assert [s['frame_index'] for s in samples] == [0, 1, 2, 4, 5, 6, 7]
    assert [s['subtask'] for s in samples
            ] == ([TARGETS[0]] + [TARGETS[1]] * 3 + [TARGETS[2]] * 3)
    for sample in samples:
        assert sample['subtask_start_frame'] <= sample['frame_index']
        assert sample['frame_index'] < sample['subtask_end_frame']
        assert sample['lerobot_video']['timestamps'] == [
            sample['frame_index'] / 25
        ]
        assert not sample['lerobot_video']['episode_meta']
        assert 'actions' not in sample


@pytest.mark.parametrize('prefix', ['dense', 'coarse'])
@pytest.mark.parametrize(
    'cameras',
    [['front'], ['overhead', 'wrist'], ['front', 'rear', 'left', 'right']])
def test_generic_registry_pipeline_with_custom_labels_and_cameras(
        tmp_path, monkeypatch, prefix, cameras):
    root = tmp_path / 'another_robot'
    keys = [f'observation.images.{name}' for name in cameras]
    labels = ['Grasp the cup.', 'Move the cup to the tray.']
    fps = 17
    row = dict(
        episode_index=43,
        length=4,
        tasks=['Put the cup on the tray.'],
        **{
            f'{prefix}_subtask_names': labels,
            f'{prefix}_subtask_start_frames': [0, 2],
            f'{prefix}_subtask_end_frames': [2, 4]
        })
    for index, key in enumerate(keys):
        row.update({
            f'videos/{key}/chunk_index': 7,
            f'videos/{key}/file_index': 9,
            f'videos/{key}/from_timestamp': 100.0 + index,
        })
    write_metadata(root, [row])
    (root / 'meta/info.json').write_text(
        json.dumps(
            dict(
                fps=fps,
                robot_type='custom_robot',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/'
                'file-{file_index:03d}.mp4',
                features={key: dict(dtype='video')
                          for key in keys})))
    calls = mock_decoder(monkeypatch)
    dataset = build_dataset_from_cfg(
        dict(
            type='LeRobotSubtaskDataset',
            data_root=str(root),
            subtask_prefix=prefix,
            frames_per_subtask=2,
            split='all',
            transforms=[dict(type='BuildSubtaskMessages', video_keys=keys)]))
    assert type(dataset) is LeRobotSubtaskDataset
    assert len(dataset) == 4 and dataset.episode_ids == (43, )
    assert not (root / 'data').exists()
    assert not (root / 'meta/stats.json').exists()
    sample = dataset[0]
    assert sample['subtask'] == labels[sample['frame_index'] // 2]
    assert [m['role'] for m in sample['messages']] == ['user', 'assistant']
    content = sample['messages'][0]['content']
    assert [item['text'] for item in content[:-1]
            if item['type'] == 'text'] == [f'Camera: {c}' for c in cameras]
    assert len([item for item in content
                if item['type'] == 'image']) == len(cameras)
    assert sample['messages'][1]['content'] == [
        dict(type='text', text=sample['subtask'])
    ]
    assert sample['full_task_description'] in content[-1]['text']
    assert all(label not in content[-1]['text'] for label in labels)
    for index, ((path, timestamps, _), key) in enumerate(zip(calls, keys)):
        assert path == str(root / f'videos/{key}/chunk-007/file-009.mp4')
        assert timestamps == [100.0 + index + sample['frame_index'] / fps]


def test_other_lerobot_data_still_requires_subtask_annotations(data_root):
    # A valid LeRobot layout without the configured annotation prefix must
    # fail explicitly, never silently use the whole task or another prefix.
    with pytest.raises(ValueError, match='Missing subtask annotations'):
        LeRobotSubtaskDataset(data_root, subtask_prefix='missing')


@pytest.mark.parametrize('options', [
    dict(frames_per_subtask=0),
    dict(frames_per_subtask=True),
    dict(validation_fraction=1),
    dict(validation_fraction=float('nan')),
    dict(split='test')
])
def test_invalid_sampling_configuration_fails(data_root, options):
    with pytest.raises(ValueError):
        LeRobotSubtaskDataset(data_root, **options)


def test_invalid_annotations_do_not_fall_back_to_whole_task(data_root):
    path = data_root / 'meta/episodes/chunk-000/file-000.parquet'
    records = pq.read_table(path).to_pylist()
    records[0]['dense_subtask_start_frames'] = [0, 2, 5]
    write_metadata(data_root, records)
    with pytest.raises(ValueError, match='Invalid subtask range'):
        LeRobotSubtaskDataset(data_root)


def test_empty_validation_split_rejected(data_root):
    with pytest.raises(ValueError, match='No samples'):
        LeRobotSubtaskDataset(data_root, validation_fraction=0, split='val')


def test_current_views_offsets_and_target_only_in_assistant(
        data_root, monkeypatch):
    sample = raw_sample(data_root)
    sample['lerobot_video']['episode_meta'] = {
        f'videos/{key}/from_timestamp': offset
        for key, offset in zip(VIDEO_KEYS, [2., 3., 4.])
    }
    original = copy.deepcopy(sample)
    calls = mock_decoder(monkeypatch)
    transform = subtask_transforms.BuildSubtaskMessages(VIDEO_KEYS)
    output = transform(sample)
    assert sample == original
    assert 'lerobot_video' not in output
    messages = output['messages']
    assert [m['role'] for m in messages] == ['user', 'assistant']
    user_text = ' '.join(p['text'] for p in messages[0]['content']
                         if p['type'] == 'text')
    assert sample['full_task_description'] in user_text
    assert all(label not in user_text for label in TARGETS)
    assert messages[-1]['content'] == [
        dict(type='text', text=sample['subtask'])
    ]
    assert [
        p['image'].getpixel((0, 0))[0] for p in messages[0]['content']
        if p['type'] == 'image'
    ] == [1, 2, 3]
    for (path, timestamps, kwargs), key, offset in zip(calls, VIDEO_KEYS,
                                                       [2., 3., 4.]):
        assert key in path
        assert timestamps == [sample['frame_index'] / 25 + offset]
        assert kwargs['tolerance_s'] < 0.5 / 25


def test_prompt_builder_generation_has_no_reference_answer():
    images = [Image.new('RGB', (4, 4)) for _ in CAMERAS]
    prompt = subtask_transforms.build_subtask_vlm_messages(
        'Stack blocks.', images, CAMERAS)
    training = subtask_transforms.build_subtask_vlm_messages(
        'Stack blocks.', images, CAMERAS, TARGETS[0])
    assert prompt == training[:-1]
    assert len(prompt) == 1 and prompt[0]['role'] == 'user'


def test_real_qwen_processor_masks_prompt_images_and_padding(
        data_root, processor_path, monkeypatch):
    mock_decoder(monkeypatch)
    transform = ProcessVLMSFTInputs(
        processor_path, processor_type='qwen3_5', enable_thinking=False)
    sample = subtask_transforms.BuildSubtaskMessages(VIDEO_KEYS)(
        raw_sample(data_root))
    processor = transform._get_processor()
    original = processor.apply_chat_template
    flags = []

    def render(*args, **kwargs):
        flags.append(kwargs.get('enable_thinking'))
        return original(*args, **kwargs)

    monkeypatch.setattr(processor, 'apply_chat_template', render)
    encoded = transform(sample)
    assert flags == [False]
    expected = processor.tokenizer.encode(
        sample['subtask'] + '<|im_end|>', add_special_tokens=False)
    targets = encoded['labels']
    assert targets[targets != -100].tolist() == expected
    assert encoded['image_grid_thw'].tolist() == [[1, 2, 2]] * 3
    batch = VLMSFTCollator(pad_token_id=0)([encoded, encoded])
    assert batch['image_grid_thw'].shape == (6, 3)
    assert batch['pixel_values'].shape == (24, 12)
    assert set(batch) <= {
        'input_ids', 'labels', 'attention_mask', 'pixel_values',
        'image_grid_thw', 'mm_token_type_ids'
    }


def test_config_uses_native_vlm_and_matching_processor(monkeypatch):
    monkeypatch.setenv('FLUXVLA_VLM_PATH', '/tmp/Qwen3.5-4B')
    monkeypatch.setenv('ROBODOJO_SUBTASK_DATA_ROOT', '/tmp/robodojo')
    cfg = Config.fromfile(CONFIG)
    assert cfg.model.type == 'VLMWrapper'
    assert cfg.model.vlm_backbone.model_path == '/tmp/Qwen3.5-4B'
    assert cfg.model.vlm_backbone.freeze_vision is False
    assert cfg.runner.train_task == 'vlm'
    assert cfg.runner.grad_accumulation_steps == 1
    assert cfg.runner.max_epochs == 3
    assert cfg.runner.lr_scheduler.type == 'linear-warmup+cosine-decay'
    train, val = cfg.train_dataloader.dataset, cfg.val_dataloader.dataset
    assert train.type == val.type == 'LeRobotSubtaskDataset'
    assert train.transforms[0].type == 'BuildSubtaskMessages'
    assert train.data_root == val.data_root == '/tmp/robodojo'
    assert train.split == 'train' and val.split == 'val'
    assert train.seed == val.seed and train.transforms == val.transforms
    assert train.transforms[-1].processor_path == '/tmp/Qwen3.5-4B'
    assert train.transforms[-1].processor_type == 'qwen3_5'
    assert train.transforms[-1].enable_thinking is False
    assert cfg.eval is cfg.themis is None


def test_native_qwen35_multiview_forward_backward_and_generation(
        data_root, processor_path, monkeypatch):
    mock_decoder(monkeypatch)
    transform = ProcessVLMSFTInputs(
        processor_path, processor_type='qwen3_5', enable_thinking=False)
    sample = subtask_transforms.BuildSubtaskMessages(VIDEO_KEYS)(
        raw_sample(data_root))
    batch = VLMSFTCollator(pad_token_id=0)([transform(sample)])
    tokenizer = transform._get_processor().tokenizer
    visual_tokens = {
        name + '_token_id': tokenizer.convert_tokens_to_ids(token)
        for name, token in [(
            'image', '<|image_pad|>'), (
                'video', '<|video_pad|>'), (
                    'vision_start',
                    '<|vision_start|>'), ('vision_end', '<|vision_end|>')]
    }
    config = dict(
        model_type='qwen3_5',
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=['linear_attention', 'full_attention'],
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            rope_parameters=dict(
                rope_type='default',
                partial_rotary_factor=1.0,
                mrope_section=[2, 1, 1])),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            num_position_embeddings=16),
        **visual_tokens)
    model = VLMWrapper(
        vlm_backbone=dict(type='CausalVLMBackbone', model_config=config))
    model.prepare_training_task('vlm')
    model.vlm_backbone.enable_gradient_checkpointing()
    assert callable(model.get_vlm_fsdp_wrapping_policy())
    output = model(**batch)
    assert torch.isfinite(output['loss']) and output['num_tokens'] > 0
    output['loss'].backward()
    for part in (model.vlm_backbone.vlm.model.visual,
                 model.vlm_backbone.vlm.model.language_model):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    model.eval()
    prompt = dict(transform._get_processor().apply_chat_template(
        sample['messages'][:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors='pt'))
    generated = model.generate_text(
        **prompt,
        max_new_tokens=2,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
        suppress_tokens=list(visual_tokens.values()))
    assert generated.shape[-1] == prompt['input_ids'].shape[-1] + 2
