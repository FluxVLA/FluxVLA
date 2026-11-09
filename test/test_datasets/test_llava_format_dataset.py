"""Local-only processor, label masking, dataset and mixed-media batch tests."""

import json
import os
import pickle
import socket

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (PreTrainedTokenizerFast, Qwen2VLImageProcessor,
                          Qwen3VLProcessor, Qwen3VLVideoProcessor)

from fluxvla.collators.vlm_sft_collator import VLMSFTCollator
from fluxvla.datasets.llava_format_dataset import (LLaVAFormatDataset,
                                                   _load_json_source)
from fluxvla.transforms import vlm_transforms

_SOCKET_CONNECT = socket.socket.connect


@pytest.fixture()
def processor(tmp_path):
    words = [
        '[UNK]', '[PAD]', '<|im_start|>', '<|im_end|>', 'assistant', 'user',
        'system', 'hello', 'answer', 'again', '<|image_pad|>', '<|video_pad|>',
        '<|vision_start|>', '<|vision_end|>'
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)))), unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token='[UNK]',
        pad_token='[PAD]',
        eos_token='<|im_end|>',
        additional_special_tokens=[
            '<|im_start|>', '<|image_pad|>', '<|video_pad|>',
            '<|vision_start|>', '<|vision_end|>'
        ])
    template = ('{% for message in messages %}'
                "{{ '<|im_start|>' + message['role'] + '\n' }}"
                "{% for content in message['content'] %}"
                "{% if content['type'] == 'image' %}"
                "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
                "{% elif content['type'] == 'video' %}"
                "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
                "{% else %}{{ content['text'] }}{% endif %}"
                "{% endfor %}{{ '<|im_end|>\n' }}{% endfor %}"
                '{% if add_generation_prompt %}'
                "{{ '<|im_start|>assistant\n' }}{% endif %}")
    result = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=2,
            temporal_patch_size=1,
            merge_size=2,
            min_pixels=16,
            max_pixels=64),
        video_processor=Qwen3VLVideoProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            size=dict(shortest_edge=16, longest_edge=64)),
        tokenizer=tokenizer,
        chat_template=template)
    result.save_pretrained(tmp_path / 'processor')
    return result


@pytest.fixture()
def video_path(tmp_path):
    import av
    path = tmp_path / 'tiny.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=8)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for index in range(32):
            pixels = np.full((16, 16, 3), index * 7, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def conversation(image=None):
    result = {
        'conversations': [{
            'from': 'system',
            'value': 'hello'
        }, {
            'from': 'human',
            'value': '<image> hello' if image else 'hello'
        }, {
            'from': 'gpt',
            'value': 'answer'
        }, {
            'from': 'human',
            'value': 'again'
        }, {
            'from': 'gpt',
            'value': 'answer again'
        }]
    }
    if image:
        result['image'] = image
    return result


def pipeline(processor_path, **kwargs):
    return [
        dict(type='BuildVLMMessages'),
        dict(
            type='ProcessVLMSFTInputs',
            processor_path=str(processor_path),
            **kwargs),
    ]


def process_sample(processor, item, data_root=None, **kwargs):
    transform = vlm_transforms.ProcessVLMSFTInputs(
        processor_path='local-test-processor', **kwargs)
    transform._processor = processor
    transform._processor_pid = os.getpid()
    data = vlm_transforms.BuildVLMMessages()(dict(item, data_root=data_root))
    return transform(data)


def test_labels_supervise_all_assistant_turns_not_prompts(processor):
    output = process_sample(processor, conversation())
    labels = output['labels']
    expected = processor.tokenizer.encode(
        'answer<|im_end|> answer again<|im_end|>', add_special_tokens=False)
    assert labels[labels != -100].tolist() == expected
    assert labels.shape == output['input_ids'].shape
    assert (labels == -100).sum() > 0


def test_labels_handle_batch_and_eos_equal_pad(processor):
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    tokens = processor.tokenizer.encode(
        '<|im_start|>user\nhello<|im_end|>'
        '<|im_start|>assistant\nanswer<|im_end|>',
        add_special_tokens=False)
    result = vlm_transforms._mask_labels_for_sft(
        torch.tensor([tokens, tokens]), processor.tokenizer)
    assert result.shape == (2, len(tokens))
    assert result[:, -1].tolist() == [processor.tokenizer.eos_token_id] * 2
    assert (result[:, :-2] == -100).all()


def test_prompt_only_sample_is_rejected(processor):
    with pytest.raises(ValueError, match='No assistant'):
        process_sample(
            processor,
            {'conversations': [{
                'from': 'human',
                'value': 'hello'
            }]})


@pytest.mark.parametrize('processor_type', ['qwen3_vl', 'qwen3_5'])
def test_generation_prompt_uses_native_shapes_and_no_labels(
        processor, tmp_path, processor_type):
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(image_path)
    transform = vlm_transforms.ProcessVLMSFTInputs(
        processor_path=str(tmp_path / 'processor'),
        processor_type=processor_type,
        image_max_pixels=16,
        truncation_max_length=128)
    inputs = transform.encode_prompt('hello', [image_path])
    assert 'labels' not in inputs
    assert inputs['input_ids'].ndim == 2
    assert inputs['attention_mask'].shape == inputs['input_ids'].shape
    assert inputs['pixel_values'].shape == (4, 12)
    assert inputs['image_grid_thw'].tolist() == [[1, 2, 2]]
    header = processor.tokenizer.encode('<|im_start|>assistant\n')
    assert inputs['input_ids'][0, -len(header):].tolist() == header
    assert processor.tokenizer.convert_tokens_to_ids('answer') not in (
        inputs['input_ids'][0].tolist())
    answer = processor.tokenizer.encode('answer<|im_end|>')
    assert transform.decode_answer(answer) == 'answer'
    assert transform.eos_token_id == processor.tokenizer.eos_token_id
    metadata = transform.generation_metadata()
    assert metadata['processor_type'] == processor_type
    assert metadata['image_max_pixels'] == 16
    assert len(metadata['tokenizer_sha256']) == 64
    transform.truncation_max_length = 2
    with pytest.raises(ValueError, match='prompt exceeds'):
        transform.encode_prompt('hello', [image_path])


def test_message_roles_whitespace_and_media_validation():
    messages = vlm_transforms.build_messages_llava(conversation('image.png'))
    assert [m['role'] for m in messages
            ] == ['system', 'user', 'assistant', 'user', 'assistant']
    assert messages[1]['content'][-1]['text'] == ' hello'
    with pytest.raises(ValueError, match='matching placeholder'):
        vlm_transforms.build_messages_llava(
            dict(conversation(), image='unused.png'))
    with pytest.raises(ValueError, match='Unsupported conversation role'):
        vlm_transforms.build_messages_llava(
            {'conversations': [{
                'from': 'typo',
                'value': 'hello'
            }]})


@pytest.mark.parametrize('extension', ['json', 'jsonl'])
def test_actual_image_processor_dataset_and_collator(tmp_path, processor,
                                                     extension):
    root = tmp_path / 'images'
    root.mkdir()
    Image.new('RGB', (4, 4), color='red').save(root / 'tiny.png')
    annotations = tmp_path / f'data.{extension}'
    rows = [conversation('tiny.png'), conversation()]
    annotations.write_text(
        json.dumps(rows) if extension == 'json' else '\n'.join(
            json.dumps(item) for item in rows))
    dataset = LLaVAFormatDataset(
        str(annotations),
        data_root_path=str(root),
        transforms=pipeline(tmp_path / 'processor'),
        max_retry_times=0)
    first, second = dataset[0], dataset[1]
    assert first['image_grid_thw'].shape == (1, 3)
    assert first['pixel_values'].shape == (4, 12)
    batch = VLMSFTCollator(processor.tokenizer.pad_token_id)([first, second])
    assert batch['input_ids'].shape[0] == 2
    assert batch['image_grid_thw'].shape == (1, 3)
    assert batch['pixel_values'].shape == (4, 12)
    assert torch.equal(batch['labels'][0, :len(first['labels'])],
                       first['labels'])


def test_truncation_must_not_orphan_visual_patches(tmp_path, processor):
    path = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(path)
    with pytest.raises(ValueError, match='visual tokens'):
        process_sample(
            processor, conversation(str(path)), truncation_max_length=2)


def test_pixel_limit_preserves_valid_minimum(tmp_path, processor):
    loaded = vlm_transforms.get_processor_for_vlm(
        'qwen3_vl', str(tmp_path / 'processor'), image_max_pixels=32)
    assert loaded.image_processor.size['longest_edge'] == 32
    assert loaded.image_processor.size['shortest_edge'] <= 32


def test_annotation_errors_weights_and_root_validation(tmp_path):
    path = tmp_path / 'data.json'
    path.write_text(json.dumps([conversation()] * 4))
    assert len(_load_json_source({'path': str(path), 'weight': 0.5})) == 2
    assert len(_load_json_source({'path': str(path), 'weight': 2})) == 8
    assert _load_json_source({'path': str(path), 'weight': 0}) == []
    with pytest.raises(ValueError):
        _load_json_source({'path': str(path), 'weight': -1})
    with pytest.raises(FileNotFoundError):
        _load_json_source(str(tmp_path / 'missing.json'))
    with pytest.raises(ValueError, match='equal lengths'):
        LLaVAFormatDataset([str(path)], [])
    path.write_text('{malformed}\n')
    with pytest.raises(json.JSONDecodeError):
        _load_json_source(str(path))


def test_retry_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / 'data.json'
    path.write_text(json.dumps([conversation(), conversation()]))
    dataset = LLaVAFormatDataset(str(path), max_retry_times=2)
    calls = []

    def fail(index):
        calls.append(index)
        raise OSError('corrupt image')

    monkeypatch.setattr(dataset, '_build_sample', fail)
    with pytest.raises(RuntimeError, match='3 attempts'):
        dataset[0]
    assert calls == [0, 1, 0]


def test_collator_mixed_media_masks_eos_and_token_types():
    image = dict(
        input_ids=torch.tensor([1, 8, 2]),
        labels=torch.tensor([-100, 8, 2]),
        attention_mask=torch.ones(3, dtype=torch.long),
        pixel_values=torch.ones(4, 12),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mm_token_type_ids=torch.tensor([0, 1, 0]))
    video = dict(
        input_ids=torch.tensor([1, 9, 7, 2]),
        labels=torch.tensor([-100, -100, 7, 2]),
        pixel_values_videos=torch.ones(8, 12),
        video_grid_thw=torch.tensor([[2, 2, 2]]),
        mm_token_type_ids=torch.tensor([0, 2, 2, 0]))
    text = dict(input_ids=torch.tensor([1, 2]), labels=torch.tensor([-100, 2]))
    batch = VLMSFTCollator(pad_token_id=2)([image, video, text])
    assert batch['attention_mask'].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1],
                                                [1, 1, 0, 0]]
    assert batch['labels'][2].tolist() == [-100, 2, -100, -100]
    assert batch['mm_token_type_ids'].tolist() == [[0, 1, 0, 0], [0, 2, 2, 0],
                                                   [0, 0, 0, 0]]
    assert batch['video_grid_thw'].shape == (1, 3)
    assert batch['pixel_values_videos'].shape == (8, 12)
    with pytest.raises(ValueError, match='patch count'):
        VLMSFTCollator(2)([dict(image, pixel_values=torch.ones(3, 12))])


@pytest.mark.parametrize('sampling', [
    dict(video_num_frames=4, video_max_frames=4),
    dict(video_fps=100, video_max_frames=4),
])
@pytest.mark.parametrize('extension', ['json', 'jsonl'])
def test_real_video_multimage_text_and_multiworker_batch(
        tmp_path, processor, video_path, sampling, extension, monkeypatch):
    # Tensor IPC uses AF_UNIX sockets. Permit only local IPC, not networking,
    # while retaining the suite's offline guard for AF_INET/AF_INET6.
    def connect_local(sock, address):
        if sock.family != socket.AF_UNIX:
            raise AssertionError('Only local tensor IPC is allowed.')
        return _SOCKET_CONNECT(sock, address)

    monkeypatch.setattr(socket.socket, 'connect', connect_local)
    Image.new('RGB', (4, 4), color='red').save(tmp_path / 'red.png')
    Image.new('RGB', (4, 4), color='blue').save(tmp_path / 'blue.png')
    images = dict(
        image=['red.png', 'blue.png'],
        conversations=[
            dict(role='user', content='<image> hello <image> again'),
            dict(role='assistant', content='answer'),
        ])
    video = dict(
        video=video_path.name,
        messages=[
            dict(role='user', content='<video> hello'),
            dict(role='assistant', content='answer'),
        ])
    interleaved = dict(
        image='red.png',
        video=[video_path.name, video_path.name],
        conversations=[
            dict(role='user', content='<image> <video> <video>'),
            dict(role='assistant', content='answer'),
        ])
    path = tmp_path / f'mixed.{extension}'
    rows = [video, images, conversation(), interleaved]
    path.write_text(
        json.dumps(rows) if extension == 'json' else '\n'.join(
            json.dumps(item) for item in rows))
    dataset = LLaVAFormatDataset(
        str(path),
        str(tmp_path),
        transforms=pipeline(tmp_path / 'processor', **sampling),
        max_retry_times=0,
        index_cache_dir=tmp_path / 'index')
    samples = [dataset[index] for index in range(4)]
    # Four sampled frames -> two temporal patches; a high FPS must not bypass
    # the bound and decode all 32 frames into model inputs.
    assert samples[0]['video_grid_thw'][0, 0] == 2
    assert samples[0]['messages'][0]['content'][0]['video'] == str(video_path)
    assert samples[1]['image_grid_thw'].shape == (2, 3)
    assert samples[3]['video_grid_thw'].shape == (2, 3)
    assert samples[3]['image_grid_thw'].shape == (1, 3)
    collator = VLMSFTCollator(processor.tokenizer.pad_token_id)
    expected = [collator(samples[:2]), collator(samples[2:])]
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=2, collate_fn=collator)
    actual = list(loader)
    for expected_batch, actual_batch in zip(expected, actual):
        for key in expected_batch:
            torch.testing.assert_close(actual_batch[key], expected_batch[key])
    # An initialized dataset remains small and pickleable for spawn workers.
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.transforms[-1]._processor is None
    torch.testing.assert_close(restored[0]['input_ids'],
                               samples[0]['input_ids'])


def test_decoded_video_preserves_frame_indices_and_short_clip_repeats(
        processor, video_path):
    frames, metadata = vlm_transforms._decode_video(
        str(video_path), processor.video_processor,
        vlm_transforms.video_sampling_kwargs(num_frames=4))
    assert metadata.frames_indices == [0, 10, 21, 31]
    assert metadata.fps == 8
    assert frames.shape == (4, 16, 16, 3)
    np.testing.assert_allclose(
        frames.mean(axis=(1, 2, 3)), np.array([0, 10, 21, 31]) * 7, atol=5)
    frames, metadata = vlm_transforms._decode_video(
        str(video_path), processor.video_processor,
        vlm_transforms.video_sampling_kwargs(num_frames=40, max_frames=40))
    assert len(frames) == len(metadata.frames_indices) == 40
    assert len(set(metadata.frames_indices)) == 32


@pytest.mark.parametrize('kwargs', [
    dict(num_frames=4, fps=2),
    dict(max_frames=0),
    dict(max_frames=True),
    dict(num_frames=33),
    dict(fps=float('nan')),
    dict(fps=-1),
])
def test_invalid_video_bounds(kwargs):
    with pytest.raises(ValueError):
        vlm_transforms.video_sampling_kwargs(**kwargs)


@pytest.mark.parametrize('extension', ['json', 'jsonl'])
def test_raw_dataset_needs_no_processor_or_media(tmp_path, monkeypatch,
                                                 extension):

    def fail(*args, **kwargs):
        pytest.fail('Raw dataset must not initialize a processor.')

    monkeypatch.setattr(vlm_transforms, 'get_processor_for_vlm', fail)
    item = conversation('not-downloaded.png')
    path = tmp_path / ('raw.' + extension)
    path.write_text(json.dumps([item] if extension == 'json' else item))
    dataset = LLaVAFormatDataset(
        path, data_root_path=tmp_path, index_cache_dir=tmp_path / 'index')
    assert dataset.transforms == []
    assert not hasattr(dataset, '_processor')
    assert not hasattr(dataset, 'model_path')
    assert dataset[0] == dict(item, data_root=str(tmp_path))
    assert 'messages' not in dataset[0]
    assert 'input_ids' not in dataset[0]


@pytest.mark.parametrize('extension', ['json', 'jsonl'])
def test_configured_transforms_run_in_order_without_mutating_source(
        tmp_path, extension):

    class AppendText:

        def __init__(self, text):
            self.text = text

        def __call__(self, data):
            data['conversations'][0]['value'] += self.text
            data.setdefault('transform_order', []).append(self.text)
            return data

    item = conversation()
    path = tmp_path / ('raw.' + extension)
    path.write_text(json.dumps([item] if extension == 'json' else item))
    dataset = LLaVAFormatDataset(
        dict(path=str(path), weight=2),
        transforms=[
            dict(type=AppendText, text='A'),
            dict(type=AppendText, text='B')
        ],
        index_cache_dir=tmp_path / 'index')
    for index in [0, 1, 0]:
        sample = dataset[index]
        assert sample['transform_order'] == ['A', 'B']
        assert sample['conversations'][0]['value'] == 'helloAB'
    assert dataset._sources[0].records[0] == item


def test_messages_transform_uses_each_source_root(tmp_path):
    first, second = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    first.write_text(json.dumps(conversation('same-name.png')))
    second.write_text(first.read_text())
    dataset = LLaVAFormatDataset(
        [str(first), str(second)],
        [str(tmp_path / 'first'),
         str(tmp_path / 'second')],
        transforms=[dict(type='BuildVLMMessages')],
        index_cache_dir=tmp_path / 'index')
    assert dataset[0]['messages'][1]['content'][0]['image'] == str(
        tmp_path / 'first/same-name.png')
    assert dataset[1]['messages'][1]['content'][0]['image'] == str(
        tmp_path / 'second/same-name.png')


def test_pipeline_allows_editing_messages_before_encoding(tmp_path, processor):

    class ChangeAnswer:

        def __call__(self, data):
            for message in data['messages']:
                if message['role'] == 'assistant':
                    message['content'] = [dict(type='text', text='again')]
            return data

    path = tmp_path / 'text.jsonl'
    path.write_text(json.dumps(conversation()))
    transforms = pipeline(tmp_path / 'processor')
    transforms.insert(1, dict(type=ChangeAnswer))
    dataset = LLaVAFormatDataset(
        path, transforms=transforms, index_cache_dir=tmp_path / 'index')
    result = dataset[0]
    expected = processor.tokenizer.encode(
        'again<|im_end|> again<|im_end|>', add_special_tokens=False)
    assert result['labels'][result['labels'] != -100].tolist() == expected
    assert dataset._sources[0].records[0] == conversation()
