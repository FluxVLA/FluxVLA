"""VLM transform registration, contracts and per-worker processor lifecycle."""

import copy
import pickle

import pytest
import torch

from fluxvla.engines import TRANSFORMS, build_transform_from_cfg
from fluxvla.transforms import vlm_transforms


def annotation():
    return dict(
        id='example',
        data_root='/data',
        conversations=[
            dict(role='user', content='<image> describe'),
            dict(role='assistant', content='A cup.'),
        ],
        image='cup.png')


def test_build_messages_is_registered_and_retains_raw_metadata():
    transform = build_transform_from_cfg(dict(type='BuildVLMMessages'))
    assert type(transform) is TRANSFORMS.get('BuildVLMMessages')
    sample = annotation()
    original = copy.deepcopy(sample)
    result = transform(sample)
    assert sample == original
    assert result['id'] == 'example'
    assert result['conversations'] == original['conversations']
    assert result['messages'][0]['content'][0] == dict(
        type='image', image='/data/cup.png')


def test_legacy_video_adapter_is_an_explicit_transform_option():
    sample = annotation()
    sample['video'] = 'clip.mp4'
    del sample['image']
    with pytest.raises(ValueError, match='Unmatched'):
        build_transform_from_cfg(dict(type='BuildVLMMessages'))(sample)
    transform = build_transform_from_cfg(
        dict(type='BuildVLMMessages', adapter='llava-video'))
    assert transform(sample)['messages'][0]['content'][0] == dict(
        type='video', video='/data/clip.mp4')


@pytest.mark.parametrize('sample', [{}, annotation(), dict(messages='bad')])
def test_encoder_requires_message_transform_before_loading(
        sample, monkeypatch):

    def fail(*args, **kwargs):
        pytest.fail('Bad pipeline must fail before loading a processor.')

    monkeypatch.setattr(vlm_transforms, 'get_processor_for_vlm', fail)
    transform = build_transform_from_cfg(
        dict(type='ProcessVLMSFTInputs', processor_path='not-downloaded'))
    assert transform._processor is None
    with pytest.raises(ValueError, match='BuildVLMMessages before it'):
        transform(sample)


def test_encoder_owns_lazy_processor_and_drops_it_from_pickle(monkeypatch):
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return object()

    def encode(processor, messages, **kwargs):
        assert processor is not None
        assert messages[0]['role'] == 'user'
        assert kwargs['video_kwargs']['max_frames'] == 8
        return dict(
            input_ids=torch.tensor([[1, 2, 3]]),
            labels=torch.tensor([[-100, 2, 3]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            pixel_values=torch.ones(4, 12))

    monkeypatch.setattr(vlm_transforms, 'get_processor_for_vlm', load)
    monkeypatch.setattr(vlm_transforms, '_process_messages_for_sft', encode)
    transform = build_transform_from_cfg(
        dict(
            type='ProcessVLMSFTInputs',
            processor_path='processor-only',
            video_max_frames=8,
            processor_kwargs=dict(local_files_only=True)))
    assert loads == []
    sample = vlm_transforms.BuildVLMMessages()(annotation())
    original = copy.deepcopy(sample)
    first = transform(sample)
    transform(sample)
    assert len(loads) == 1
    assert loads[0][0] == ('qwen3_vl', 'processor-only')
    assert loads[0][1]['local_files_only'] is True
    assert first['input_ids'].shape == (3, )
    assert first['image_grid_thw'].shape == (1, 3)
    assert first['messages'] == original['messages']
    assert first['id'] == original['id']
    assert sample == original
    restored = pickle.loads(pickle.dumps(transform))
    assert restored._processor is None and restored._processor_pid is None
    restored(sample)
    assert len(loads) == 2
    # A forked worker must not reuse a processor initialized in its parent.
    transform._processor_pid = -1
    transform(sample)
    assert len(loads) == 3


@pytest.mark.parametrize('kwargs', [
    dict(processor_type='unsupported'),
    dict(truncation_max_length=-1),
    dict(truncation_max_length=1.5),
    dict(video_num_frames=4, video_fps=2),
])
def test_encoder_rejects_invalid_options_without_loading(kwargs):
    with pytest.raises(ValueError):
        build_transform_from_cfg(
            dict(
                type='ProcessVLMSFTInputs',
                processor_path='not-loaded',
                **kwargs))
