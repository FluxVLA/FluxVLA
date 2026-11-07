# Copyright 2026 Limx Dynamics
"""Shared VLM collator registration, text padding and visual layouts."""

from pathlib import Path

import pytest
import torch
from mmengine import Config

from fluxvla.collators import VLMSFTCollator
from fluxvla.engines import build_collator_from_cfg


def _sample(**kwargs):
    return dict(
        input_ids=torch.tensor([2, 7, 1]),
        labels=torch.tensor([-100, 7, 1]),
        **kwargs)


def test_views_collator_builds_from_registry_and_pads_missing_images():
    samples = [_sample(images=torch.ones(6, 8, 8)), _sample()]
    collator = build_collator_from_cfg(
        dict(type='VLMSFTCollator', pad_token_id=0, image_layout='views'))
    assert type(collator) is VLMSFTCollator
    batch = collator(samples)
    assert batch['images'].shape == (2, 6, 8, 8)
    torch.testing.assert_close(batch['images'][0], samples[0]['images'])
    assert batch['images'][1].count_nonzero() == 0
    assert batch['img_masks'].tolist() == [[True, True], [False, False]]


@pytest.mark.parametrize('layout', ['patches', 'views'])
def test_text_only_padding_preserves_eos_when_it_is_also_pad(layout):
    samples = [
        _sample(),
        dict(input_ids=torch.tensor([2, 1]), labels=torch.tensor([-100, 1]))
    ]
    batch = VLMSFTCollator(pad_token_id=1, image_layout=layout)(samples)
    assert set(batch) == {'input_ids', 'labels', 'attention_mask'}
    assert batch['input_ids'].tolist() == [[2, 7, 1], [2, 1, 1]]
    assert batch['labels'].tolist() == [[-100, 7, 1], [-100, 1, -100]]
    assert batch['attention_mask'].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_default_layout_keeps_native_image_video_patches_and_token_types():
    image = _sample(
        images=['raw/image.jpg'],
        pixel_values=torch.ones(4, 12),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mm_token_type_ids=torch.tensor([0, 1, 0]))
    video = _sample(
        pixel_values_videos=torch.ones(8, 12),
        video_grid_thw=torch.tensor([[2, 2, 2]]),
        mm_token_type_ids=torch.tensor([0, 2, 0]))
    batch = VLMSFTCollator(pad_token_id=0)([image, _sample(), video, image])
    assert 'images' not in batch and 'img_masks' not in batch
    assert batch['pixel_values'].shape == (8, 12)
    assert batch['pixel_values_videos'].shape == (8, 12)
    assert batch['image_grid_thw'].tolist() == [[1, 2, 2], [1, 2, 2]]
    assert batch['video_grid_thw'].tolist() == [[2, 2, 2]]
    assert batch['mm_token_type_ids'].tolist() == [[0, 1, 0], [0, 0, 0],
                                                   [0, 2, 0], [0, 1, 0]]


@pytest.mark.parametrize('layout,sample,message', [
    ('patches', _sample(images=torch.ones(3, 8, 8)), 'image_layout'),
    ('views', _sample(pixel_values=torch.ones(4, 12)), 'Cannot mix'),
    ('views', _sample(images=['raw/image.jpg']), 'float tensors'),
    ('views', _sample(images=torch.ones(4, 8, 8)), 'float tensors'),
])
def test_mismatched_image_layouts_are_rejected(layout, sample, message):
    with pytest.raises(ValueError, match=message):
        VLMSFTCollator(pad_token_id=0, image_layout=layout)([sample])


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match='image_layout'):
        VLMSFTCollator(pad_token_id=0, image_layout='auto')


@pytest.mark.parametrize('name,section', [
    ('llava665k', 'runner'),
    ('robocasa_llava665k_mixed', 'vlm_dataloader'),
    ('robotwin_llava665k_mixed', 'vlm_dataloader'),
])
def test_pi05_recipes_build_registered_shared_views_collator(name, section):
    root = Path(__file__).resolve().parents[2] / 'configs/pi05'
    cfg = Config.fromfile(root / f'pi05_paligemma_{name}_full_finetune.py')
    collator = build_collator_from_cfg(cfg[section].collator)
    assert type(collator) is VLMSFTCollator
    assert collator.image_layout == 'views'
    assert collator.pad_token_id == 0
    batch = collator([_sample(images=torch.ones(3, 8, 8))])
    assert batch['images'].shape == (1, 3, 8, 8)
    assert batch['img_masks'].tolist() == [[True]]
