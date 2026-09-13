# Copyright 2026 Limx Dynamics
"""Check the supported LIBERO configs, not a removed RoboCasa experiment."""

from pathlib import Path

import pytest
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(('suffix', 'use_cache'), [('', False),
                                                   ('_w_cache', True)])
def test_dreamzero_temporal_contract(suffix, use_cache):
    cfg = Config.fromfile(ROOT / 'configs/dreamzero' /
                          f'dreamzero_libero_10_full_finetune{suffix}.py')
    model = cfg.model
    head = model.vla_head
    dataset = cfg.train_dataloader.dataset.datasets
    prepare_video = next(t for t in dataset.transforms
                         if t.type == 'PrepareVideo')
    latent_frames = 1 + (model.frame_window_size - 1) // 4
    blocks = (latent_frames - 1) // head.num_frame_per_block

    assert model.use_cache is use_cache
    assert model.frame_window_size == head.num_frames
    assert head.num_frames == dataset.frame_window_size
    assert prepare_video.frame_window_size == dataset.frame_window_size
    assert dataset.action_window_size == blocks * head.num_action_per_block
    assert dataset.window_start_idx == 0
    assert cfg.eval.eval_chunk_size == head.action_horizon
    assert {'actions', 'action_masks', 'lang_tokens', 'lang_masks'} <= set(
        cfg.runner.collator['keys'])
