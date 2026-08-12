# Copyright 2026 Limx Dynamics

import numpy as np
from mmengine import Config

from fluxvla.engines.runners.libero_eval_runner import LiberoEvalRunner


def test_truncate_action_chunk_at_rollout_horizon():
    actions = np.arange(8 * 7).reshape(8, 7)

    selected = LiberoEvalRunner._truncate_action_chunk(
        actions, current_step=527, step_limit=530)

    np.testing.assert_array_equal(selected, actions[:3])


def test_truncate_action_chunk_when_horizon_is_reached():
    actions = np.arange(8 * 7).reshape(8, 7)

    selected = LiberoEvalRunner._truncate_action_chunk(
        actions, current_step=530, step_limit=530)

    assert selected.shape == (0, 7)


def test_dit4dit_config_enables_horizon_truncation():
    cfg = Config.fromfile(
        'configs/dit4dit/dit4dit_libero_all_full_finetune.py')

    assert cfg.eval.type == 'LiberoEvalRunner'
    assert cfg.eval.truncate_action_chunk_at_horizon is True
