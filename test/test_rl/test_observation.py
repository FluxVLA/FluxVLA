import numpy as np
import pytest
import torch

from fluxvla.engines import build_transform_from_cfg
from fluxvla.rl.benchmarks.libero.adapter import LiberoObservationAdapter


def test_raw_flux_and_canonical_rlinf_preprocessing_match(tiny_assets):
    cfg, _, _, _ = tiny_assets
    rng = np.random.default_rng(1)
    raw = {
        'agentview_image':
        rng.integers(0, 256, (20, 24, 3), dtype=np.uint8),
        'robot0_eye_in_hand_image':
        rng.integers(0, 256, (20, 24, 3), dtype=np.uint8),
        'robot0_eef_pos':
        np.array([0.2, 0.1, 0.8], dtype=np.float32),
        'robot0_eef_quat':
        np.array([0., 0., 0., 1.], dtype=np.float32),
        'robot0_gripper_qpos':
        np.array([0.02, -0.02], dtype=np.float32),
        'task_description':
        'pick the red cube',
        'norm_stats':
        cfg.train_dataloader.dataset.dataset_statistics.libero_10_no_noops
    }
    expected = dict(raw)
    for transform in cfg.eval.dataset.transforms:
        expected = build_transform_from_cfg(transform)(expected)
    obs = {
        'main_images':
        raw['agentview_image'][::-1, ::-1].copy()[None],
        'wrist_images':
        raw['robot0_eye_in_hand_image'][::-1, ::-1].copy()[None],
        'states':
        np.concatenate(
            [raw['robot0_eef_pos'],
             np.zeros(3), raw['robot0_gripper_qpos']])[None],
        'task_descriptions': [raw['task_description']]
    }
    adapter = LiberoObservationAdapter(cfg)
    actual = adapter(obs)
    for key, original_key in [('images', 'pixel_values'), ('states', 'states'),
                              ('lang_tokens', 'lang_tokens'),
                              ('lang_masks', 'lang_masks')]:
        torch.testing.assert_close(
            actual[key][0],
            torch.as_tensor(expected[original_key]).to(actual[key].dtype))
    assert actual['images'].shape == (1, 6, 16, 16)
    assert not actual['states'][:, 8:].any()


def test_action_denormalization_and_gripper_once(tiny_assets):
    cfg, _, _, _ = tiny_assets
    adapter = LiberoObservationAdapter(cfg)
    actions = torch.zeros(1, 3, 32)
    actions[0, :, 6] = torch.tensor([-5., 0., 5.])
    actual = adapter.env_actions(actions, 3, 7)
    expected = np.stack([
        adapter.action_transform({
            'action': a[:7].numpy().copy(),
            'norm_stats_key': adapter.stats_key
        }) for a in actions[0]
    ])
    np.testing.assert_allclose(actual[0], expected)
    assert actual[0, 0, 6] == 1 and actual[0, 2, 6] == -1


def test_invalid_image_layout_rejected(policy, env_obs):
    env_obs['main_images'] = env_obs['main_images'].permute(0, 3, 1, 2)
    with pytest.raises(ValueError, match='RGB uint8'):
        policy.observation_adapter(env_obs)
