from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'configs/pi05/pi05_rebot_dual_inference.py'


def test_rebot_config_covers_training_and_inference():
    cfg = Config.fromfile(CONFIG_PATH)
    dataset = cfg.train_dataloader.dataset
    train_dataset = dataset.datasets[0]
    expected_mask = [True] * 6 + [False] + [True] * 6 + [False]

    assert cfg.model.ori_action_dim == 14
    assert cfg.inference_model.ori_action_dim == 14
    assert dataset.auto_compute_statistics.delta_mask == expected_mask
    assert train_dataset.type == 'ParquetDatasetV3'
    assert train_dataset.data_root_path == [
        './datasets/RealRobot_reBot_lerobot_v3/rebot_example'
    ]
    relative = next(transform for transform in train_dataset.transforms
                    if transform.type == 'RelativeActions')
    assert relative.mask == expected_mask

    assert cfg.inference.type == 'RebotDualInferenceRunner'
    assert cfg.inference.state_dim == 14
    assert cfg.inference.async_execution is True
    assert cfg.inference.execute_horizon < cfg.inference.action_chunk
    assert cfg.inference.denormalize_action.delta_action_mask == expected_mask
    assert cfg.inference.dataset.img_keys == [
        'cam_front', 'cam_wrist_left', 'cam_wrist_right'
    ]


def test_rebot_operator_reorders_joint_states_by_name():
    from fluxvla.engines.operators.rebot_dual_operator import (
        JOINT_NAMES, RebotDualOperator)

    message = SimpleNamespace(
        name=list(reversed(JOINT_NAMES)), position=list(range(7)))

    positions = RebotDualOperator.joint_positions(message)

    assert np.array_equal(positions, np.arange(6, -1, -1))


def test_rebot_runner_dispatches_async_dual_arm_trajectory():
    from fluxvla.engines.runners.rebot_dual_inference_runner import \
        RebotDualInferenceRunner

    runner = object.__new__(RebotDualInferenceRunner)
    runner.disable_puppet_arm = False
    runner.async_execution = True
    runner.execute_horizon = 2
    runner.dt = 1.0 / 30.0
    runner._prev_ctx = None
    runner._action_ctx = SimpleNamespace()
    runner.ros_operator = SimpleNamespace(execute_trajectory=Mock())
    actions = np.arange(42, dtype=np.float32).reshape(3, 14)

    with patch(
            'fluxvla.engines.runners.rebot_dual_inference_runner.time.sleep'):
        runner._execute_actions(actions, rate=None)

    kwargs = runner.ros_operator.execute_trajectory.call_args.kwargs
    assert np.array_equal(kwargs['arm_trajectories']['left'], actions[:2, :7])
    assert np.array_equal(kwargs['arm_trajectories']['right'], actions[:2,
                                                                       7:14])
    assert kwargs['dt'] == runner.dt
    assert kwargs['async_exec'] is True
