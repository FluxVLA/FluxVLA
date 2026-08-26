# Copyright 2026 Limx Dynamics

from types import SimpleNamespace

import numpy as np
import pytest

from fluxvla.engines.runners.tron2_inference_runner import Tron2InferenceRunner


def _runner(layout):
    runner = object.__new__(Tron2InferenceRunner)
    runner.action_layout = layout
    return runner


def test_interleaved_gripper_action_layout():
    runner = _runner('interleaved_grippers')
    actions = np.arange(32).reshape(2, 16)

    left, right, head, left_gripper, right_gripper = (
        runner._split_action_components(actions))

    np.testing.assert_array_equal(left, actions[:, :7])
    np.testing.assert_array_equal(right, actions[:, 8:15])
    assert head is None
    np.testing.assert_array_equal(left_gripper, actions[:, 7])
    np.testing.assert_array_equal(right_gripper, actions[:, 15])


def test_interleaved_gripper_observation_layout():
    runner = _runner('interleaved_grippers')
    left = SimpleNamespace(position=np.arange(7))
    right = SimpleNamespace(position=np.arange(10, 17))
    head = SimpleNamespace(position=np.array([20, 21]))
    gripper = SimpleNamespace(position=np.array([30, 31]))

    qpos = runner._compose_qpos(left, right, head, gripper)

    np.testing.assert_array_equal(
        qpos,
        np.array([0, 1, 2, 3, 4, 5, 6, 30, 10, 11, 12, 13, 14, 15, 16, 31]))


def test_interleaved_gripper_layout_rejects_18d_actions():
    runner = _runner('interleaved_grippers')

    with pytest.raises(ValueError, match='expects 16D actions'):
        runner._split_action_components(np.zeros((2, 18)))


def test_legacy_action_layout_is_preserved():
    runner = _runner('arms_head_grippers')
    actions = np.arange(18)

    left, right, head, left_gripper, right_gripper = (
        runner._split_action_components(actions))

    np.testing.assert_array_equal(left, actions[:7])
    np.testing.assert_array_equal(right, actions[7:14])
    np.testing.assert_array_equal(head, actions[14:16])
    assert left_gripper == actions[16]
    assert right_gripper == actions[17]
