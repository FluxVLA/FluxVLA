# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mask-driven SO(3) delta helpers must stay embodiment-agnostic."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fluxvla.datasets.kinematics.eef_so3_actions import (make_bool_mask,
                                                         so3_delta)

# Single-arm layout such as DROID/Piper: [pos(3), rotvec(3), gripper(1)].
SINGLE_DELTA = make_bool_mask(6, -1)
SINGLE_ROTATION = make_bool_mask(-3, 3, -1)


def test_single_arm_roundtrip_and_gripper_passthrough():
    state = np.array([0.1, 0.2, 0.3, 0.4, -0.7, 0.2, 0.5])
    actions = np.zeros((2, 7))
    actions[:, :3] = [[0.2, 0.1, 0.4], [0.3, 0.0, 0.5]]
    actions[:, 3:6] = [[0.1, 0.8, 0.2], [0.3, 0.5, -0.6]]
    actions[:, 6] = 0.9
    delta = so3_delta(actions, state, SINGLE_DELTA, SINGLE_ROTATION)
    np.testing.assert_allclose(
        delta[:, :3], actions[:, :3] - state[:3], atol=1e-7)
    # Rotations compose as Ra Rs^T against the same state for every step,
    # which differs from component-wise subtraction of rotation vectors.
    expected = Rotation.from_rotvec(actions[:, 3:6]).as_matrix() @ (
        Rotation.from_rotvec(state[3:6]).as_matrix().T)
    np.testing.assert_allclose(
        Rotation.from_rotvec(delta[:, 3:6]).as_matrix(), expected, atol=1e-7)
    assert not np.allclose(delta[:, 3:6], actions[:, 3:6] - state[3:6])
    np.testing.assert_allclose(delta[:, 6], actions[:, 6])
    restored = so3_delta(
        delta, state, SINGLE_DELTA, SINGLE_ROTATION, inverse=True)
    np.testing.assert_allclose(restored, actions, atol=1e-7)


def test_so3_wrap_boundary():
    # 179 deg -> -179 deg about one axis is a 2 deg delta, not -358 deg.
    state = np.zeros(7)
    state[5] = np.deg2rad(179)
    action = state.copy()
    action[5] = np.deg2rad(-179)
    action[6] = 0.2
    delta = so3_delta(action, state, SINGLE_DELTA, SINGLE_ROTATION)
    np.testing.assert_allclose(delta[5], np.deg2rad(2), atol=1e-7)
    np.testing.assert_allclose(delta[6], 0.2)
    restored = so3_delta(
        delta, state, SINGLE_DELTA, SINGLE_ROTATION, inverse=True)
    np.testing.assert_allclose(restored, action, atol=3e-7)


def test_mask_semantics_and_validation():
    assert make_bool_mask(2, -2, 1) == (True, True, False, False, True)
    assert make_bool_mask(
        6, -1, 6, -1) == (True, ) * 6 + (False, ) + (True, ) * 6 + (False, )
    with pytest.raises(ValueError, match='non-zero'):
        make_bool_mask(2, 0)
    values = np.zeros(7)
    with pytest.raises(ValueError, match='lengths differ'):
        so3_delta(values, values, make_bool_mask(6), SINGLE_ROTATION)
    with pytest.raises(ValueError, match='subset'):
        so3_delta(values, values, make_bool_mask(3, -4), SINGLE_ROTATION)
    with pytest.raises(ValueError, match='groups of 3'):
        so3_delta(values, values, SINGLE_DELTA, make_bool_mask(-3, 2, -2))
    with pytest.raises(ValueError, match='groups of 3'):
        so3_delta(values, values, SINGLE_DELTA, make_bool_mask(-4, 2, -1))
    with pytest.raises(ValueError, match='last dimension 7'):
        so3_delta(np.zeros(6), values, SINGLE_DELTA, SINGLE_ROTATION)
    with pytest.raises(ValueError, match='finite values'):
        so3_delta(np.full(7, np.nan), values, SINGLE_DELTA, SINGLE_ROTATION)
