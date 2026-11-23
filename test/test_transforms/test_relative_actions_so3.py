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
"""SO(3) rotation composition in RelativeActions/AbsoluteActions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fluxvla.transforms.transform_actions import (AbsoluteActions,
                                                  RelativeActions)

# Single-arm layout such as DROID/Piper: [pos(3), rotvec(3), gripper(1)].
MASK = [True] * 6 + [False]
ROTATION = [False] * 3 + [True] * 3 + [False]


def test_rotation_triples_compose_and_roundtrip():
    state = np.array([0.1, 0.2, 0.3, 0.4, -0.7, 0.2, 0.5])
    actions = np.zeros((2, 7))
    actions[:, :3] = [[0.2, 0.1, 0.4], [0.3, 0.0, 0.5]]
    actions[:, 3:6] = [[0.1, 0.8, 0.2], [0.3, 0.5, -0.6]]
    actions[:, 6] = 0.9
    relative = RelativeActions(MASK, rotation_mask=ROTATION)
    delta = relative(dict(states=state, actions=actions.copy()))['actions']
    np.testing.assert_allclose(
        delta[:, :3], actions[:, :3] - state[:3], atol=1e-7)
    # Ra Rs^T against the same state for every step; this differs from
    # component-wise subtraction of rotation vectors.
    expected = Rotation.from_rotvec(actions[:, 3:6]).as_matrix() @ (
        Rotation.from_rotvec(state[3:6]).as_matrix().T)
    np.testing.assert_allclose(
        Rotation.from_rotvec(delta[:, 3:6]).as_matrix(), expected, atol=1e-7)
    assert not np.allclose(delta[:, 3:6], actions[:, 3:6] - state[3:6])
    np.testing.assert_allclose(delta[:, 6], actions[:, 6])
    absolute = AbsoluteActions(MASK, rotation_mask=ROTATION)
    restored = absolute(dict(states=state, actions=delta.copy()))['actions']
    np.testing.assert_allclose(restored, actions, atol=1e-7)


def test_batched_states_and_wrap_boundary():
    rng = np.random.default_rng(3)
    states = rng.uniform(-0.5, 0.5, (3, 7))
    actions = rng.uniform(-0.5, 0.5, (3, 4, 7))
    relative = RelativeActions(MASK, rotation_mask=ROTATION)
    absolute = AbsoluteActions(MASK, rotation_mask=ROTATION)
    delta = relative(dict(states=states, actions=actions.copy()))['actions']
    np.testing.assert_allclose(
        absolute(dict(states=states, actions=delta.copy()))['actions'],
        actions,
        atol=1e-7)
    # 179 deg -> -179 deg about one axis is a 2 deg delta, not -358 deg.
    state = np.zeros(7)
    state[5] = np.deg2rad(179)
    action = state.copy()
    action[5] = np.deg2rad(-179)
    delta = relative(dict(states=state, actions=action[None]))['actions'][0]
    np.testing.assert_allclose(delta[5], np.deg2rad(2), atol=1e-7)


def test_rotation_mask_validation_and_default_behavior():
    with pytest.raises(ValueError, match='same length'):
        RelativeActions(MASK, rotation_mask=[True] * 6)
    with pytest.raises(ValueError, match='subset'):
        RelativeActions(MASK, rotation_mask=[False] * 6 + [True])
    with pytest.raises(ValueError, match='groups of 3'):
        RelativeActions(MASK, rotation_mask=[False] * 4 + [True] * 2 + [False])
    # Without a rotation mask the transform is the plain OpenPI subtraction.
    state = np.arange(7.0)
    actions = np.ones((2, 7))
    plain = RelativeActions(MASK)(dict(states=state,
                                       actions=actions.copy()))['actions']
    np.testing.assert_allclose(plain[:, :6], np.tile(1 - state[:6], (2, 1)))
    np.testing.assert_allclose(plain[:, 6], 1)
