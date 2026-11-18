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
"""EEF transforms: normalized predictions to qpos and IK failure reporting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluxvla.datasets.kinematics.robotwin_eef import RobotwinEEF
from fluxvla.engines.runners import robotwin_eval_runner
from fluxvla.transforms.normalize import NormalizeStatesAndActions
from fluxvla.transforms.transform_eef_actions import (DenormalizeEEFAction,
                                                      RelativeEEFInputs)

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'test/data/datasets/robotwin_arx5_kinematic.urdf'


def _identity_stats():
    fields = dict(q01=[-1.] * 14, q99=[1.] * 14)
    return dict(private=dict(proprio=fields, action=fields))


def test_normalized_prediction_to_qpos_execution(monkeypatch):
    kin = RobotwinEEF(URDF)
    q = np.array([0.1, 0.6, 0.3, -0.2, 0.1, 0.2, 0.7] * 2)
    targets = np.tile(q, (3, 1))
    targets[:, 0] += [0.005, 0.01, 0.015]
    targets[:, [6, 13]] = [0.4, 0.8]
    stats = _identity_stats()
    item = RelativeEEFInputs('robotwin-eef', URDF)(
        dict(states=q, actions=targets, stats=stats['private']))
    normalized = NormalizeStatesAndActions(
        action_dim=None,
        state_dim=None,
        state_key='proprio',
        action_key='action',
        norm_type='quantile',
        output_dtype='float32')(
            item)
    decoder = DenormalizeEEFAction(
        'robotwin-eef',
        URDF,
        norm_stats=stats,
        action_dim=14,
        norm_type='quantile')
    padded = np.pad(normalized['actions'], ((0, 0), (0, 18)))
    # The eval runner passes only the executed prefix to the decoder.
    result = decoder(dict(action=padded[None, :2], state=q))
    assert result.shape == (2, 14)
    assert decoder.ik_failures == 0
    np.testing.assert_allclose(
        kin.forward(result), kin.forward(targets[:2]), atol=0.002)
    np.testing.assert_allclose(
        result[:, [6, 13]], targets[:2, [6, 13]], atol=2e-6)
    commands = []
    env = SimpleNamespace(
        eval_success=False,
        take_action_cnt=0,
        step_lim=100,
        get_instruction=lambda: 'test',
        take_action=lambda action, action_type: commands.append(
            (action, action_type)))
    model = SimpleNamespace(max_episode_steps=None, predict=lambda *_: result)
    robotwin_eval_runner.eval(env, model, {})
    assert len(commands) == 2
    assert all(kind == 'qpos' for _, kind in commands)


def test_unreachable_predictions_report_failures():
    decoder = DenormalizeEEFAction(
        'robotwin-eef',
        URDF,
        norm_stats=_identity_stats(),
        action_dim=14,
        norm_type='quantile',
        max_nfev=5)
    delta = np.zeros((1, 14))
    delta[:, [0, 7]] = 100
    state = np.zeros(14)
    with pytest.warns(RuntimeWarning, match='IK held 2'):
        result = decoder(dict(action=delta, state=state))
    assert decoder.ik_failures == 2
    np.testing.assert_array_equal(result[:, :6], 0)
    np.testing.assert_array_equal(result[:, 7:13], 0)
