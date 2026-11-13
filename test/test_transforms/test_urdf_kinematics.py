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
"""URDF kinematics and the joint <-> EEF transforms on the RoboTwin ARX5."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from fluxvla.engines.runners import robotwin_eval_runner
from fluxvla.policies.eval_io import RobotwinInferenceIO
from fluxvla.transforms.normalize import (DenormalizePrivateAction,
                                          NormalizeStatesAndActions)
from fluxvla.transforms.transform_actions import (AbsoluteActions,
                                                  RelativeActions)
from fluxvla.transforms.urdf_kinematics import (EEFPoseToJoint, JointToEEFPose,
                                                URDFKinematics)

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'test/data/datasets/robotwin_arx5_kinematic.urdf'
ARM_JOINTS = [[f'fl_joint{i}' for i in range(1, 7)],
              [f'fr_joint{i}' for i in range(1, 7)]]
ARM_SLICES = [[0, 6], [7, 13]]
MASK = [True] * 6 + [False] + [True] * 6 + [False]
ROTATION = ([False] * 3 + [True] * 3 + [False]) * 2
KIN = dict(urdf_path=URDF, arm_joints=ARM_JOINTS, joint_slices=ARM_SLICES)


def test_fk_is_vectorized_and_does_not_modify_inputs():
    kin = URDFKinematics(URDF, ARM_JOINTS)
    q = np.random.default_rng(5).uniform(-0.3, 0.3, (4, 14))
    original = q.copy()
    transform = JointToEEFPose(**KIN)
    batch = transform.convert(q)
    np.testing.assert_allclose(batch,
                               np.stack([transform.convert(x) for x in q]))
    np.testing.assert_array_equal(q, original)
    np.testing.assert_allclose(batch[:, [6, 13]], q[:, [6, 13]])
    np.testing.assert_allclose(batch[:, :6], kin.arm_pose(q[:, :6], 0))
    result = transform(dict(states=q[0], actions=q.copy()))
    np.testing.assert_allclose(result['states'], batch[0])
    np.testing.assert_allclose(result['actions'], batch)
    # Evaluation form: convert the state into its own key, leave actions.
    eval_form = JointToEEFPose(
        state_key='state',
        action_key=None,
        output_state_key='eef_state',
        **KIN)
    data = eval_form(dict(state=q[0], action=np.ones((2, 14))))
    np.testing.assert_allclose(data['eef_state'], batch[0])
    np.testing.assert_array_equal(data['state'], q[0])
    np.testing.assert_array_equal(data['action'], 1)


def test_ik_reaches_nearby_targets_within_limits_and_holds_unreachable():
    kin = URDFKinematics(URDF, ARM_JOINTS)
    q = np.array([0.1, 0.6, 0.3, -0.2, 0.1, 0.2, 0.7] * 2)
    for arm, start in enumerate((0, 7)):
        target = kin.arm_pose(q[start:start + 6], arm)
        got, ok, pe, re = kin.inverse_arm(target, q[start:start + 6] + 0.02,
                                          arm)
        assert ok and pe < 0.002 and re < 0.02
        assert np.all(got >= kin.limits[arm][:, 0])
        assert np.all(got <= kin.limits[arm][:, 1])
        p, r = kin.arm_fk(got, arm)
        np.testing.assert_allclose(p, target[:3], atol=0.002)
        assert (Rotation.from_matrix(r).inv() *
                Rotation.from_rotvec(target[3:6])).magnitude() < 0.02
    seed = np.zeros(6)
    got, ok, _, _ = kin.inverse_arm([100, 100, 100, 0, 0, 0],
                                    seed,
                                    0,
                                    max_nfev=5)
    assert not ok
    np.testing.assert_array_equal(got, seed)


def test_invalid_arm_configuration_is_rejected():
    with pytest.raises(ValueError, match='missing from URDF'):
        URDFKinematics(URDF, [['fl_joint1', 'no_such_joint']])
    with pytest.raises(ValueError, match='exactly 6 joints'):
        JointToEEFPose(URDF, [ARM_JOINTS[0][:5]], [[0, 5]])
    with pytest.raises(ValueError, match='one \\[start, stop\\) per arm'):
        JointToEEFPose(URDF, ARM_JOINTS, [[0, 6]])


def _identity_stats():
    fields = dict(q01=[-1.] * 14, q99=[1.] * 14)
    return dict(private=dict(proprio=fields, action=fields))


def _denormalize_chain(max_nfev=100):
    return [
        DenormalizePrivateAction(
            norm_stats=_identity_stats(), action_dim=14, norm_type='quantile'),
        JointToEEFPose(
            state_key='state',
            action_key=None,
            output_state_key='eef_state',
            **KIN),
        AbsoluteActions(
            MASK,
            rotation_mask=ROTATION,
            state_key='eef_state',
            action_key='action'),
        EEFPoseToJoint(
            state_key='state', action_key='action', max_nfev=max_nfev, **KIN),
    ]


def test_normalized_prediction_to_qpos_execution():
    joint_to_eef = JointToEEFPose(**KIN)
    q = np.array([0.1, 0.6, 0.3, -0.2, 0.1, 0.2, 0.7] * 2)
    targets = np.tile(q, (3, 1))
    targets[:, 0] += [0.005, 0.01, 0.015]
    targets[:, [6, 13]] = [0.4, 0.8]
    item = joint_to_eef(dict(states=q, actions=targets.copy()))
    item = RelativeActions(MASK, rotation_mask=ROTATION)(item)
    normalized = NormalizeStatesAndActions(
        action_dim=None,
        state_dim=None,
        state_key='proprio',
        action_key='action',
        norm_type='quantile',
        output_dtype='float32')(
            dict(item, stats=_identity_stats()['private']))
    padded = np.pad(normalized['actions'], ((0, 0), (0, 18)))
    stages = _denormalize_chain()
    # The assembled predictor passes only the executed prefix through IK.
    io = RobotwinInferenceIO(
        vla=None,
        dataset=None,
        denormalize_action=stages,
        execute_horizon=2,
        unnorm_key='private',
        mixed_precision_dtype=torch.bfloat16,
        enable_mixed_precision=False,
        device=torch.device('cpu'))
    result, _ = io.process_actions(torch.from_numpy(padded[None]), raw_state=q)
    assert result.shape == (2, 14)
    assert stages[-1].ik_failures == 0
    np.testing.assert_allclose(
        joint_to_eef.convert(result),
        joint_to_eef.convert(targets[:2]),
        atol=0.002)
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
    acknowledged = []
    model = SimpleNamespace(
        max_episode_steps=None,
        predict=lambda *_, **__: result,
        last_output=SimpleNamespace(status='ready'),
        acknowledge=acknowledged.append)
    robotwin_eval_runner.eval(env, model, {})
    assert len(commands) == 2
    assert all(kind == 'qpos' for _, kind in commands)
    assert acknowledged == [2]
    # A single-step [1, 32] prediction is squeezed to [14] by
    # DenormalizePrivateAction; the chain must still yield a [1, 14] chunk.
    io.execute_horizon = 1
    io.denormalize_action = _denormalize_chain()
    single, _ = io.process_actions(torch.from_numpy(padded[:1]), raw_state=q)
    assert single.shape == (1, 14)
    np.testing.assert_allclose(single, result[:1], atol=1e-5)


def test_unreachable_predictions_hold_joints_and_warn():
    decoder = EEFPoseToJoint(
        state_key='state', action_key='action', max_nfev=5, **KIN)
    targets = np.zeros((1, 14))
    targets[:, [0, 7]] = 100
    with pytest.warns(RuntimeWarning, match='IK held 2'):
        result = decoder(dict(state=np.zeros(14), action=targets))['action']
    assert decoder.ik_failures == 2
    np.testing.assert_array_equal(result[:, :6], 0)
    np.testing.assert_array_equal(result[:, 7:13], 0)
