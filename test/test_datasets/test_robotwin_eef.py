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
"""RoboTwin ARX5 kinematics and EEF statistics without model/training."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from fluxvla.datasets.kinematics.robotwin_eef import (DELTA_MASK,
                                                      ROTATION_MASK,
                                                      RobotwinEEF,
                                                      relative_eef)
from fluxvla.datasets.utils.transformed_statistics import (
    _build_action_chunks, compute_statistics_from_dataset_config)
from fluxvla.transforms.transform_eef_actions import RelativeEEFInputs

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'test/data/datasets/robotwin_arx5_kinematic.urdf'


def test_fk_vectorized_and_inputs_are_not_modified():
    kin = RobotwinEEF(URDF)
    q = np.random.default_rng(5).uniform(-0.3, 0.3, (4, 14))
    original = q.copy()
    batch = kin.forward(q)
    np.testing.assert_allclose(batch, np.stack([kin.forward(x) for x in q]))
    np.testing.assert_array_equal(q, original)
    np.testing.assert_allclose(batch[:, [6, 13]], q[:, [6, 13]])
    transform = RelativeEEFInputs('robotwin-eef', URDF)
    result = transform(dict(states=q[0], actions=q))
    np.testing.assert_allclose(result['states'], batch[0])
    np.testing.assert_allclose(result['actions'],
                               relative_eef(batch, batch[0]))
    np.testing.assert_array_equal(q, original)


def test_ik_small_targets_both_arms_and_limits():
    kin = RobotwinEEF(URDF)
    q = np.array([0.1, 0.6, 0.3, -0.2, 0.1, 0.2, 0.7] * 2)
    target = kin.forward(q)
    for arm, start in enumerate((0, 7)):
        got, ok, pe, re = kin.inverse_arm(target[start:start + 6],
                                          q[start:start + 6] + 0.02, arm)
        assert ok and pe < 0.002 and re < 0.02
        assert np.all(got >= kin.limits[arm][:, 0])
        assert np.all(got <= kin.limits[arm][:, 1])
        p, r = kin.arm_fk(got, arm)
        np.testing.assert_allclose(p, target[start:start + 3], atol=0.002)
        assert (Rotation.from_matrix(r).inv() * Rotation.from_rotvec(
            target[start + 3:start + 6])).magnitude() < 0.02


def test_ik_unreachable_holds_seed():
    kin = RobotwinEEF(URDF)
    seed = np.zeros(6)
    got, ok, _, _ = kin.inverse_arm([100, 100, 100, 0, 0, 0],
                                    seed,
                                    0,
                                    max_nfev=5)
    assert not ok
    np.testing.assert_array_equal(got, seed)


def test_statistics_match_training_pipeline_and_terminal_padding(tmp_path):
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text('{}')
    (tmp_path / 'data').mkdir()
    q = np.random.default_rng(11).uniform(-0.2, 0.2, (5, 14)).astype('float32')
    a = q + 0.015
    pq.write_table(
        pa.table({
            'observation.state': q.tolist(),
            'action': a.tolist(),
            'episode_index': [0] * 5,
            'frame_index': list(range(5))
        }), tmp_path / 'data/episode.parquet')
    cfg = dict(datasets=[
        dict(
            type='ParquetDataset',
            data_root_path=str(tmp_path),
            action_window_size=3,
            window_start_idx=0,
            supervise_terminal_padding=True)
    ])
    stats, meta = compute_statistics_from_dataset_config(
        cfg, dict(profile='robotwin-eef-so3', urdf_path=str(URDF)), tmp_path)
    transform = RelativeEEFInputs('robotwin-eef', URDF)
    expected_states, expected_actions = [], []
    for t in range(5):
        result = transform(
            dict(
                states=q[t],
                actions=a[np.minimum(t + np.arange(3), 4)],
                stats=stats['private']))
        expected_states.append(result['states'])
        expected_actions.extend(result['actions'])
    for key, values in [('proprio', expected_states),
                        ('action', expected_actions)]:
        values = np.asarray(values)
        for name, quantile in [('q01', 0.01), ('q99', 0.99)]:
            np.testing.assert_allclose(
                stats['private'][key][name],
                np.quantile(values, quantile, axis=0),
                atol=1e-7)
        np.testing.assert_allclose(
            stats['private'][key]['mean'], values.mean(axis=0), atol=1e-7)
    assert meta['action_samples'] == 15
    no_pad = _build_action_chunks(
        transform.adapter.forward(a),
        transform.adapter.forward(q),
        3,
        1,
        DELTA_MASK,
        False,
        rotation_mask=ROTATION_MASK)
    expected = [
        transform(dict(states=q[t], actions=a[t + 1:t + 4]))['actions']
        for t in range(4)
    ]
    np.testing.assert_allclose(no_pad, np.concatenate(expected), atol=1e-7)
