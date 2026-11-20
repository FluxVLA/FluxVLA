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
"""RoboTwin EEF/SO(3) statistics must match the training transforms."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fluxvla.datasets.utils.transformed_statistics import (
    PROFILES, _build_action_chunks, compute_statistics_from_dataset_config)
from fluxvla.transforms.transform_actions import RelativeActions
from fluxvla.transforms.urdf_kinematics import JointToEEFPose

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'test/data/datasets/robotwin_arx5_kinematic.urdf'
PROFILE = PROFILES['robotwin-eef-so3']


def _training_transforms():
    joint_to_eef = JointToEEFPose(URDF, **PROFILE.joint_to_eef)
    relative = RelativeActions(
        list(PROFILE.delta_mask), rotation_mask=list(PROFILE.rotation_mask))
    return lambda data: relative(joint_to_eef(data))


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
    transform = _training_transforms()
    expected_states, expected_actions = [], []
    for t in range(5):
        result = transform(
            dict(states=q[t], actions=a[np.minimum(t + np.arange(3), 4)]))
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
    assert meta['rotation_mask'] == list(PROFILE.rotation_mask)
    joint_to_eef = JointToEEFPose(URDF, **PROFILE.joint_to_eef)
    no_pad = _build_action_chunks(
        joint_to_eef.convert(a),
        joint_to_eef.convert(q),
        3,
        1,
        PROFILE.delta_mask,
        False,
        rotation_mask=PROFILE.rotation_mask)
    expected = [
        transform(dict(states=q[t], actions=a[t + 1:t + 4]))['actions']
        for t in range(4)
    ]
    np.testing.assert_allclose(no_pad, np.concatenate(expected), atol=1e-7)
