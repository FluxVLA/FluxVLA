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
"""Unscored CPU geometry checks against SAPIEN's native Pinocchio backend.

Does NOT construct a policy, train a model, or launch a benchmark rollout.
Uses real observation/action samples and a mesh-free copy of the same URDF.
Run from the repository root with ``python -m tools.validate_robotwin_eef``.
"""

import argparse
import json
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sapien.core as sapien
from scipy.spatial.transform import Rotation

from fluxvla.transforms.transform_actions import (AbsoluteActions,
                                                  RelativeActions)
from fluxvla.transforms.urdf_kinematics import JointToEEFPose

ARM_JOINTS = [[f'fl_joint{i}' for i in range(1, 7)],
              [f'fr_joint{i}' for i in range(1, 7)]]
ARM_SLICES = [[0, 6], [7, 13]]
MASK = [True] * 6 + [False] + [True] * 6 + [False]
ROTATION = ([False] * 3 + [True] * 3 + [False]) * 2


def validate(urdf, parquet_files, samples=16):
    joint_to_eef = JointToEEFPose(urdf, ARM_JOINTS, ARM_SLICES)
    kin = joint_to_eef.kinematics
    relative = RelativeActions(MASK, rotation_mask=ROTATION)
    absolute = AbsoluteActions(MASK, rotation_mask=ROTATION)
    tree = ET.parse(urdf)
    for link in tree.getroot().findall('link'):
        for tag in ('visual', 'collision'):
            for element in link.findall(tag):
                link.remove(element)
    engine = sapien.Engine()
    scene = engine.create_scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    with tempfile.TemporaryDirectory(prefix='eef-native-') as temp:
        path = Path(temp) / 'kinematic.urdf'
        tree.write(path)
        robot = loader.load(str(path))
    native = robot.create_pinocchio_model()
    names = [j.name for j in robot.get_active_joints()]
    links = {link.name: i for i, link in enumerate(robot.get_links())}
    indices = [[names.index(f'{prefix}_joint{i}') for i in range(1, 7)]
               for prefix in ('fl', 'fr')]
    native.compute_forward_kinematics(np.zeros(robot.dof))
    origin = np.mean([
        native.get_link_pose(links[f'{prefix}_link1']).p
        for prefix in ('fl', 'fr')
    ],
                     axis=0)
    assert np.max(np.abs(origin - kin.origin)) < 2e-6
    max_position_error = max_rotation_error = 0.
    ik_max_position = ik_max_rotation = 0.
    ik_failures = ik_total = checked = 0
    started = time.monotonic()
    for file in parquet_files:
        table = pq.read_table(file, columns=['observation.state', 'action'])
        states = np.asarray(table['observation.state'].to_pylist())
        actions = np.asarray(table['action'].to_pylist())
        selected = np.linspace(
            0, len(states) - 1, min(samples, len(states)), dtype=int)
        for q in np.concatenate([states[selected], actions[selected]]):
            full = np.zeros(robot.dof)
            for arm, start in enumerate((0, 7)):
                full[indices[arm]] = q[start:start + 6]
            native.compute_forward_kinematics(full)
            for arm, prefix in enumerate(('fl', 'fr')):
                pose = native.get_link_pose(links[f'{prefix}_link6'])
                p, r = kin.arm_fk(q[arm * 7:arm * 7 + 6], arm)
                actual = Rotation.from_quat(np.asarray(pose.q)[[1, 2, 3, 0]])
                max_position_error = max(
                    max_position_error,
                    float(np.linalg.norm(p - (pose.p - origin))))
                angle = (Rotation.from_matrix(r) * actual.inv()).magnitude()
                max_rotation_error = max(max_rotation_error, float(angle))
            checked += 1
        for t in selected:
            # Target frames share one observation, including terminal holds.
            horizon = actions[np.minimum(t + np.arange(10), len(actions) - 1)]
            eef = joint_to_eef.convert(horizon)
            state = joint_to_eef.convert(states[t])
            delta = relative(dict(states=state, actions=eef))['actions']
            restored = absolute(dict(states=state, actions=delta))['actions']
            np.testing.assert_allclose(
                restored[:, [0, 1, 2, 7, 8, 9]],
                eef[:, [0, 1, 2, 7, 8, 9]],
                atol=2e-7)
            seed = states[t].copy()
            for target in restored:
                for arm, start in enumerate((0, 7)):
                    sl = slice(start, start + 6)
                    q, ok, pe, re = kin.inverse_arm(target[sl], seed[sl], arm)
                    seed[sl] = q
                    ik_failures += int(not ok)
                    ik_total += 1
                    ik_max_position = max(ik_max_position, float(pe))
                    ik_max_rotation = max(ik_max_rotation, float(re))
    assert max_position_error < 2e-6, max_position_error
    assert max_rotation_error < 2e-6, max_rotation_error
    assert ik_failures == 0, (
        f'{ik_failures}/{ik_total} demonstration IK failures')
    # Unscored physics smoke: execute an IK result with articulation drives.
    # There are no task objects/cameras or policy rollouts in this scene.
    initial = states[selected[0]]
    target = actions[min(selected[0] + 5, len(actions) - 1)]
    desired = joint_to_eef.convert(target)
    full_initial, full_target = np.zeros(robot.dof), np.zeros(robot.dof)
    for arm, start in enumerate((0, 7)):
        sl = slice(start, start + 6)
        q, ok, _, _ = kin.inverse_arm(desired[sl], initial[sl], arm)
        assert ok
        full_initial[indices[arm]] = initial[sl]
        full_target[indices[arm]] = q
    robot.set_qpos(full_initial)
    scene.set_timestep(1 / 250)
    for index in np.concatenate(indices):
        joint = robot.get_active_joints()[index]
        joint.set_drive_property(1000, 200)
        joint.set_drive_target(full_target[index])
    for _ in range(500):
        robot.set_qf(
            robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True))
        scene.step()
    measured = robot.get_qpos()
    joint_error = float(
        np.max(
            np.abs(measured[np.concatenate(indices)] -
                   full_target[np.concatenate(indices)])))
    assert joint_error < 0.005, joint_error
    return dict(
        parquet_files=list(parquet_files),
        checked_fk_states=checked,
        ik_targets=ik_total,
        fk_max_position_error_m=max_position_error,
        fk_max_rotation_error_rad=max_rotation_error,
        ik_failures=ik_failures,
        ik_max_position_error_m=ik_max_position,
        ik_max_rotation_error_rad=ik_max_rotation,
        physics_max_joint_error_rad=joint_error,
        elapsed_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--parquet', nargs='+', required=True)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    report = validate(args.urdf, args.parquet, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
