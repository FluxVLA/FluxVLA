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
"""ARX5 joint6 FK and bounded IK: the RoboTwin pose adapter.

``RobotwinEEF`` maps 14D absolute joint values onto the shared
``[position, rotation vector, gripper]`` layout per arm, converts absolute
EEF targets back to qpos, and carries the delta/rotation masks used by the
generic SO(3) delta helpers. No simulator/Pinocchio import is needed in
dataloader workers. URDF origin rotations are extrinsic xyz; all non-arm
joints are fixed at zero, as in SLA's reduced Pinocchio model. Rotations are
in radians, positions in meters.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fluxvla.datasets.kinematics.eef_so3_actions import (make_bool_mask,
                                                         so3_delta,
                                                         validate_values)

__all__ = [
    'DELTA_MASK', 'ROTATION_MASK', 'RobotwinEEF', 'get_robotwin_eef',
    'relative_eef'
]

# Arm poses are chunk-relative, both grippers stay absolute.
DELTA_MASK = make_bool_mask(6, -1, 6, -1)
ROTATION_MASK = make_bool_mask(-3, 3, -1, -3, 3, -1)


def relative_eef(actions, state, inverse=False):
    """ARX5 dual-arm alias of :func:`so3_delta` using the module masks."""
    return so3_delta(actions, state, DELTA_MASK, ROTATION_MASK, inverse)


class RobotwinEEF:
    """Vectorized URDF FK plus bounded, warm-started IK for both ARX5 arms."""

    delta_mask = DELTA_MASK
    rotation_mask = ROTATION_MASK

    def __init__(self, urdf_path):
        path = Path(urdf_path).expanduser().resolve(strict=True)
        tree = ET.fromstring(path.read_bytes())
        joints = {j.attrib['name']: j for j in tree.findall('joint')}
        parents = {j.find('child').attrib['link']: j for j in joints.values()}
        self.chains, self.limits = [], []
        roots = []
        for prefix in ('fl', 'fr'):
            names = [f'{prefix}_joint{i}' for i in range(1, 7)]
            chain, joint = [], joints[names[-1]]
            while joint is not None:
                name = joint.attrib['name']
                origin = joint.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                axis_node = joint.find('axis')
                axis = np.fromstring(
                    axis_node.get('xyz', '1 0 0')
                    if axis_node is not None else '1 0 0',
                    sep=' ')
                index = names.index(name) if name in names else None
                if index is not None and joint.get('type') != 'revolute':
                    raise ValueError(f'{name} must be a revolute joint')
                chain.append((Rotation.from_euler('xyz', rpy).as_matrix(), xyz,
                              axis / np.linalg.norm(axis), index))
                joint = parents.get(joint.find('parent').attrib['link'])
            self.chains.append(chain[::-1])
            limits = [joints[name].find('limit') for name in names]
            self.limits.append(
                np.array([[float(x.get('lower')),
                           float(x.get('upper'))] for x in limits]))
            r, p = np.eye(3), np.zeros(3)
            for rotation, position, _, index in chain[::-1]:
                p = p + r @ position
                r = r @ rotation
                if index == 0:
                    roots.append(p)
                    break
        self.origin = np.mean(roots, axis=0)

    def arm_fk(self, joints, arm):
        joints = np.asarray(joints, dtype=np.float64)
        rotation = np.broadcast_to(np.eye(3), (*joints.shape[:-1], 3, 3))
        position = np.zeros((*joints.shape[:-1], 3))
        for r, p, axis, index in self.chains[arm]:
            position = position + (rotation @ p[..., None])[..., 0]
            rotation = rotation @ r
            if index is not None:
                angles = joints[..., index, None] * axis
                motion = Rotation.from_rotvec(angles.reshape(-1, 3))
                rotation = rotation @ motion.as_matrix().reshape(
                    (*joints.shape[:-1], 3, 3))
        return position - self.origin, rotation

    def forward(self, values):
        values = validate_values(values, 14)
        result = values.copy()
        for arm, start in enumerate((0, 7)):
            pos, rot = self.arm_fk(values[..., start:start + 6], arm)
            result[..., start:start + 3] = pos
            result[..., start + 3:start + 6] = Rotation.from_matrix(
                rot.reshape(-1, 3, 3)).as_rotvec().reshape(pos.shape)
        return result.astype(np.float32)

    def inverse_arm(self,
                    target,
                    seed,
                    arm,
                    position_tolerance=0.002,
                    rotation_tolerance=0.02,
                    max_nfev=100):
        """Return (q, success, position_error, angle_error).

        On failure the caller holds the preceding arm target and reports it.
        Small successive steps and the previous solution select a local IK
        branch. This is kinematic IK, not a collision-avoidance planner.
        """
        target, seed = np.asarray(target), np.asarray(seed)
        desired = Rotation.from_rotvec(target[3:6])

        def residual(q):
            pos, rot = self.arm_fk(q, arm)
            error = (desired * Rotation.from_matrix(rot).inv()).as_rotvec()
            return np.concatenate([pos - target[:3], error])

        bounds = self.limits[arm]
        initial = np.clip(seed, bounds[:, 0], bounds[:, 1])
        fit = least_squares(
            residual,
            initial,
            bounds=(bounds[:, 0], bounds[:, 1]),
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9)
        error = residual(fit.x)
        pe, re = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        ok = bool(pe <= position_tolerance and re <= rotation_tolerance)
        return (fit.x if ok else seed.copy()), ok, pe, re

    def inverse(self,
                targets,
                seed,
                position_tolerance=0.002,
                rotation_tolerance=0.02,
                max_nfev=100):
        """Absolute EEF targets [T, 14] + raw qpos seed [14] -> qpos [T, 14].

        Grippers pass through unchanged. Each arm warm-starts from the
        previous solution; a failed arm target holds the preceding qpos.
        Returns ``(qpos, failures, attempts)``.
        """
        targets = np.asarray(targets)
        seed = validate_values(seed, 14).copy()
        result = targets.copy()
        failures = attempts = 0
        for i, target in enumerate(targets):
            for arm, start in enumerate((0, 7)):
                sl = slice(start, start + 6)
                q, ok, _, _ = self.inverse_arm(target[sl], seed[sl], arm,
                                               position_tolerance,
                                               rotation_tolerance, max_nfev)
                attempts += 1
                failures += int(not ok)
                result[i, sl] = seed[sl] = q
        return result, failures, attempts


@lru_cache(maxsize=8)
def get_robotwin_eef(urdf_path):
    return RobotwinEEF(urdf_path)
