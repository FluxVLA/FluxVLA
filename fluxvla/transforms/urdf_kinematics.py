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
"""URDF kinematics and the joint <-> EEF pose transforms built on it.

``URDFKinematics`` parses a URDF without a simulator or Pinocchio, so it can
run inside dataloader workers.  Each arm is an ordered list of revolute joint
names; non-arm joints along the chain are held at zero.  Poses are
``[position (m), rotation vector (rad)]`` of the last arm joint frame,
expressed relative to the midpoint of the arms' root joints at the zero
configuration.

``JointToEEFPose`` replaces each arm's joint block with that pose (training
inputs and statistics); ``EEFPoseToJoint`` inverts it with bounded,
warm-started IK for controllers that execute joint targets.
"""

import warnings
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fluxvla.engines import TRANSFORMS

__all__ = [
    'URDFKinematics', 'get_urdf_kinematics', 'JointToEEFPose', 'EEFPoseToJoint'
]

POSE_DIM = 6


def _finite(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{name} must be finite.')
    return values


class URDFKinematics:
    """Vectorized URDF forward kinematics and bounded IK for serial arms.

    Args:
        urdf_path (str): URDF file; only ``<joint>`` elements are read.
        arm_joints (Sequence[Sequence[str]]): Per arm, the revolute joint
            names from root to tip. The last joint frame is the EEF frame.
    """

    def __init__(self, urdf_path: str,
                 arm_joints: Sequence[Sequence[str]]) -> None:
        path = Path(urdf_path).expanduser().resolve(strict=True)
        tree = ET.fromstring(path.read_bytes())
        joints = {j.attrib['name']: j for j in tree.findall('joint')}
        parents = {j.find('child').attrib['link']: j for j in joints.values()}
        self.arm_joints = [tuple(names) for names in arm_joints]
        if not self.arm_joints or any(not names for names in self.arm_joints):
            raise ValueError('arm_joints must list at least one joint per arm')
        self.chains, self.limits = [], []
        roots = []
        for names in self.arm_joints:
            missing = [name for name in names if name not in joints]
            if missing:
                raise ValueError(f'Joints missing from URDF: {missing}')
            chain, joint = [], joints[names[-1]]
            while joint is not None:
                name = joint.attrib['name']
                origin = joint.find('origin')
                xyz = np.fromstring(
                    '0 0 0' if origin is None else origin.get('xyz', '0 0 0'),
                    sep=' ')
                rpy = np.fromstring(
                    '0 0 0' if origin is None else origin.get('rpy', '0 0 0'),
                    sep=' ')
                axis_node = joint.find('axis')
                axis = np.fromstring(
                    '1 0 0' if axis_node is None else axis_node.get(
                        'xyz', '1 0 0'),
                    sep=' ')
                index = names.index(name) if name in names else None
                if index is not None and joint.get('type') != 'revolute':
                    raise ValueError(f'{name} must be a revolute joint')
                chain.append((Rotation.from_euler('xyz', rpy).as_matrix(), xyz,
                              axis / np.linalg.norm(axis), index))
                joint = parents.get(joint.find('parent').attrib['link'])
            chain = chain[::-1]
            if sum(index is not None for *_, index in chain) != len(names):
                raise ValueError(f'Arm joints {names} are not on one chain')
            self.chains.append(chain)
            limits = [joints[name].find('limit') for name in names]
            if any(limit is None for limit in limits):
                raise ValueError(f'Arm joints {names} need <limit> elements')
            self.limits.append(
                np.array([[float(x.get('lower')),
                           float(x.get('upper'))] for x in limits]))
            r, p = np.eye(3), np.zeros(3)
            for rotation, position, _, index in chain:
                p = p + r @ position
                r = r @ rotation
                if index == 0:
                    roots.append(p)
                    break
        self.origin = np.mean(roots, axis=0)

    @property
    def num_arms(self) -> int:
        return len(self.chains)

    def arm_fk(self, joints, arm: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(position, rotation_matrix)`` for ``joints[..., n]``."""
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

    def arm_pose(self, joints, arm: int) -> np.ndarray:
        """Return ``[..., 6]`` position + rotation-vector poses."""
        position, rotation = self.arm_fk(joints, arm)
        rotvec = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_rotvec()
        return np.concatenate(
            [position, rotvec.reshape(position.shape)], axis=-1)

    def inverse_arm(self,
                    target,
                    seed,
                    arm: int,
                    position_tolerance: float = 0.002,
                    rotation_tolerance: float = 0.02,
                    max_nfev: int = 100):
        """Return ``(q, success, position_error, angle_error)``.

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


@lru_cache(maxsize=8)
def _cached_kinematics(urdf_path: str, arm_joints: Tuple[Tuple[str, ...],
                                                         ...]):
    return URDFKinematics(urdf_path, arm_joints)


def get_urdf_kinematics(urdf_path, arm_joints) -> URDFKinematics:
    """Share one parsed URDF between transforms built from the same config."""
    return _cached_kinematics(
        str(urdf_path), tuple(tuple(names) for names in arm_joints))


def _arm_slices(kinematics: URDFKinematics,
                joint_slices: Sequence[Sequence[int]]) -> List[slice]:
    if len(joint_slices) != kinematics.num_arms:
        raise ValueError('joint_slices must list one [start, stop) per arm')
    slices = []
    for names, (start, stop) in zip(kinematics.arm_joints, joint_slices):
        if stop - start != len(names) or len(names) != POSE_DIM:
            raise ValueError(
                'Each arm needs exactly 6 joints so its joint block can be '
                f'replaced by a 6D pose; got {len(names)} joints for '
                f'[{start}, {stop}).')
        slices.append(slice(int(start), int(stop)))
    return slices


@TRANSFORMS.register_module()
class JointToEEFPose:
    """Replace each arm's joint block with its absolute EEF pose.

    Other dimensions (for example grippers) pass through unchanged, so the
    vector width is preserved and the usual mask-based transforms apply.

    Args:
        urdf_path (str): Robot URDF.
        arm_joints (list[list[str]]): Per arm, ordered revolute joint names.
        joint_slices (list[list[int]]): Per arm, the ``[start, stop)`` block
            of the state/action vector holding those joints.
        state_key (str): State key to convert. Defaults to ``'states'``.
        action_key (str | None): Action key to convert; ``None`` leaves actions
            untouched (evaluation, where actions are already EEF deltas).
        output_state_key (str | None): Write the converted state here instead
            of in place, keeping the raw joints available for later IK.
    """

    def __init__(self,
                 urdf_path: str,
                 arm_joints: Sequence[Sequence[str]],
                 joint_slices: Sequence[Sequence[int]],
                 state_key: str = 'states',
                 action_key: Optional[str] = 'actions',
                 output_state_key: Optional[str] = None) -> None:
        self.kinematics = get_urdf_kinematics(urdf_path, arm_joints)
        self.slices = _arm_slices(self.kinematics, joint_slices)
        self.state_key = state_key
        self.action_key = action_key
        self.output_state_key = output_state_key or state_key

    def convert(self, values) -> np.ndarray:
        values = _finite(values, 'Joint values')
        result = values.copy()
        for arm, block in enumerate(self.slices):
            result[...,
                   block] = self.kinematics.arm_pose(values[..., block], arm)
        return result.astype(np.float32)

    def __call__(self, data: Dict) -> Dict:
        if self.state_key in data:
            data[self.output_state_key] = self.convert(data[self.state_key])
        if self.action_key is not None and self.action_key in data:
            data[self.action_key] = self.convert(data[self.action_key])
        return data


@TRANSFORMS.register_module()
class EEFPoseToJoint:
    """Absolute EEF pose targets -> joint targets via bounded IK.

    ``state_key`` holds the raw joint observation that seeds the IK; each arm
    warm-starts from its previous solution.  A target that fails the
    tolerances holds the preceding joints; failures are counted and reported
    through warnings so they surface in worker logs.
    """

    def __init__(self,
                 urdf_path: str,
                 arm_joints: Sequence[Sequence[str]],
                 joint_slices: Sequence[Sequence[int]],
                 state_key: str = 'state',
                 action_key: str = 'action',
                 position_tolerance: float = 0.002,
                 rotation_tolerance: float = 0.02,
                 max_nfev: int = 100) -> None:
        if max_nfev <= 0:
            raise ValueError('max_nfev must be positive')
        if position_tolerance <= 0 or rotation_tolerance <= 0:
            raise ValueError('IK tolerances must be positive')
        self.kinematics = get_urdf_kinematics(urdf_path, arm_joints)
        self.slices = _arm_slices(self.kinematics, joint_slices)
        self.state_key = state_key
        self.action_key = action_key
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance
        self.max_nfev = max_nfev
        self.ik_failures = 0
        self.ik_attempts = 0

    def __call__(self, data: Dict) -> Dict:
        targets = _finite(data[self.action_key], 'EEF targets')
        if targets.ndim == 1:
            targets = targets[None]
        if targets.ndim != 2:
            raise ValueError(
                f'Expected EEF targets shaped [T, D], got {targets.shape}')
        seed = _finite(data[self.state_key], 'Joint state').reshape(-1).copy()
        if seed.shape[0] != targets.shape[1]:
            raise ValueError('Joint state and EEF targets must share width')
        result = targets.copy()
        failures = 0
        for i, target in enumerate(targets):
            for arm, block in enumerate(self.slices):
                q, ok, _, _ = self.kinematics.inverse_arm(
                    target[block], seed[block], arm, self.position_tolerance,
                    self.rotation_tolerance, self.max_nfev)
                self.ik_attempts += 1
                failures += int(not ok)
                result[i, block] = seed[block] = q
        self.ik_failures += failures
        if failures:
            warnings.warn(
                f'EEF IK held {failures} arm targets; '
                f'cumulative={self.ik_failures}/{self.ik_attempts}',
                RuntimeWarning)
        data[self.action_key] = result.astype(np.float32)
        return data
