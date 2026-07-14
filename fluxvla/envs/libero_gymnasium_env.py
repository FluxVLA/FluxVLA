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
#
# Portions adapted from NVIDIA Isaac-GR00T's Apache-2.0 LIBERO Gymnasium
# wrapper.

"""Gymnasium wrapper and registration helpers for LIBERO simulation."""

from __future__ import annotations

import math
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

LIBERO_GYM_NAMESPACE = 'libero_sim'
LIBERO_SUITES = (
    'libero_10',
    'libero_spatial',
    'libero_object',
    'libero_goal',
    'libero_90',
)
LOCAL_ENTRY_POINT = 'fluxvla.envs.libero_gymnasium_env:LiberoEnv'


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert an ``xyzw`` quaternion to an axis-angle rotation vector."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action: np.ndarray,
                             binarize: bool = True) -> np.ndarray:
    """Convert gripper action from ``[0, 1]`` to ``[-1, 1]``."""
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (
        orig_high - orig_low) - 1

    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip the gripper convention used by LIBERO's robosuite env."""
    action[..., -1] = action[..., -1] * -1.0
    return action


class LiberoEnv(gym.Env):
    """LIBERO OffScreenRenderEnv exposed through the Gymnasium API."""

    metadata = {'render_modes': []}

    def __init__(self, task_bddl_file: str, task_description: str):
        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
            ignore_done=True,
        )
        self._task_description = task_description
        self.observation_space = gym.spaces.Dict({
            'video.image':
            gym.spaces.Box(low=0, high=255, shape=(256, 256, 3),
                           dtype=np.uint8),
            'video.wrist_image':
            gym.spaces.Box(low=0, high=255, shape=(256, 256, 3),
                           dtype=np.uint8),
            'state.x':
            gym.spaces.Box(low=-1, high=1, shape=(1,)),
            'state.y':
            gym.spaces.Box(low=-1, high=1, shape=(1,)),
            'state.z':
            gym.spaces.Box(low=-1, high=1, shape=(1,)),
            'state.roll':
            gym.spaces.Box(low=-1, high=1, shape=(1,)),
            'state.pitch':
            gym.spaces.Box(low=-1, high=1, shape=(1,)),
            'state.yaw':
            gym.spaces.Box(low=-1, high=1, shape=(1,)),
            'state.gripper':
            gym.spaces.Box(low=-1, high=1, shape=(2,)),
            'annotation.human.action.task_description':
            gym.spaces.Text(max_length=512),
        })
        self.action_space = spaces.Dict({
            'action.x': spaces.Box(low=-1, high=1, shape=(1,)),
            'action.y': spaces.Box(low=-1, high=1, shape=(1,)),
            'action.z': spaces.Box(low=-1, high=1, shape=(1,)),
            'action.roll': spaces.Box(low=-1, high=1, shape=(1,)),
            'action.pitch': spaces.Box(low=-1, high=1, shape=(1,)),
            'action.yaw': spaces.Box(low=-1, high=1, shape=(1,)),
            'action.gripper': spaces.Box(low=-1, high=1, shape=(1,)),
        })

    def close(self):
        self._env.close()

    def _process_observation(self, obs):
        xyz = obs['robot0_eef_pos']
        rpy = quat2axisangle(obs['robot0_eef_quat'])
        gripper = obs['robot0_gripper_qpos']
        return {
            'video.image': obs['agentview_image'][::-1, ::-1],
            'video.wrist_image': obs['robot0_eye_in_hand_image'][::-1, ::-1],
            'state.x': [xyz[0]],
            'state.y': [xyz[1]],
            'state.z': [xyz[2]],
            'state.roll': [rpy[0]],
            'state.pitch': [rpy[1]],
            'state.yaw': [rpy[2]],
            'state.gripper': gripper,
            'annotation.human.action.task_description': self._task_description,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        if seed is not None:
            self._env.seed(int(seed))
        observation = self._env.reset()
        observation = self._process_observation(observation)
        info = {'success': self._env.check_success()}
        return observation, info

    def set_init_state(self, init_state):
        """Set a LIBERO benchmark initial state and return processed obs."""
        return self._process_observation(self._env.set_init_state(init_state))

    def step(self, action):
        action_vector = np.concatenate([
            action['action.x'],
            action['action.y'],
            action['action.z'],
            action['action.roll'],
            action['action.pitch'],
            action['action.yaw'],
            action['action.gripper'],
        ],
                                       axis=0)
        action_vector = normalize_gripper_action(action_vector)
        action_vector = invert_gripper_action(action_vector)
        observation, reward, done, info = self._env.step(action_vector)
        observation = self._process_observation(observation)
        info['success'] = self._env.check_success()
        truncated = False
        return observation, reward, done, truncated, info


def _env_id(task_name: str) -> str:
    return f'{LIBERO_GYM_NAMESPACE}/{task_name}'


def register_libero_envs() -> None:
    """Register LIBERO tasks as ``libero_sim/<task_name>`` Gymnasium envs."""
    benchmark_dict = benchmark.get_benchmark_dict()
    for task_suite_name in LIBERO_SUITES:
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            task_bddl_file = os.path.join(
                get_libero_path('bddl_files'),
                task.problem_folder,
                task.bddl_file,
            )
            env_id = _env_id(task.name)
            if env_id in registry:
                spec = registry[env_id]
                if spec.entry_point == LOCAL_ENTRY_POINT:
                    continue
                registry.pop(env_id, None)
            register(
                id=env_id,
                entry_point=LOCAL_ENTRY_POINT,
                kwargs={
                    'task_bddl_file': task_bddl_file,
                    'task_description': task.language,
                },
            )
