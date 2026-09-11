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
"""OpenAI Responses API policy for checkpoint-free robot evaluation."""

from __future__ import annotations
import base64
import copy
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from numbers import Real
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial.transform import Rotation

from fluxvla.engines import VLAS, initialize_overwatch

overwatch = initialize_overwatch(__name__)

DEFAULT_LIBERO_SYSTEM_PROMPT = """You control one Panda robot arm in the
LIBERO simulator. At every turn you receive an external camera image, a wrist
camera image, and the current end-effector state. Use exactly one move_to tool
call to make progress on the instruction. The simulator, not you, decides
when the task succeeds.

The move_to tool accepts an absolute end-effector position in MuJoCo world
coordinates and an optional rotation_delta [rx, ry, rz]. This is a relative
axis-angle rotation vector in radians about WORLD axes, NOT Euler angles or
an absolute orientation. Its direction is the rotation axis and its length
is the angle, using the right-hand rule. For example [0, 0, 0.2] turns the
gripper about world z; [0, 0, -0.2] turns it the other way. It is the total
requested rotation for this call, not a delta to repeat at every step.
Omitted position dimensions and omitted rotation hold their previous value.
Rotate to align the fingertips with an object's grasp axis or a handle before
descending; tilt only when needed. Motions are speed-limited and may only
partially reach a requested target. Re-check the reported pose and images
after every command rather than assuming the target was reached. Approach an
object from above, descend only after it is centered between the fingertips,
close the gripper, lift clear of obstacles, move above the destination,
descend, open the gripper, and retreat when appropriate.

Coordinate guide for the upright external camera: the robot base is near the
top of the image; decreasing x generally moves away from the robot toward the
bottom of the image, increasing y generally moves toward the left side of the
image, and increasing z moves upward. Position units are meters. The allowed
workspace is shown in the tool description. Gripper target 0 means closed and
1 means open. Use the external camera to inspect side labels and the wrist
camera mainly for grasp alignment; moving a top-down wrist camera around a
closed container will often continue to show only its lid. Do not substitute a
nearby object based only on color, but a readable label fragment (for example,
the requested category word) plus distinctive packaging is enough to commit.
Spend at most five calls comparing candidate objects, then grasp the best
supported match. Include a short note describing what you see and why you
chose the target. Do not guess object coordinates from simulator-only
metadata; use the images and the reported robot state."""

DEFAULT_ROBOCASA_SYSTEM_PROMPT = """You control a Fourier GR-1 humanoid in the
RoboCasa tabletop simulator. At every turn you receive one or more ego-camera
images and the current left-arm, right-arm, hand, and waist joint states. Use
exactly one control_gr1 tool call to make progress on the instruction. The
simulator, not you, decides when the task succeeds.

The two arm vectors use this joint order: shoulder pitch, shoulder roll,
shoulder yaw, elbow pitch, wrist yaw, wrist roll, wrist pitch. The waist order
is yaw, pitch, roll. Arm and waist values supplied to the tool are incremental
joint changes from the reported state; the controller clips them to a safe
per-call magnitude and converts them to absolute joint targets. Omit an arm or
waist vector to hold it. Each hand accepts open, close, or hold. Use small,
deliberate motions, change only the joints needed for the current sub-goal,
and inspect the next image before correcting the motion.

The ego camera is mounted on the robot and faces the work surface. Plan a
short sequence: identify the source and destination, choose the nearer arm,
approach, close the selected hand, lift, move to the destination, open the
hand, and retreat. Keep the unused hand open and away from the workspace.
Include a short note describing what you see and why you chose the command.
Use only the camera images and reported robot state; do not assume hidden
simulator metadata."""

DEFAULT_OLI_SYSTEM_PROMPT = """You control an OLI humanoid through relative
keypoint and base motions. At every turn you receive available camera images,
the current absolute keypoint and base poses, and both hand states.
Use exactly one control_oli tool call. Omitted targets hold their current
values.

Infer the current task phase from the images, robot state, and prior turns.
Continue from the observed state; do not restart a fixed grasping sequence or
infer the phase from finger closure alone. Make one small, deliberate move.
Each hand target contains its six native finger channels in this order:
[thumb flexion, thumb lateral splay, index, middle, ring, little].
Useful reference profiles are open [0, 98, 0, 0, 0, 0], normal grasp
[30, 92, 40, 98, 98, 98], and firm grasp [58, 92, 58, 98, 98, 98]. Values
between profiles provide partial closure. Channel 1 controls lateral thumb
splay rather than finger flexion, so its open value intentionally stays near
98. Omit an unused hand to hold all six current channels. Keep unused
keypoints still. Position units are metres in the robot base frame: +x points
forward from the robot, +y points to the robot's left, and +z points upward.
All dx/dy/dz commands are relative displacements. Base dx/dy and yaw are total
body-frame changes over the next action chunk; base dz changes its height.
Base roll_deg and pitch_deg are absolute world-frame orientation targets.
Keypoint rotations are local relative [roll, pitch, yaw] changes. A wrist's
local +x points from wrist to fingers, +y lies laterally across the palm, and
+z is the outward palm normal. A foot's local +x points forward, +y points
left, and +z points upward. Roll twists about +x, pitch rotates about +y, and
yaw rotates about +z. Positive angles follow the right-hand rule. Use base and
feet when locomotion, turning, or changing stance is needed, coordinating them
with the upper body. Include a short note explaining the visible evidence and
requested move. Runtime limits each command; use the next observation to verify
what was executed."""


@VLAS.register_module()
class OpenAIResponsesVLA(nn.Module):
    """Inference-only VLA backed by the OpenAI Responses API.

    The model emits a high-level absolute Cartesian ``move_to`` tool call.
    This wrapper converts that target into a short chunk of normalized LIBERO
    OSC pose actions. It intentionally has no local trainable weights.
    """

    def __init__(self,
                 model: str = 'gpt-6-astra',
                 base_url: str = 'https://api.openai.com/v1',
                 api_key_env: str = 'OPENAI_API_KEY',
                 reasoning_effort: str = 'medium',
                 max_output_tokens: int = None,
                 request_timeout: float = 120.0,
                 max_retries: int = 2,
                 retry_backoff: float = 2.0,
                 image_detail: str = 'low',
                 image_format: str = 'JPEG',
                 jpeg_quality: int = 85,
                 image_horizon: int = 2,
                 max_llm_calls: int = 20,
                 action_horizon: int = 10,
                 max_speed_fraction: float = 0.25,
                 position_action_scale: float = 0.01,
                 rotation_action_scale: float = 0.1,
                 max_rotation_speed_fraction: float = 0.25,
                 gripper_settle_steps: int = 8,
                 workspace_bounds: Sequence[Sequence[float]] = ((-0.45, 0.45),
                                                                (-0.45, 0.45),
                                                                (-0.05, 1.40)),
                 system_prompt: str = DEFAULT_LIBERO_SYSTEM_PROMPT,
                 task_visual_hints: Dict[str, str] = None,
                 device: str = None,
                 torch_dtype=None) -> None:
        super().__init__()
        del device, torch_dtype
        if action_horizon < 1:
            raise ValueError('action_horizon must be at least 1')
        if not 0 < max_speed_fraction <= 1:
            raise ValueError('max_speed_fraction must be in (0, 1]')
        if (not math.isfinite(position_action_scale)
                or position_action_scale <= 0):
            raise ValueError(
                'position_action_scale must be finite and positive')
        if (not math.isfinite(rotation_action_scale)
                or rotation_action_scale <= 0):
            raise ValueError(
                'rotation_action_scale must be finite and positive')
        if not 0 < max_rotation_speed_fraction <= 1:
            raise ValueError('max_rotation_speed_fraction must be in (0, 1]')
        bounds_array = np.asarray(workspace_bounds, dtype=np.float64)
        if (bounds_array.shape != (3, 2)
                or not np.all(np.isfinite(bounds_array))
                or np.any(bounds_array[:, 0] >= bounds_array[:, 1])):
            raise ValueError(
                'workspace_bounds must contain finite, ordered x/y/z bounds')

        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key_env = api_key_env
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.image_detail = image_detail
        self.image_format = image_format.upper()
        self.jpeg_quality = int(jpeg_quality)
        self.image_horizon = int(image_horizon)
        self.max_llm_calls = int(max_llm_calls)
        self.action_horizon = int(action_horizon)
        self.max_speed_fraction = float(max_speed_fraction)
        self.position_action_scale = float(position_action_scale)
        self.rotation_action_scale = float(rotation_action_scale)
        self.max_rotation_speed_fraction = float(max_rotation_speed_fraction)
        self.gripper_settle_steps = int(gripper_settle_steps)
        self.workspace_bounds = tuple((float(bounds[0]), float(bounds[1]))
                                      for bounds in workspace_bounds)
        self.system_prompt = system_prompt
        self.task_visual_hints = {
            str(task).strip().lower(): str(hint)
            for task, hint in (task_visual_hints or {}).items()
        }

        # Keep nn.Module device/dtype methods valid without loading a model.
        self.register_buffer(
            '_device_anchor', torch.zeros(0), persistent=False)
        self.norm_stats = None
        self.freeze_vision_backbone = True
        self.freeze_llm_backbone = True
        self.freeze_projector = True
        self.freeze_vlm_backbone = True
        self.last_note = ''
        self.last_response_metadata = {}
        self._history: List[Dict[str, Any]] = []
        self._task_description = None
        self._llm_calls = 0
        self._last_gripper_action = -1.0
        self._budget_warning_emitted = False

    @property
    def tools(self) -> List[Dict[str, Any]]:
        x_bounds, y_bounds, z_bounds = self.workspace_bounds
        max_rotation = (
            self.action_horizon * self.rotation_action_scale *
            self.max_rotation_speed_fraction)
        description = (
            'Move the robot end effector to an absolute Cartesian target. '
            'Omitted x/y/z dimensions keep their current value. Optional '
            'rotation_delta is a relative WORLD-frame axis-angle vector '
            'in radians for the whole call, not Euler angles. Omit it to '
            'hold orientation. Rotation magnitude is limited to approximately '
            f'{max_rotation:.3f} '
            'radians per call; inspect the next observation for the '
            'achieved pose. '
            'Position bounds (meters): '
            f'x=[{x_bounds[0]}, {x_bounds[1]}], '
            f'y=[{y_bounds[0]}, {y_bounds[1]}], '
            f'z=[{z_bounds[0]}, {z_bounds[1]}]. Gripper target 0 is fully '
            'closed and 1 is fully open.')
        return [{
            'type': 'function',
            'name': 'move_to',
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': {
                    'targets': {
                        'type': 'object',
                        'properties': {
                            'x': {
                                'type': 'number'
                            },
                            'y': {
                                'type': 'number'
                            },
                            'z': {
                                'type': 'number'
                            },
                            'rotation_delta': {
                                'type':
                                'array',
                                'items': {
                                    'type': 'number'
                                },
                                'minItems':
                                3,
                                'maxItems':
                                3,
                                'description':
                                ('Relative world-axis rotation vector '
                                 '[rx, ry, rz], radians, right-hand rule. '
                                 'For yaw alignment use [0, 0, angle]. '
                                 'Not Euler angles; omit to hold '
                                 'orientation.'),
                            },
                            'gripper': {
                                'type': 'number',
                                'minimum': 0,
                                'maximum': 1,
                            },
                        },
                        'additionalProperties': False,
                    },
                    'note': {
                        'type':
                        'string',
                        'description':
                        ('One or two sentences describing the current '
                         'observation and why this motion was chosen.'),
                    },
                },
                'required': ['targets', 'note'],
                'additionalProperties': False,
            },
            'strict': False,
        }]

    def forward(self, *args, **kwargs):
        raise RuntimeError('OpenAIResponsesVLA is inference-only.')

    def get_fsdp_wrapping_policy(self):
        return None

    def freeze_backbones(self) -> None:
        return

    def from_pretrained(self) -> None:
        return

    def _reset_episode(self, task_description: str) -> None:
        self._task_description = task_description
        self._llm_calls = 0
        self._last_gripper_action = -1.0
        self.last_note = ''
        self.last_response_metadata = {}
        self._budget_warning_emitted = False
        goal_content = f'Goal: {task_description}'
        visual_hint = self.task_visual_hints.get(
            task_description.strip().lower())
        if visual_hint:
            goal_content += f'\nVisual grounding hint: {visual_hint}'
        self._history = [
            {
                'role': 'system',
                'content': self.system_prompt,
            },
            {
                'role': 'user',
                'content': goal_content,
            },
        ]

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _unbatch_array(cls, value: Any) -> np.ndarray:
        value = cls._as_numpy(value)
        if value.ndim > 1 and value.shape[0] == 1:
            value = value[0]
        return value

    @staticmethod
    def _unbatch_text(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        return str(value)

    def _image_data_url(self, image: Any) -> str:
        if torch.is_tensor(image):
            image = image.detach().float().cpu().numpy()
        if isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            array = np.asarray(image)
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))
            if np.issubdtype(array.dtype, np.floating):
                if array.size and float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            elif array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array).convert('RGB')

        buffer = io.BytesIO()
        save_kwargs = {}
        if self.image_format == 'JPEG':
            save_kwargs['quality'] = self.jpeg_quality
        pil_image.save(buffer, format=self.image_format, **save_kwargs)
        mime = 'jpeg' if self.image_format == 'JPEG' else \
            self.image_format.lower()
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        return f'data:image/{mime};base64,{encoded}'

    def _image_content(self, images, image_names):
        content = []
        for name, image in zip(image_names, images):
            image_item = {
                'type': 'input_image',
                'image_url': self._image_data_url(image),
            }
            if self.image_detail:
                image_item['detail'] = self.image_detail
            content.extend([{
                'type': 'input_text',
                'text': f"camera '{name}':",
            }, image_item])
        return content

    def _observation_message(self,
                             images: Sequence[Any],
                             image_names: Sequence[str],
                             task_description: str,
                             eef_position: Any,
                             eef_quaternion: Any,
                             gripper_position: Any,
                             joint_position: Any = None) -> Dict[str, Any]:
        position = self._state_vector(eef_position, 3, 'eef_position')
        quaternion = self._state_vector(eef_quaternion, 4, 'eef_quaternion')
        if np.linalg.norm(quaternion) < 1e-8:
            raise RuntimeError('eef_quaternion must be nonzero')
        gripper = self._unbatch_array(gripper_position).reshape(-1)
        identify, grasp, release = [
            max(1, math.ceil(self.max_llm_calls * fraction))
            for fraction in (0.2, 0.5, 0.9)
        ]
        lines = [
            'Current observation.',
            f'Instruction: {task_description}',
            (f'Action call: {self._llm_calls + 1}/{self.max_llm_calls}. '
             'Use the call budget efficiently: identify the target by call '
             f'{identify}, aim to grasp by call {grasp}, and release it at '
             f'the destination by call {release}.'),
            'robot0_eef_pos (x, y, z meters): ' +
            np.array2string(position, precision=5, separator=', '),
            'robot0_eef_quat (x, y, z, w): ' +
            np.array2string(quaternion, precision=5, separator=', '),
            'robot0_gripper_qpos: ' +
            np.array2string(gripper, precision=5, separator=', '),
        ]
        if joint_position is not None:
            joints = self._unbatch_array(joint_position).reshape(-1)
            lines.append('robot0_joint_pos: ' +
                         np.array2string(joints, precision=5, separator=', '))

        content: List[Dict[str, Any]] = [{
            'type': 'input_text',
            'text': '\n'.join(lines),
        }]
        content.extend(self._image_content(images, image_names))
        return {'role': 'user', 'content': content}

    def _compact_history(self) -> List[Dict[str, Any]]:
        history = copy.deepcopy(self._history)
        image_messages = [
            index for index, item in enumerate(history)
            if item.get('role') == 'user'
            and isinstance(item.get('content'), list) and any(
                content.get('type') == 'input_image'
                for content in item['content'])
        ]
        keep = set(image_messages[-self.image_horizon:]) \
            if self.image_horizon > 0 else set()
        for index in image_messages:
            if index in keep:
                continue
            content = history[index]['content']
            num_images = sum(
                item.get('type') == 'input_image' for item in content)
            history[index]['content'] = [
                item for item in content if item.get('type') != 'input_image'
            ]
            history[index]['content'].append({
                'type':
                'input_text',
                'text':
                f'[{num_images} prior camera frame(s) omitted]',
            })
        return history

    def _request_body(self) -> Dict[str, Any]:
        body = {
            'model': self.model,
            'input': self._compact_history(),
            'tools': self.tools,
            'tool_choice': 'required',
            'parallel_tool_calls': False,
        }
        if self.reasoning_effort is not None:
            body['reasoning'] = {'effort': self.reasoning_effort}
        if self.max_output_tokens is not None:
            body['max_output_tokens'] = self.max_output_tokens
        return body

    def _post_json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f'Missing OpenAI API key in environment variable '
                f'{self.api_key_env!r}.')

        request = urllib.request.Request(
            f'{self.base_url}/responses',
            data=json.dumps(body).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'FluxVLA/OpenAIResponsesVLA',
            },
            method='POST')
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                        request, timeout=self.request_timeout) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode('utf-8', errors='replace')
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(
                        f'OpenAI Responses API returned HTTP {exc.code}: '
                        f'{error_body[:1000]}') from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f'OpenAI Responses API request failed: {exc}') from exc
            time.sleep(self.retry_backoff * (2**attempt))
        raise AssertionError('unreachable')

    @staticmethod
    def _function_call(response: Dict[str, Any],
                       name: str = 'move_to') -> Dict[str, Any]:
        calls = [
            item for item in response.get('output', [])
            if item.get('type') == 'function_call' and item.get('name') == name
        ]
        if len(calls) != 1:
            output_types = [
                item.get('type') for item in response.get('output', [])
            ]
            raise RuntimeError(
                f'Expected exactly one {name} tool call from the OpenAI '
                f'Responses API, got {len(calls)}; output types={output_types}'
            )
        return calls[0]

    @classmethod
    def _state_vector(cls, value: Any, length: int, name: str) -> np.ndarray:
        vector = cls._unbatch_array(value).astype(np.float64)
        if vector.shape != (length, ) or not np.all(np.isfinite(vector)):
            raise RuntimeError(f'{name} must contain {length} finite values')
        return vector

    @staticmethod
    def _target_number(value: Any, name: str) -> float:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value)):
            raise RuntimeError(f'move_to.{name} must be a finite number')
        return float(value)

    def _actions_from_targets(self, targets: Dict[str, Any],
                              eef_position: Any) -> torch.Tensor:
        if not isinstance(targets, dict):
            raise RuntimeError('move_to.targets must be an object')
        unknown = set(targets) - {'x', 'y', 'z', 'rotation_delta', 'gripper'}
        if unknown:
            raise RuntimeError(
                f'Unknown move_to target fields: {sorted(unknown)}')
        current = self._state_vector(eef_position, 3, 'eef_position')
        target = current.copy()
        for index, key in enumerate(('x', 'y', 'z')):
            if key in targets:
                lo, hi = self.workspace_bounds[index]
                target[index] = np.clip(
                    self._target_number(targets[key], key), lo, hi)

        delta = target - current
        max_step = self.position_action_scale * self.max_speed_fraction
        move_steps = 1
        if np.any(delta):
            move_steps = math.ceil(
                min(self.action_horizon,
                    np.max(np.abs(delta)) / max_step))

        rotation_delta = np.zeros(3, dtype=np.float64)
        if 'rotation_delta' in targets:
            value = targets['rotation_delta']
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise RuntimeError(
                    'move_to.rotation_delta must contain 3 finite numbers')
            rotation_delta = np.array([
                self._target_number(item, 'rotation_delta') for item in value
            ])
        rotation_angle = math.hypot(*rotation_delta)
        if not math.isfinite(rotation_angle):
            raise RuntimeError(
                'move_to.rotation_delta magnitude must be finite')
        max_rotation_step = (
            self.rotation_action_scale * self.max_rotation_speed_fraction)
        # Bound the vector's norm, not its individual coordinates, to preserve
        # the requested world rotation axis. LIBERO's fixed Panda base is
        # world-aligned. Native OSC rotation inputs are axis-angle deltas, not
        # Euler angles or deltas in the gripper's local frame.
        max_rotation = self.action_horizon * max_rotation_step
        if rotation_angle > max_rotation:
            rotation_delta *= max_rotation / rotation_angle
            rotation_angle = max_rotation
        rotation_steps = max(1, math.ceil(rotation_angle / max_rotation_step))

        if 'gripper' not in targets:
            gripper_action = self._last_gripper_action
            gripper_steps = 1
        else:
            gripper_target = self._target_number(targets['gripper'], 'gripper')
            if not 0 <= gripper_target <= 1:
                raise RuntimeError('move_to.gripper must be in [0, 1]')
            gripper_action = -1.0 if gripper_target >= 0.5 else 1.0
            gripper_steps = self.gripper_settle_steps
        self._last_gripper_action = gripper_action

        num_steps = min(self.action_horizon,
                        max(1, move_steps, rotation_steps, gripper_steps))
        xyz_action = np.clip(delta / (num_steps * self.position_action_scale),
                             -self.max_speed_fraction, self.max_speed_fraction)
        rotation_action = rotation_delta / (
            num_steps * self.rotation_action_scale)
        action = np.concatenate(
            [xyz_action, rotation_action,
             np.array([gripper_action])])
        actions = np.repeat(action[None], num_steps, axis=0)
        return torch.from_numpy(actions.astype(np.float32)).unsqueeze(0)

    def _hold_actions(self) -> torch.Tensor:
        action = np.array([0, 0, 0, 0, 0, 0, self._last_gripper_action],
                          dtype=np.float32)
        actions = np.repeat(action[None], self.action_horizon, axis=0)
        return torch.from_numpy(actions).unsqueeze(0)

    @torch.inference_mode()
    def predict_action(self,
                       images: Sequence[Any],
                       task_description: str,
                       eef_position: Any,
                       eef_quaternion: Any,
                       gripper_position: Any,
                       image_names: Sequence[str] = None,
                       joint_position: Any = None,
                       reset_history: bool = False,
                       **kwargs) -> torch.Tensor:
        del kwargs
        task_description = self._unbatch_text(task_description)
        if reset_history or self._task_description != task_description:
            self._reset_episode(task_description)

        if image_names is None:
            image_names = [f'camera_{index}' for index in range(len(images))]
        elif (isinstance(image_names, (list, tuple)) and len(image_names) == 1
              and isinstance(image_names[0], (list, tuple))):
            image_names = image_names[0]

        if self._llm_calls >= self.max_llm_calls:
            if not self._budget_warning_emitted:
                overwatch.warning(
                    f'OpenAI call budget ({self.max_llm_calls}) exhausted; '
                    'returning hold actions for the rest of the episode.')
                self._budget_warning_emitted = True
            return self._hold_actions()

        self._history.append(
            self._observation_message(images, image_names, task_description,
                                      eef_position, eef_quaternion,
                                      gripper_position, joint_position))

        start = time.monotonic()
        response = self._post_json(self._request_body())
        latency = time.monotonic() - start
        self._llm_calls += 1
        call = self._function_call(response)
        try:
            arguments = json.loads(call.get('arguments', '{}'))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f'Invalid move_to arguments: {call.get("arguments")!r}') \
                from exc
        if not isinstance(arguments, dict):
            raise RuntimeError('move_to arguments must be an object')
        unknown = set(arguments) - {'targets', 'note'}
        if unknown:
            raise RuntimeError(f'Unknown move_to arguments: {sorted(unknown)}')
        targets = arguments.get('targets')
        if not isinstance(targets, dict):
            raise RuntimeError('move_to.targets must be an object')
        self.last_note = str(arguments.get('note', ''))
        usage = response.get('usage') or {}
        self.last_response_metadata = {
            'id': response.get('id'),
            'model': response.get('model', self.model),
            'latency_seconds': latency,
            'input_tokens': usage.get('input_tokens'),
            'output_tokens': usage.get('output_tokens'),
        }
        overwatch.info(f'GPT action {self._llm_calls}/{self.max_llm_calls}: '
                       f'{targets} | {self.last_note}')

        call_id = call.get('call_id')
        self._history.append({
            'type': 'function_call',
            'call_id': call_id,
            'name': 'move_to',
            'arguments': call.get('arguments', '{}'),
        })
        actions = self._actions_from_targets(targets, eef_position)
        self._history.append({
            'type':
            'function_call_output',
            'call_id':
            call_id,
            'output':
            f'executing move_to over {actions.shape[1]} steps',
        })
        return actions


@VLAS.register_module()
class OpenAIResponsesOliVLA(OpenAIResponsesVLA):
    """Checkpoint-free GPT controller for OLI's keypoint action wire."""

    ACTION_SEGMENTS = {
        'head': (0, 9),
        'left_foot': (9, 18),
        'right_foot': (18, 27),
        'left_wrist': (27, 36),
        'right_wrist': (36, 45),
        'base': (45, 54),
        'hands': (54, 66),
    }

    def __init__(self,
                 max_wrist_delta: float = 0.08,
                 max_head_delta: float = 0.03,
                 max_foot_delta: float = 0.08,
                 max_base_xy_delta: float = 0.10,
                 max_base_height_delta: float = 0.08,
                 max_rotation_delta_deg: float = 15.0,
                 max_base_tilt_deg: float = 15.0,
                 workspace_bounds: Sequence[Sequence[float]] = ((-0.30, 0.80),
                                                                (-0.65, 0.65),
                                                                (-0.30, 1.50)),
                 system_prompt: str = DEFAULT_OLI_SYSTEM_PROMPT,
                 **kwargs) -> None:
        defaults = {
            'image_detail': 'high',
            'jpeg_quality': 95,
            'max_llm_calls': 50,
            'action_horizon': 50,
        }
        for name, value in defaults.items():
            kwargs.setdefault(name, value)
        super().__init__(
            workspace_bounds=workspace_bounds,
            system_prompt=system_prompt,
            **kwargs)
        if (max_wrist_delta <= 0 or max_head_delta <= 0 or max_foot_delta <= 0
                or max_base_xy_delta <= 0 or max_base_height_delta <= 0
                or max_rotation_delta_deg <= 0 or max_base_tilt_deg <= 0):
            raise ValueError('OLI motion limits must be positive')
        if self.action_horizon < 2:
            raise ValueError('OLI action_horizon must be at least 2')
        self.max_wrist_delta = float(max_wrist_delta)
        self.max_head_delta = float(max_head_delta)
        self.max_foot_delta = float(max_foot_delta)
        self.max_base_xy_delta = float(max_base_xy_delta)
        self.max_base_height_delta = float(max_base_height_delta)
        self.max_rotation_delta_deg = float(max_rotation_delta_deg)
        self.max_base_tilt_deg = float(max_base_tilt_deg)

    @property
    def tools(self) -> List[Dict[str, Any]]:
        hand = {
            'type':
            'array',
            'items': {
                'type': 'number',
                'minimum': 0,
                'maximum': 100,
            },
            'minItems':
            6,
            'maxItems':
            6,
            'description':
            ('Six channels ordered [thumb flexion, thumb lateral splay, '
             'index, middle, ring, little]. References: open '
             '[0,98,0,0,0,0], normal grasp [30,92,40,98,98,98], firm '
             'grasp [58,92,58,98,98,98]. Omit to hold current values.'),
        }
        translation_delta = {
            'type': 'object',
            'properties':
            {axis: {
                'type': 'number'
            }
             for axis in ('dx', 'dy', 'dz')},
            'additionalProperties': False,
        }
        keypoint = {
            'type': 'object',
            'properties':
            {axis: {
                'type': 'number'
            }
             for axis in ('dx', 'dy', 'dz')},
            'additionalProperties': False,
        }
        keypoint['properties']['rotation_delta_rpy_deg'] = {
            'type':
            'array',
            'items': {
                'type': 'number'
            },
            'minItems':
            3,
            'maxItems':
            3,
            'description':
            ('Local relative [roll, pitch, yaw] in degrees. For a wrist, +x '
             'points wrist-to-fingers, +y runs laterally across the palm, '
             'and +z is the outward palm normal. For a foot, +x points '
             'forward, +y left, and +z up. Omit to hold rotation.'),
        }
        base = {
            'type': 'object',
            'properties': {
                **{axis: {
                    'type': 'number'
                }
                   for axis in ('dx', 'dy', 'dz')},
                'yaw_delta_deg': {
                    'type': 'number',
                    'description': 'Total relative yaw over this chunk.',
                },
                'roll_deg': {
                    'type': 'number',
                    'minimum': -self.max_base_tilt_deg,
                    'maximum': self.max_base_tilt_deg,
                    'description': 'Absolute world-frame roll target.',
                },
                'pitch_deg': {
                    'type': 'number',
                    'minimum': -self.max_base_tilt_deg,
                    'maximum': self.max_base_tilt_deg,
                    'description': 'Absolute world-frame pitch target.',
                },
            },
            'additionalProperties': False,
        }
        return [{
            'type':
            'function',
            'name':
            'control_oli',
            'description':
            ('Request bounded relative keypoint and base motion, and native '
             'hand channels. Omitted fields hold their current values.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'targets': {
                        'type': 'object',
                        'properties': {
                            'left_wrist': keypoint,
                            'right_wrist': keypoint,
                            'left_foot': keypoint,
                            'right_foot': keypoint,
                            'head': translation_delta,
                            'base': base,
                            'left_hand': hand,
                            'right_hand': hand,
                        },
                        'additionalProperties': False,
                    },
                    'note': {
                        'type': 'string'
                    },
                },
                'required': ['targets', 'note'],
                'additionalProperties': False,
            },
            'strict':
            False,
        }]

    @classmethod
    def _current_oli_action(cls, poses: Dict[str, Any], base_pose: Any,
                            hands: Any):
        required = {
            'left_wrist', 'right_wrist', 'head', 'left_foot', 'right_foot'
        }
        if not isinstance(poses, dict) or not required.issubset(poses):
            raise RuntimeError(f'OLI poses must contain {sorted(required)}')
        action = np.empty(66, dtype=np.float64)
        for name in ('left_wrist', 'right_wrist', 'head', 'left_foot',
                     'right_foot'):
            pose = np.asarray(poses[name], dtype=np.float64).reshape(-1)
            if pose.shape != (9, ) or not np.isfinite(pose).all():
                raise RuntimeError(
                    f'{name} must contain nine finite xyz+rot6d values')
            start, _ = cls.ACTION_SEGMENTS[name]
            action[start:start + 9] = pose
        base_pose = np.asarray(base_pose, dtype=np.float64).reshape(-1)
        if base_pose.shape != (9, ) or not np.isfinite(base_pose).all():
            raise RuntimeError(
                'base must contain nine finite xyz+rot6d values')
        start, end = cls.ACTION_SEGMENTS['base']
        # The all-zero rot6d is the Operator's hold sentinel. If base control
        # is exposed later, accept yaw_delta/pitch/roll in the tool and convert
        # them to rot6d here; rot6d is not a natural interface for an LLM.
        action[start:end] = 0.0
        action[start + 2] = base_pose[2]
        hands = np.asarray(hands, dtype=np.float64).reshape(-1)
        if hands.shape != (12, ) or not np.isfinite(hands).all():
            raise RuntimeError('OLI hands must contain 12 finite values')
        hand_start, hand_end = cls.ACTION_SEGMENTS['hands']
        action[hand_start:hand_end] = hands
        return action

    def _oli_observation_message(self, images, image_names, task_description,
                                 poses, base_pose, hands):
        lines = [
            'Current OLI observation.',
            f'Instruction: {task_description}',
            f'Action call: {self._llm_calls + 1}/{self.max_llm_calls}.',
            f'Max wrist displacement per call: {self.max_wrist_delta:.3f} m.',
            f'Max head displacement per call: {self.max_head_delta:.3f} m.',
            f'Max foot displacement per call: {self.max_foot_delta:.3f} m.',
            ('Max base xy displacement per call: '
             f'{self.max_base_xy_delta:.3f} m.'),
            ('Max base height change per call: '
             f'{self.max_base_height_delta:.3f} m.'),
            ('Max keypoint rotation and base yaw per call: '
             f'{self.max_rotation_delta_deg:.1f} degrees.'),
            ('Max absolute base roll/pitch: '
             f'{self.max_base_tilt_deg:.1f} degrees.'),
            'Workspace xyz bounds: ' + np.array2string(
                np.asarray(self.workspace_bounds), precision=3,
                separator=', '),
        ]
        for name in ('left_wrist', 'right_wrist', 'head', 'left_foot',
                     'right_foot'):
            pose = np.asarray(poses[name], dtype=np.float64).reshape(-1)
            lines.append(
                f'{name} xyz in base frame: ' +
                np.array2string(pose[:3], precision=5, separator=', '))
        base_pose = np.asarray(base_pose, dtype=np.float64).reshape(-1)
        if base_pose.shape != (9, ) or not np.isfinite(base_pose).all():
            raise RuntimeError(
                'base must contain nine finite xyz+rot6d values')
        lines.append(
            'base xyz in world frame: ' +
            np.array2string(base_pose[:3], precision=5, separator=', '))
        base_ypr = self._rotation_from_rot6d(base_pose[3:]).as_euler(
            'ZYX', degrees=True)
        lines.append(
            'base world-frame roll/pitch/yaw in degrees: ' +
            np.array2string(base_ypr[[2, 1, 0]], precision=3, separator=', '))
        hands = np.asarray(hands).reshape(-1)
        lines.append(
            'left_hand [thumb_flex, thumb_splay, index, middle, ring, '
            'little]: ' +
            np.array2string(hands[0::2], precision=1, separator=', '))
        lines.append(
            'right_hand [thumb_flex, thumb_splay, index, middle, ring, '
            'little]: ' +
            np.array2string(hands[1::2], precision=1, separator=', '))
        content = [{
            'type': 'input_text',
            'text': '\n'.join(lines),
        }]
        content.extend(self._image_content(images, image_names))
        return {'role': 'user', 'content': content}

    def _apply_oli_targets(self, current, targets, base_pose):
        if not isinstance(targets, dict):
            raise RuntimeError('control_oli.targets must be an object')
        allowed = {
            'left_wrist', 'right_wrist', 'left_foot', 'right_foot', 'head',
            'base', 'left_hand', 'right_hand'
        }
        unknown = set(targets) - allowed
        if unknown:
            raise RuntimeError(
                f'Unknown control_oli targets: {sorted(unknown)}')
        target = current.copy()
        for name in ('left_wrist', 'right_wrist', 'left_foot', 'right_foot'):
            requested = targets.get(name)
            if requested is None:
                continue
            if not isinstance(requested, dict):
                raise RuntimeError(f'control_oli.{name} must be an object')
            allowed_axes = {'dx', 'dy', 'dz', 'rotation_delta_rpy_deg'}
            unknown_axes = set(requested) - allowed_axes
            if unknown_axes:
                raise RuntimeError(
                    f'Unknown {name} axes: {sorted(unknown_axes)}')
            start, _ = self.ACTION_SEGMENTS[name]
            old = current[start:start + 3]
            delta_base = np.array([
                self._target_number(requested.get(key, 0.0), f'{name}.{key}')
                for key in ('dx', 'dy', 'dz')
            ])
            max_delta = (
                self.max_wrist_delta
                if 'wrist' in name else self.max_foot_delta)
            norm = float(np.linalg.norm(delta_base))
            if norm > max_delta:
                delta_base *= max_delta / norm
            new = old + delta_base
            if 'wrist' in name:
                new = np.clip(new,
                              [bounds[0] for bounds in self.workspace_bounds],
                              [bounds[1] for bounds in self.workspace_bounds])
            target[start:start + 3] = new
            rpy = requested.get('rotation_delta_rpy_deg')
            if rpy is not None:
                rpy = np.asarray(rpy, dtype=np.float64).reshape(-1)
                if rpy.shape != (3, ) or not np.isfinite(rpy).all():
                    raise RuntimeError(
                        f'{name}.rotation_delta_rpy_deg must be finite (3,)')
                delta_rotation = Rotation.from_euler('xyz', rpy, degrees=True)
                angle = np.degrees(delta_rotation.magnitude())
                if angle > self.max_rotation_delta_deg:
                    delta_rotation = Rotation.from_rotvec(
                        delta_rotation.as_rotvec() *
                        self.max_rotation_delta_deg / angle)
                current_rotation = self._rotation_from_rot6d(
                    current[start + 3:start + 9])
                target[start + 3:start + 9] = self._rotation_to_rot6d(
                    current_rotation * delta_rotation)
        requested = targets.get('head')
        if requested is not None:
            if not isinstance(requested, dict):
                raise RuntimeError('control_oli.head must be an object')
            unknown_axes = set(requested) - {'dx', 'dy', 'dz'}
            if unknown_axes:
                raise RuntimeError(
                    f'Unknown head axes: {sorted(unknown_axes)}')
            start, _ = self.ACTION_SEGMENTS['head']
            old = current[start:start + 3]
            delta = np.array([
                self._target_number(requested.get(key, 0.0), f'head.{key}')
                for key in ('dx', 'dy', 'dz')
            ])
            norm = float(np.linalg.norm(delta))
            if norm > self.max_head_delta:
                delta *= self.max_head_delta / norm
            new = old + delta
            target[start:start + 3] = np.clip(
                new, [bounds[0] for bounds in self.workspace_bounds],
                [bounds[1] for bounds in self.workspace_bounds])
        requested = targets.get('base')
        if requested is not None:
            if not isinstance(requested, dict):
                raise RuntimeError('control_oli.base must be an object')
            allowed_axes = {
                'dx', 'dy', 'dz', 'yaw_delta_deg', 'roll_deg', 'pitch_deg'
            }
            unknown_axes = set(requested) - allowed_axes
            if unknown_axes:
                raise RuntimeError(
                    f'Unknown base axes: {sorted(unknown_axes)}')
            start, _ = self.ACTION_SEGMENTS['base']
            delta_xy = np.array([
                self._target_number(requested.get(key, 0.0), f'base.{key}')
                for key in ('dx', 'dy')
            ])
            norm = float(np.linalg.norm(delta_xy))
            if norm > self.max_base_xy_delta:
                delta_xy *= self.max_base_xy_delta / norm
            target[start:start + 2] = delta_xy
            dz = np.clip(
                self._target_number(requested.get('dz', 0.0), 'base.dz'),
                -self.max_base_height_delta, self.max_base_height_delta)
            target[start + 2] = np.clip(base_pose[2] + dz,
                                        self.workspace_bounds[2][0],
                                        self.workspace_bounds[2][1])
            rotation_keys = {'yaw_delta_deg', 'roll_deg', 'pitch_deg'}
            if rotation_keys.intersection(requested):
                measured_rotation = self._rotation_from_rot6d(base_pose[3:])
                measured_rpy = measured_rotation.as_euler('ZYX', degrees=True)
                yaw = np.clip(
                    self._target_number(
                        requested.get('yaw_delta_deg', 0.0),
                        'base.yaw_delta_deg'), -self.max_rotation_delta_deg,
                    self.max_rotation_delta_deg)
                pitch = np.clip(
                    self._target_number(
                        requested.get('pitch_deg', measured_rpy[1]),
                        'base.pitch_deg'), -self.max_base_tilt_deg,
                    self.max_base_tilt_deg)
                roll = np.clip(
                    self._target_number(
                        requested.get('roll_deg', measured_rpy[2]),
                        'base.roll_deg'), -self.max_base_tilt_deg,
                    self.max_base_tilt_deg)
                target_rotation = Rotation.from_euler(
                    'ZYX', [yaw, pitch, roll], degrees=True)
                target[start + 3:start +
                       9] = self._rotation_to_rot6d(target_rotation)
        hand_start, hand_end = self.ACTION_SEGMENTS['hands']
        for index, key in enumerate(('left_hand', 'right_hand')):
            command = targets.get(key)
            if command is None:
                continue
            command = np.asarray(command, dtype=np.float64).reshape(-1)
            if (command.shape != (6, ) or not np.isfinite(command).all()
                    or np.any(command < 0.0) or np.any(command > 100.0)):
                raise RuntimeError(
                    f'{key} must contain six values in [0, 100]')
            target[hand_start + index:hand_end:2] = command
        return target

    @staticmethod
    def _rotation_from_rot6d(rot6d):
        a1, a2 = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 /= np.linalg.norm(b2)
        return Rotation.from_matrix(np.stack((b1, b2, np.cross(b1, b2))))

    @staticmethod
    def _rotation_to_rot6d(rotation):
        return rotation.as_matrix()[:2].reshape(6)

    def _oli_trajectory(self, current, target, base_pose):
        blend = np.linspace(0.0, 1.0, self.action_horizon)
        actions = np.repeat(current[None], self.action_horizon, axis=0)
        for name in ('left_wrist', 'right_wrist', 'left_foot', 'right_foot',
                     'head'):
            start, _ = self.ACTION_SEGMENTS[name]
            actions[:, start:start + 3] = (
                current[start:start + 3] + blend[:, None] *
                (target[start:start + 3] - current[start:start + 3]))
            current_rotation = self._rotation_from_rot6d(current[start +
                                                                 3:start + 9])
            target_rotation = self._rotation_from_rot6d(target[start +
                                                               3:start + 9])
            rotation_delta = (current_rotation.inv() *
                              target_rotation).as_rotvec()
            rotations = current_rotation * Rotation.from_rotvec(
                blend[:, None] * rotation_delta)
            actions[:, start + 3:start + 9] = (
                rotations.as_matrix()[:, :2].reshape(-1, 6))
        base_start, _ = self.ACTION_SEGMENTS['base']
        actions[:, base_start:base_start + 2] = (
            target[base_start:base_start + 2] / self.action_horizon)
        actions[:, base_start + 2] = (
            current[base_start + 2] + blend *
            (target[base_start + 2] - current[base_start + 2]))
        if np.any(target[base_start + 3:base_start + 9]):
            current_ypr = self._rotation_from_rot6d(base_pose[3:]).as_euler(
                'ZYX', degrees=True)
            target_ypr = self._rotation_from_rot6d(
                target[base_start + 3:base_start + 9]).as_euler(
                    'ZYX', degrees=True)
            ypr = np.column_stack((
                np.full(self.action_horizon,
                        target_ypr[0] / self.action_horizon),
                current_ypr[1] + blend * (target_ypr[1] - current_ypr[1]),
                current_ypr[2] + blend * (target_ypr[2] - current_ypr[2]),
            ))
            actions[:, base_start + 3:base_start + 9] = (
                Rotation.from_euler('ZYX', ypr,
                                    degrees=True).as_matrix()[:, :2].reshape(
                                        -1, 6))
        hand_start, hand_end = self.ACTION_SEGMENTS['hands']
        actions[:, hand_start:hand_end] = (
            current[hand_start:hand_end] + blend[:, None] *
            (target[hand_start:hand_end] - current[hand_start:hand_end]))
        return torch.from_numpy(actions.astype(np.float32)).unsqueeze(0)

    def record_execution_outcome(self, executed: bool, detail: str = None):
        if (not self._history
                or self._history[-1].get('type') != 'function_call_output'):
            raise RuntimeError('No pending OLI command outcome')
        result = 'executed' if executed else 'not executed'
        if detail:
            result += f': {detail}'
        self._history[-1]['output'] = result

    @torch.inference_mode()
    def predict_action(self,
                       images: Sequence[Any],
                       task_description: str,
                       poses: Dict[str, Any],
                       base_pose: Any,
                       hands: Any,
                       image_names: Sequence[str] = None,
                       reset_history: bool = False,
                       **kwargs) -> torch.Tensor:
        del kwargs
        task_description = self._unbatch_text(task_description)
        if reset_history or self._task_description != task_description:
            self._reset_episode(task_description)
        image_names = image_names or [
            f'camera_{index}' for index in range(len(images))
        ]
        current = self._current_oli_action(poses, base_pose, hands)
        if self._llm_calls >= self.max_llm_calls:
            hold = np.repeat(current[None], self.action_horizon, axis=0)
            return torch.from_numpy(hold.astype(np.float32)).unsqueeze(0)
        self._history.append(
            self._oli_observation_message(images, image_names,
                                          task_description, poses, base_pose,
                                          hands))
        started = time.monotonic()
        response = self._post_json(self._request_body())
        latency = time.monotonic() - started
        self._llm_calls += 1
        call = self._function_call(response, 'control_oli')
        try:
            arguments = json.loads(call.get('arguments', '{}'))
        except json.JSONDecodeError as exc:
            raise RuntimeError('Invalid control_oli arguments') from exc
        if not isinstance(arguments, dict):
            raise RuntimeError('control_oli arguments must be an object')
        unknown = set(arguments) - {'targets', 'note'}
        if unknown:
            raise RuntimeError(
                f'Unknown control_oli arguments: {sorted(unknown)}')
        target = self._apply_oli_targets(current, arguments.get('targets'),
                                         base_pose)
        actions = self._oli_trajectory(current, target, base_pose)
        self.last_note = str(arguments.get('note', ''))
        usage = response.get('usage') or {}
        self.last_response_metadata = {
            'id': response.get('id'),
            'model': response.get('model', self.model),
            'latency_seconds': latency,
            'input_tokens': usage.get('input_tokens'),
            'output_tokens': usage.get('output_tokens'),
        }
        call_id = call.get('call_id')
        self._history.extend([{
            'type': 'function_call',
            'call_id': call_id,
            'name': 'control_oli',
            'arguments': call.get('arguments', '{}'),
        }, {
            'type': 'function_call_output',
            'call_id': call_id,
            'output': 'execution pending',
        }])
        return actions


@VLAS.register_module()
class OpenAIResponsesRobocasaVLA(OpenAIResponsesVLA):
    """Checkpoint-free Responses API policy for RoboCasa GR1 evaluation.

    The API emits bounded joint deltas plus discrete hand commands. They are
    converted to native, absolute GR1 controller targets in N1.5 order:
    left arm, right arm, left hand, right hand, then waist.
    """

    OPEN_HAND_ACTION = np.array([-1.5, -1.5, -1.5, -1.5, -3.0, 3.0],
                                dtype=np.float32)
    CLOSE_HAND_ACTION = np.array([1.5, 1.5, 1.5, 1.5, 3.0, 3.0],
                                 dtype=np.float32)

    def __init__(self,
                 model: str = 'gpt-6-astra',
                 base_url: str = 'https://api.openai.com/v1',
                 api_key_env: str = 'OPENAI_API_KEY',
                 reasoning_effort: str = 'medium',
                 max_output_tokens: int = None,
                 request_timeout: float = 120.0,
                 max_retries: int = 2,
                 retry_backoff: float = 2.0,
                 image_detail: str = 'high',
                 image_format: str = 'JPEG',
                 jpeg_quality: int = 95,
                 image_horizon: int = 2,
                 max_llm_calls: int = 90,
                 action_horizon: int = 8,
                 max_arm_joint_delta: float = 0.12,
                 max_waist_joint_delta: float = 0.05,
                 left_arm_joint_bounds: Sequence[Sequence[float]] = (
                     (-3.0, 3.0), (0.0, 3.0), (-3.0, 3.0), (-3.0, 0.0),
                     (-3.0, 3.0), (-1.5, 1.5), (-1.5, 1.5)),
                 right_arm_joint_bounds: Sequence[Sequence[float]] = (
                     (-3.0, 3.0), (-3.0, 0.0), (-3.0, 3.0), (-3.0, 0.0),
                     (-3.0, 3.0), (-1.5, 1.5), (-1.5, 1.5)),
                 waist_joint_bounds: Sequence[Sequence[float]] = ((-1.05,
                                                                   1.05),
                                                                  (-0.52,
                                                                   1.22),
                                                                  (-0.70,
                                                                   0.70)),
                 system_prompt: str = DEFAULT_ROBOCASA_SYSTEM_PROMPT,
                 task_visual_hints: Dict[str, str] = None,
                 device: str = None,
                 torch_dtype=None) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            image_detail=image_detail,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            image_horizon=image_horizon,
            max_llm_calls=max_llm_calls,
            action_horizon=action_horizon,
            system_prompt=system_prompt,
            task_visual_hints=task_visual_hints,
            device=device,
            torch_dtype=torch_dtype)
        if max_arm_joint_delta <= 0 or max_waist_joint_delta <= 0:
            raise ValueError('RoboCasa joint delta limits must be positive')
        if len(left_arm_joint_bounds) != 7:
            raise ValueError('left_arm_joint_bounds must contain seven pairs')
        if len(right_arm_joint_bounds) != 7:
            raise ValueError('right_arm_joint_bounds must contain seven pairs')
        if len(waist_joint_bounds) != 3:
            raise ValueError('waist_joint_bounds must contain three pairs')
        self.max_arm_joint_delta = float(max_arm_joint_delta)
        self.max_waist_joint_delta = float(max_waist_joint_delta)
        self.left_arm_joint_bounds = tuple((float(bounds[0]), float(bounds[1]))
                                           for bounds in left_arm_joint_bounds)
        self.right_arm_joint_bounds = tuple(
            (float(bounds[0]), float(bounds[1]))
            for bounds in right_arm_joint_bounds)
        self.waist_joint_bounds = tuple((float(bounds[0]), float(bounds[1]))
                                        for bounds in waist_joint_bounds)
        self._last_left_hand_action = self.OPEN_HAND_ACTION.copy()
        self._last_right_hand_action = self.OPEN_HAND_ACTION.copy()

    @property
    def tools(self) -> List[Dict[str, Any]]:
        arm_description = (
            'Seven incremental joint changes in radians, ordered as shoulder '
            'pitch, shoulder roll, shoulder yaw, elbow pitch, wrist yaw, '
            'wrist roll, wrist pitch. Values are clipped to '
            f'+/-{self.max_arm_joint_delta} per call.')
        waist_description = (
            'Three incremental joint changes in radians, ordered as yaw, '
            f'pitch, roll. Values are clipped to '
            f'+/-{self.max_waist_joint_delta} per call.')
        vector = lambda length, description: {  # noqa: E731
            'type': 'array',
            'items': {
                'type': 'number'
            },
            'minItems': length,
            'maxItems': length,
            'description': description,
        }
        return [{
            'type':
            'function',
            'name':
            'control_gr1',
            'description':
            ('Apply one bounded GR1 joint-space command and then observe '
             'again. Omitted joint groups hold their current position.'),
            'parameters': {
                'type': 'object',
                'properties': {
                    'left_arm_delta': vector(7, arm_description),
                    'right_arm_delta': vector(7, arm_description),
                    'waist_delta': vector(3, waist_description),
                    'left_hand': {
                        'type': 'string',
                        'enum': ['hold', 'open', 'close'],
                    },
                    'right_hand': {
                        'type': 'string',
                        'enum': ['hold', 'open', 'close'],
                    },
                    'note': {
                        'type':
                        'string',
                        'description':
                        ('One or two sentences describing the current '
                         'observation and why this command was chosen.'),
                    },
                },
                'required': ['note'],
                'additionalProperties': False,
            },
            'strict':
            False,
        }]

    def _reset_episode(self, task_description: str) -> None:
        super()._reset_episode(task_description)
        self._last_left_hand_action = self.OPEN_HAND_ACTION.copy()
        self._last_right_hand_action = self.OPEN_HAND_ACTION.copy()

    @staticmethod
    def _joint_vector(value: Any, length: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape[0] != length:
            raise RuntimeError(
                f'{name} must contain exactly {length} values, got '
                f'{vector.shape[0]}.')
        if not np.all(np.isfinite(vector)):
            raise RuntimeError(f'{name} contains a non-finite value.')
        return vector

    @staticmethod
    def _apply_delta(current: np.ndarray, command: Dict[str, Any], key: str,
                     max_delta: float, bounds) -> np.ndarray:
        if key not in command:
            return current.copy()
        delta = OpenAIResponsesRobocasaVLA._joint_vector(
            command[key], current.shape[0], key)
        target = current + np.clip(delta, -max_delta, max_delta)
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if bounds_array.ndim == 1:
            return np.clip(target, bounds_array[0], bounds_array[1])
        return np.clip(target, bounds_array[:, 0], bounds_array[:, 1])

    def _hand_action(self, command: str, side: str) -> np.ndarray:
        command = str(command or 'hold').lower()
        attr = f'_last_{side}_hand_action'
        if command == 'open':
            action = self.OPEN_HAND_ACTION.copy()
        elif command == 'close':
            action = self.CLOSE_HAND_ACTION.copy()
        elif command == 'hold':
            action = getattr(self, attr).copy()
        else:
            raise RuntimeError(
                f'{side}_hand must be open, close, or hold; got {command!r}.')
        setattr(self, attr, action.copy())
        return action

    def _robocasa_observation_message(self, images: Sequence[Any],
                                      image_names: Sequence[str],
                                      task_description: str, left_arm: Any,
                                      left_hand: Any, right_arm: Any,
                                      right_hand: Any,
                                      waist: Any) -> Dict[str, Any]:
        state_lines = [
            'Current observation.',
            f'Instruction: {task_description}',
            f'Action call: {self._llm_calls + 1}/{self.max_llm_calls}.',
        ]
        for name, value in (
            ('left_arm', left_arm),
            ('left_hand', left_hand),
            ('right_arm', right_arm),
            ('right_hand', right_hand),
            ('waist', waist),
        ):
            vector = self._unbatch_array(value).reshape(-1)
            state_lines.append(
                f'{name}: ' +
                np.array2string(vector, precision=5, separator=', '))
        content: List[Dict[str, Any]] = [{
            'type': 'input_text',
            'text': '\n'.join(state_lines),
        }]
        content.extend(self._image_content(images, image_names))
        return {'role': 'user', 'content': content}

    def _robocasa_actions(self, command: Dict[str, Any], left_arm: Any,
                          right_arm: Any, waist: Any) -> torch.Tensor:
        left_arm = self._unbatch_array(left_arm).astype(np.float64).reshape(-1)
        right_arm = self._unbatch_array(right_arm).astype(
            np.float64).reshape(-1)
        waist = self._unbatch_array(waist).astype(np.float64).reshape(-1)
        if left_arm.shape[0] != 7 or right_arm.shape[0] != 7:
            raise RuntimeError(
                'RoboCasa arm states must each contain 7 values.')
        if waist.shape[0] != 3:
            raise RuntimeError('RoboCasa waist state must contain 3 values.')

        left_target = self._apply_delta(left_arm, command, 'left_arm_delta',
                                        self.max_arm_joint_delta,
                                        self.left_arm_joint_bounds)
        right_target = self._apply_delta(right_arm, command, 'right_arm_delta',
                                         self.max_arm_joint_delta,
                                         self.right_arm_joint_bounds)
        waist_target = self._apply_delta(waist, command, 'waist_delta',
                                         self.max_waist_joint_delta,
                                         self.waist_joint_bounds)
        left_hand_target = self._hand_action(
            command.get('left_hand', 'hold'), 'left')
        right_hand_target = self._hand_action(
            command.get('right_hand', 'hold'), 'right')
        action = np.concatenate([
            left_target, right_target, left_hand_target, right_hand_target,
            waist_target
        ]).astype(np.float32)
        actions = np.repeat(action[None], self.action_horizon, axis=0)
        return torch.from_numpy(actions).unsqueeze(0)

    def _hold_robocasa_actions(self, left_arm: Any, right_arm: Any,
                               waist: Any) -> torch.Tensor:
        return self._robocasa_actions({}, left_arm, right_arm, waist)

    @torch.inference_mode()
    def predict_action(self,
                       images: Sequence[Any],
                       task_description: str,
                       left_arm: Any,
                       left_hand: Any,
                       right_arm: Any,
                       right_hand: Any,
                       waist: Any,
                       image_names: Sequence[str] = None,
                       reset_history: bool = False,
                       **kwargs) -> torch.Tensor:
        del kwargs
        task_description = self._unbatch_text(task_description)
        if reset_history or self._task_description != task_description:
            self._reset_episode(task_description)

        if image_names is None:
            image_names = [f'camera_{index}' for index in range(len(images))]
        elif (isinstance(image_names, (list, tuple)) and len(image_names) == 1
              and isinstance(image_names[0], (list, tuple))):
            image_names = image_names[0]

        if self._llm_calls >= self.max_llm_calls:
            if not self._budget_warning_emitted:
                overwatch.warning(
                    f'OpenAI call budget ({self.max_llm_calls}) exhausted; '
                    'returning GR1 hold actions for the rest of the episode.')
                self._budget_warning_emitted = True
            return self._hold_robocasa_actions(left_arm, right_arm, waist)

        self._history.append(
            self._robocasa_observation_message(images, image_names,
                                               task_description, left_arm,
                                               left_hand, right_arm,
                                               right_hand, waist))
        start = time.monotonic()
        response = self._post_json(self._request_body())
        latency = time.monotonic() - start
        self._llm_calls += 1
        call = self._function_call(response, 'control_gr1')
        try:
            command = json.loads(call.get('arguments', '{}'))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Invalid control_gr1 arguments: '
                               f'{call.get("arguments")!r}') from exc
        if not isinstance(command, dict):
            raise RuntimeError('control_gr1 arguments must be an object')
        self.last_note = str(command.get('note', ''))
        usage = response.get('usage') or {}
        self.last_response_metadata = {
            'id': response.get('id'),
            'model': response.get('model', self.model),
            'latency_seconds': latency,
            'input_tokens': usage.get('input_tokens'),
            'output_tokens': usage.get('output_tokens'),
        }
        logged_command = {
            key: value
            for key, value in command.items() if key != 'note'
        }
        overwatch.info(
            f'GPT RoboCasa action {self._llm_calls}/{self.max_llm_calls}: '
            f'{logged_command} | {self.last_note}')

        call_id = call.get('call_id')
        self._history.append({
            'type': 'function_call',
            'call_id': call_id,
            'name': 'control_gr1',
            'arguments': call.get('arguments', '{}'),
        })
        actions = self._robocasa_actions(command, left_arm, right_arm, waist)
        self._history.append({
            'type':
            'function_call_output',
            'call_id':
            call_id,
            'output':
            f'executing control_gr1 over {actions.shape[1]} steps',
        })
        return actions
