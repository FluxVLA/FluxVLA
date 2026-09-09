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
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from fluxvla.engines import VLAS, initialize_overwatch

overwatch = initialize_overwatch(__name__)

DEFAULT_LIBERO_SYSTEM_PROMPT = """You control one Panda robot arm in the
LIBERO simulator. At every turn you receive an external camera image, a wrist
camera image, and the current end-effector state. Use exactly one move_to tool
call to make progress on the instruction. The simulator, not you, decides
when the task succeeds.

The move_to tool accepts an absolute end-effector position in MuJoCo world
coordinates. Unspecified dimensions hold their current value. The default
downward-facing gripper orientation is held fixed. Use small, deliberate
motions and re-check the next observation after every command. Approach an
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
        if position_action_scale <= 0:
            raise ValueError('position_action_scale must be positive')
        if len(workspace_bounds) != 3:
            raise ValueError('workspace_bounds must contain x/y/z bounds')

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
        description = (
            'Move the robot end effector to an absolute Cartesian target. '
            'Omitted x/y/z dimensions keep their current value. The fixed '
            'downward gripper orientation is preserved. Bounds: '
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

    def _observation_message(self,
                             images: Sequence[Any],
                             image_names: Sequence[str],
                             task_description: str,
                             eef_position: Any,
                             eef_quaternion: Any,
                             gripper_position: Any,
                             joint_position: Any = None) -> Dict[str, Any]:
        position = self._unbatch_array(eef_position).reshape(-1)
        quaternion = self._unbatch_array(eef_quaternion).reshape(-1)
        gripper = self._unbatch_array(gripper_position).reshape(-1)
        lines = [
            'Current observation.',
            f'Instruction: {task_description}',
            (f'Action call: {self._llm_calls + 1}/{self.max_llm_calls}. '
             'Use the call budget efficiently: identify the target by call '
             '10, aim to grasp by call 25, and release it at the destination '
             'by call 45.'),
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
        for name, image in zip(image_names, images):
            content.append({
                'type': 'input_text',
                'text': f"camera '{name}':",
            })
            image_item = {
                'type': 'input_image',
                'image_url': self._image_data_url(image),
            }
            if self.image_detail:
                image_item['detail'] = self.image_detail
            content.append(image_item)
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
    def _function_call(response: Dict[str, Any]) -> Dict[str, Any]:
        calls = [
            item for item in response.get('output', [])
            if item.get('type') == 'function_call'
            and item.get('name') == 'move_to'
        ]
        if len(calls) != 1:
            output_types = [
                item.get('type') for item in response.get('output', [])
            ]
            raise RuntimeError(
                'Expected exactly one move_to tool call from the OpenAI '
                f'Responses API, got {len(calls)}; output types={output_types}'
            )
        return calls[0]

    def _actions_from_targets(self, targets: Dict[str, Any],
                              eef_position: Any) -> torch.Tensor:
        current = self._unbatch_array(eef_position).astype(
            np.float64).reshape(-1)[:3]
        target = current.copy()
        for index, key in enumerate(('x', 'y', 'z')):
            if key in targets:
                lo, hi = self.workspace_bounds[index]
                target[index] = np.clip(float(targets[key]), lo, hi)

        delta = target - current
        max_step = self.position_action_scale * self.max_speed_fraction
        move_steps = int(math.ceil(np.max(np.abs(delta)) / max_step)) \
            if np.any(delta) else 1

        gripper_target = targets.get('gripper')
        if gripper_target is None:
            gripper_action = self._last_gripper_action
            gripper_steps = 1
        elif float(gripper_target) >= 0.5:
            gripper_action = -1.0
            gripper_steps = self.gripper_settle_steps
        else:
            gripper_action = 1.0
            gripper_steps = self.gripper_settle_steps
        self._last_gripper_action = gripper_action

        num_steps = min(self.action_horizon, max(1, move_steps, gripper_steps))
        xyz_action = np.clip(delta / (num_steps * self.position_action_scale),
                             -self.max_speed_fraction, self.max_speed_fraction)
        action = np.concatenate(
            [xyz_action, np.zeros(3),
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
        targets = arguments.get('targets', {})
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
        for name, image in zip(image_names, images):
            content.append({
                'type': 'input_text',
                'text': f"camera '{name}':",
            })
            image_item = {
                'type': 'input_image',
                'image_url': self._image_data_url(image),
            }
            if self.image_detail:
                image_item['detail'] = self.image_detail
            content.append(image_item)
        return {'role': 'user', 'content': content}

    @staticmethod
    def _robocasa_function_call(response: Dict[str, Any]) -> Dict[str, Any]:
        calls = [
            item for item in response.get('output', [])
            if item.get('type') == 'function_call'
            and item.get('name') == 'control_gr1'
        ]
        if len(calls) != 1:
            output_types = [
                item.get('type') for item in response.get('output', [])
            ]
            raise RuntimeError(
                'Expected exactly one control_gr1 tool call from the OpenAI '
                f'Responses API, got {len(calls)}; output types={output_types}'
            )
        return calls[0]

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
        call = self._robocasa_function_call(response)
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
