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
