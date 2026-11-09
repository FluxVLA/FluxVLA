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
"""OpenAI Responses transport for the shared tool-calling robot policies."""

from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from fluxvla.engines import VLAS
# Re-export shared prompts alongside the concrete OpenAI policies.
from .tool_calling_vla import DEFAULT_LIBERO_SYSTEM_PROMPT  # noqa: F401
from .tool_calling_vla import DEFAULT_ROBOCASA_SYSTEM_PROMPT  # noqa: F401
from .tool_calling_vla import ToolCallingRobocasaVLA, ToolCallingVLA


class OpenAIResponsesProtocolMixin:
    """OpenAI defaults and transport; no robot-specific logic."""

    def __init__(self,
                 model: str = 'gpt-6-astra',
                 base_url: str = 'https://api.openai.com/v1',
                 api_key_env: str = 'OPENAI_API_KEY',
                 *args,
                 **kwargs) -> None:
        # Positional arguments after these three retain their original order
        # in the LIBERO/RoboCasa constructor, including each task's defaults.
        super().__init__(model, base_url, api_key_env, *args, **kwargs)

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
                'User-Agent': 'FluxVLA/OpenAILiberoVLA',
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


@VLAS.register_module()
class OpenAILiberoVLA(OpenAIResponsesProtocolMixin, ToolCallingVLA):
    """LIBERO tool-calling policy backed by the OpenAI Responses API."""


@VLAS.register_module()
class OpenAIRobocasaVLA(OpenAIResponsesProtocolMixin, ToolCallingRobocasaVLA):
    """RoboCasa GR1 tool-calling policy backed by the OpenAI Responses API."""
