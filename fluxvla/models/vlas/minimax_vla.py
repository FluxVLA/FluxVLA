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
"""MiniMax API policies for checkpoint-free robot evaluation.

Request bodies follow the MiniMax API references, which are identical for
the China (platform.minimax.cn) and international (platform.minimax.io)
platforms:

- Messages: https://platform.minimax.cn/docs/api-reference/text-chat-anthropic
- Chat: https://platform.minimax.cn/docs/api-reference/text-chat-openai
- Responses: https://platform.minimax.cn/docs/api-reference/responses-create
"""

from __future__ import annotations
import copy
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from fluxvla.engines import VLAS, initialize_overwatch
from .tool_calling_vla import ToolCallingRobocasaVLA, ToolCallingVLA

overwatch = initialize_overwatch(__name__)

DEFAULT_MINIMAX_MODEL = 'MiniMax-M3.1-Flash'
# Keys are issued per platform, so each region reads its own variable and
# never falls back to the other region's key.
MINIMAX_REGIONS = {
    'cn':
    dict(
        base_url='https://api.minimax.cn/v1',
        anthropic_base_url='https://api.minimax.cn/anthropic',
        api_key_env='MINIMAX_CN_API_KEY',
    ),
    'intl':
    dict(
        base_url='https://api.minimax.io/v1',
        anthropic_base_url='https://api.minimax.io/anthropic',
        api_key_env='MINIMAX_INTL_API_KEY',
    ),
}
API_PROTOCOLS = ('anthropic', 'chat_completions', 'responses')
TOOL_CHOICES = ('auto', 'required')

_TRUNCATION_REASONS = ('max_tokens', 'length', 'max_output_tokens')
# base_resp codes for timeouts, rate limits and internal errors.
_RETRYABLE_STATUS_CODES = (1001, 1002, 1013)


class MiniMaxProtocolMixin:
    """Serve an :class:`ToolCallingVLA` policy over MiniMax protocols.

    The policy history stays in Responses format, so observation building,
    image compaction, tool-call validation and action conversion are all
    inherited. Only the request body and the transport are protocol specific;
    every reply is normalized back to a Responses-style ``output`` list.

    MiniMax requires the complete assistant turn, including thinking blocks
    or ``reasoning_content``, to be sent back during multi-turn tool use.
    Raw assistant turns are kept per tool-call id and replayed verbatim.

    Args:
        region (str): ``'cn'`` (api.minimax.cn) or ``'intl'``
            (api.minimax.io). Selects the defaults of ``base_url``,
            ``anthropic_base_url`` and ``api_key_env`` from
            :data:`MINIMAX_REGIONS`; explicit values override them, but an
            official endpoint or key variable of the other region is
            rejected so that one region's key is never sent to the other.
        api_protocol (str): ``'anthropic'`` (Messages, MiniMax's recommended
            protocol), ``'chat_completions'`` or ``'responses'``.
        anthropic_base_url (str): Base URL of the Anthropic-compatible API;
            ``base_url`` serves the two OpenAI-compatible protocols.
        tool_choice (str): ``'auto'`` is the only value the Messages and
            Responses APIs accept; the system prompt already demands exactly
            one tool call. Messages gets one corrective retry for an
            ``end_turn`` without a tool call, within ``max_llm_calls``;
            persistent failure explicitly stops further API calls for this
            episode and holds until the environment ends it.
            ``'required'`` is sent
            only over Chat Completions, whose reference does not document
            ``tool_choice``; ``'auto'`` omits the field there.
        replay_assistant_turns (bool): Replay raw assistant turns. Disable
            only to send the minimal reconstructed tool call instead.
        max_output_tokens (int): Generation cap. Thinking tokens count
            towards it and MiniMax-M3.1-Flash always thinks, so keep ample
            headroom above the size of one tool call.
    """

    def __init__(self,
                 model: str = DEFAULT_MINIMAX_MODEL,
                 region: str = 'cn',
                 base_url: Optional[str] = None,
                 api_key_env: Optional[str] = None,
                 api_protocol: str = 'anthropic',
                 anthropic_base_url: Optional[str] = None,
                 tool_choice: str = 'auto',
                 replay_assistant_turns: bool = True,
                 reasoning_effort: Optional[str] = 'low',
                 max_output_tokens: Optional[int] = 16384,
                 image_detail: str = 'default',
                 **kwargs) -> None:
        base_url, anthropic_base_url, api_key_env = self._resolve_region(
            region, base_url, anthropic_base_url, api_key_env)
        if api_protocol not in API_PROTOCOLS:
            raise ValueError(f'api_protocol must be one of {API_PROTOCOLS}, '
                             f'got {api_protocol!r}')
        if tool_choice not in TOOL_CHOICES:
            raise ValueError(f'tool_choice must be one of {TOOL_CHOICES}, '
                             f'got {tool_choice!r}')
        if tool_choice == 'required' and api_protocol != 'chat_completions':
            raise ValueError(f'The MiniMax {api_protocol} API only accepts '
                             "tool_choice='auto'")
        if max_output_tokens is not None:
            max_output_tokens = int(max_output_tokens)
            if max_output_tokens < 1:
                raise ValueError('max_output_tokens must be positive')
        elif api_protocol == 'anthropic':
            raise ValueError(
                'max_output_tokens is required by the Anthropic protocol')
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            image_detail=image_detail,
            **kwargs)
        self.region = region
        self.api_protocol = api_protocol
        self.anthropic_base_url = anthropic_base_url.rstrip('/')
        self.tool_choice = tool_choice
        self.replay_assistant_turns = bool(replay_assistant_turns)
        self._assistant_turns: Dict[str, Any] = {}

    @staticmethod
    def _resolve_region(region: str, base_url: Optional[str],
                        anthropic_base_url: Optional[str],
                        api_key_env: Optional[str]):
        if region not in MINIMAX_REGIONS:
            raise ValueError(f'region must be one of {tuple(MINIMAX_REGIONS)}'
                             f', got {region!r}')
        defaults = MINIMAX_REGIONS[region]
        base_url = base_url or defaults['base_url']
        anthropic_base_url = (
            anthropic_base_url or defaults['anthropic_base_url'])
        api_key_env = api_key_env or defaults['api_key_env']
        for other, values in MINIMAX_REGIONS.items():
            if other == region:
                continue
            other_hosts = {
                urlsplit(values[key]).hostname
                for key in ('base_url', 'anthropic_base_url')
            }
            for url in (base_url, anthropic_base_url):
                if urlsplit(url).hostname in other_hosts:
                    raise ValueError(
                        f'{url!r} is the official {other!r} endpoint but '
                        f'region={region!r}; set region={other!r} instead.')
            if api_key_env == values['api_key_env']:
                raise ValueError(
                    f'{api_key_env!r} holds the {other!r} key but '
                    f'region={region!r}; set region={other!r} instead.')
        return base_url, anthropic_base_url, api_key_env

    def _reset_episode(self, task_description: str) -> None:
        super()._reset_episode(task_description)
        self._assistant_turns = {}

    def _replayed_turn(self, item: Dict[str, Any]) -> Any:
        if not self.replay_assistant_turns:
            return None
        return self._assistant_turns.get(item.get('call_id'))

    # ---- Request bodies -------------------------------------------------

    def _request_body(self) -> Dict[str, Any]:
        if self.api_protocol == 'responses':
            return self._responses_body()
        history = self._compact_history()
        if self.api_protocol == 'chat_completions':
            return self._chat_completions_body(history)
        return self._anthropic_body(history)

    def _responses_body(self) -> Dict[str, Any]:
        # Keep provider-specific options out of the shared robot policy.
        body = {
            'model': self.model,
            'input': self._compact_history(),
            'tools': self.tools,
            'tool_choice': 'auto',
        }
        if self.reasoning_effort is not None:
            body['reasoning'] = {'effort': self.reasoning_effort}
        if self.max_output_tokens is not None:
            body['max_output_tokens'] = self.max_output_tokens
        items = []
        for item in body['input']:
            turn = (
                self._replayed_turn(item)
                if item.get('type') == 'function_call' else None)
            if turn is not None:
                items.extend(turn)
            elif isinstance(item.get('content'), list):
                items.append({
                    **item, 'content':
                    [self._responses_part(part) for part in item['content']]
                })
            else:
                items.append(item)
        body['input'] = items
        return body

    @staticmethod
    def _responses_part(part: Dict[str, Any]) -> Dict[str, Any]:
        # MiniMax reads ``detail`` inside the image_url object.
        if part.get('type') != 'input_image':
            return part
        image = {'url': part['image_url']}
        if part.get('detail'):
            image['detail'] = part['detail']
        return {'type': 'input_image', 'image_url': image}

    def _chat_completions_body(
            self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = []
        for item in history:
            role = item.get('role')
            if role in ('system', 'user'):
                messages.append({
                    'role': role,
                    'content': self._chat_content(item['content'])
                })
            elif item.get('type') == 'function_call':
                messages.append(
                    self._replayed_turn(item) or {
                        'role':
                        'assistant',
                        'content':
                        '',
                        'tool_calls': [{
                            'id': item['call_id'],
                            'type': 'function',
                            'function': {
                                'name': item['name'],
                                'arguments': item['arguments'],
                            },
                        }],
                    })
            elif item.get('type') == 'function_call_output':
                messages.append({
                    'role': 'tool',
                    'tool_call_id': item['call_id'],
                    'content': item['output'],
                })
            else:
                raise RuntimeError(f'Unsupported history item: {item!r}')
        body = {
            'model':
            self.model,
            'messages':
            messages,
            'tools': [{
                'type': 'function',
                'function': {
                    'name': tool['name'],
                    'description': tool['description'],
                    'parameters': tool['parameters'],
                },
            } for tool in self.tools],
        }
        if self.tool_choice == 'required':
            body['tool_choice'] = 'required'
        if self.reasoning_effort is not None:
            body['reasoning_effort'] = self.reasoning_effort
        if self.max_output_tokens is not None:
            body['max_completion_tokens'] = self.max_output_tokens
        return body

    @staticmethod
    def _chat_content(content: Any) -> Any:
        if isinstance(content, str):
            return content
        parts = []
        for item in content:
            if item['type'] == 'input_text':
                parts.append({'type': 'text', 'text': item['text']})
            elif item['type'] == 'input_image':
                image = {'url': item['image_url']}
                if item.get('detail'):
                    image['detail'] = item['detail']
                parts.append({'type': 'image_url', 'image_url': image})
            else:
                raise RuntimeError(f'Unsupported content item: {item!r}')
        return parts

    def _anthropic_body(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        system = None
        messages: List[Dict[str, Any]] = []

        # Messages must alternate roles. Merging consecutive user content
        # also places each tool_result first in the following user turn.
        def push(role: str, blocks: List[Dict[str, Any]]) -> None:
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'].extend(blocks)
            else:
                messages.append({'role': role, 'content': list(blocks)})

        for item in history:
            role = item.get('role')
            if role == 'system':
                system = item['content']
            elif role == 'user':
                push('user', self._anthropic_content(item['content']))
            elif item.get('type') == 'function_call':
                push(
                    'assistant',
                    self._replayed_turn(item)
                    or [{
                        'type': 'tool_use',
                        'id': item['call_id'],
                        'name': item['name'],
                        'input': json.loads(item['arguments']),
                    }])
            elif item.get('type') == 'function_call_output':
                push('user', [{
                    'type': 'tool_result',
                    'tool_use_id': item['call_id'],
                    'content': item['output'],
                }])
            else:
                raise RuntimeError(f'Unsupported history item: {item!r}')
        body = {
            'model':
            self.model,
            'max_tokens':
            self.max_output_tokens,
            'messages':
            messages,
            'tools': [{
                'name': tool['name'],
                'description': tool['description'],
                'input_schema': tool['parameters'],
            } for tool in self.tools],
            'tool_choice': {
                'type': 'auto'
            },
        }
        if system:
            body['system'] = system
        if self.reasoning_effort is not None:
            body['output_config'] = {'effort': self.reasoning_effort}
        return body

    @staticmethod
    def _anthropic_content(content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{'type': 'text', 'text': content}]
        blocks = []
        for item in content:
            if item['type'] == 'input_text':
                blocks.append({'type': 'text', 'text': item['text']})
            elif item['type'] == 'input_image':
                url = item['image_url']
                if url.startswith('data:'):
                    header, data = url.split(',', 1)
                    source = {
                        'type': 'base64',
                        'media_type': header[len('data:'):].split(';')[0],
                        'data': data,
                    }
                else:
                    source = {'type': 'url', 'url': url}
                if item.get('detail'):
                    source['detail'] = item['detail']
                blocks.append({'type': 'image', 'source': source})
            else:
                raise RuntimeError(f'Unsupported content item: {item!r}')
        return blocks

    # ---- Transport and response normalization ---------------------------

    def _post_json(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f'Missing MiniMax {self.region!r} API key: set the '
                f'environment variable {self.api_key_env!r}.')
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'FluxVLA/{type(self).__name__}',
        }
        if self.api_protocol == 'anthropic':
            start = time.monotonic()
            for attempt in range(2):
                raw = self._http_post(f'{self.anthropic_base_url}/v1/messages',
                                      headers, body, 'MiniMax Messages API')
                self._raise_api_error(raw)
                blocks = raw.get('content') or []
                if raw.get('stop_reason') != 'end_turn' or any(
                        block.get('type') == 'tool_use' for block in blocks):
                    break
                # predict_action counts the successful reply. Count each
                # discarded reply here so repair cannot exceed its budget.
                self._llm_calls += 1
                retry = attempt == 0 and self._llm_calls < self.max_llm_calls
                reply_text = ' '.join(
                    block.get('text', '') for block in blocks
                    if block.get('type') == 'text')[:400]
                overwatch.warning(
                    f'MiniMax end_turn without a tool call; '
                    f'id={raw.get("id")!r}, retry={retry}, '
                    f'calls={self._llm_calls}/{self.max_llm_calls}, '
                    f'text={reply_text!r}')
                if not retry:
                    self._action_generation_stopped = True
                    usage = raw.get('usage') or {}
                    self.last_response_metadata = {
                        'id': raw.get('id'),
                        'model': raw.get('model', self.model),
                        'latency_seconds': time.monotonic() - start,
                        'input_tokens': usage.get('input_tokens'),
                        'output_tokens': usage.get('output_tokens'),
                        'stop_reason': 'missing_tool_call',
                        'api_stop_reason': raw.get('stop_reason'),
                        'policy_calls': self._llm_calls,
                        'response_text': reply_text,
                    }
                    self.last_note = (
                        'MiniMax returned no tool call; stopping API calls '
                        'for this episode and holding the previous control '
                        'state until the environment ends it. No tool call '
                        'or task success was fabricated.')
                    overwatch.warning(self.last_note)
                    return None
                # Re-request the unchanged observation with a reminder, not
                # a fabricated tool result or an action parsed from prose.
                body = copy.deepcopy(body)
                body['messages'][-1]['content'].append({
                    'type':
                    'text',
                    'text':
                    ('No tool call was returned and no action was '
                     'executed. The simulator has not ended this episode. '
                     f'Use exactly one {self.tools[0]["name"]} tool call '
                     'for your next action; put explanations in its note '
                     f'field. API calls used: {self._llm_calls}/'
                     f'{self.max_llm_calls}.'),
                })
            return self._normalize_anthropic(raw)
        if self.api_protocol == 'chat_completions':
            raw = self._http_post(f'{self.base_url}/chat/completions', headers,
                                  body, 'MiniMax Chat Completions API')
            return self._normalize_chat_completions(raw)
        raw = self._http_post(f'{self.base_url}/responses', headers, body,
                              'MiniMax Responses API')
        return self._normalize_responses(raw)

    def _http_post(self, url: str, headers: Dict[str, str],
                   body: Dict[str, Any], api_name: str) -> Dict[str, Any]:
        """POST JSON, retrying rate limits, server errors and timeouts.

        MiniMax can also report these conditions with HTTP 200 and a non-zero
        ``base_resp.status_code``, so those codes are retried as well.
        """
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode('utf-8'),
            headers=headers,
            method='POST')
        for attempt in range(self.max_retries + 1):
            last_attempt = attempt >= self.max_retries
            try:
                with urllib.request.urlopen(
                        request, timeout=self.request_timeout) as response:
                    raw = json.loads(response.read().decode('utf-8'))
                status = (raw.get('base_resp') or {}).get('status_code')
                if status not in _RETRYABLE_STATUS_CODES or last_attempt:
                    return raw
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode('utf-8', errors='replace')
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or last_attempt:
                    raise RuntimeError(f'{api_name} returned HTTP {exc.code}: '
                                       f'{error_body[:1000]}') from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if last_attempt:
                    raise RuntimeError(
                        f'{api_name} request failed: {exc}') from exc
            time.sleep(self.retry_backoff * (2**attempt))
        raise AssertionError('unreachable')

    @staticmethod
    def _raise_api_error(raw: Dict[str, Any]) -> None:
        base_resp = raw.get('base_resp') or {}
        if base_resp.get('status_code') not in (None, 0):
            raise RuntimeError(
                f'MiniMax API error {base_resp.get("status_code")}: '
                f'{base_resp.get("status_msg")}')
        if raw.get('type') == 'error' or raw.get('error'):
            raise RuntimeError(f'MiniMax API error: {raw.get("error")}')

    @staticmethod
    def _raise_without_tool_call(output: List[Dict[str, Any]],
                                 reason: Optional[str]) -> None:
        if output:
            return
        if reason in _TRUNCATION_REASONS:
            raise RuntimeError(
                f'MiniMax stopped ({reason}) before emitting a tool call; '
                'thinking tokens count towards the limit, so increase '
                'max_output_tokens or lower reasoning_effort.')
        if reason == 'content_filter':
            raise RuntimeError(
                'MiniMax withheld the reply (content_filter) before '
                'emitting a tool call.')
        raise RuntimeError(
            f'MiniMax returned no tool call (stop_reason={reason!r}); '
            'no action was executed.')

    def _store_turn(self, output: List[Dict[str, Any]], turn: Any) -> None:
        for item in output:
            self._assistant_turns[item['call_id']] = turn

    def _normalize_anthropic(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        self._raise_api_error(raw)
        blocks = raw.get('content') or []
        output = [{
            'type': 'function_call',
            'name': block.get('name'),
            'call_id': block.get('id'),
            'arguments': json.dumps(block.get('input', {})),
        } for block in blocks if block.get('type') == 'tool_use']
        self._raise_without_tool_call(output, raw.get('stop_reason'))
        self._store_turn(output, blocks)
        usage = raw.get('usage') or {}
        return {
            'id': raw.get('id'),
            'model': raw.get('model', self.model),
            'output': output,
            'usage': {
                'input_tokens': usage.get('input_tokens'),
                'output_tokens': usage.get('output_tokens'),
            },
        }

    def _normalize_chat_completions(self, raw: Dict[str,
                                                    Any]) -> Dict[str, Any]:
        self._raise_api_error(raw)
        choices = raw.get('choices') or []
        if not choices:
            raise RuntimeError('MiniMax Chat Completions API returned no '
                               f'choices: {json.dumps(raw)[:1000]}')
        message = choices[0].get('message') or {}
        output = []
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            arguments = function.get('arguments', '{}')
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            output.append({
                'type': 'function_call',
                'name': function.get('name'),
                'call_id': call.get('id'),
                'arguments': arguments,
            })
        self._raise_without_tool_call(output, choices[0].get('finish_reason'))
        self._store_turn(output, message)
        usage = raw.get('usage') or {}
        return {
            'id': raw.get('id'),
            'model': raw.get('model', self.model),
            'output': output,
            'usage': {
                'input_tokens': usage.get('prompt_tokens'),
                'output_tokens': usage.get('completion_tokens'),
            },
        }

    def _normalize_responses(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        self._raise_api_error(raw)
        items = raw.get('output') or []
        output = [
            item for item in items if item.get('type') == 'function_call'
        ]
        reason = (raw.get('incomplete_details') or {}).get('reason')
        self._raise_without_tool_call(output, reason)
        self._store_turn(output, items)
        return raw


@VLAS.register_module()
class MiniMaxLiberoVLA(MiniMaxProtocolMixin, ToolCallingVLA):
    """Checkpoint-free MiniMax policy emitting LIBERO ``move_to`` calls.

    Accepts every :class:`ToolCallingVLA` argument plus the protocol
    options of :class:`MiniMaxProtocolMixin`.
    """


@VLAS.register_module()
class MiniMaxRobocasaVLA(MiniMaxProtocolMixin, ToolCallingRobocasaVLA):
    """Checkpoint-free MiniMax policy emitting RoboCasa ``control_gr1`` calls.

    Accepts every :class:`ToolCallingRobocasaVLA` argument plus the
    protocol options of :class:`MiniMaxProtocolMixin`.
    """
