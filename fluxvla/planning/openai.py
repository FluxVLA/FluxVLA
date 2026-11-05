# Copyright 2026 Limx Dynamics
"""Online GPT planning via the same Responses protocol as OpenAILiberoVLA.

This client shares the existing GPT policies' model/connection conventions,
not their robot action tool. It emits closed-schema task decisions only.
"""

import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fluxvla.engines.utils.root import PLANNERS, TASK_MONITORS
from .backends import (_camera_data_urls, _StructuredPredicateMonitor,
                       _StructuredSubtaskPlanner, _validate_image_keys,
                       strict_json_loads)


class OpenAIResponsesClient:
    """Strict Responses JSON output, with no plain-JSON or model fallback.

    ``base_url`` and ``api_key_env`` follow the existing GPT configs, including
    OPENAI_BASE_URL / OPENAI_API_KEY_ENV environment overrides. Credentials are
    read at request time and are never stored in configs or response events.
    Retries are opt-in and share a single request time budget.
    """

    def __init__(self,
                 model='gpt-6-astra',
                 base_url=None,
                 api_key_env=None,
                 reasoning_effort='medium',
                 max_output_tokens=4096,
                 request_timeout=10.0,
                 max_retries=0,
                 retry_backoff=0.5,
                 image_keys=(),
                 image_detail='high',
                 max_image_size=640):
        if base_url is None:
            base_url = (
                os.environ.get('OPENAI_BASE_URL')
                or 'https://api.openai.com/v1')
        if api_key_env is None:
            api_key_env = (
                os.environ.get('OPENAI_API_KEY_ENV') or 'OPENAI_API_KEY')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('A nonempty OpenAI model is required')
        if not isinstance(base_url, str):
            raise ValueError('An HTTP(S) OpenAI base_url is required')
        parsed = urlparse(base_url)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError('base_url must be HTTP(S) without credentials, '
                             'query parameters or fragments')
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError('api_key_env must name a credential environment')
        if reasoning_effort is not None and (not isinstance(
                reasoning_effort, str) or not reasoning_effort.strip()):
            raise ValueError('reasoning_effort must be nonempty or None')
        if (not math.isfinite(request_timeout) or request_timeout <= 0
                or not math.isfinite(retry_backoff) or retry_backoff < 0
                or type(max_retries) is not int or max_retries < 0
                or type(max_image_size) is not int or max_image_size < 1
                or type(max_output_tokens) is not int
                or max_output_tokens < 1):
            raise ValueError('Invalid OpenAI client limits')
        if image_detail not in {'low', 'high', 'auto'}:
            raise ValueError('image_detail must be low, high or auto')
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key_env = api_key_env
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.image_keys = _validate_image_keys(image_keys)
        self.image_detail = image_detail
        self.max_image_size = max_image_size

    def _content(self, payload, observation):
        content = [
            dict(type='input_text', text=json.dumps(payload, allow_nan=False))
        ]
        for key, url in _camera_data_urls(self.image_keys, observation,
                                          self.max_image_size):
            content.extend([
                dict(type='input_text', text=f'Camera: {key}'),
                dict(
                    type='input_image',
                    image_url=url,
                    detail=self.image_detail),
            ])
        return content

    def _post_json(self, body):
        secret = os.environ.get(self.api_key_env)
        if not secret:
            raise ValueError(
                f'Missing OpenAI API key in environment: {self.api_key_env}')
        request = Request(
            f'{self.base_url}/responses',
            data=json.dumps(body, allow_nan=False).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {secret}',
                'Content-Type': 'application/json',
                'User-Agent': 'FluxVLA/HierarchicalPlanning',
            },
            method='POST')
        deadline = time.monotonic() + self.request_timeout
        for attempt in range(self.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('OpenAI request time budget exhausted')
            try:
                with urlopen(request, timeout=remaining) as response:
                    raw = response.read(1024 * 1024 + 1)
                if time.monotonic() > deadline:
                    raise TimeoutError('OpenAI request time budget exhausted')
                if len(raw) > 1024 * 1024:
                    raise ValueError('OpenAI response exceeds size limit')
                return strict_json_loads(raw)
            except HTTPError as exc:
                code = exc.code
                exc.close()
                # Never echo provider error bodies: they may contain the
                # request's images, credentials or untrusted instructions.
                if (code != 429 and not 500 <= code < 600
                        or attempt >= self.max_retries):
                    raise RuntimeError(
                        f'OpenAI Responses API returned HTTP {code}; check '
                        'credentials, model and strict JSON schema support'
                    ) from None
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    error = (
                        TimeoutError
                        if isinstance(exc, TimeoutError) else RuntimeError)
                    raise error(
                        'OpenAI Responses API request failed') from None
            delay = self.retry_backoff * (2**attempt)
            if time.monotonic() + delay >= deadline:
                raise TimeoutError('OpenAI request time budget exhausted')
            time.sleep(delay)
        raise AssertionError('unreachable')

    @staticmethod
    def _parse_response(reply):
        if (not isinstance(reply, dict) or reply.get('status') != 'completed'
                or reply.get('error') is not None
                or reply.get('incomplete_details') is not None):
            raise ValueError('Missing, incomplete or failed OpenAI response')
        output = reply.get('output')
        if not isinstance(output, list) or not output:
            raise ValueError('OpenAI response must contain output items')
        messages = []
        for item in output:
            if not isinstance(item, dict):
                raise ValueError('Invalid OpenAI output item')
            if item.get('type') == 'reasoning':
                continue
            if (item.get('type') != 'message'
                    or item.get('role') != 'assistant'
                    or item.get('status') != 'completed'):
                raise ValueError(
                    'Unexpected tool call or unfinished OpenAI message')
            messages.append(item)
        if len(messages) != 1:
            raise ValueError('Expected one OpenAI assistant message')
        content = messages[0].get('content')
        if (not isinstance(content, list) or len(content) != 1
                or not isinstance(content[0], dict)
                or content[0].get('type') != 'output_text'
                or not isinstance(content[0].get('text'), str)):
            raise ValueError('Refused or invalid OpenAI structured output')
        return strict_json_loads(content[0]['text'])

    def generate(self, system, payload, observation, schema):
        body = dict(
            model=self.model,
            instructions=system,
            input=[
                dict(role='user', content=self._content(payload, observation))
            ],
            text=dict(
                format=dict(
                    type='json_schema',
                    name='fluxvla_planning',
                    strict=True,
                    schema=schema)),
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        if self.reasoning_effort is not None:
            body['reasoning'] = {'effort': self.reasoning_effort}
        return self._parse_response(self._post_json(body))


@PLANNERS.register_module()
class OpenAISubtaskPlanner(_StructuredSubtaskPlanner):
    """Select a catalog subtask through the online GPT Responses API."""

    client_type = OpenAIResponsesClient


@TASK_MONITORS.register_module()
class OpenAIPredicateMonitor(_StructuredPredicateMonitor):
    """Check completion against current camera images using online GPT."""

    client_type = OpenAIResponsesClient
