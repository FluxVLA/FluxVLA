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

import io
import json
import urllib.error
from pathlib import Path

import pytest
from mmengine import Config

from fluxvla.engines import VLAS, build_vla_from_cfg
from fluxvla.models.vlas import openai_vla as openai
from fluxvla.models.vlas import tool_calling_vla as common
from fluxvla.models.vlas.minimax_vla import (MiniMaxLiberoVLA,
                                             MiniMaxRobocasaVLA)


@pytest.mark.parametrize('cls, base', [
    (openai.OpenAILiberoVLA, common.ToolCallingVLA),
    (MiniMaxLiberoVLA, common.ToolCallingVLA),
    (openai.OpenAIRobocasaVLA, common.ToolCallingRobocasaVLA),
    (MiniMaxRobocasaVLA, common.ToolCallingRobocasaVLA),
])
def test_providers_inherit_shared_actions_not_each_other(cls, base):
    assert issubclass(cls, base)
    assert cls.predict_action is base.predict_action
    assert VLAS.get(cls.__name__) is cls
    assert VLAS.get(base.__name__) is None  # No new concrete config types.
    assert not issubclass(MiniMaxLiberoVLA, openai.OpenAILiberoVLA)
    assert not issubclass(MiniMaxRobocasaVLA, openai.OpenAIRobocasaVLA)


@pytest.mark.parametrize('cls, horizon, budget, detail', [
    (openai.OpenAILiberoVLA, 10, 20, 'low'),
    (openai.OpenAIRobocasaVLA, 8, 90, 'high'),
])
def test_openai_defaults_and_positional_arguments(cls, horizon, budget,
                                                  detail):
    model = cls()
    assert model.model == 'gpt-6-astra'
    assert model.base_url == 'https://api.openai.com/v1'
    assert model.api_key_env == 'OPENAI_API_KEY'
    assert model.action_horizon == horizon
    assert model.max_llm_calls == budget
    assert model.image_detail == detail
    assert model.reasoning_effort == 'medium'
    assert model.max_output_tokens is None
    positional = cls('custom', 'https://example.test/v1/', 'CUSTOM_KEY', 'low',
                     1024, 3.0)
    assert positional.model == 'custom'
    assert positional.base_url == 'https://example.test/v1'
    assert positional.api_key_env == 'CUSTOM_KEY'
    assert positional.reasoning_effort == 'low'
    assert positional.max_output_tokens == 1024
    assert positional.request_timeout == 3.0


@pytest.mark.parametrize('cls',
                         [openai.OpenAILiberoVLA, openai.OpenAIRobocasaVLA])
def test_openai_transport_stays_in_provider_and_preserves_requests(
        monkeypatch, cls):
    model = cls(max_retries=1, request_timeout=3.0)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-key')
    requests = []
    reply = {'id': 'test-response', 'output': []}

    def urlopen(request, timeout):
        requests.append((request, timeout))
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, 'rate limit',
                                         {}, io.BytesIO(b'{}'))
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(openai.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(openai.time, 'sleep', lambda _: None)
    model._reset_episode('test goal')
    body = model._request_body()
    assert body['tool_choice'] == 'required'
    assert body['parallel_tool_calls'] is False
    assert body['tools'] == model.tools
    assert body['input'] == model._compact_history()
    assert body['reasoning'] == {'effort': 'medium'}
    assert 'max_output_tokens' not in body
    assert model._post_json(body) == reply
    assert len(requests) == 2
    request, timeout = requests[-1]
    assert timeout == 3.0
    assert request.full_url == 'https://api.openai.com/v1/responses'
    assert request.get_header('Authorization') == 'Bearer offline-test-key'
    assert request.get_header('User-agent') == 'FluxVLA/OpenAILiberoVLA'
    assert json.loads(request.data) == body
    monkeypatch.delenv('OPENAI_API_KEY')
    with pytest.raises(RuntimeError, match='Missing OpenAI API key'):
        model._post_json(body)
    assert len(requests) == 2


@pytest.mark.parametrize('name, cls', [
    ('gpt6_astra_libero_inference.py', openai.OpenAILiberoVLA),
    ('gpt6_astra_robocasa_inference.py', openai.OpenAIRobocasaVLA),
])
def test_existing_openai_configs_still_build(name, cls):
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/openai' / name, lazy_import=False)
    model = build_vla_from_cfg(cfg.inference_model)
    assert type(model) is cls
    assert model.action_horizon == cfg.eval.eval_chunk_size


def test_shared_prompts_are_reexported_by_openai():
    assert (openai.DEFAULT_LIBERO_SYSTEM_PROMPT is
            common.DEFAULT_LIBERO_SYSTEM_PROMPT)
    assert (openai.DEFAULT_ROBOCASA_SYSTEM_PROMPT is
            common.DEFAULT_ROBOCASA_SYSTEM_PROMPT)
