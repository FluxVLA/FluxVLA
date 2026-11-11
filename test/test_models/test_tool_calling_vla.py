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
import torch
from mmengine import Config

from fluxvla.engines import VLAS, build_dataset_from_cfg, build_vla_from_cfg
from fluxvla.models.vlas import minimax_vla as minimax
from fluxvla.models.vlas import openai_vla as openai
from fluxvla.models.vlas import tool_calling_vla as common
from fluxvla.models.vlas.minimax_vla import MiniMaxVLA


def test_only_provider_level_vlas_are_public_and_registered():
    assert VLAS.get('OpenAIVLA') is openai.OpenAIVLA
    assert VLAS.get('MiniMaxVLA') is MiniMaxVLA
    for old_name in ('OpenAILiberoVLA', 'OpenAIRobocasaVLA', 'OpenAIOliVLA',
                     'MiniMaxLiberoVLA', 'MiniMaxRobocasaVLA'):
        assert VLAS.get(old_name) is None
        assert not hasattr(openai, old_name)
        assert not hasattr(minimax, old_name)


@pytest.mark.parametrize('cls, embodiment, policy_base', [
    (openai.OpenAIVLA, 'libero', common.LiberoToolPolicy),
    (openai.OpenAIVLA, 'robocasa', common.RobocasaToolPolicy),
    (openai.OpenAIVLA, 'oli', common.OliToolPolicy),
    (MiniMaxVLA, 'libero', common.LiberoToolPolicy),
    (MiniMaxVLA, 'robocasa', common.RobocasaToolPolicy),
])
def test_provider_vla_selects_private_action_adapter(cls, embodiment,
                                                     policy_base):
    model = cls(embodiment=embodiment)
    assert model.embodiment == embodiment
    assert isinstance(model._policy, policy_base)
    assert VLAS.get(cls.__name__) is cls
    assert model.tools == model._policy.tools
    model.eval()
    assert not model.training
    assert not model._policy.training


def test_provider_vla_exposes_the_common_predict_action_contract(monkeypatch):
    model = openai.OpenAIVLA(embodiment='libero')
    expected = torch.zeros(1, 2, 7)
    received = {}

    def predict_action(**batch):
        received.update(batch)
        return expected

    monkeypatch.setattr(model._policy, 'predict_action', predict_action)
    assert model.predict_action(observation='test') is expected
    assert received == {'observation': 'test'}


@pytest.mark.parametrize('cls', [openai.OpenAIVLA, MiniMaxVLA])
def test_provider_vla_rejects_unknown_embodiment(cls):
    with pytest.raises(ValueError, match='Unsupported'):
        cls(embodiment='unknown')


@pytest.mark.parametrize('embodiment, horizon, budget, detail', [
    ('libero', 10, 20, 'low'),
    ('robocasa', 8, 90, 'high'),
])
def test_openai_defaults_and_provider_arguments(embodiment, horizon, budget,
                                                detail):
    model = openai.OpenAIVLA(embodiment=embodiment)
    assert model.model == 'gpt-6-astra'
    assert model.base_url == 'https://api.openai.com/v1'
    assert model.api_key_env == 'OPENAI_API_KEY'
    assert model.action_horizon == horizon
    assert model.max_llm_calls == budget
    assert model.image_detail == detail
    assert model.reasoning_effort == 'medium'
    assert model.max_output_tokens is None
    configured = openai.OpenAIVLA(
        embodiment=embodiment,
        model='custom',
        base_url='https://example.test/v1/',
        api_key_env='CUSTOM_KEY',
        reasoning_effort='low',
        max_output_tokens=1024,
        request_timeout=3.0)
    assert configured.model == 'custom'
    assert configured.base_url == 'https://example.test/v1'
    assert configured.api_key_env == 'CUSTOM_KEY'
    assert configured.reasoning_effort == 'low'
    assert configured.max_output_tokens == 1024
    assert configured.request_timeout == 3.0


@pytest.mark.parametrize('embodiment', ['libero', 'robocasa'])
def test_openai_transport_stays_in_provider_and_preserves_requests(
        monkeypatch, embodiment):
    model = openai.OpenAIVLA(
        embodiment=embodiment, max_retries=1, request_timeout=3.0)
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
    assert request.get_header('User-agent') == 'FluxVLA/OpenAIVLA'
    assert json.loads(request.data) == body
    monkeypatch.delenv('OPENAI_API_KEY')
    with pytest.raises(RuntimeError, match='Missing OpenAI API key'):
        model._post_json(body)
    assert len(requests) == 2


@pytest.mark.parametrize('name, embodiment, tool_name', [
    ('gpt6_astra_libero_inference.py', 'libero', 'move_to'),
    ('gpt6_astra_robocasa_inference.py', 'robocasa', 'control_gr1'),
])
def test_openai_configs_build_one_provider_vla(name, embodiment, tool_name):
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/openai' / name, lazy_import=False)
    model = build_vla_from_cfg(cfg.inference_model)
    assert type(model) is openai.OpenAIVLA
    assert model.embodiment == embodiment
    assert model.tools[0]['name'] == tool_name
    assert model.action_horizon == cfg.eval.eval_chunk_size


def test_openai_inference_config_uses_the_same_provider_vla_contract():
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(
        root / 'configs/openai/gpt6_astra_oli_keypoint_inference.py',
        lazy_import=False)
    model_cfg = cfg.inference_model.copy()
    model_cfg.trace_dir = None
    model_cfg.trace_console = False
    model = build_vla_from_cfg(model_cfg)
    dataset = build_dataset_from_cfg(cfg.inference.dataset)

    assert type(model) is openai.OpenAIVLA
    assert model.embodiment == 'oli'
    assert isinstance(model._policy, common.OliToolPolicy)
    assert model.action_horizon == cfg.inference.action_chunk
    assert type(dataset).__name__ == 'ToolCallingOliDataset'


def test_shared_prompts_are_reexported_by_openai():
    assert (openai.DEFAULT_LIBERO_SYSTEM_PROMPT is
            common.DEFAULT_LIBERO_SYSTEM_PROMPT)
    assert (openai.DEFAULT_ROBOCASA_SYSTEM_PROMPT is
            common.DEFAULT_ROBOCASA_SYSTEM_PROMPT)
