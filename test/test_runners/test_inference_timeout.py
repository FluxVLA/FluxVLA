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
"""Timeout budgets and opt-in diagnostics; no GPU, API or model loading."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fluxvla.engines.runners.serving import distributed_server as serving


@pytest.mark.parametrize('server_timeout,client_timeout,expected', [
    (3600, 120, 120),
    (120, 3600, 120),
    (3600, 3600, 3600),
])
def test_timeout_reports_effective_budget_and_request(monkeypatch,
                                                      server_timeout,
                                                      client_timeout,
                                                      expected):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(serving.time, 'monotonic', lambda: clock.now)
    server = serving.FluxVLAInferenceServer(
        frontend_bind='inproc://front',
        backend_bind='inproc://back',
        request_timeout_s=server_timeout)
    errors = Mock()
    monkeypatch.setattr(server, '_send_frontend_error', errors)
    server._queue_prediction((b'client', ),
                             dict(
                                 request_id='slow-request',
                                 policy_seed=0,
                                 deadline_ms=client_timeout * 1000), b'')
    pending = server._pending_predictions.popleft()
    assert pending.timeout_s == expected
    assert pending.received_at == 100.0
    worker = serving._ModelWorker(
        identity=b'worker',
        worker_id='model-worker-0',
        node_id='test',
        device='cpu',
        prediction=pending)
    server._workers[worker.identity] = worker
    clock.now = pending.deadline + .1
    with pytest.raises(TimeoutError) as caught:
        server._expire_predictions()
    message = str(caught.value)
    assert 'request=slow-request' in message
    assert f'effective_budget={expected}s' in message
    assert f'server_limit={server_timeout}s' in message
    assert 'elapsed=' in message
    assert errors.call_args.args[2] is caught.value
    assert errors.call_args.kwargs['error_type'] == 'timeout'


@pytest.mark.parametrize('value', [-1, True, '60', float('inf'), float('nan')])
def test_bad_stack_dump_interval_fails_explicitly(value):
    cfg = dict(themis=dict(server=dict(debug_request_stacks_after_s=value)))
    with pytest.raises(ValueError, match='finite nonnegative'):
        serving._request_stack_dump_after(cfg)


def test_diagnostics_are_disabled_by_default(monkeypatch, capsys):
    assert serving._request_stack_dump_after({}) == 0
    assert serving._request_stack_dump_after(dict(themis=dict(server={}))) == 0
    dump, cancel = Mock(), Mock()
    monkeypatch.setattr(serving.faulthandler, 'dump_traceback_later', dump)
    monkeypatch.setattr(serving.faulthandler, 'cancel_dump_traceback_later',
                        cancel)
    with serving._prediction_diagnostics('worker', 'request', 0):
        pass
    dump.assert_not_called()
    cancel.assert_not_called()
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('fail', [False, True])
def test_stack_timer_is_cancelled_after_success_or_error(
        monkeypatch, capsys, fail):
    dump, cancel = Mock(), Mock()
    monkeypatch.setattr(serving.faulthandler, 'dump_traceback_later', dump)
    monkeypatch.setattr(serving.faulthandler, 'cancel_dump_traceback_later',
                        cancel)
    original = ValueError('original model error')

    def predict():
        with serving._prediction_diagnostics('worker', 'request', 60):
            if fail:
                raise original

    if fail:
        with pytest.raises(ValueError) as caught:
            predict()
        assert caught.value is original
    else:
        predict()
    dump.assert_called_once_with(60, repeat=False)
    cancel.assert_called_once_with()
    output = capsys.readouterr().out
    assert 'request begin worker=worker request=request' in output
    assert 'request leave worker=worker request=request' in output
    assert 'diagnostic, not a fatal timeout' in output
