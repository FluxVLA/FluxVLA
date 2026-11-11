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
"""Incremental, human-readable HTML traces for OpenAI robot control."""

from __future__ import annotations
import base64
import copy
import html
import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


class OpenAITraceWriter:
    """Journal requests before transport and retain failed model turns."""

    def __init__(self,
                 root_dir: str,
                 model: str,
                 console: bool = True) -> None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.run_dir = Path(
            root_dir).expanduser() / f'{timestamp}-{os.getpid()}'
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / 'trace.html'
        self.model = model
        self.console = bool(console)
        self.calls = 0
        self.records = []
        self.system_prompt = ''
        self.tools = []
        if self.console:
            print(f'[gpt-trace] live trace: {self.path}', flush=True)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def _save_request(self, request: Dict[str, Any], call_dir: Path):
        """Externalize embedded images and return a readable request copy."""
        request = copy.deepcopy(request)
        images = []
        image_messages = [
            message for message in request.get('input', [])
            if isinstance(message.get('content'), list) and any(
                item.get('type') == 'input_image'
                for item in message['content'])
        ]
        for message_index, message in enumerate(image_messages):
            age = len(image_messages) - message_index - 1
            frame_name = 'current' if age == 0 else f'history-{age}'
            camera_name = None
            for item in message['content']:
                if item.get('type') == 'input_text':
                    match = re.fullmatch(r"camera '(.+)':",
                                         item.get('text', ''))
                    if match:
                        camera_name = match.group(1)
                if item.get('type') != 'input_image':
                    continue
                image_url = item.get('image_url', '')
                match = re.match(r'data:image/([^;]+);base64,(.+)', image_url,
                                 re.DOTALL)
                if not match:
                    continue
                suffix = 'jpg' if match.group(1) == 'jpeg' else match.group(1)
                name = camera_name or 'camera'
                safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
                image_path = call_dir / (
                    f'image_{len(images):02d}_{safe_name}.{suffix}')
                image_path.write_bytes(base64.b64decode(match.group(2)))
                item['image_url'] = image_path.name
                images.append({
                    'label': f'{frame_name}/{name}',
                    'path': f'{call_dir.name}/{image_path.name}',
                    'data_url': image_url if age == 0 else None,
                    'current': age == 0,
                })
        return request, images

    @staticmethod
    def _data_url(media_type: str, content: str) -> str:
        encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
        return f'data:{media_type};base64,{encoded}'

    @staticmethod
    def _observation_text(request: Dict[str, Any]) -> str:
        for message in reversed(request.get('input', [])):
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for item in content:
                text = item.get('text', '')
                if text.startswith('Current OLI observation.'):
                    return text
        return ''

    @staticmethod
    def _task_prompt(observation: str) -> str:
        marker = 'Instruction: '
        start = observation.find(marker)
        if start < 0:
            return ''
        prompt = observation[start + len(marker):]
        end = prompt.find('\nAction call:')
        return (prompt if end < 0 else prompt[:end]).strip()

    def _render_call(self, record: Dict[str, Any], active: bool) -> str:
        index = record['index']
        images = ''.join(
            '<figure><img src="{}" alt="{}"><figcaption>{}</figcaption>'
            '</figure>'.format(
                self._escape(image['data_url']), self._escape(image['label']),
                self._escape(image['label'].split('/', 1)[-1]))
            for image in record['images'] if image['current'])
        history_count = sum(not image['current'] for image in record['images'])
        context = ('current images' if history_count == 0 else
                   f'{history_count} history images + current images')
        targets = self._escape(
            self._json(record['arguments'].get('targets', {})))
        note = self._escape(record['arguments'].get('note', ''))
        task_prompt = self._escape(record['task_prompt'])
        observation = self._escape(record['observation'])
        hidden = '' if active else ' hidden'
        latency = record['latency']
        input_tokens = record['input_tokens']
        output_tokens = record['output_tokens']
        shape = record['action_shape']
        chunk_label = (f'{shape[0]} × {shape[1]}'
                       if shape else 'not generated')
        artifacts = record['artifacts']
        links = ' · '.join(
            f'<a href="{self._escape(url)}">{self._escape(name)}</a>'
            for name, url in artifacts.items())
        status = self._escape(record['status'])
        stage = self._escape(record['stage'])
        error = record.get('error')
        error_panel = ('<section><h3>Failure</h3><pre><code>' +
                       self._escape(self._json(error)) +
                       '</code></pre></section>' if error is not None else '')
        return f'''<section id="call-panel-{index}" role="tabpanel"
  aria-labelledby="call-tab-{index}"{hidden}>
  <p>Status: <strong>{status}</strong> · Stage: {stage}.
    This model trace does not record robot execution.</p>
  {error_panel}
  <div class="meta">
    <span>Latency <strong>{latency:.2f} s</strong></span>
    <span>Tokens <strong>{input_tokens} + {output_tokens}</strong></span>
    <span>Chunk <strong>{chunk_label}</strong></span>
  </div>
  <div class="context">
    <span class="tag">{context}</span><span>→ GPT call {index}</span>
  </div>
  <section class="task-prompt">
    <h3>Task prompt</h3><p>{task_prompt}</p>
  </section>
  <div class="images">{images}</div>
  <div class="columns">
    <section>
      <h3>GPT tool call</h3><pre><code>{targets}</code></pre>
    </section>
    <section><h3>Reason given</h3><p class="note">{note}</p>
    </section>
  </div>
  <details><summary>Observation</summary>
    <pre><code>{observation}</code></pre>
  </details>
  <p class="artifacts">Full artifacts: {links}</p>
</section>'''

    def _render(self) -> str:
        call_count = len(self.records)
        last_index = self.records[-1]['index'] if self.records else 0
        total_latency = sum(record['latency'] for record in self.records)
        input_tokens = sum(record['input_tokens'] for record in self.records)
        output_tokens = sum(record['output_tokens'] for record in self.records)
        tabs = ''.join(
            '<button id="call-tab-{0}" role="tab" '
            'aria-controls="call-panel-{0}" '
            'aria-selected="{1}" class="{2}" type="button">Call {0}</button>'.
            format(record['index'],
                   str(record['index'] == last_index).lower(),
                   'active' if record['index'] == last_index else '')
            for record in self.records)
        panels = ''.join(
            self._render_call(record, record['index'] == last_index)
            for record in self.records)
        system_prompt = self._escape(self.system_prompt)
        tools = self._escape(self._json(self.tools))
        updated = self._escape(self._utc_now())
        page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OLI GPT Trace</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f7f7f5; --fg: #20201e; --muted: #6b6b66;
  --surface: #fff; --line: #deded8; --accent: #315efb;
  --code: #f0f0ec;
}
@media(prefers-color-scheme: dark) {
  :root {
    --bg: #171716; --fg: #eeeeea; --muted: #aaa9a2;
    --surface: #222220; --line: #3b3b37; --accent: #8aa4ff;
    --code: #292927;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 system-ui, sans-serif;
}
main { max-width: 1180px; margin: auto; padding: 28px 22px 60px; }
h1 { font-size: 25px; margin: 0; }
h2 { font-size: 18px; }
h3 { font-size: 14px; margin: 0 0 8px; }
.top {
  display: flex; justify-content: space-between; gap: 16px;
  align-items: flex-start;
}
.sub, .meta, .artifacts { color: var(--muted); }
.stats {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 12px; margin: 20px 0;
}
.stat {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 13px;
}
.stat strong { display: block; font-size: 20px; font-weight: 500; }
.tabs {
  display: flex; gap: 6px; flex-wrap: wrap;
  border-bottom: 1px solid var(--line); padding-bottom: 10px;
}
.tabs button {
  font: inherit; color: var(--muted); background: transparent;
  border: 0; border-radius: 7px; padding: 8px 12px; cursor: pointer;
}
.tabs button.active { background: var(--fg); color: var(--bg); }
.meta, .context {
  display: flex; gap: 18px; flex-wrap: wrap; margin: 14px 0;
}
.tag {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 99px; padding: 2px 9px;
}
.task-prompt {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 9px; padding: 12px 14px; margin: 14px 0;
}
.task-prompt p { margin: 0; }
.images {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 12px; margin: 14px 0 22px;
}
figure { margin: 0; }
img {
  display: block; width: 100%; aspect-ratio: 4/3;
  object-fit: cover; border-radius: 9px; background: var(--surface);
}
figcaption { text-align: center; color: var(--muted); margin-top: 4px; }
.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.note { border-left: 3px solid var(--accent); padding-left: 12px; margin: 0; }
pre {
  white-space: pre-wrap; word-break: break-word; background: var(--code);
  padding: 13px; border-radius: 8px; overflow: auto; margin: 0;
}
details { margin-top: 18px; }
summary { cursor: pointer; color: var(--muted); }
details pre { margin-top: 10px; }
.artifacts a { color: var(--accent); }
.refresh {
  display: flex; align-items: center; gap: 7px; color: var(--muted);
}
@media(max-width: 700px) {
  main { padding: 18px 14px; }
  .top { display: block; }
  .stats, .images, .columns { grid-template-columns: 1fr; }
  .refresh { margin-top: 10px; }
}
</style></head><body><main>
<div class="top"><div><h1>OLI GPT Trace</h1>
<div class="sub">Updated: __UPDATED__</div></div>
<label class="refresh"><input id="auto-refresh" type="checkbox" checked>
Auto-refresh</label></div>
<div class="stats">
<div class="stat"><span>Calls</span><strong>__CALL_COUNT__</strong></div>
<div class="stat"><span>API latency</span>
<strong>__TOTAL_LATENCY__</strong></div>
<div class="stat"><span>Tokens</span><strong>__TOKENS__</strong>
<span class="sub">input + output</span></div></div>
<nav class="tabs" role="tablist" aria-label="GPT calls">__TABS__</nav>
__PANELS__
<details><summary>System prompt and tool schema</summary>
<h2>System prompt</h2><pre><code>__SYSTEM_PROMPT__</code></pre>
<h2>Tool schema</h2><pre><code>__TOOLS__</code></pre></details>
</main><script>
const buttons = [...document.querySelectorAll('[role="tab"]')];
buttons.forEach(button => button.addEventListener('click', () => {
  buttons.forEach(item => {
    const selected = item === button;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-selected', selected);
    document.getElementById(item.getAttribute('aria-controls')).hidden =
      !selected;
  });
}));
const refresh = document.getElementById('auto-refresh');
setInterval(() => { if (refresh.checked) location.reload(); }, 3000);
</script></body></html>'''
        replacements = {
            '__UPDATED__': updated,
            '__CALL_COUNT__': str(call_count),
            '__TOTAL_LATENCY__': f'{total_latency:.1f} s',
            '__TOKENS__': f'{input_tokens:,} + {output_tokens:,}',
            '__TABS__': tabs,
            '__PANELS__': panels,
            '__SYSTEM_PROMPT__': system_prompt,
            '__TOOLS__': tools,
        }
        for marker, value in replacements.items():
            page = page.replace(marker, value)
        return page

    def _write_html(self) -> None:
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(self._render())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _write_json(self, path: Path, value: Any) -> str:
        text = self._json(value)
        self._write_text(path, text)
        return text

    def _record(self, call_dir: Path) -> Dict[str, Any]:
        for record in self.records:
            if record['directory'] == call_dir:
                return record
        raise ValueError('Unknown trace call directory')

    def _write_status(self, record: Dict[str, Any]) -> None:
        self._write_json(
            record['directory'] / 'status.json', {
                'index': record['index'],
                'started_at': record['time'],
                'updated_at': self._utc_now(),
                'status': record['status'],
                'stage': record['stage'],
                'execution_status': 'not_recorded',
            })

    def begin_call(self, request: Dict[str, Any]) -> Path:
        """Save the prepared request before any API operation is attempted."""
        self.calls += 1
        call_dir = self.run_dir / f'call_{self.calls:04d}'
        call_dir.mkdir()
        # Keep a complete request even if externalizing an image fails.
        self._write_json(call_dir / 'request.json', request)
        saved_request, images = self._save_request(request, call_dir)
        request_text = self._write_json(call_dir / 'request.json',
                                        saved_request)
        if not self.records:
            self.system_prompt = next((item.get('content', '')
                                       for item in request.get('input', [])
                                       if item.get('role') == 'system'), '')
            self.tools = request.get('tools', [])
        observation = self._observation_text(request)
        record = {
            'index': self.calls,
            'directory': call_dir,
            'time': self._utc_now(),
            'status': 'pending',
            'stage': 'request',
            'latency': 0.0,
            'input_tokens': 0,
            'output_tokens': 0,
            'action_shape': None,
            'observation': observation,
            'task_prompt': self._task_prompt(observation),
            'arguments': {},
            'images': images,
            'artifacts': {
                'request': self._data_url('application/json', request_text),
            },
        }
        self.records.append(record)
        try:
            self._write_status(record)
            self._write_html()
        except BaseException as exc:
            record['stage'] = 'trace_prepare'
            try:
                self.fail_call(call_dir, exc)
            except Exception as trace_error:
                print(f'[OLI_TRACE_WRITE_FAILED] {trace_error}', flush=True)
            raise
        return call_dir

    def mark_stage(self, call_dir: Path, stage: str) -> None:
        record = self._record(call_dir)
        record['stage'] = stage
        self._write_status(record)

    @staticmethod
    def _metric(value: Any) -> float:
        try:
            number = float(value)
        except (ValueError, TypeError):
            return 0.0
        return number if np.isfinite(number) else 0.0

    def save_response(self, call_dir: Path, response: Any,
                      metadata: Dict[str, Any]) -> None:
        """Persist the response before parsing or validating its tool call."""
        record = self._record(call_dir)
        record['stage'] = 'response_save'
        text = self._write_json(call_dir / 'response.json', response)
        record['artifacts']['response'] = self._data_url(
            'application/json', text)
        self._write_json(call_dir / 'metadata.json', metadata)
        record['latency'] = self._metric(metadata.get('latency_seconds'))
        record['input_tokens'] = int(
            self._metric(metadata.get('input_tokens')))
        record['output_tokens'] = int(
            self._metric(metadata.get('output_tokens')))
        record['status'] = 'response_received'
        self.mark_stage(call_dir, 'response_parse')
        self._write_html()

    def save_arguments(self, call_dir: Path, arguments: Dict[str,
                                                             Any]) -> None:
        record = self._record(call_dir)
        record['arguments'] = copy.deepcopy(arguments)
        self._write_json(call_dir / 'arguments.json', arguments)

    def finish_call(self, call_dir: Path, actions: np.ndarray) -> Path:
        """Record generated actions, without claiming they were executed."""
        record = self._record(call_dir)
        text = self._write_json(call_dir / 'action_chunk.json',
                                actions.tolist())
        record['artifacts']['action_chunk'] = self._data_url(
            'application/json', text)
        record['action_shape'] = list(actions.shape)
        record['status'] = 'generated'
        record['stage'] = 'actions_generated'
        self._write_status(record)
        self._write_html()
        if self.console:
            targets = json.dumps(
                record['arguments'].get('targets', {}),
                ensure_ascii=False,
                separators=(',', ':'))
            note = str(record['arguments'].get('note', ''))
            print(
                f'[gpt-trace] call={record["index"]} status=generated '
                f'latency={record["latency"]:.2f}s '
                f'tokens={record["input_tokens"]}+'
                f'{record["output_tokens"]} targets={targets} '
                f'note={note!r}',
                flush=True)
        return call_dir

    def fail_call(self, call_dir: Path, error: BaseException) -> None:
        """Retain the failure and its chained transport cause."""
        record = self._record(call_dir)
        causes = []
        cause = error.__cause__ or error.__context__
        seen = {id(error)}
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            item = {'type': type(cause).__name__, 'message': str(cause)}
            if isinstance(getattr(cause, 'code', None), int):
                item['http_status'] = cause.code
            causes.append(item)
            cause = cause.__cause__ or cause.__context__
        details = {
            'type':
            type(error).__name__,
            'message':
            str(error),
            'stage':
            record['stage'],
            'time':
            self._utc_now(),
            'causes':
            causes,
            'traceback':
            ''.join(
                traceback.format_exception(
                    type(error), error, error.__traceback__)),
        }
        if isinstance(getattr(error, 'code', None), int):
            details['http_status'] = error.code
        text = self._write_json(call_dir / 'error.json', details)
        if (record['stage'] == 'request'
                and isinstance(error, json.JSONDecodeError)):
            self._write_text(call_dir / 'response.txt', error.doc)
        record['error'] = details
        record['artifacts']['error'] = self._data_url('application/json', text)
        record['status'] = 'failed'
        self._write_status(record)
        self._write_html()
        print(
            f'[gpt-trace] call={record["index"]} status=failed '
            f'stage={record["stage"]} error={call_dir / "error.json"}',
            flush=True)

    def append(self, request: Dict[str, Any], response: Dict[str, Any],
               arguments: Dict[str, Any], metadata: Dict[str, Any],
               actions: np.ndarray) -> Path:
        """Compatibility entry for callers saving an already completed turn."""
        call_dir = self.begin_call(request)
        try:
            self.save_response(call_dir, response, metadata)
            self.save_arguments(call_dir, arguments)
            self.mark_stage(call_dir, 'trace_finalize')
            return self.finish_call(call_dir, actions)
        except BaseException as exc:
            try:
                self.fail_call(call_dir, exc)
            except Exception as trace_error:
                print(f'[OLI_TRACE_WRITE_FAILED] {trace_error}', flush=True)
            raise
