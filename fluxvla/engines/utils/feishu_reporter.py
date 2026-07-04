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
"""Report evaluation summaries to Feishu/Lark Sheets."""

from __future__ import annotations
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

LIBERO_RESULT_COLUMNS = [
    'libero_10',
    'libero_goal',
    'libero_object',
    'libero_spatial',
    'all',
]
ROBOCASA_RESULT_COLUMNS = [
    'Cabinet',
    'Drawer',
    'Microwave',
    'Generalization',
    'all',
]

REPORT_SPECS = {
    'libero': {
        'sheet_title':
        'libero',
        'headers':
        ['id', 'commit id', 'config', 'ckpt_path'] + LIBERO_RESULT_COLUMNS,
    },
    'robocasa': {
        'sheet_title':
        'robocasa',
        'headers':
        ['id', 'commit id', 'config', 'ckpt_path'] + ROBOCASA_RESULT_COLUMNS,
    },
}

_ALLOWED_HOST_RE = re.compile(
    r'(^|\.)((feishu|larksuite|larkoffice|larkenterprise)\.com|feishu\.cn)$')
_TOKEN_RE = re.compile(r'^[A-Za-z0-9_-]+$')
_PERMISSION_DENIED_CODES = {91403}


class FeishuReportError(RuntimeError):
    """Raised when a Feishu report cannot be written safely."""


def _normalize_feishu_sheet_url(sheet_url: str) -> str:
    """Normalize common shell-escaped Feishu URL separators."""
    return str(sheet_url).strip().replace('\\?', '?').replace('\\=',
                                                              '=').replace(
                                                                  '\\&', '&')


@dataclass
class FeishuReportResult:
    wrote: bool
    reason: str
    report_kind: str
    sheet_title: str = ''
    sheet_id: str = ''


def parse_feishu_spreadsheet_token(sheet_url: str) -> Optional[str]:
    """Return the spreadsheet token from a Feishu Sheets URL, if valid."""
    if not sheet_url:
        return None
    try:
        parsed = urllib.parse.urlparse(_normalize_feishu_sheet_url(sheet_url))
    except ValueError:
        return None
    if parsed.scheme not in ('http', 'https'):
        return None
    hostname = (parsed.hostname or '').lower()
    if _ALLOWED_HOST_RE.search(hostname) is None:
        return None
    parts = [
        urllib.parse.unquote(part) for part in parsed.path.split('/') if part
    ]
    if 'sheets' not in parts:
        return None
    idx = parts.index('sheets')
    if idx + 1 >= len(parts):
        return None
    token = parts[idx + 1]
    if _TOKEN_RE.match(token) is None:
        return None
    return token


def parse_feishu_sheet_id(sheet_url: str) -> Optional[str]:
    """Return the worksheet id from a Feishu Sheets URL query, if present."""
    if parse_feishu_spreadsheet_token(sheet_url) is None:
        return None
    try:
        parsed = urllib.parse.urlparse(_normalize_feishu_sheet_url(sheet_url))
    except ValueError:
        return None
    query = urllib.parse.parse_qs(parsed.query)
    for key in ('sheet', 'gid'):
        for value in query.get(key, []):
            value = urllib.parse.unquote(str(value).strip())
            if value and _TOKEN_RE.match(value) is not None:
                return value
    return None


def _feishu_target_sheet_url(sheet_url: str, sheet_id: str) -> str:
    """Return a browser URL pointing at the actual target worksheet."""
    normalized_url = _normalize_feishu_sheet_url(sheet_url)
    if not sheet_id:
        return normalized_url
    parsed = urllib.parse.urlparse(normalized_url)
    query_items = [(key, value)
                   for key, value in urllib.parse.parse_qsl(parsed.query)
                   if key not in ('sheet', 'gid')]
    query_items.append(('sheet', sheet_id))
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query_items)))


def open_api_base_from_url(sheet_url: str) -> str:
    """Choose the matching OpenAPI host for Feishu or Lark URLs."""
    hostname = (urllib.parse.urlparse(
        _normalize_feishu_sheet_url(sheet_url)).hostname or '').lower()
    if 'larksuite.com' in hostname:
        return 'https://open.larksuite.com/open-apis'
    return 'https://open.feishu.cn/open-apis'


def get_git_commit_id(repo_dir: Optional[str] = None) -> str:
    """Return the current git commit id, falling back to an empty string."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_dir or os.getcwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ''


def _column_name(index: int) -> str:
    """Convert a 1-based column index into an A1 notation column name."""
    if index <= 0:
        raise ValueError('column index must be positive')
    name = ''
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(ord('A') + rem) + name
    return name


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate_from_stats(stats: Optional[Dict[str, Any]]) -> Optional[float]:
    if not stats:
        return None
    successes = (
        stats.get('total_successes')
        if 'total_successes' in stats else stats.get('successes'))
    trials = (
        stats.get('total_trials')
        if 'total_trials' in stats else stats.get('total_episodes'))
    successes = _safe_float(successes)
    trials = _safe_float(trials)
    if successes is not None and trials not in (None, 0):
        return successes / max(trials, 1.0) * 100.0
    return _safe_float(stats.get('success_rate'))


def _format_rate(rate: Optional[float]) -> str:
    if rate is None:
        return ''
    return f'{rate:.2f}%'


def _weighted_rate(
        stats_iter: Iterable[Optional[Dict[str, Any]]]) -> Optional[float]:
    successes = 0.0
    trials = 0.0
    fallback_rates = []
    for stats in stats_iter:
        if not stats:
            continue
        succ = _safe_float(
            stats.get('total_successes') if 'total_successes' in
            stats else stats.get('successes'))
        total = _safe_float(
            stats.get('total_trials') if 'total_trials' in
            stats else stats.get('total_episodes'))
        if succ is not None and total not in (None, 0):
            successes += succ
            trials += total
        else:
            rate = _rate_from_stats(stats)
            if rate is not None:
                fallback_rates.append(rate)
    if trials > 0:
        return successes / trials * 100.0
    if fallback_rates:
        return sum(fallback_rates) / len(fallback_rates)
    return None


def _lower_keyed_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key).lower(): value for key, value in stats.items()}


def build_report_row(summary: Dict[str, Any],
                     report_kind: str,
                     config: Optional[str] = None,
                     commit_id: Optional[str] = None,
                     repo_dir: Optional[str] = None) -> List[Any]:
    """Build a row without the leading auto-increment id."""
    report_kind = report_kind.lower()
    if report_kind not in REPORT_SPECS:
        raise ValueError(f'Unsupported report kind: {report_kind}')
    commit = (
        commit_id if commit_id is not None else get_git_commit_id(repo_dir))
    cfg = config if config is not None else str(summary.get('config', ''))
    ckpt_path = str(summary.get('ckpt') or summary.get('ckpt_path') or '')

    if report_kind == 'libero':
        suite_stats = summary.get('suite_stats', {})
        values = [
            _format_rate(_rate_from_stats(suite_stats.get(suite)))
            for suite in LIBERO_RESULT_COLUMNS[:-1]
        ]
        values.append(_format_rate(_weighted_rate(suite_stats.values())))
    else:
        group_stats = summary.get('group_stats') or summary.get(
            'suite_stats', {})
        lower_stats = _lower_keyed_stats(group_stats)
        values = [
            _format_rate(_rate_from_stats(lower_stats.get(group.lower())))
            for group in ROBOCASA_RESULT_COLUMNS[:-1]
        ]
        overall = _weighted_rate(lower_stats.values())
        if overall is None:
            overall = _safe_float(
                summary.get('overall', {}).get('success_rate'))
        values.append(_format_rate(overall))

    return [commit, cfg, ckpt_path] + values


def _normalize_row(row: Sequence[Any], width: int) -> List[str]:
    values = [_cell_text(cell) for cell in row]
    if len(values) < width:
        values.extend([''] * (width - len(values)))
    return values[:width]


def _cell_text(cell: Any) -> str:
    return '' if cell is None else str(cell).strip()


def _table_is_empty(values: Sequence[Sequence[Any]]) -> bool:
    for row in values:
        for cell in row:
            if _cell_text(cell):
                return False
    return True


def _next_row_id(values: Sequence[Sequence[Any]]) -> int:
    max_id = 0
    data_rows = 0
    for row in values[1:]:
        if not any(_cell_text(cell) for cell in row):
            continue
        data_rows += 1
        try:
            max_id = max(max_id, int(float(str(row[0]).strip())))
        except (IndexError, TypeError, ValueError):
            continue
    return max_id + 1 if max_id > 0 else data_rows + 1


def _next_write_row(values: Sequence[Sequence[Any]]) -> int:
    """Return the 1-based row after the last non-empty row."""
    last_non_empty = 0
    for idx, row in enumerate(values, start=1):
        if any(_cell_text(cell) for cell in row):
            last_non_empty = idx
    return last_non_empty + 1


def _feishu_error_hint(code: Optional[int], status_code: int) -> str:
    if status_code != 403 and code not in _PERMISSION_DENIED_CODES:
        return ''
    return (
        ' Permission denied. The app credentials are valid, but the app '
        'identity cannot read or edit this spreadsheet. Grant the Feishu '
        'custom app the required Sheets/Drive document permissions, publish '
        'or re-approve the app permission change, and share this spreadsheet '
        'with the app/bot using edit permission.')


def _feishu_error_message(status_code: int, detail: str) -> str:
    code = None
    msg = ''
    try:
        payload = json.loads(detail) if detail else {}
        code = payload.get('code')
        msg = payload.get('msg', '')
    except json.JSONDecodeError:
        payload = {}
    try:
        numeric_code = int(code) if code is not None else None
    except (TypeError, ValueError):
        numeric_code = None
    hint = _feishu_error_hint(numeric_code, status_code)
    if payload:
        return (f'Feishu HTTP {status_code}: code={code}, msg={msg}.{hint} '
                f'Raw response: {detail}')
    return f'Feishu HTTP {status_code}: {detail}{hint}'


def _feishu_api_error_message(payload: Dict[str, Any]) -> str:
    code = payload.get('code', 0)
    msg = payload.get('msg', '')
    try:
        numeric_code = int(code)
    except (TypeError, ValueError):
        numeric_code = None
    hint = _feishu_error_hint(numeric_code, status_code=200)
    return f'Feishu API error {code}: {msg}.{hint}'


class FeishuSheetClient:
    """Small urllib-based Feishu Sheets client for experiment reporting."""

    def __init__(self,
                 app_id: str,
                 app_secret: str,
                 api_base: str = 'https://open.feishu.cn/open-apis',
                 timeout: float = 10.0) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.api_base = api_base.rstrip('/')
        self.timeout = float(timeout)
        self._tenant_access_token = None

    def _request(self,
                 method: str,
                 path: str,
                 body: Optional[Dict[str, Any]] = None,
                 query: Optional[Dict[str, str]] = None,
                 auth: bool = True) -> Dict[str, Any]:
        url = f'{self.api_base}{path}'
        if query:
            url = f'{url}?{urllib.parse.urlencode(query)}'
        data = None
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        if body is not None:
            data = json.dumps(body).encode('utf-8')
        if auth:
            headers['Authorization'] = f'Bearer {self.tenant_access_token()}'
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise FeishuReportError(_feishu_error_message(exc.code,
                                                          detail)) from exc
        except urllib.error.URLError as exc:
            raise FeishuReportError(f'Feishu request failed: {exc}') from exc

        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise FeishuReportError(
                f'Feishu returned non-JSON response: {raw[:200]}') from exc
        code = payload.get('code', 0)
        if code not in (0, None):
            raise FeishuReportError(_feishu_api_error_message(payload))
        return payload

    def tenant_access_token(self) -> str:
        if self._tenant_access_token:
            return self._tenant_access_token
        payload = self._request(
            'POST',
            '/auth/v3/tenant_access_token/internal',
            body={
                'app_id': self.app_id,
                'app_secret': self.app_secret
            },
            auth=False)
        token = payload.get('tenant_access_token')
        if token is None:
            token = payload.get('data', {}).get('tenant_access_token')
        if not token:
            raise FeishuReportError(
                'Feishu tenant_access_token missing from response')
        self._tenant_access_token = token
        return token

    def list_sheets(self, spreadsheet_token: str) -> List[Dict[str, Any]]:
        payload = self._request(
            'GET', f'/sheets/v3/spreadsheets/{spreadsheet_token}/sheets/query')
        sheets = payload.get('data', {}).get('sheets', [])
        normalized = []
        for sheet in sheets:
            sheet_id = sheet.get('sheet_id') or sheet.get('sheetId')
            title = sheet.get('title')
            if sheet_id and title:
                normalized.append({
                    'sheet_id': sheet_id,
                    'title': title,
                    'index': sheet.get('index'),
                })
        return normalized

    def add_sheet(self, spreadsheet_token: str, title: str) -> str:
        payload = self._request(
            'POST',
            f'/sheets/v2/spreadsheets/{spreadsheet_token}/sheets_batch_update',
            body={
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': title
                        }
                    }
                }]
            })
        replies = payload.get('data', {}).get('replies', [])
        if replies:
            props = replies[0].get('addSheet', {}).get('properties', {})
            sheet_id = props.get('sheetId') or props.get('sheet_id')
            if sheet_id:
                return sheet_id
        for sheet in self.list_sheets(spreadsheet_token):
            if sheet['title'] == title:
                return sheet['sheet_id']
        raise FeishuReportError(f'Cannot create Feishu sheet: {title}')

    def read_values(self, spreadsheet_token: str, sheet_id: str,
                    cell_range: str) -> List[List[Any]]:
        range_ref = f'{sheet_id}!{cell_range}'
        quoted_range = urllib.parse.quote(range_ref, safe='')
        payload = self._request(
            'GET', f'/sheets/v2/spreadsheets/{spreadsheet_token}/values/'
            f'{quoted_range}',
            query={'valueRenderOption': 'ToString'})
        value_range = payload.get('data', {}).get('valueRange', {})
        return value_range.get('values') or []

    def write_values(self, spreadsheet_token: str, sheet_id: str,
                     start_row: int, values: List[List[Any]]) -> None:
        if not values:
            return
        width = max(len(row) for row in values)
        end_column = _column_name(width)
        end_row = start_row + len(values) - 1
        range_ref = f'{sheet_id}!A{start_row}:{end_column}{end_row}'
        self._request(
            'PUT',
            f'/sheets/v2/spreadsheets/{spreadsheet_token}/values',
            body={'valueRange': {
                'range': range_ref,
                'values': values
            }})


class FeishuExperimentReporter:
    """Write one experiment result row into the expected Feishu worksheet."""

    def __init__(self,
                 client: FeishuSheetClient,
                 spreadsheet_token: str,
                 preferred_sheet_id: Optional[str] = None) -> None:
        self.client = client
        self.spreadsheet_token = spreadsheet_token
        self.preferred_sheet_id = (
            str(preferred_sheet_id).strip() if preferred_sheet_id else '')

    def _ensure_sheet(self, title: str) -> Optional[Dict[str, str]]:
        sheets = self.client.list_sheets(self.spreadsheet_token)
        if self.preferred_sheet_id:
            for sheet in sheets:
                sheet_id = str(sheet['sheet_id'])
                if sheet_id == self.preferred_sheet_id:
                    return {'sheet_id': sheet_id, 'title': str(sheet['title'])}
            return None

        for sheet in sheets:
            if str(sheet['title']).lower() == title.lower():
                return {
                    'sheet_id': str(sheet['sheet_id']),
                    'title': str(sheet['title'])
                }

        sheet_id = self.client.add_sheet(self.spreadsheet_token, title)
        return {'sheet_id': str(sheet_id), 'title': title}

    def write_row(self, report_kind: str,
                  row_without_id: List[Any]) -> FeishuReportResult:
        report_kind = report_kind.lower()
        if report_kind not in REPORT_SPECS:
            return FeishuReportResult(
                False, f'unsupported report kind: {report_kind}', report_kind)

        spec = REPORT_SPECS[report_kind]
        headers = list(spec['headers'])
        sheet_title = str(spec['sheet_title'])
        sheet = self._ensure_sheet(sheet_title)
        if sheet is None:
            return FeishuReportResult(
                False,
                f'worksheet id from URL not found: {self.preferred_sheet_id}',
                report_kind,
                sheet_title,
                self.preferred_sheet_id,
            )
        sheet_id = sheet['sheet_id']
        actual_sheet_title = sheet['title']
        width = len(headers)
        end_column = _column_name(width)
        values = self.client.read_values(self.spreadsheet_token, sheet_id,
                                         f'A1:{end_column}2000')

        if _table_is_empty(values):
            first_id = 1
            row = [first_id] + row_without_id
            self.client.write_values(self.spreadsheet_token, sheet_id, 1,
                                     [headers, row])
            return FeishuReportResult(True,
                                      'wrote header and first result row',
                                      report_kind, actual_sheet_title,
                                      sheet_id)

        current_header = _normalize_row(values[0], width) if values else []
        if current_header != headers:
            return FeishuReportResult(
                False,
                'header mismatch on sheet '
                f'{actual_sheet_title} ({sheet_id}); expected {headers}; '
                f'got {current_header}; Feishu sheet was left unchanged',
                report_kind,
                actual_sheet_title,
                sheet_id,
            )

        row_id = _next_row_id(values)
        row = [row_id] + row_without_id
        self.client.write_values(self.spreadsheet_token, sheet_id,
                                 _next_write_row(values), [row])
        return FeishuReportResult(True, 'wrote result row', report_kind,
                                  actual_sheet_title, sheet_id)


def _log(logger: Optional[Callable[[str], None]], message: str) -> None:
    if logger is not None:
        logger(message)


def maybe_report_summary_to_feishu(
        summary_path: str,
        report_kind: str,
        sheet_url: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        config: Optional[str] = None,
        commit_id: Optional[str] = None,
        repo_dir: Optional[str] = None,
        timeout: float = 10.0,
        logger: Optional[Callable[[str], None]] = None,
        log_unconfigured: bool = False) -> FeishuReportResult:
    """Best-effort upload of one summary JSON to Feishu Sheets."""
    sheet_url = sheet_url or os.environ.get('FEISHU_SHEET_URL', '')
    app_id = app_id or os.environ.get('FEISHU_APP_ID', '')
    app_secret = app_secret or os.environ.get('FEISHU_APP_SECRET', '')
    report_kind = report_kind.lower()

    if not sheet_url and not app_id and not app_secret:
        result = FeishuReportResult(False,
                                    'Feishu reporting is not configured',
                                    report_kind)
        if log_unconfigured:
            _log(logger, f'[feishu] skip: {result.reason}')
        return result
    if not sheet_url or not app_id or not app_secret:
        result = FeishuReportResult(
            False,
            'Feishu reporting requires FEISHU_SHEET_URL, FEISHU_APP_ID, '
            'and FEISHU_APP_SECRET',
            report_kind,
        )
        _log(logger, f'[feishu] skip: {result.reason}')
        return result

    spreadsheet_token = parse_feishu_spreadsheet_token(sheet_url)
    if spreadsheet_token is None:
        result = FeishuReportResult(
            False,
            'invalid Feishu Sheets URL; expected a /sheets/<token> link',
            report_kind,
        )
        _log(logger, f'[feishu] skip: {result.reason}')
        return result

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        row = build_report_row(
            summary,
            report_kind,
            config=config,
            commit_id=commit_id,
            repo_dir=repo_dir)
        client = FeishuSheetClient(
            app_id=app_id,
            app_secret=app_secret,
            api_base=open_api_base_from_url(sheet_url),
            timeout=timeout,
        )
        preferred_sheet_id = parse_feishu_sheet_id(sheet_url)
        _log(
            logger,
            f'[feishu] report enabled: kind={report_kind}, '
            f'url_sheet_id={preferred_sheet_id or "none"}',
        )
        reporter = FeishuExperimentReporter(
            client,
            spreadsheet_token,
            preferred_sheet_id=preferred_sheet_id,
        )
        result = reporter.write_row(report_kind, row)
        if result.wrote:
            target_url = _feishu_target_sheet_url(sheet_url, result.sheet_id)
            selection = ('url sheet'
                         if preferred_sheet_id else 'report kind sheet')
            _log(
                logger,
                f'[feishu] {result.reason}: {result.sheet_title} '
                f'(selection={selection}, sheet_id={result.sheet_id}, '
                f'url={target_url})',
            )
        else:
            _log(logger, f'[feishu] skip: {result.reason}')
        return result
    except Exception as exc:  # noqa: BLE001
        result = FeishuReportResult(False, f'{type(exc).__name__}: {exc}',
                                    report_kind)
        _log(logger, f'[feishu] skip: {result.reason}')
        return result
