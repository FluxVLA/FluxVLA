import copy
import importlib.util
import sys
from pathlib import Path


def _load_feishu_reporter():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / 'fluxvla' / 'engines' / 'utils' / 'feishu_reporter.py')
    spec = importlib.util.spec_from_file_location(
        'fluxvla_feishu_reporter_test', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


feishu_reporter = _load_feishu_reporter()
FeishuExperimentReporter = feishu_reporter.FeishuExperimentReporter
build_report_row = feishu_reporter.build_report_row
parse_feishu_sheet_id = feishu_reporter.parse_feishu_sheet_id
parse_feishu_spreadsheet_token = \
    feishu_reporter.parse_feishu_spreadsheet_token


class FakeFeishuClient:

    def __init__(self, sheets):
        self.sheets = copy.deepcopy(sheets)
        self.writes = []

    def list_sheets(self, spreadsheet_token):
        return [{
            'sheet_id': sheet_id,
            'title': data['title']
        } for sheet_id, data in self.sheets.items()]

    def add_sheet(self, spreadsheet_token, title):
        sheet_id = f'sheet_{len(self.sheets) + 1}'
        self.sheets[sheet_id] = {'title': title, 'values': []}
        return sheet_id

    def read_values(self, spreadsheet_token, sheet_id, cell_range):
        return copy.deepcopy(self.sheets[sheet_id]['values'])

    def write_values(self, spreadsheet_token, sheet_id, start_row, values):
        table = self.sheets[sheet_id]['values']
        while len(table) < start_row - 1:
            table.append([])
        for offset, row in enumerate(values):
            target = start_row - 1 + offset
            if len(table) <= target:
                table.append(list(row))
            else:
                table[target] = list(row)
        self.writes.append((sheet_id, start_row, copy.deepcopy(values)))


def test_parse_feishu_spreadsheet_token_accepts_sheet_url():
    token = parse_feishu_spreadsheet_token(
        'https://example.feishu.cn/sheets/abc_123?sheet=sht1')

    assert token == 'abc_123'


def test_parse_feishu_sheet_id_accepts_sheet_query():
    sheet_id = parse_feishu_sheet_id(
        'https://example.feishu.cn/sheets/abc_123?sheet=fuw4tc')

    assert sheet_id == 'fuw4tc'


def test_parse_feishu_spreadsheet_token_rejects_non_sheet_url():
    assert parse_feishu_spreadsheet_token(
        'https://example.feishu.cn/docx/abc_123') is None
    assert parse_feishu_spreadsheet_token(
        'https://example.com/sheets/abc_123') is None


def test_permission_error_message_explains_action():
    message = feishu_reporter._feishu_error_message(
        403, '{"code":91403,"msg":"Forbidden","data":{}}')

    assert 'code=91403' in message
    assert 'share this spreadsheet with the app/bot' in message


def test_empty_libero_sheet_gets_header_and_first_row():
    client = FakeFeishuClient({'s1': {'title': 'libero', 'values': []}})
    reporter = FeishuExperimentReporter(client, 'spreadsheet_token')
    result = reporter.write_row(
        'libero', ['commit', 'configs/foo.py', '90.00%', '', '', '', '90.00%'])

    assert result.wrote
    assert client.sheets['s1']['values'][0] == [
        'id', 'commit id', 'config', 'libero_10', 'libero_goal',
        'libero_object', 'libero_spatial', 'all'
    ]
    assert client.sheets['s1']['values'][1] == [
        1, 'commit', 'configs/foo.py', '90.00%', '', '', '', '90.00%'
    ]


def test_preferred_sheet_id_uses_linked_empty_sheet():
    client = FakeFeishuClient({
        'old': {
            'title': 'libero',
            'values': [['id', 'wrong']]
        },
        'fuw4tc': {
            'title': 'Sheet1',
            'values': []
        },
    })
    reporter = FeishuExperimentReporter(
        client, 'spreadsheet_token', preferred_sheet_id='fuw4tc')
    result = reporter.write_row(
        'libero', ['commit', 'configs/foo.py', '90.00%', '', '', '', '90.00%'])

    assert result.wrote
    assert client.sheets['old']['values'] == [['id', 'wrong']]
    assert client.sheets['fuw4tc']['values'][0] == [
        'id', 'commit id', 'config', 'libero_10', 'libero_goal',
        'libero_object', 'libero_spatial', 'all'
    ]


def test_header_mismatch_skips_write():
    client = FakeFeishuClient(
        {'s1': {
            'title': 'robocasa',
            'values': [['id', 'wrong']]
        }})
    reporter = FeishuExperimentReporter(client, 'spreadsheet_token')
    result = reporter.write_row(
        'robocasa',
        ['commit', 'configs/foo.py', '10.00%', '', '', '', '10.00%'])

    assert not result.wrote
    assert client.writes == []


def test_build_report_row_uses_weighted_all_rate():
    summary = {
        'config': 'configs/libero.py',
        'suite_stats': {
            'libero_10': {
                'total_successes': 8,
                'total_trials': 10
            },
            'libero_goal': {
                'total_successes': 1,
                'total_trials': 10
            },
        }
    }

    row = build_report_row(summary, 'libero', commit_id='abc')

    assert row == [
        'abc', 'configs/libero.py', '80.00%', '10.00%', '', '', '45.00%'
    ]
