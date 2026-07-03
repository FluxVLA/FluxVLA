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
"""Aggregate FluxVLA LIBERO eval outputs into cross-suite summaries.

Each ``LiberoEvalRunner`` run evaluates a single suite and writes per-task
``<run_dir>/<suite>/task{id}_results.json``. This tool scans one or more such
run directories, groups the per-task JSONs by suite, and emits a combined
multi-suite ``summary.csv`` (with an ``Overall`` column), ``summary.txt`` and
``summary.json`` for downstream comparison and reporting.
"""

from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

SUITE_ORDER = [
    'libero_spatial',
    'libero_object',
    'libero_goal',
    'libero_10',
    'libero_90',
]


def _load_feishu_reporter():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / 'fluxvla' / 'engines' / 'utils' / 'feishu_reporter.py')
    spec = importlib.util.spec_from_file_location('fluxvla_feishu_reporter',
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.maybe_report_summary_to_feishu


def format_time(seconds: float) -> str:
    """Format seconds as ``SS`` / ``MMmSSs`` / ``HHhMMmSSs``."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds:02d}s'
    if seconds < 3600:
        return f'{seconds // 60:02d}m{seconds % 60:02d}s'
    hours, rem = divmod(seconds, 3600)
    return f'{hours:02d}h{rem // 60:02d}m{rem % 60:02d}s'


def _iter_result_files(root: str):
    """Yield ``*_results.json`` paths under per-suite subdirs of ``root``."""
    for suite in SUITE_ORDER:
        suite_dir = os.path.join(root, suite)
        if not os.path.isdir(suite_dir):
            continue
        for name in sorted(os.listdir(suite_dir)):
            if name.endswith('_results.json'):
                yield suite, os.path.join(suite_dir, name)


def _collect_run_dirs(args: argparse.Namespace) -> List[str]:
    run_dirs = list(args.run_dir or [])
    if args.scan_root:
        for root, dirnames, _ in os.walk(args.scan_root):
            dirnames.sort()
            if os.path.basename(root).startswith('EVAL-'):
                run_dirs.append(root)
                dirnames[:] = []
    if not run_dirs:
        raise SystemExit(
            'No run directories given. Pass --run-dir and/or --scan-root.')
    return run_dirs


def summarize(run_dirs: List[str]) -> Dict:
    """Aggregate per-task JSONs across run dirs into per-suite statistics."""
    suite_stats = defaultdict(
        lambda: {
            'total_tasks': 0,
            'total_trials': 0,
            'total_successes': 0,
            'total_time': 0.0,
            'max_time': 0.0,
        })
    task_results: Dict[str, Dict] = {}
    for run_dir in run_dirs:
        for suite, path in _iter_result_files(run_dir):
            with open(path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            task_id = int(result['task_id'])
            task_key = f'{suite}_{task_id}'
            eps = int(result['total_episodes'])
            if eps == 0:
                continue
            succ = int(result['successes'])
            dur = float(result.get('duration', 0.0))
            stats = suite_stats[suite]
            stats['total_tasks'] += 1
            stats['total_trials'] += eps
            stats['total_successes'] += succ
            stats['total_time'] += dur
            stats['max_time'] = max(stats['max_time'], dur)
            task_results[task_key] = {
                'success_rate': succ / eps * 100,
                'duration': dur,
                'total_episodes': eps,
                'successes': succ,
                'task_description': result.get('task_description', ''),
            }
    return {'suite_stats': dict(suite_stats), 'task_results': task_results}


def write_summaries(summary: Dict, output_dir: str, title: str) -> str:
    """Write combined ``summary.{csv,txt,json}`` to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    suite_stats = summary['suite_stats']
    task_results = summary['task_results']
    ordered_suites = [s for s in SUITE_ORDER if s in suite_stats]

    columns: List[str] = []
    rows = {'Success Rate (%)': [], 'Average Time (s)': [], 'Max Time (s)': []}
    txt_lines = [
        '=== Evaluation Results Summary ===', '',
        'Statistics for each task suite:'
    ]
    total_success_rate = 0.0
    total_time = 0.0
    total_tasks = 0
    overall_max_time = 0.0
    for suite in ordered_suites:
        stats = suite_stats[suite]
        if stats['total_trials'] == 0:
            continue
        rate = stats['total_successes'] / stats['total_trials'] * 100
        avg_time = stats['total_time'] / stats['total_tasks']
        columns.append(suite)
        rows['Success Rate (%)'].append(f'{rate:.2f}')
        rows['Average Time (s)'].append(f'{avg_time:.2f}')
        rows['Max Time (s)'].append(f"{stats['max_time']:.2f}")
        txt_lines += [
            f'\n{suite}:',
            f"- Tasks completed: {stats['total_tasks']}",
            f"- Total attempts: {stats['total_trials']}",
            f"- Successful attempts: {stats['total_successes']}",
            f'- Success rate: {rate:.2f}%',
            f"- Total time: {format_time(stats['total_time'])}",
            f'- Average time per task: {format_time(avg_time)}',
            f"- Longest task time: {format_time(stats['max_time'])}",
        ]
        total_success_rate += rate
        total_time += stats['total_time']
        total_tasks += stats['total_tasks']
        overall_max_time = max(overall_max_time, stats['max_time'])

    num_suites = len(columns)
    if num_suites == 0:
        raise SystemExit('No completed tasks found in the given run dirs.')
    overall_rate = total_success_rate / num_suites
    overall_avg_time = total_time / total_tasks if total_tasks else 0.0
    columns.append('Overall')
    rows['Success Rate (%)'].append(f'{overall_rate:.2f}')
    rows['Average Time (s)'].append(f'{overall_avg_time:.2f}')
    rows['Max Time (s)'].append(f'{overall_max_time:.2f}')
    txt_lines += [
        '\nOverall statistics:',
        f'- Average success rate: {overall_rate:.2f}%',
        f'- Total time: {format_time(total_time)}',
        f'- Average time per task: {format_time(overall_avg_time)}',
        f'- Longest task time: {format_time(overall_max_time)}',
    ]

    summary_csv = os.path.join(output_dir, 'summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        f.write(f'{title}\n')
        writer = csv.writer(f)
        writer.writerow([''] + columns)
        for metric in ('Success Rate (%)', 'Average Time (s)', 'Max Time (s)'):
            writer.writerow([metric] + rows[metric])

    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    suite_tasks = defaultdict(list)
    for task_key in task_results:
        suite = task_key.rsplit('_', 1)[0]
        suite_tasks[suite].append(task_key)
    task_csv = os.path.join(output_dir, 'task_success_rates.csv')
    task_rows = []
    with open(task_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Description', 'Success Rate (%)'])
        for suite in ordered_suites:
            for task_key in sorted(
                    suite_tasks.get(suite, []),
                    key=lambda x: int(x.rsplit('_', 1)[1])):
                res = task_results[task_key]
                row = [
                    task_key, res['task_description'],
                    f"{res['success_rate']:.2f}"
                ]
                task_rows.append(row)
                writer.writerow(row)

    summary_json = os.path.join(output_dir, 'summary.json')
    with open(summary_json, 'w') as f:
        json.dump(
            {
                'run_id': os.path.basename(output_dir),
                'ckpt': os.environ.get('CKPT', ''),
                'config': os.environ.get('CONFIG', ''),
                'suite_stats':
                {suite: suite_stats[suite]
                 for suite in ordered_suites},
                'task_results': task_results,
                'overall': {
                    'average_success_rate': overall_rate,
                    'total_time': total_time,
                    'average_task_time': overall_avg_time,
                },
            },
            f,
            indent=4)
    print('\n'.join(txt_lines))
    print('\n=== Run Information ===')
    print(f'Run ID: {os.path.basename(output_dir)}')
    print(f'Results directory: {output_dir}')
    print(f'Summary file: {summary_json}')
    print(f'Summary CSV: {summary_csv}')
    print(f'Task success rates CSV: {task_csv}')
    print('\n=== Task Success Rates ===')
    print('Task,Description,Success Rate (%)')
    for row in task_rows:
        print(','.join(str(item) for item in row))
    print('\n=== Results Table ===')
    print(','.join([''] + columns))
    for metric in ('Success Rate (%)', 'Average Time (s)', 'Max Time (s)'):
        print(','.join([metric] + rows[metric]))
    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate FluxVLA LIBERO eval outputs.')
    parser.add_argument(
        '--run-dir',
        action='append',
        help='A per-run eval dir holding <suite>/task*_results.json. '
        'Repeatable.')
    parser.add_argument(
        '--scan-root',
        default=None,
        help='Parent dir; every EVAL-* subdir is treated as a run dir.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Where to write the combined summary.{csv,txt,json}.')
    parser.add_argument(
        '--title',
        default='Results',
        help='Title line written at the top of summary.csv.')
    parser.add_argument(
        '--feishu-sheet-url',
        default=os.environ.get('FEISHU_SHEET_URL', ''),
        help='Optional Feishu Sheets URL for uploading LIBERO results.')
    parser.add_argument(
        '--feishu-app-id',
        default=os.environ.get('FEISHU_APP_ID', ''),
        help='Optional Feishu custom app App ID.')
    parser.add_argument(
        '--feishu-app-secret',
        default=os.environ.get('FEISHU_APP_SECRET', ''),
        help='Optional Feishu custom app App Secret.')
    parser.add_argument(
        '--feishu-timeout',
        type=float,
        default=float(os.environ.get('FEISHU_TIMEOUT', '10')),
        help='Feishu API timeout in seconds.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _collect_run_dirs(args)
    summary = summarize(run_dirs)
    summary_json = write_summaries(summary, args.output_dir, args.title)
    if args.feishu_sheet_url or args.feishu_app_id or args.feishu_app_secret:
        maybe_report_summary_to_feishu = _load_feishu_reporter()
        maybe_report_summary_to_feishu(
            summary_json,
            'libero',
            sheet_url=args.feishu_sheet_url,
            app_id=args.feishu_app_id,
            app_secret=args.feishu_app_secret,
            config=os.environ.get('CONFIG', ''),
            timeout=args.feishu_timeout,
            logger=print)


if __name__ == '__main__':
    main()
