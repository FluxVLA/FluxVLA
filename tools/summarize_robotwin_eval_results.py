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
"""Aggregate FluxVLA RoboTwin eval outputs into per-difficulty summaries.

Each ``RobotwinEvalRunner`` worker launched by the manager evaluates one
task under a single condition (``clean`` or ``random``) and writes a
``summary.json``. This tool scans one manager run directory and merges
worker summaries for each configured evaluation condition. It emits
``summary.csv``, ``summary.txt`` and ``summary.json`` for downstream
comparison and reporting.
"""

from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    """Format seconds as ``SSs`` / ``MMmSSs`` / ``HHhMMmSSs``."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds:02d}s'
    if seconds < 3600:
        return f'{seconds // 60:02d}m{seconds % 60:02d}s'
    hours, rem = divmod(seconds, 3600)
    return f'{hours:02d}h{rem // 60:02d}m{rem % 60:02d}s'


def _load_worker_summaries(run_dir: Path) -> Dict[str, Dict]:
    """Return ``{task: worker summary payload}`` for every worker found."""
    worker_summaries = {}
    for path in sorted((run_dir / 'workers').glob('*/**/summary.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        task_dir = path.relative_to(run_dir / 'workers').parts[0]
        worker_summaries[task_dir] = payload
    return worker_summaries


def _collect_expected_tasks(args: argparse.Namespace, run_dir: Path,
                            worker_summaries: Dict[str, Dict]) -> List[str]:
    """Expected task names: ``--tasks``, then ``tasks.txt``, then workers."""
    if args.tasks:
        return list(args.tasks)
    task_file = run_dir / 'tasks.txt'
    if task_file.is_file():
        return [
            line.strip()
            for line in task_file.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
    return sorted(worker_summaries)


def _build_task_results(expected_tasks: List[str],
                        worker_summaries: Dict[str, Dict]) -> Dict[str, Dict]:
    """Validate worker task metrics and mark missing expected tasks."""
    task_results: Dict[str, Dict] = {}
    for task in expected_tasks:
        if task in task_results:
            raise SystemExit(f'Duplicate RoboTwin task: {task}')
        payload = worker_summaries.get(task)
        worker_results = payload['task_results'] if payload else {}
        result = worker_results.get(task)
        if not result or result['status'] != 'COMPLETED':
            task_results[task] = {
                'status': (result or {}).get('status') or 'MISSING',
                'successes': 0,
                'total_episodes': 0,
                'success_rate': None,
            }
            continue
        eps = int(result['total_episodes'])
        succ = int(result['successes'])
        if eps <= 0 or not 0 <= succ <= eps:
            raise SystemExit(f'Invalid RoboTwin episode counts for {task}')
        task_results[task] = dict(
            result,
            successes=succ,
            total_episodes=eps,
            success_rate=succ / eps * 100)
    return task_results


def _collect_worker_settings(expected_tasks: List[str],
                             worker_summaries: Dict[str, Dict],
                             settings=None) -> Dict:
    """Validate selected worker settings and return their shared metadata."""
    worker_summaries = {
        task: worker_summaries[task]
        for task in expected_tasks if task in worker_summaries
    }
    for key in ('ckpt', 'task_suite_name', 'instruction_type', 'seed',
                'eval_chunk_size', 'trials_per_task'):
        values = {payload[key] for payload in worker_summaries.values()}
        if len(values) > 1:
            raise SystemExit(f'Inconsistent RoboTwin worker field: {key}')
    return next(iter(worker_summaries.values()), settings or {})


def summarize(expected_tasks: List[str], worker_summaries: Dict[str, Dict],
              task_suite_name: str) -> Dict:
    """Aggregate worker task results into group statistics and task details."""
    group = {
        'clean': 'Easy',
        'random': 'Hard',
    }.get(task_suite_name)
    group_stats = {
        group: {
            'total_tasks': 0,
            'total_trials': 0,
            'total_successes': 0,
            'total_time': 0,
            'max_time': 0.0,
        }
    }
    task_results: Dict[str,
                       Dict] = _build_task_results(expected_tasks,
                                                   worker_summaries)
    stats = group_stats[group]
    for result in task_results.values():
        if result['status'] != 'COMPLETED':
            continue
        eps = result['total_episodes']
        succ = result['successes']
        dur = float(result['duration'] or 0)
        stats['total_tasks'] += 1
        stats['total_trials'] += eps
        stats['total_successes'] += succ
        stats['total_time'] += dur
        stats['max_time'] = max(stats['max_time'], dur)
    return {'group_stats': group_stats, 'task_results': task_results}


def _write_summary_csv(output_dir: Path, title: str, columns: List[str],
                       rows: Dict[str, List]) -> str:
    """Write the summary table to ``summary.csv``."""
    summary_csv = os.path.join(output_dir, 'summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        f.write(f'{title}\n')
        writer = csv.writer(f)
        writer.writerow([''] + columns)
        for metric, values in rows.items():
            writer.writerow([metric] + values)
    return summary_csv


def _write_summary_json(output_dir: Path, result: Dict) -> str:
    """Write ``summary.json`` and return its path."""
    summary_json = os.path.join(output_dir, 'summary.json')
    with open(summary_json, 'w') as f:
        json.dump(result, f, indent=4)
    return summary_json


def _print_results_table(columns: List[str], rows: Dict[str, List]) -> None:
    """Print the summary table to the terminal."""
    print('\n=== Results Table ===')
    print(','.join([''] + columns))
    for metric, values in rows.items():
        print(','.join([metric] + [str(value) for value in values]))


def write_summaries(summary: Dict,
                    output_dir: Path,
                    title: str,
                    settings: Dict,
                    ckpt: str = '',
                    conditions: Optional[List[str]] = None) -> str:
    """Write single-condition or comparison ``summary.{csv,txt,json}``
    reports to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    combined = conditions is not None
    summary_items = (((condition, summary[condition])
                      for condition in conditions)
                     if combined else [(None, summary)])
    entries = {}
    columns: List[str] = []
    rows = {'Success Rate (%)': []}
    if not combined:
        rows.update({'Average Time (s)': [], 'Max Time (s)': []})
    rows.update({
        'Episodes': [],
        'Successes': [],
        'Tasks Completed': [],
        'Tasks Expected': [],
    })
    txt_lines = ['=== Evaluation Results Summary ===', '']
    txt_lines += ([
        'Statistics for each task suite:',
        'Success rates cover completed tasks in each difficulty.',
    ] if combined else ['Overall statistics:'])
    for condition, condition_summary in summary_items:
        group_stats = condition_summary['group_stats']
        task_results = condition_summary['task_results']
        stats = next(iter(group_stats.values()))
        rate = (
            stats['total_successes'] / stats['total_trials'] *
            100 if stats['total_trials'] else None)
        avg_time = (
            stats['total_time'] /
            stats['total_tasks'] if stats['total_tasks'] else 0.0)
        if combined:
            difficulty = {
                'clean': 'Easy',
                'random': 'Hard'
            }.get(condition, condition)
            entries[condition] = {
                'difficulty': difficulty,
                'success_rate': rate,
                'successes': stats['total_successes'],
                'episodes': stats['total_trials'],
                'total_tasks': len(task_results),
                'completed_tasks': stats['total_tasks'],
                'task_time_seconds': stats['total_time'],
                'max_time': stats['max_time'],
            }
            columns.append(difficulty)
            txt_lines.append(f'\n{difficulty} ({condition}):')
        else:
            columns.append('Overall')
            rows['Average Time (s)'].append(f'{avg_time:.2f}')
            rows['Max Time (s)'].append(f"{stats['max_time']:.2f}")
        rows['Success Rate (%)'].append('' if rate is None else f'{rate:.2f}')
        rows['Episodes'].append(stats['total_trials'])
        rows['Successes'].append(stats['total_successes'])
        rows['Tasks Completed'].append(stats['total_tasks'])
        rows['Tasks Expected'].append(len(task_results))
        display_rate = 'N/A' if rate is None else f'{rate:.2f}%'
        txt_lines += [
            f"- Tasks completed: {stats['total_tasks']}",
            f'- Tasks expected: {len(task_results)}',
            f"- Total attempts: {stats['total_trials']}",
            f"- Successful attempts: {stats['total_successes']}",
            f'- Success rate: {display_rate}',
            f"- Total time: {format_time(stats['total_time'])}",
            f'- Average time per task: {format_time(avg_time)}',
            f"- Longest task time: {format_time(stats['max_time'])}",
        ]

    summary_csv = _write_summary_csv(output_dir, title, columns, rows)
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')

    if not combined:
        task_csv = os.path.join(output_dir, 'task_success_rates.csv')
        task_rows = []
        with open(task_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Task', 'Successes', 'Episodes', 'Success Rate (%)'])
            for task, res in task_results.items():
                task_rate = res['success_rate']
                row = [
                    task,
                    res['successes'],
                    res['total_episodes'],
                    '' if task_rate is None else f'{task_rate:.2f}',
                ]
                task_rows.append(row)
                writer.writerow(row)

    if combined:
        result = {
            'run_id': output_dir.name,
            'config': os.environ.get('CONFIG', ''),
            'ckpt': ckpt,
            'conditions': entries,
        }
    else:
        result = {
            'run_id': output_dir.name,
            'ckpt': ckpt or settings.get('ckpt', ''),
            'config': os.environ.get('CONFIG', ''),
            'group_stats': {
                group: {
                    'total_successes': stats['total_successes'],
                    'total_trials': stats['total_trials'],
                }
                for group, stats in group_stats.items() if group
            },
            'task_results': task_results,
            'overall': {
                'success_rate': rate,
                'total_tasks': len(task_results),
                'completed_tasks': stats['total_tasks'],
                'total_time': stats['total_time'],
                'average_task_time': avg_time,
                'max_time': stats['max_time'],
            },
        }
    summary_json = _write_summary_json(output_dir, result)
    print('\n'.join(txt_lines))
    print('\n=== Run Information ===')
    print(f'Run ID: {output_dir.name}')
    print(f'Results directory: {output_dir}')
    print(f'Summary file: {summary_json}')
    print(f'Summary CSV: {summary_csv}')
    if not combined:
        print(f'Task success rates CSV: {task_csv}')
        print('\n=== Task Success Rates ===')
        print('Task,Successes,Episodes,Success Rate (%)')
        for row in task_rows:
            print(','.join(str(item) for item in row))
    _print_results_table(columns, rows)
    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate FluxVLA RoboTwin eval outputs.')
    parser.add_argument(
        '--run-dir',
        required=True,
        help='Manager output directory containing workers/.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Where to write the combined summary.{csv,txt,json}.')
    parser.add_argument(
        '--title',
        default='Results',
        help='Title line written at the top of summary.csv.')
    parser.add_argument(
        '--tasks',
        nargs='*',
        default=None,
        help='Expected task names in order. Default: tasks.txt in run-dir, '
        'else the discovered workers.')
    parser.add_argument(
        '--ckpt',
        default=os.environ.get('CKPT', ''),
        help='Checkpoint path saved into summary.json. Default: the '
        'checkpoint recorded by the workers.')
    parser.add_argument(
        '--feishu-sheet-url',
        default=os.environ.get('FEISHU_SHEET_URL', ''),
        help='Optional Feishu Sheets URL for uploading RoboTwin results.')
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


def _load_run_summary(run_dir: Path, args: argparse.Namespace,
                      condition: Optional[str]) -> Tuple[Optional[Dict], Dict]:
    """Load and validate worker results, then summarize the expected tasks."""
    settings = {'task_suite_name': condition} if condition is not None else {}
    worker_summaries = _load_worker_summaries(run_dir)
    expected = _collect_expected_tasks(args, run_dir, worker_summaries)
    if not expected:
        return None, settings
    settings = _collect_worker_settings(expected, worker_summaries, settings)
    summary = summarize(expected, worker_summaries,
                        settings.get('task_suite_name'))
    return summary, settings


def _collect_conditions(run_dir: Path) -> List[str]:
    """Identify evaluation conditions from the result directories."""
    if (run_dir / 'workers').is_dir():
        return []
    return [
        condition for condition in ('clean', 'random')
        if (run_dir / condition).is_dir()
    ]


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    conditions = _collect_conditions(run_dir)
    combined = len(conditions) > 1
    condition_summaries = {}
    summary_paths = []
    complete = True
    for condition in conditions or [None]:
        inner = run_dir / condition if combined else run_dir
        summary, settings = _load_run_summary(inner, args, condition)
        if summary is None:
            print(f'No tasks found under {inner}.', file=sys.stderr)
            return 1
        summary_json = write_summaries(
            summary,
            inner if combined else output_dir,
            args.title,
            settings,
            ckpt=args.ckpt)
        summary_paths.append(summary_json)
        if combined:
            condition_summaries[condition] = summary
        completed_tasks = sum(stats['total_tasks']
                              for stats in summary['group_stats'].values())
        complete = complete and (completed_tasks == len(
            summary['task_results']))
    if combined:
        write_summaries(
            condition_summaries,
            output_dir,
            args.title, {},
            ckpt=args.ckpt,
            conditions=conditions)
    # The reporter consumes condition reports, not the top-level comparison.
    if args.feishu_sheet_url or args.feishu_app_id or args.feishu_app_secret:
        maybe_report_summary_to_feishu = _load_feishu_reporter()
        for summary_json in summary_paths:
            maybe_report_summary_to_feishu(
                summary_json,
                'robotwin',
                sheet_url=args.feishu_sheet_url,
                app_id=args.feishu_app_id,
                app_secret=args.feishu_app_secret,
                config=os.environ.get('CONFIG', ''),
                timeout=args.feishu_timeout,
                logger=print)
    # Non-zero when any expected task is missing so callers can retry.
    return 0 if complete else 1


if __name__ == '__main__':
    sys.exit(main())
