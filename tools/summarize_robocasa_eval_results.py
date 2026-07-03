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
"""Aggregate FluxVLA RoboCasa eval outputs into one summary.

Each ``RobocasaEvalRunner`` worker writes per-task
``robocasa/*_results.json`` files. This tool scans one or more run roots,
groups those per-task results into RoboCasa's reporting buckets, and emits a
combined ``summary.csv``, ``summary.txt``, ``task_success_rates.csv`` and
``summary.json`` suitable for Feishu reporting.
"""

from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

GROUP_ORDER = ['Cabinet', 'Drawer', 'Microwave', 'Generalization']


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


def _result_dirs(root: str) -> Iterable[Path]:
    root_path = Path(root).expanduser().resolve()
    if root_path.name == 'robocasa':
        yield root_path
    candidate = root_path / 'robocasa'
    if candidate.is_dir():
        yield candidate


def _iter_result_files(root: str):
    """Yield per-task ``*_results.json`` paths under ``root``."""
    for result_dir in _result_dirs(root):
        for path in sorted(result_dir.glob('*_results.json')):
            yield path


def _collect_run_dirs(args: argparse.Namespace) -> List[str]:
    run_dirs = list(args.run_dir or [])
    if args.scan_root:
        scan_root = Path(args.scan_root).expanduser().resolve()
        for root, dirnames, _ in os.walk(scan_root):
            dirnames.sort()
            root_path = Path(root)
            if (root_path / 'robocasa').is_dir():
                run_dirs.append(str(root_path))
                dirnames[:] = []
    if not run_dirs:
        raise SystemExit(
            'No run directories given. Pass --run-dir and/or --scan-root.')
    return run_dirs


def summarize(run_dirs: List[str]) -> Dict:
    """Aggregate per-task JSON files into RoboCasa group statistics."""
    group_stats = {
        group: {
            'total_tasks': 0,
            'total_trials': 0,
            'total_successes': 0,
            'total_time': 0.0,
            'max_time': 0.0,
        }
        for group in GROUP_ORDER
    }
    task_stats: Dict[str, Dict] = {}
    seen_task_ids = set()
    for run_dir in run_dirs:
        for path in _iter_result_files(run_dir):
            with open(path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            task_id = int(result['task_id'])
            if task_id in seen_task_ids:
                raise SystemExit(
                    f'Duplicate RoboCasa task id {task_id} from {path}.')
            seen_task_ids.add(task_id)
            eps = int(result.get('total_episodes', 0))
            if eps == 0:
                continue
            succ = int(result.get('successes', 0))
            dur = float(result.get('duration', 0.0))
            group = result.get('group') or 'Generalization'
            if group not in group_stats:
                group = 'Generalization'
            env_name = str(result.get('env_name') or f'task{task_id}')
            rate = succ / max(eps, 1) * 100
            stats = group_stats[group]
            stats['total_tasks'] += 1
            stats['total_trials'] += eps
            stats['total_successes'] += succ
            stats['total_time'] += dur
            stats['max_time'] = max(stats['max_time'], dur)
            task_stats[env_name] = {
                'task_id': task_id,
                'group': group,
                'total_episodes': eps,
                'successes': succ,
                'success_rate': rate,
                'duration': dur,
                'gpu_id': result.get('gpu_id'),
            }
    return {'group_stats': group_stats, 'task_stats': task_stats}


def write_summaries(summary: Dict,
                    output_dir: str,
                    title: str,
                    config: str = '',
                    ckpt: str = '') -> str:
    """Write combined ``summary.{csv,txt,json}`` to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    group_stats = summary['group_stats']
    task_stats = summary['task_stats']

    total_trials = 0
    total_successes = 0
    total_time = 0.0
    overall_max_time = 0.0
    rows = {'Success Rate (%)': [], 'Episodes': [], 'Successes': []}
    txt_lines = [
        '=== RoboCasa Evaluation Results Summary ===', '',
        'Statistics for each group:'
    ]
    for group in GROUP_ORDER:
        stats = group_stats[group]
        trials = int(stats['total_trials'])
        successes = int(stats['total_successes'])
        rate = successes / max(trials, 1) * 100 if trials else 0.0
        rows['Success Rate (%)'].append(f'{rate:.2f}' if trials else '')
        rows['Episodes'].append(trials)
        rows['Successes'].append(successes)
        txt_lines += [
            f'\n{group}:',
            f"- Tasks completed: {stats['total_tasks']}",
            f'- Total attempts: {trials}',
            f'- Successful attempts: {successes}',
            f'- Success rate: {rate:.2f}%',
            f"- Total time: {format_time(stats['total_time'])}",
            f"- Longest task time: {format_time(stats['max_time'])}",
        ]
        total_trials += trials
        total_successes += successes
        total_time += float(stats['total_time'])
        overall_max_time = max(overall_max_time, float(stats['max_time']))

    if total_trials == 0:
        raise SystemExit('No completed RoboCasa tasks found.')

    overall_rate = total_successes / max(total_trials, 1) * 100
    rows['Success Rate (%)'].append(f'{overall_rate:.2f}')
    rows['Episodes'].append(total_trials)
    rows['Successes'].append(total_successes)
    columns = GROUP_ORDER + ['all']
    txt_lines += [
        '\nOverall statistics:',
        f'- Success rate: {overall_rate:.2f}%',
        f'- Total attempts: {total_trials}',
        f'- Successful attempts: {total_successes}',
        f'- Total time: {format_time(total_time)}',
        f'- Longest task time: {format_time(overall_max_time)}',
    ]

    summary_csv = os.path.join(output_dir, 'summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        f.write(f'{title}\n')
        writer = csv.writer(f)
        writer.writerow([''] + columns)
        for metric in ('Success Rate (%)', 'Episodes', 'Successes'):
            writer.writerow([metric] + rows[metric])

    summary_txt = os.path.join(output_dir, 'summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines) + '\n')

    task_csv = os.path.join(output_dir, 'task_success_rates.csv')
    task_rows = []
    with open(task_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task ID', 'Task', 'Group', 'Successes', 'Episodes',
            'Success Rate (%)', 'GPU'
        ])
        for env_name, stats in sorted(
                task_stats.items(), key=lambda item: item[1]['task_id']):
            row = [
                stats['task_id'],
                env_name,
                stats['group'],
                stats['successes'],
                stats['total_episodes'],
                f"{stats['success_rate']:.2f}",
                stats.get('gpu_id', ''),
            ]
            task_rows.append(row)
            writer.writerow(row)

    summary_json = os.path.join(output_dir, 'summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'run_id': os.path.basename(os.path.abspath(output_dir)),
                'ckpt': ckpt,
                'config': config,
                'group_stats': group_stats,
                'task_stats': task_stats,
                'overall': {
                    'success_rate': overall_rate,
                    'total_episodes': total_trials,
                    'successes': total_successes,
                    'total_time': total_time,
                    'max_time': overall_max_time,
                },
            },
            f,
            indent=4)

    print('\n'.join(txt_lines))
    print('\n=== Run Information ===')
    print(f'Run ID: {os.path.basename(os.path.abspath(output_dir))}')
    print(f'Results directory: {output_dir}')
    print(f'Summary file: {summary_json}')
    print(f'Summary CSV: {summary_csv}')
    print(f'Task success rates CSV: {task_csv}')
    print('\n=== Task Success Rates ===')
    print('Task ID,Task,Group,Successes,Episodes,Success Rate (%),GPU')
    for row in task_rows:
        print(','.join(str(item) for item in row))
    print('\n=== Results Table ===')
    print(','.join([''] + columns))
    for metric in ('Success Rate (%)', 'Episodes', 'Successes'):
        print(','.join([metric] + [str(v) for v in rows[metric]]))
    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate FluxVLA RoboCasa eval outputs.')
    parser.add_argument(
        '--run-dir',
        action='append',
        help='A run root holding robocasa/*_results.json. Repeatable.')
    parser.add_argument(
        '--scan-root',
        default=None,
        help='Parent dir; every child with robocasa/*_results.json is used.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Where to write the combined summary.{csv,txt,json}.')
    parser.add_argument(
        '--title',
        default='Results',
        help='Title line written at the top of summary.csv.')
    parser.add_argument(
        '--config',
        default=os.environ.get('CONFIG', ''),
        help='Config path saved into summary.json and Feishu.')
    parser.add_argument(
        '--ckpt',
        default=os.environ.get('CKPT', ''),
        help='Checkpoint path saved into summary.json.')
    parser.add_argument(
        '--feishu-sheet-url',
        default=os.environ.get('FEISHU_SHEET_URL', ''),
        help='Optional Feishu Sheets URL for uploading RoboCasa results.')
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
    summary_json = write_summaries(
        summary,
        args.output_dir,
        args.title,
        config=args.config,
        ckpt=args.ckpt)
    if args.feishu_sheet_url or args.feishu_app_id or args.feishu_app_secret:
        maybe_report_summary_to_feishu = _load_feishu_reporter()
        maybe_report_summary_to_feishu(
            summary_json,
            'robocasa',
            sheet_url=args.feishu_sheet_url,
            app_id=args.feishu_app_id,
            app_secret=args.feishu_app_secret,
            config=args.config,
            timeout=args.feishu_timeout,
            logger=print)


if __name__ == '__main__':
    main()
