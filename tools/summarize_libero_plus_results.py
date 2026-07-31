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
"""Summarize LIBERO-Plus results by the seven perturbation dimensions.

This tool consumes the same per-task JSON or persistent per-worker results as
``summarize_libero_eval_results.py`` and joins zero-based evaluation task IDs
to the one-based IDs in LIBERO-Plus ``task_classification.json``.
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
from typing import Dict, Iterable, List, Optional, Tuple

SUITE_ORDER = [
    'libero_spatial',
    'libero_object',
    'libero_goal',
    'libero_10',
]

# Leaderboard column -> task_classification.json category.
CATEGORY_COLUMNS = [
    ('Camera', 'Camera Viewpoints'),
    ('Robot', 'Robot Initial States'),
    ('Language', 'Language Instructions'),
    ('Light', 'Light Conditions'),
    ('Background', 'Background Textures'),
    ('Noise', 'Sensor Noise'),
    ('Layout', 'Objects Layout'),
]


def _load_standard_summarizer():
    module_path = Path(__file__).with_name('summarize_libero_eval_results.py')
    spec = importlib.util.spec_from_file_location(
        'fluxvla_summarize_libero_eval_results', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load summarizer from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_feishu_reporter():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = (
        repo_root / 'fluxvla' / 'engines' / 'utils' / 'feishu_reporter.py')
    spec = importlib.util.spec_from_file_location(
        'fluxvla_libero_plus_feishu_reporter', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load reporter from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.maybe_report_summary_to_feishu


def _deduplicate_paths(paths: Iterable[Path]) -> List[str]:
    result = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        if not resolved.is_dir():
            raise SystemExit(f'Run directory does not exist: {resolved}')
        seen.add(resolved)
        result.append(str(resolved))
    return result


def _collect_run_dirs(args: argparse.Namespace) -> List[str]:
    paths = [Path(path) for path in (args.run_dir or [])]
    for root_value in args.run_root or []:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f'Run root does not exist: {root}')
        if not any((root / suite).is_dir() for suite in SUITE_ORDER):
            raise SystemExit(
                f'Run root contains no LIBERO suite directories: {root}')
        paths.append(root)
    run_dirs = _deduplicate_paths(paths)
    if not run_dirs:
        raise SystemExit(
            'No result directories found. Pass --run-root and/or --run-dir.')
    return run_dirs


def _load_classification(path: str) -> Tuple[Dict, Dict]:
    classification_path = Path(path).expanduser().resolve()
    if not classification_path.is_file():
        raise SystemExit(
            f'Task classification file does not exist: {classification_path}')
    with classification_path.open(encoding='utf-8') as classification_file:
        raw = json.load(classification_file)

    task_map = {}
    expected_counts = defaultdict(lambda: defaultdict(int))
    known_categories = {category for _, category in CATEGORY_COLUMNS}
    for suite in SUITE_ORDER:
        rows = raw.get(suite)
        if not isinstance(rows, list):
            raise SystemExit(f'Missing classification rows for {suite}.')
        suite_map = {}
        for row in rows:
            classification_id = int(row['id'])
            task_id = classification_id - 1
            if task_id < 0:
                raise SystemExit(
                    f'Classification IDs must be one-based: {suite} id='
                    f'{classification_id}')
            if task_id in suite_map:
                raise SystemExit(f'Duplicate classification ID: {suite} id='
                                 f'{classification_id}')
            category = row['category']
            if category not in known_categories:
                raise SystemExit(
                    f'Unknown category {category!r} for {suite} id='
                    f'{classification_id}')
            suite_map[task_id] = {
                'classification_id': classification_id,
                'name': row.get('name', ''),
                'category': category,
                'difficulty_level': row.get('difficulty_level'),
            }
            expected_counts[suite][category] += 1
        expected_ids = set(range(len(rows)))
        actual_ids = set(suite_map)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise SystemExit(f'Non-contiguous classification IDs for {suite}: '
                             f'missing={missing[:10]}, extra={extra[:10]}')
        task_map[suite] = suite_map
    return task_map, dict(expected_counts)


def _new_counter() -> Dict[str, float]:
    return {
        'expected_tasks': 0,
        'completed_tasks': 0,
        'total_trials': 0,
        'successes': 0,
    }


def _rate(counter: Dict[str, float]) -> Optional[float]:
    trials = int(counter['total_trials'])
    if trials == 0:
        return None
    return int(counter['successes']) / trials * 100.0


def _format_rate(value: Optional[float]) -> str:
    return '' if value is None else f'{value:.2f}'


def _format_task_ranges(task_ids: List[int]) -> str:
    """Format zero-based task IDs as compact inclusive ranges."""
    if not task_ids:
        return ''
    ranges = []
    start = previous = task_ids[0]
    for task_id in task_ids[1:]:
        if task_id == previous + 1:
            previous = task_id
            continue
        ranges.append(
            str(start) if start == previous else f'{start}-{previous}')
        start = previous = task_id
    ranges.append(str(start) if start == previous else f'{start}-{previous}')
    return ','.join(ranges)


def _build_breakdown(summary: Dict, task_map: Dict,
                     expected_counts: Dict) -> Dict:
    by_suite = {
        suite: {category: _new_counter()
                for _, category in CATEGORY_COLUMNS}
        for suite in SUITE_ORDER
    }
    overall = {category: _new_counter() for _, category in CATEGORY_COLUMNS}
    suite_totals = {suite: _new_counter() for suite in SUITE_ORDER}
    overall_total = _new_counter()

    for suite in SUITE_ORDER:
        for _, category in CATEGORY_COLUMNS:
            expected = int(expected_counts[suite][category])
            by_suite[suite][category]['expected_tasks'] = expected
            overall[category]['expected_tasks'] += expected
            suite_totals[suite]['expected_tasks'] += expected
            overall_total['expected_tasks'] += expected

    completed_ids = {suite: set() for suite in SUITE_ORDER}
    extra_results = []
    non_single_trial_tasks = []
    for task_key, result in summary.get('task_results', {}).items():
        suite, task_id_text = task_key.rsplit('_', 1)
        task_id = int(task_id_text)
        if suite not in task_map or task_id not in task_map[suite]:
            extra_results.append(task_key)
            continue
        if task_id in completed_ids[suite]:
            raise SystemExit(f'Duplicate result after aggregation: {task_key}')
        completed_ids[suite].add(task_id)

        trials = int(result['total_episodes'])
        successes = int(result['successes'])
        if trials <= 0 or successes < 0 or successes > trials:
            raise SystemExit(f'Invalid result counts for {task_key}: '
                             f'{successes}/{trials}')
        if trials != 1:
            non_single_trial_tasks.append(task_key)

        category = task_map[suite][task_id]['category']
        for counter in (by_suite[suite][category], overall[category],
                        suite_totals[suite], overall_total):
            counter['completed_tasks'] += 1
            counter['total_trials'] += trials
            counter['successes'] += successes

    missing_ids = {
        suite: sorted(set(task_map[suite]) - completed_ids[suite])
        for suite in SUITE_ORDER
    }
    complete = not extra_results and all(not task_ids
                                         for task_ids in missing_ids.values())
    return {
        'complete': complete,
        'by_suite': by_suite,
        'overall': overall,
        'suite_totals': suite_totals,
        'overall_total': overall_total,
        'missing_task_ids': missing_ids,
        'extra_results': sorted(extra_results),
        'non_single_trial_tasks': sorted(non_single_trial_tasks),
    }


def _counter_payload(counter: Dict[str, float]) -> Dict:
    payload = dict(counter)
    payload['success_rate'] = _rate(counter)
    return payload


def _rates_for(categories: Dict[str, Dict], total: Dict) -> List[str]:
    rates = [
        _format_rate(_rate(categories[category]))
        for _, category in CATEGORY_COLUMNS
    ]
    rates.append(_format_rate(_rate(total)))
    return rates


def _write_outputs(breakdown: Dict, output_dir: str, title: str,
                   classification_path: str, run_dirs: List[str], config: str,
                   ckpt: str) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    completed = int(breakdown['overall_total']['completed_tasks'])
    expected = int(breakdown['overall_total']['expected_tasks'])
    status = 'complete' if breakdown['complete'] else 'partial'
    display_title = title
    if not breakdown['complete']:
        display_title += f' [PARTIAL {completed}/{expected}]'

    columns = [column for column, _ in CATEGORY_COLUMNS] + ['Total']
    leaderboard_path = output_path / 'libero_plus_summary.csv'
    with leaderboard_path.open(
            'w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Model'] + columns)
        writer.writerow([
            display_title,
            *_rates_for(breakdown['overall'], breakdown['overall_total'])
        ])

    by_suite_path = output_path / 'libero_plus_by_suite.csv'
    with by_suite_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            'Suite', *columns, 'Completed Tasks', 'Expected Tasks',
            'Coverage (%)'
        ])
        for suite in SUITE_ORDER:
            suite_total = breakdown['suite_totals'][suite]
            suite_completed = int(suite_total['completed_tasks'])
            suite_expected = int(suite_total['expected_tasks'])
            coverage = suite_completed / suite_expected * 100
            writer.writerow([
                suite,
                *_rates_for(breakdown['by_suite'][suite], suite_total),
                suite_completed,
                suite_expected,
                f'{coverage:.2f}',
            ])
        writer.writerow([
            'Overall',
            *_rates_for(breakdown['overall'], breakdown['overall_total']),
            completed,
            expected,
            f'{completed / expected * 100:.2f}',
        ])

    missing_path = output_path / 'libero_plus_missing_tasks.csv'
    with missing_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ['Suite', 'Missing Count', 'Missing Task IDs (0-based)'])
        for suite in SUITE_ORDER:
            missing = breakdown['missing_task_ids'][suite]
            writer.writerow(
                [suite, len(missing),
                 _format_task_ranges(missing)])

    json_path = output_path / 'libero_plus_summary.json'
    payload = {
        'run_id': output_path.name,
        'title': title,
        'config': config,
        'ckpt': ckpt,
        'status': status,
        'task_classification': str(Path(classification_path).resolve()),
        'task_id_mapping':
        'evaluation task_id is zero-based; classification id is one-based',
        'run_dirs': run_dirs,
        'overall': {
            'categories': {
                column: _counter_payload(breakdown['overall'][category])
                for column, category in CATEGORY_COLUMNS
            },
            'total': _counter_payload(breakdown['overall_total']),
        },
        'by_suite': {
            suite: {
                'categories': {
                    column:
                    _counter_payload(breakdown['by_suite'][suite][category])
                    for column, category in CATEGORY_COLUMNS
                },
                'total': _counter_payload(breakdown['suite_totals'][suite]),
            }
            for suite in SUITE_ORDER
        },
        'missing_task_ids': breakdown['missing_task_ids'],
        'extra_results': breakdown['extra_results'],
        'non_single_trial_tasks': breakdown['non_single_trial_tasks'],
    }
    with json_path.open('w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write('\n')

    text_path = output_path / 'libero_plus_summary.txt'
    text_lines = [
        '=== LIBERO-Plus Perturbation Summary ===',
        f'Title: {display_title}',
        f'Status: {status}',
        f'Coverage: {completed}/{expected} '
        f'({completed / expected * 100:.2f}%)',
        '',
        ','.join(['Suite'] + columns),
    ]
    for suite in SUITE_ORDER:
        text_lines.append(','.join([
            suite,
            *_rates_for(breakdown['by_suite'][suite],
                        breakdown['suite_totals'][suite]),
        ]))
    text_lines.append(','.join([
        'Overall',
        *_rates_for(breakdown['overall'], breakdown['overall_total']),
    ]))
    text_lines += [
        '',
        'Total is micro-averaged over task trials, not the arithmetic mean '
        'of the seven displayed percentages.',
        'Official LIBERO-Plus comparison requires all 10,030 tasks and one '
        'trial per task.',
    ]
    if not breakdown['complete']:
        text_lines += [
            '',
            'WARNING: This is a partial result and is not directly comparable '
            'to the LIBERO-Plus leaderboard.',
        ]
    text_path.write_text('\n'.join(text_lines) + '\n', encoding='utf-8')

    print('\n'.join(text_lines))
    print(f'CSV: {leaderboard_path}')
    print(f'By-suite CSV: {by_suite_path}')
    print(f'JSON: {json_path}')
    print(f'Missing tasks: {missing_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate LIBERO-Plus results into leaderboard columns.')
    parser.add_argument(
        '--run-root',
        action='append',
        help='Run root containing per-suite output dirs (libero_spatial, '
        'libero_object, libero_goal, libero_10). Repeatable.')
    parser.add_argument(
        '--run-dir',
        action='append',
        help='A standard LIBERO run root accepted by '
        'summarize_libero_eval_results.py. Repeatable.')
    parser.add_argument(
        '--task-classification',
        required=True,
        help='Path to LIBERO-Plus task_classification.json.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory for libero_plus_summary.{csv,json,txt}.')
    parser.add_argument(
        '--title', default='Results', help='Model/checkpoint name in the CSV.')
    parser.add_argument(
        '--config',
        default='',
        help='Config path saved into summary.json files.')
    parser.add_argument(
        '--ckpt',
        default='',
        help='Checkpoint path saved into summary.json files.')
    parser.add_argument(
        '--allow-incomplete',
        action='store_true',
        help='Write explicitly marked partial outputs instead of failing when '
        'one or more of the 10,030 task results are missing.')
    parser.add_argument(
        '--feishu-sheet-url',
        default=os.environ.get('FEISHU_SHEET_URL', ''),
        help='Optional Feishu Sheets URL for uploading LIBERO-Plus results.')
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
    task_map, expected_counts = _load_classification(args.task_classification)
    standard_summarizer = _load_standard_summarizer()
    summary = standard_summarizer.summarize(run_dirs, reject_duplicates=True)
    breakdown = _build_breakdown(summary, task_map, expected_counts)

    if breakdown['extra_results']:
        raise SystemExit(
            'Results contain task IDs absent from task_classification.json: ' +
            ', '.join(breakdown['extra_results'][:20]))
    if breakdown['non_single_trial_tasks']:
        raise SystemExit(
            'LIBERO-Plus requires exactly one trial per task; invalid '
            'results: ' + ', '.join(breakdown['non_single_trial_tasks'][:20]))
    if not breakdown['complete'] and not args.allow_incomplete:
        missing_summary = ', '.join(
            f'{suite}={len(task_ids)}'
            for suite, task_ids in breakdown['missing_task_ids'].items()
            if task_ids)
        raise SystemExit(
            'LIBERO-Plus results are incomplete (' + missing_summary + '). '
            'Finish/resume evaluation, or pass --allow-incomplete to write '
            'clearly marked partial statistics.')

    standard_summarizer.write_summaries(
        summary,
        args.output_dir,
        args.title,
        config=args.config,
        ckpt=args.ckpt,
        print_task_rows=False,
    )

    _write_outputs(
        breakdown,
        args.output_dir,
        args.title,
        args.task_classification,
        run_dirs,
        args.config,
        args.ckpt,
    )
    if (breakdown['complete'] and (args.feishu_sheet_url or args.feishu_app_id
                                   or args.feishu_app_secret)):
        maybe_report_summary_to_feishu = _load_feishu_reporter()
        maybe_report_summary_to_feishu(
            str(Path(args.output_dir).resolve() / 'libero_plus_summary.json'),
            'libero_plus',
            sheet_url=args.feishu_sheet_url,
            app_id=args.feishu_app_id,
            app_secret=args.feishu_app_secret,
            config=args.config,
            timeout=args.feishu_timeout,
            logger=print,
        )


if __name__ == '__main__':
    main()
