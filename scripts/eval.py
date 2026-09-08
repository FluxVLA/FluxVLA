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

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch.distributed as dist
from mmengine import Config, DictAction

from fluxvla.engines import build_runner_from_cfg, initialize_overwatch
from fluxvla.engines.utils.feishu_reporter import \
    maybe_report_summary_to_feishu
from fluxvla.engines.utils.torch_utils import \
    configure_inference_attention_defaults

overwatch = initialize_overwatch(__name__)


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get_eval_runner_cfg(cfg):
    """Return runner config from old or namespaced eval config."""
    if hasattr(cfg.eval, 'runner'):
        return copy.deepcopy(cfg.eval.runner)
    return copy.deepcopy(cfg.eval)


def _get_eval_value(cfg, key, default=None):
    """Read eval runner value from old or namespaced eval config."""
    if hasattr(cfg.eval, 'runner') and hasattr(cfg.eval.runner, key):
        return getattr(cfg.eval.runner, key)
    return cfg.eval.get(key, default)


def _get_cfg_value(cfg_obj, key, default=None):
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _drop_runner_report_cfg(eval_cfg):
    """Ignore legacy report-only keys before constructing eval runners."""
    for key in (
            'feishu_sheet_url',
            'feishu_app_id',
            'feishu_app_secret',
            'feishu_timeout',
    ):
        if isinstance(eval_cfg, dict):
            eval_cfg.pop(key, None)
        elif hasattr(eval_cfg, key):
            delattr(eval_cfg, key)


def _is_libero_eval_cfg(eval_cfg):
    return _get_cfg_value(eval_cfg, 'type') == 'LiberoEvalRunner'


def _is_robocasa_eval_cfg(eval_cfg):
    return _get_cfg_value(eval_cfg, 'type') == 'RobocasaEvalRunner'


def _resolve_suite_max_steps(max_steps, suite):
    if isinstance(max_steps, dict):
        return max_steps.get(suite)
    return max_steps


def _cleanup_eval_runner(eval_runner):
    if eval_runner is None:
        return
    cleanup = getattr(eval_runner, 'cleanup', None)
    if callable(cleanup):
        cleanup()


def _destroy_distributed_process_group():
    """Release distributed resources initialized by Overwatch/Accelerate."""
    if dist.is_available() and dist.is_initialized():
        # Do not add a barrier here. This function also runs after exceptions,
        # when another rank may already have left the process group.
        dist.destroy_process_group()


def _run_eval(cfg, args, suite_name=None):
    eval_cfg = _get_eval_runner_cfg(cfg)
    _drop_runner_report_cfg(eval_cfg)
    is_libero_eval = _is_libero_eval_cfg(eval_cfg)
    is_robocasa_eval = _is_robocasa_eval_cfg(eval_cfg)
    if suite_name is not None:
        eval_cfg.task_suite_name = suite_name
        eval_cfg.max_steps = _resolve_suite_max_steps(
            _get_eval_value(cfg, 'max_steps'), suite_name)
    eval_cfg.cfg = cfg
    eval_cfg.ckpt_path = args.ckpt_path
    if hasattr(eval_cfg,
               'processor') and not hasattr(eval_cfg.processor, 'model_path'):
        eval_cfg.processor.model_path = args.ckpt_path
    eval_runner = None
    try:
        eval_runner = build_runner_from_cfg(eval_cfg)
        eval_runner.run_setup()
        eval_runner.run()
        if (is_libero_eval or is_robocasa_eval) and hasattr(
                eval_runner, 'run_dir'):
            summary_path = os.path.join(eval_runner.run_dir, 'summary.json')
            if os.path.exists(summary_path):
                return ('libero' if is_libero_eval else 'robocasa',
                        summary_path)
    finally:
        _cleanup_eval_runner(eval_runner)
    return None


def _combine_libero_summary_paths(summary_paths, args, cfg):
    summary_paths = [path for path in summary_paths if path]
    if len(summary_paths) == 0:
        return None
    if len(summary_paths) == 1:
        return summary_paths[0]

    suite_stats = {}
    task_results = {}
    total_time = 0.0
    total_tasks = 0
    total_successes = 0
    total_trials = 0
    for summary_path in summary_paths:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        suite_stats.update(summary.get('suite_stats', {}))
        task_results.update(summary.get('task_results', {}))
    for stats in suite_stats.values():
        total_time += float(stats.get('total_time', 0.0))
        total_tasks += int(stats.get('total_tasks', 0))
        total_successes += int(stats.get('total_successes', 0))
        total_trials += int(stats.get('total_trials', 0))

    out_dir = Path(summary_paths[0]).resolve().parent.parent
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'libero_eval_summary_{timestamp}.json'
    average_success_rate = (
        total_successes / max(total_trials, 1) * 100 if total_trials else 0.0)
    average_task_time = total_time / max(total_tasks, 1)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'run_id': out_path.stem,
                'ckpt':
                os.path.abspath(args.ckpt_path) if args.ckpt_path else '',
                'config': getattr(cfg, 'filename', None) or args.config,
                'suite_stats': suite_stats,
                'task_results': task_results,
                'overall': {
                    'average_success_rate': average_success_rate,
                    'total_time': total_time,
                    'average_task_time': average_task_time,
                },
            },
            f,
            indent=4)
    return str(out_path)


def _maybe_report_libero_eval(summary_paths, args, cfg):
    if not overwatch.is_rank_zero() or _get_eval_value(cfg, 'task_ids') \
            is not None:
        return
    summary_path = _combine_libero_summary_paths(summary_paths, args, cfg)
    if summary_path is None:
        return
    maybe_report_summary_to_feishu(
        summary_path,
        'libero',
        config=getattr(cfg, 'filename', None) or args.config,
        logger=overwatch.warning,
        log_unconfigured=True)


def _maybe_report_robocasa_eval(summary_paths, args, cfg):
    if not overwatch.is_rank_zero() or _get_eval_value(cfg, 'task_ids') \
            is not None:
        return
    summary_paths = [path for path in summary_paths if path]
    if len(summary_paths) == 0:
        return
    maybe_report_summary_to_feishu(
        summary_paths[0],
        'robocasa',
        config=getattr(cfg, 'filename', None) or args.config,
        logger=overwatch.warning,
        log_unconfigured=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model with the given configuration.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.',
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help='Path to the checkpoint file.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=  # noqa: E251
        'override some settings in the used config, the key-value pair in xxx=yyy format'  # noqa: E501
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def _run_main():
    configure_inference_attention_defaults()
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    suite_names = _as_list(_get_eval_value(cfg, 'task_suite_name'))
    summary_paths = {'libero': [], 'robocasa': []}
    for suite_name in suite_names:
        result = _run_eval(cfg, args, suite_name=suite_name)
        if result is not None:
            report_kind, summary_path = result
            summary_paths.setdefault(report_kind, []).append(summary_path)
    _maybe_report_libero_eval(summary_paths.get('libero', []), args, cfg)
    _maybe_report_robocasa_eval(summary_paths.get('robocasa', []), args, cfg)


def main():
    try:
        _run_main()
    finally:
        _destroy_distributed_process_group()


if __name__ == '__main__':
    main()
