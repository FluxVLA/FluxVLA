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

from mmengine import Config, DictAction

from fluxvla.engines import build_runner_from_cfg, initialize_overwatch
from fluxvla.engines.runners.hierarchical_inference_runner import \
    HierarchicalInferenceRunner
from fluxvla.engines.utils.torch_utils import \
    configure_inference_attention_defaults
from fluxvla.planning.backends import strict_json_loads
from fluxvla.planning.catalog import validate_conditions

overwatch = initialize_overwatch(__name__)


def parse_goal_conditions(value):
    try:
        return validate_conditions(strict_json_loads(value), 'goal_conditions')
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference with the given configuration.')
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
        '--task-description',
        type=str,
        default=None,
        help='Overall task for this hierarchical inference episode.')
    parser.add_argument(
        '--goal-conditions',
        type=parse_goal_conditions,
        default=None,
        help='Completion predicates as a nonempty JSON object of booleans; '
        'required when supplying a new task description.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=  # noqa: E251
        'override some settings in the used config, the key-value pair in xxx=yyy format'  # noqa: E501
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    configure_inference_attention_defaults()
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.inference.cfg = cfg
    cfg.inference.ckpt_path = args.ckpt_path
    inference_runner = build_runner_from_cfg(cfg.inference)
    if isinstance(inference_runner, HierarchicalInferenceRunner):
        # run() validates the task before performing its own idempotent setup.
        inference_runner.run(
            task_description=args.task_description,
            goal_conditions=args.goal_conditions)
    else:
        if (args.task_description is not None
                or args.goal_conditions is not None):
            raise ValueError('--task-description/--goal-conditions require '
                             'HierarchicalInferenceRunner')
        inference_runner.run_setup()
        inference_runner.run()
    exit_code = getattr(inference_runner, 'exit_code', None)
    if exit_code is not None:
        raise SystemExit(exit_code)
