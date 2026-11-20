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
"""Startup-time assembly of native evaluation predictors.

The legacy model/checkpoint loaders remain unchanged. Once loaded, their
components are composed here, not by an eval runner or its rollout loop.
Only this construction boundary reads the top-level policy configuration.
"""

from pathlib import Path

from fluxvla.policies.eval_io import (LiberoInferenceIO, RobocasaInferenceIO,
                                      RobotwinInferenceIO)
from fluxvla.policies.runtime import (DeferredPredictor,
                                      build_action_predictor, copy_options,
                                      predictor_fingerprint,
                                      resolve_policy_options)


def build_eval_predictor(runner, options=None, init_args=None):
    """Wrap already loaded inference components; never build a second VLA."""
    options = options or {}
    args = init_args or {}
    kind = runner.inference_backend
    common = dict(
        vla=runner.vla,
        dataset=runner.dataset,
        denormalize_action=runner.denormalize_action,
        execute_horizon=runner.eval_chunk_size,
        unnorm_key=(runner.task_suite_name
                    if kind == 'libero' else runner.unnorm_key),
        mixed_precision_dtype=runner.mixed_precision_dtype,
        enable_mixed_precision=(
            runner.enable_mixed_precision_training and
            args.get('model_build_device',
                     getattr(runner, 'model_build_device', None)) != 'cpu'))
    if kind == 'libero':
        composed = any(
            options.get(key) is not None
            for key in ('policy', 'observation_adapter'))
        io = LiberoInferenceIO(
            **common,
            norm_stats_key=runner.norm_stats_key,
            controller_use_delta=getattr(runner, 'controller_use_delta', None),
            num_inference_steps=runner.num_inference_steps,
            inference_seed=runner.inference_seed,
            prefetch_observations=runner.preprocess_every_step
            and not composed,
            record_history=composed)
    elif kind == 'robocasa':
        grouped = runner.grouped_norm_stats
        io = RobocasaInferenceIO(
            **common,
            group_names=runner.norm_stats_group_names,
            group_stats=runner._stats_full_by_group if grouped else None,
            group_denormalizers=runner._denorm_by_group if grouped else None)
    else:
        raise ValueError(f'Unsupported evaluation backend: {kind}')
    return build_action_predictor(io, **options)


def build_robotwin_model(cfg, settings, options=None):
    """Shared by the native runner and the standalone RoboTwin entrypoint."""
    from fluxvla.engines.runners.robotwin_eval_runner import LocalFluxVLAPolicy

    model = LocalFluxVLAPolicy(cfg=cfg, **settings)
    try:
        io = RobotwinInferenceIO(
            vla=model.vla,
            dataset=model.dataset,
            denormalize_action=model.denormalize_action,
            execute_horizon=model.eval_chunk_size,
            unnorm_key=model.unnorm_key,
            mixed_precision_dtype=model.mixed_precision_dtype,
            enable_mixed_precision=(model.enable_mixed_precision_training
                                    and model.device.type == 'cuda'),
            device=model.device)
        model.predictor = build_action_predictor(io, **(options or {}))
        return model
    except BaseException:
        model.close()
        raise


def resolve_robotwin_settings(usr_args):
    """Resolve standalone hook configuration outside environment runners."""
    cfg = usr_args.get('_fluxvla_cfg')
    if cfg is None:
        config_path = usr_args.get('fluxvla_config')
        if not config_path:
            raise ValueError(
                'Unified RoboTwin evaluation requires _fluxvla_cfg or '
                'fluxvla_config')
        from mmengine import Config
        cfg = Config.fromfile(config_path)

    dataset = usr_args.get('dataset')
    denormalize_action = usr_args.get('denormalize_action')
    if dataset is None or denormalize_action is None:
        # Standalone use through ``fluxvla_config``: read the runner section.
        eval_cfg = getattr(cfg, 'eval', None) or {}
        dataset = dataset if dataset is not None else eval_cfg.get('dataset')
        denormalize_action = (
            denormalize_action if denormalize_action is not None else
            eval_cfg.get('denormalize_action'))

    options = resolve_policy_options(cfg, 'eval', overrides=usr_args)
    eval_chunk_size = usr_args.get('eval_chunk_size', 32)

    settings = dict(
        ckpt_path=str(Path(usr_args['ckpt_path']).expanduser().resolve()),
        dataset=dataset,
        denormalize_action=denormalize_action,
        eval_chunk_size=eval_chunk_size,
        unnorm_key=usr_args.get('unnorm_key', 'private'),
        device=usr_args.get('device', 'cuda:0'),
        mixed_precision_dtype=usr_args.get('mixed_precision_dtype', 'bf16'),
        enable_mixed_precision_training=usr_args.get(
            'enable_mixed_precision_training', True),
        norm_stats_path=usr_args.get('norm_stats_path'),
        max_episode_steps=usr_args.get('max_episode_steps'),
        enable_model_dtype_cast=usr_args.get('enable_model_dtype_cast', False))
    # Include preprocessing/statistics overrides, not only model weights.
    key = (id(cfg), repr(settings), predictor_fingerprint(options))
    return cfg, settings, options, key


def _deferred_robotwin_predictor(args, options):
    # The adapter must choose CUDA/Warp/Vulkan settings before loading models.
    names = ('ckpt_path', 'dataset', 'denormalize_action', 'eval_chunk_size',
             'unnorm_key', 'mixed_precision_dtype',
             'enable_mixed_precision_training', 'norm_stats_path',
             'max_episode_steps', 'enable_model_dtype_cast')
    settings = {name: args[name] for name in names if name in args}
    settings = copy_options(settings)
    cfg = args['cfg']

    def construct(device):
        return build_robotwin_model(cfg,
                                    dict(settings, device=device or 'cuda:0'),
                                    options)

    return DeferredPredictor(construct, predictor_fingerprint(options))


def build_native_eval_runner(runner_cfg, default_args=None):
    """Resolve config and inject a runtime at the construction boundary."""
    from fluxvla.engines.utils.builder import build_from_cfg
    from fluxvla.engines.utils.root import RUNNERS

    args = dict(default_args or {})
    args.update(runner_cfg)
    options = resolve_policy_options(args.get('cfg'), 'eval', args)
    args.pop('policy', None)
    args.pop('observation_adapter', None)
    target = args['type']
    cls = RUNNERS.get(target) if isinstance(target, str) else target
    if (cls.inference_backend == 'robotwin' and args.get('predictor') is None):
        args['predictor'] = _deferred_robotwin_predictor(args, options)
    runner = build_from_cfg(args, RUNNERS)
    try:
        if runner.predictor is None:
            runner.predictor = build_eval_predictor(runner, options, args)
        return runner
    except BaseException:
        cleanup = getattr(runner, 'cleanup', None)
        if callable(cleanup):
            cleanup()
        raise
