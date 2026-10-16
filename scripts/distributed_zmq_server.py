#!/usr/bin/env python
"""Launch a single- or multi-node stateless FluxVLA ZMQ server."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_FRONTEND_BIND = 'tcp://0.0.0.0:15555'
DEFAULT_MODEL_WORKER_BIND = 'tcp://0.0.0.0:15556'
DEFAULT_MAX_PENDING_REQUESTS = 128
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_STARTUP_TIMEOUT_S = 900.0


def _torchrun_environment() -> tuple[int, int, int, str]:
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size < 1:
        raise ValueError('WORLD_SIZE must be positive')
    if rank < 0 or rank >= world_size:
        raise ValueError('RANK must be in [0, WORLD_SIZE)')
    if local_rank < 0:
        raise ValueError('LOCAL_RANK must be non-negative')
    return (rank, world_size, local_rank,
            os.environ.get('MASTER_ADDR', '127.0.0.1'))


def parse_args(argv=None):
    from mmengine import DictAction

    parser = argparse.ArgumentParser(
        description='Launch the stateless FluxVLA ZMQ inference service.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', default=None)
    networking = parser.add_argument_group('advanced networking')
    networking.add_argument(
        '--frontend-bind',
        default=DEFAULT_FRONTEND_BIND,
        help=f'Public client-facing bind address (default: '
        f'{DEFAULT_FRONTEND_BIND}).')
    networking.add_argument(
        '--backend-bind',
        default=DEFAULT_MODEL_WORKER_BIND,
        help=f'Internal model-worker bind address (default: '
        f'{DEFAULT_MODEL_WORKER_BIND}).')
    networking.add_argument(
        '--backend-endpoint',
        default=None,
        help='Address model workers connect to; defaults to the backend port '
        'on MASTER_ADDR.')
    reliability = parser.add_argument_group('advanced reliability')
    reliability.add_argument(
        '--max-pending-requests',
        type=int,
        default=DEFAULT_MAX_PENDING_REQUESTS)
    reliability.add_argument(
        '--request-timeout-s', type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    reliability.add_argument(
        '--startup-timeout-s', type=float, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None)
    return parser.parse_args(argv)


def _connect_endpoint(bind: str, host: str) -> str:
    parsed = urlsplit(bind)
    if parsed.scheme != 'tcp' or parsed.port is None:
        raise ValueError('distributed endpoints must use tcp://host:port')
    return 'tcp://{}:{}'.format(host, parsed.port)


def main(argv=None) -> int:
    from mmengine import Config

    from fluxvla.engines.runners.serving.distributed_server import \
        launch_server_task
    from fluxvla.engines.runners.serving.serving_runner import (
        config_get, get_server_config, require_mapping,
        resolve_checkpoint_path)

    args = parse_args(argv)
    rank, world_size, local_rank, master_addr = _torchrun_environment()
    config_path = str(Path(args.config).expanduser().resolve(strict=True))
    cfg = Config.fromfile(config_path)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    themis_cfg = require_mapping(config_get(cfg, 'themis'), 'config.themis')
    server_cfg = get_server_config(themis_cfg)
    section_name = server_cfg.get('dataset_section', 'eval')
    section_cfg = require_mapping(
        config_get(cfg, section_name), f'config.{section_name}')
    checkpoint = resolve_checkpoint_path(
        args.ckpt_path or server_cfg.get('ckpt_path')
        or section_cfg.get('ckpt_path'))
    print(
        '[FluxVLA] rank={} starting model worker on cuda:{}'.format(
            rank, local_rank),
        flush=True)
    frontend_bind = args.frontend_bind
    backend_bind = args.backend_bind
    backend_endpoint = (
        args.backend_endpoint or _connect_endpoint(backend_bind, master_addr))

    launch_server_task(
        config_path=config_path,
        ckpt_path=str(checkpoint),
        device='cuda:{}'.format(local_rank),
        frontend_bind=frontend_bind,
        backend_bind=backend_bind,
        backend_endpoint=backend_endpoint,
        max_pending_requests=args.max_pending_requests,
        request_timeout_s=args.request_timeout_s,
        startup_timeout_s=args.startup_timeout_s,
        cfg_options=args.cfg_options,
        rank=rank,
        world_size=world_size,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
