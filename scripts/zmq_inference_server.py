#!/usr/bin/env python
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
"""Launch FluxVLA inference as a FluxThemis ZMQ evaluation server."""

import argparse


def parse_args():
    from mmengine import DictAction

    parser = argparse.ArgumentParser(
        description='Serve FluxVLA inference to FluxThemis through ZMQ.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--host', default='*')
    parser.add_argument('--port', type=int, default=5555)
    worker_group = parser.add_mutually_exclusive_group()
    worker_group.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of full-model inference replicas on cuda:0..cuda:N-1.')
    worker_group.add_argument(
        '--devices',
        type=_parse_devices,
        default=None,
        help='Comma-separated inference devices, for example cuda:0,cuda:1.')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None)
    return parser.parse_args()


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(',') if item.strip())
    if not devices:
        raise argparse.ArgumentTypeError('--devices cannot be empty')
    if len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError('--devices cannot contain duplicates')
    return devices


def main() -> int:
    from mmengine import Config

    from fluxvla.engines.runners.serving.zmq_eval_server import \
        build_zmq_eval_server_from_config
    from fluxvla.engines.utils.torch_utils import \
        configure_inference_attention_defaults

    args = parse_args()
    configure_inference_attention_defaults()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    server = build_zmq_eval_server_from_config(
        cfg,
        ckpt_path=args.ckpt_path,
        device=args.device,
        host=args.host,
        port=args.port,
        worker_devices=args.devices,
        num_workers=args.num_workers,
        config_path=args.config,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        close = getattr(server, 'close', None)
        if callable(close):
            close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
