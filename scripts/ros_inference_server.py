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
"""Launch FluxVLA inference as a FluxThemis ROS 1 or ROS 2 service."""

import argparse

from mmengine import Config, DictAction

from fluxvla.engines.runners.serving.ros_server import \
    build_ros_server_from_config
from fluxvla.engines.utils.torch_utils import \
    configure_inference_attention_defaults


def parse_args():
    parser = argparse.ArgumentParser(
        description='Serve FluxVLA inference to FluxThemis through ROS.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--service-name', default=None)
    parser.add_argument('--node-name', default=None)
    parser.add_argument(
        '--ros-version',
        type=int,
        choices=(1, 2),
        default=None,
        help='ROS transport version; overrides themis.ros_server.ros_version.')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_inference_attention_defaults()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    server = build_ros_server_from_config(
        cfg,
        ckpt_path=args.ckpt_path,
        device=args.device,
        service_name=args.service_name,
        node_name=args.node_name,
        ros_version=args.ros_version,
        config_path=args.config,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
