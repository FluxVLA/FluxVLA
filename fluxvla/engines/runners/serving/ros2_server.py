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
"""Native ROS 2 transport for FluxThemis inference and reporting services.

ROS 2 dependencies are imported only by :meth:`FluxVLAROS2Server.run`.  The
policy, request validation, episode tracking, preprocessing, and action
denormalization contracts remain shared with the ROS 1 implementation.
"""
from __future__ import annotations
import importlib
from typing import Any

from .ros_server import FluxVLAROSServer


class _ROS2Time:
    """Expose the ROS 1-style ``Time.now`` hook used by the shared handler."""

    def __init__(self, node: Any) -> None:
        self._node = node

    def now(self) -> Any:
        return self._node.get_clock().now().to_msg()


class _ROS2Runtime:
    """Small adapter for the transport-neutral request handler."""

    def __init__(self, node: Any) -> None:
        self._node = node
        self.Time = _ROS2Time(node)

    def logerr(self, message: str) -> None:
        self._node.get_logger().error(message)

    def loginfo(self, message: str) -> None:
        self._node.get_logger().info(message)


class FluxVLAROS2Server(FluxVLAROSServer):
    """Expose policy inference and optional evaluation reporting over ROS 2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rclpy = None
        self._node = None
        self._owns_rclpy_context = False

    def run(self) -> None:
        """Import ROS 2, advertise the service, and block in ``rclpy.spin``."""
        try:
            rclpy = importlib.import_module('rclpy')
            cv_bridge = importlib.import_module('cv_bridge')
            service_module = importlib.import_module('fluxthemis_msgs.srv')
            bridge = getattr(cv_bridge, 'CvBridge')()
            service_type = getattr(service_module, 'PredictAction')
            response_type = getattr(service_type, 'Response')
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                'FluxVLA ROS 2 serving requires rclpy, cv_bridge and the '
                'generated fluxthemis_msgs/PredictAction service. Source the '
                'ROS 2 installation and interface workspace before launching '
                'the server.') from exc

        report_service_type = None
        report_response_type = None
        if self.report_service_name is not None:
            try:
                report_service_type = getattr(service_module,
                                              'ReportEvaluation')
                report_response_type = getattr(report_service_type, 'Response')
            except AttributeError as exc:
                raise ImportError(
                    'FluxVLA evaluation reporting is configured, but the '
                    'generated fluxthemis_msgs/srv/ReportEvaluation ROS 2 '
                    'service is unavailable. Rebuild and source the '
                    'FluxThemis ROS 2 interface workspace.') from exc

        owns_context = not rclpy.ok()
        node = None
        service = None
        report_service = None
        if owns_context:
            rclpy.init(args=None)
        try:
            node = rclpy.create_node(self.node_name)
            runtime = _ROS2Runtime(node)
            self._rclpy = rclpy
            self._node = node
            self._owns_rclpy_context = owns_context
            self._bind_ros(
                runtime,
                bridge,
                response_type,
                report_response_type=report_response_type,
            )
            service = node.create_service(service_type, self.service_name,
                                          self._handle_ros2_request)
            self._service = service
            node.get_logger().info(
                f'FluxVLA ROS 2 inference ready on {self.service_name}')
            if self.report_service_name is not None:
                report_service = node.create_service(
                    report_service_type,
                    self.report_service_name,
                    self._handle_ros2_report_request,
                )
                self._report_service = report_service
                node.get_logger().info(
                    'FluxVLA ROS 2 evaluation reporting ready on '
                    f'{self.report_service_name}')
            rclpy.spin(node)
        finally:
            if node is not None:
                if report_service is not None:
                    node.destroy_service(report_service)
                if service is not None:
                    node.destroy_service(service)
                node.destroy_node()
            self._service = None
            self._report_service = None
            self._node = None
            self._rclpy = None
            self._owns_rclpy_context = False
            self._rospy = None
            self._bridge = None
            self._response_type = None
            self._report_response_type = None
            if owns_context and rclpy.ok():
                rclpy.shutdown()

    def _handle_ros2_request(self, request: Any, response: Any) -> Any:
        return self.handle_request(request, response=response)

    def _handle_ros2_report_request(self, request: Any, response: Any) -> Any:
        return self.handle_report_request(request, response=response)
