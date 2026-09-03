"""Serving subpackage for remote VLA inference.

All imports are lazy -- msgpack, zmq, and other serving dependencies are
only loaded when a symbol from this package is actually accessed.  This
avoids forcing ``pip install pyzmq msgpack`` on users who only do local
inference.
"""


def __getattr__(name):
    _public = {
        'FORMAT_MSGPACK',
        'FORMAT_PROTOBUF',
        'MsgSerializer',
        'ObsSerializer',
        'ObsSerializerProto',
        'decode_predict_request',
        'decode_predict_response',
        'detect_format',
        'encode_predict_request',
        'encode_predict_response',
    }
    _server = {'PolicyServer', 'create_server', 'serialize_actions'}
    _ros_server = {
        'FluxVLAROSPolicy',
        'FluxVLAROSServer',
        'build_ros_policy_from_config',
        'build_ros_server_from_config',
    }
    _ros2_server = {'FluxVLAROS2Server'}
    _ros_worker_pool = {
        'EpisodeAffinityPolicyPool',
        'spawn_ros_policy_pool',
    }
    _evaluation_reporter = {
        'EvaluationEventError',
        'FluxVLAROSEvaluationReporter',
    }
    _zmq_eval_server = {
        'FluxVLAZMQEvalServer',
        'build_zmq_eval_server_from_config',
    }

    if name in _public:
        from . import serializers
        return getattr(serializers, name)
    if name in _server:
        from . import zmq_server
        return getattr(zmq_server, name)
    if name in _ros_server:
        from . import ros_server
        return getattr(ros_server, name)
    if name in _ros2_server:
        from . import ros2_server
        return getattr(ros2_server, name)
    if name in _ros_worker_pool:
        from . import ros_worker_pool
        return getattr(ros_worker_pool, name)
    if name in _evaluation_reporter:
        from . import evaluation_reporter
        return getattr(evaluation_reporter, name)
    if name in _zmq_eval_server:
        from . import zmq_eval_server
        return getattr(zmq_eval_server, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'FORMAT_MSGPACK',
    'FORMAT_PROTOBUF',
    'EvaluationEventError',
    'FluxVLAROSPolicy',
    'FluxVLAROSEvaluationReporter',
    'FluxVLAZMQEvalServer',
    'build_zmq_eval_server_from_config',
    'FluxVLAROS2Server',
    'FluxVLAROSServer',
    'EpisodeAffinityPolicyPool',
    'MsgSerializer',
    'ObsSerializer',
    'ObsSerializerProto',
    'PolicyServer',
    'create_server',
    'build_ros_policy_from_config',
    'build_ros_server_from_config',
    'decode_predict_request',
    'decode_predict_response',
    'detect_format',
    'encode_predict_request',
    'encode_predict_response',
    'serialize_actions',
    'spawn_ros_policy_pool',
]
