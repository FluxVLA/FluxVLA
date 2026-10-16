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
    _serving_runner = {
        'FluxVLAServingRunner',
        'build_serving_runner_from_config',
    }
    _evaluation_report_writer = {
        'EvaluationEventError',
        'FluxVLAEvaluationReportWriter',
    }
    _distributed_server = {
        'FluxVLAInferenceServer',
        'build_evaluation_report_writer_from_config',
        'launch_server_task',
        'run_model_worker',
    }

    if name in _public:
        from . import serializers
        return getattr(serializers, name)
    if name in _server:
        from . import zmq_server
        return getattr(zmq_server, name)
    if name in _serving_runner:
        from . import serving_runner
        return getattr(serving_runner, name)
    if name in _evaluation_report_writer:
        from . import evaluation_report_writer
        return getattr(evaluation_report_writer, name)
    if name in _distributed_server:
        from . import distributed_server
        return getattr(distributed_server, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'FORMAT_MSGPACK',
    'FORMAT_PROTOBUF',
    'EvaluationEventError',
    'FluxVLAServingRunner',
    'FluxVLAEvaluationReportWriter',
    'MsgSerializer',
    'ObsSerializer',
    'ObsSerializerProto',
    'PolicyServer',
    'create_server',
    'FluxVLAInferenceServer',
    'build_evaluation_report_writer_from_config',
    'build_serving_runner_from_config',
    'decode_predict_request',
    'decode_predict_response',
    'detect_format',
    'encode_predict_request',
    'encode_predict_response',
    'serialize_actions',
    'launch_server_task',
    'run_model_worker',
]
