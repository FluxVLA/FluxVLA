# Copyright 2026 Limx Dynamics
"""Episode-local planning worker for clients with a different Python runtime.

The private inherited pipe is local IPC, never a network pickle endpoint.
This process uses FluxVLA's planning backends but does not construct a VLA.
RoboDojo can keep its own Isaac/Transformers versions.
"""

import argparse
import os
from multiprocessing.connection import Connection


def start_debugger():
    """Opt-in localhost attachment; normal evaluation needs no debugpy."""
    port = os.environ.get('FLUXVLA_PLANNER_DEBUG_PORT')
    if not port:
        return None
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError('FLUXVLA_PLANNER_DEBUG_PORT must be in [1, 65535]')
    import debugpy
    debugpy.listen(('127.0.0.1', port))
    print(f'[planner] debugger listening on 127.0.0.1:{port}', flush=True)
    if os.environ.get('FLUXVLA_PLANNER_DEBUG_WAIT') == '1':
        return debugpy.wait_for_client
    return None


def serve(connection):
    from fluxvla.engines.utils import PLANNERS, TASK_MONITORS, build_from_cfg
    from fluxvla.planning import (SkillCatalog, SubtaskController,
                                  TrainingSubtaskCatalog,
                                  TrainingSubtaskController)

    controller = None
    wait_for_debugger = start_debugger()
    try:
        while True:
            request = connection.recv()
            command = request['command']
            if command == 'close':
                break
            try:
                if command == 'configure':
                    cfg = request['policy']
                    controller = None
                    settings = dict(
                        planner=build_from_cfg(cfg['planner'], PLANNERS),
                        monitor=(build_from_cfg(cfg['monitor'], TASK_MONITORS)
                                 if cfg.get('monitor') is not None else None),
                        goal_conditions=cfg.get('goal_conditions'),
                        stable_observations=cfg.get('stable_observations', 1),
                        max_rejections=cfg.get('max_rejections', 1))
                    if 'subtask_source' in cfg:
                        if 'skill_catalog' in cfg:
                            raise ValueError(
                                'Choose subtask_source OR skill_catalog')
                        controller = TrainingSubtaskController(
                            catalog=TrainingSubtaskCatalog(
                                **cfg['subtask_source']),
                            replan_every_chunks=cfg.get(
                                'replan_every_chunks', 1),
                            max_episode_chunks=cfg.get('max_episode_chunks',
                                                       70),
                            max_control_replans=cfg.get(
                                'max_control_replans', 0),
                            max_episode_control_replans=cfg.get(
                                'max_episode_control_replans', 12),
                            completion_requirements=cfg.get(
                                'completion_requirements'),
                            completion_guidance=cfg.get(
                                'completion_guidance', ''),
                            completion_mode=cfg.get('completion_mode',
                                                    'environment'),
                            planner_image_keys=cfg.get('planner_image_keys',
                                                       ()),
                            **settings)
                    else:
                        controller = SubtaskController(
                            catalog=SkillCatalog(**cfg['skill_catalog']),
                            **settings)
                    result = None
                elif controller is None:
                    raise RuntimeError('Configure the planner first')
                elif command == 'reset':
                    controller.reset(request['goal'])
                    result = None
                elif command == 'step':
                    # Wait at the first prediction, NOT during worker startup:
                    # the simulator supervisor has a short startup watchdog.
                    if wait_for_debugger is not None:
                        print(
                            '[planner] attach the Planner debugger to '
                            'continue the first subtask',
                            flush=True)
                        wait_for_debugger()
                        wait_for_debugger = None
                    if request['executed_previous_chunk']:
                        controller.record_dispatch(controller.version)
                    result = controller.step(request['observation'],
                                             request['observation_id'])
                else:
                    raise ValueError('Unknown planner command')
                connection.send(dict(ok=True, result=result))
            except Exception as exc:
                # Do not echo arbitrary provider payloads/credentials.
                connection.send(dict(ok=False, error=type(exc).__name__))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    serve(Connection(parser.parse_args().fd))
