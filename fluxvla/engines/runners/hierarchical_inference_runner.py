# Copyright 2026 Limx Dynamics
"""Config-driven planner -> existing VLA runner, safe-by-default preview."""

import copy
import json
import math
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fluxvla.planning import SkillCatalog, SubtaskController
from fluxvla.planning.catalog import validate_conditions
from ..utils.builder import build_from_cfg
from ..utils.root import PLANNERS, RUNNERS, TASK_MONITORS


@RUNNERS.register_module()
class HierarchicalInferenceRunner:
    """Compose a synchronous real-robot runner without changing VLA outputs.

    ``observations`` is an offline replay for dry-run only. Otherwise an
    ``executor`` config constructs an existing BaseInferenceRunner subclass.
    Preview mode does not load a VLA, connect to a VLA server, or publish
    actions. Live mode deliberately rejects asynchronous/RTC/mobile-base
    execution: those require a stronger handoff and hardware-stop contract.

    Supply the episode's task and completion conditions to ``run()``. The
    constructor may carry paired defaults for a fixed demo, but capabilities
    and executor setup are independent of the task being run.
    """

    def __init__(self,
                 skill_catalog,
                 planner,
                 monitor,
                 executor=None,
                 observations=None,
                 dry_run=True,
                 stable_observations=2,
                 max_rejections=3,
                 max_steps=200,
                 max_observation_age_s=10.0,
                 poll_interval_s=0.1,
                 event_log=None,
                 cfg=None,
                 ckpt_path=None,
                 task_description=None,
                 goal_conditions=None):
        if task_description is not None and (not isinstance(
                task_description, str) or not task_description.strip()):
            raise ValueError('task_description must be nonempty')
        if type(dry_run) is not bool:
            raise ValueError('dry_run must be a boolean')
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        if (not math.isfinite(max_observation_age_s)
                or not math.isfinite(poll_interval_s)
                or max_observation_age_s <= 0 or poll_interval_s < 0):
            raise ValueError('Invalid observation timing limits')
        if observations is not None and not dry_run:
            raise ValueError(
                'Replay observations cannot drive real robot actions')
        if observations is None and executor is None:
            raise ValueError('Provide replay observations or a robot executor')
        self.catalog = SkillCatalog(**skill_catalog)
        self.planner = build_from_cfg(planner, PLANNERS)
        self.monitor = build_from_cfg(monitor, TASK_MONITORS)
        self.controller = None
        self.stable_observations = stable_observations
        self.max_rejections = max_rejections
        # Defaults only: a runtime task must not overwrite the next episode's
        # defaults or silently inherit another task's completion conditions.
        self.task_description = task_description
        self.goal_conditions = (
            validate_conditions(goal_conditions, 'goal_conditions')
            if goal_conditions is not None else None)
        self.executor_config = executor
        self.executor = None
        self.observations = observations
        self.dry_run = dry_run
        self.max_steps = max_steps
        self.max_observation_age_s = max_observation_age_s
        self.poll_interval_s = poll_interval_s
        self.event_log = event_log
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self._setup = False
        self._log = None
        self.episode_id = None

    def run_setup(self):
        if self._setup:
            return
        if self.observations is None:
            if isinstance(self.executor_config, dict):
                config = copy.deepcopy(self.executor_config)
                config['cfg'] = self.cfg
                if self.dry_run:
                    config['ckpt_path'] = None
                    config['remote_inference'] = None
                elif self.ckpt_path is not None:
                    config['ckpt_path'] = self.ckpt_path
                self.executor = build_from_cfg(config, RUNNERS)
            else:
                self.executor = self.executor_config
            for name in ('update_observation_window',
                         '_preprocess_observation', '_predict_action',
                         '_postprocess_actions', '_execute_actions',
                         'reset_inference_history'):
                if not callable(getattr(self.executor, name, None)):
                    raise ValueError(f'Executor must implement {name}')
            if not self.dry_run:
                rtc = getattr(self.executor, 'rtc_config', None) or {}
                if (getattr(self.executor, 'async_execution', False)
                        or rtc.get('enabled', False)
                        or getattr(self.executor, 'use_robot_base', False)):
                    raise ValueError(
                        'V1 supports synchronous, non-RTC arms only')
                horizon = getattr(self.executor, 'execute_horizon', None)
                if horizon is not None and horizon <= 0:
                    raise ValueError(
                        'Synchronous execute_horizon must be positive')
                if getattr(self.executor, 'disable_puppet_arm', False):
                    raise ValueError(
                        'Live execution requires enabled robot arms')
                operator = getattr(self.executor, 'ros_operator', None)
                if not callable(getattr(operator, 'stop_trajectory', None)):
                    raise ValueError('Live executor requires stop_trajectory')
                self.executor.run_setup()
        self._setup = True

    def _emit(self, event):
        event = dict(event, dry_run=self.dry_run, episode_id=self.episode_id)
        text = json.dumps(event, ensure_ascii=False, allow_nan=False)
        print(text, flush=True)
        if self._log:
            self._log.write(text + '\n')
            self._log.flush()
        return event

    def _stop(self):
        if self.executor is not None and not self.dry_run:
            operator = getattr(self.executor, 'ros_operator', None)
            stop = getattr(operator, 'stop_trajectory', None)
            if callable(stop):
                stop()

    def _is_stale(self, captured_at):
        return time.monotonic() - captured_at > self.max_observation_age_s

    def _execute(self, event, observation, captured_at):
        executor = self.executor
        if event['decision']['decision'] == 'execute':
            self._stop()
            executor.reset_inference_history()
        executor._action_ctx = SimpleNamespace(
            instruction=event['instruction'])
        vla_observation = dict(observation)
        vla_observation.pop('predicates', None)
        inputs = executor._preprocess_observation(event['instruction'],
                                                  vla_observation)
        if self._is_stale(captured_at):
            raise TimeoutError('Observation expired during preprocessing')
        with torch.inference_mode(), torch.autocast(
                'cuda',
                dtype=executor.mixed_precision_dtype,
                enabled=executor.enable_mixed_precision
                and not executor._use_remote):
            if executor._use_remote:
                # Some robot-specific overrides call their local VLA directly.
                # Use the shared transport explicitly in remote mode.
                raw_actions = executor._predict_action_remote(inputs)
            else:
                raw_actions = executor._predict_action(inputs)
        actions = executor._postprocess_actions(raw_actions)
        if self._is_stale(captured_at):
            raise TimeoutError('Observation expired during VLA inference')
        if (not isinstance(actions, np.ndarray) or actions.ndim != 2
                or not actions.shape[0] or not np.isfinite(actions).all()):
            raise ValueError(
                'Executor must produce finite [T, action_dim] actions')
        import rospy
        if rospy.is_shutdown():
            raise InterruptedError('ROS shutdown before action publication')
        executor._execute_actions(actions, rospy.Rate(executor.publish_rate))
        executor._prev_ctx = executor._action_ctx

    def run(self, task_description=None, *, goal_conditions=None):
        """Run one task, reusing the catalog, backends and loaded executor.

        Omitted arguments use the configured defaults, never the previous
        episode's runtime arguments. A new task requires explicit completion
        conditions: v1 cannot derive them from natural language. Validate the
        request before setting up hardware or loading the VLA.
        """
        self.controller = None
        if task_description is None:
            task_description = self.task_description
        if not isinstance(task_description,
                          str) or not task_description.strip():
            raise ValueError('Pass a nonempty task_description to run()')
        if goal_conditions is None:
            if task_description != self.task_description:
                raise ValueError(
                    'A new task_description requires explicit '
                    'goal_conditions; completion conditions are not inferred '
                    'from task text')
            goal_conditions = self.goal_conditions
        self.controller = SubtaskController(self.catalog, self.planner,
                                            self.monitor, goal_conditions,
                                            self.stable_observations,
                                            self.max_rejections)
        self.controller.reset(task_description)
        self.episode_id = uuid.uuid4().hex
        executed_chunks = preview_chunks = 0
        last = None
        try:
            self.run_setup()
            if self.event_log:
                path = Path(self.event_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._log = path.open('a', encoding='utf-8')
            replay = iter(
                self.observations) if self.observations is not None else None
            for index in range(self.max_steps):
                if replay is not None:
                    try:
                        observation = next(replay)
                    except StopIteration:
                        last = self.controller.finish('timeout',
                                                      'replay_exhausted')
                        break
                else:
                    import rospy
                    if rospy.is_shutdown(
                    ) or self.executor._poll_keyboard_pause():
                        last = self.controller.finish('cancelled',
                                                      'operator_stop')
                        break
                    observation = self.executor.update_observation_window()
                captured_at = time.monotonic()
                # Retain a stable snapshot even if an observation provider
                # reuses buffers while the remote planner is in flight.
                observation = copy.deepcopy(observation)
                last = self.controller.step(observation, index)
                if self._is_stale(captured_at):
                    last = self.controller.finish('timeout',
                                                  'observation_expired')
                if last['status'] == 'ready':
                    if self.dry_run:
                        preview_chunks += 1
                    else:
                        self._execute(last, observation, captured_at)
                        executed_chunks += 1
                    self.controller.record_dispatch(last['subtask_version'])
                self._emit(last)
                if self.controller.terminal:
                    break
                if replay is None:
                    time.sleep(self.poll_interval_s)
            if not self.controller.terminal:
                last = self.controller.finish('timeout', 'max_steps_exhausted')
        except (KeyboardInterrupt, InterruptedError):
            last = self.controller.finish('cancelled', 'interrupted')
        except TimeoutError as exc:
            last = self.controller.finish('timeout', str(exc))
        except Exception as exc:
            last = self.controller.finish('failed',
                                          f'{type(exc).__name__}: {exc}')
        finally:
            try:
                self._stop()
            except Exception as exc:
                last = self.controller.finish('failed',
                                              f'Executor stop failed: {exc}')
        result = dict(
            last,
            executed_chunks=executed_chunks,
            preview_chunks=preview_chunks)
        try:
            return self._emit(result)
        finally:
            if self._log:
                self._log.close()
                self._log = None

    def cleanup(self):
        self._stop()
        if self.executor is not None and not self.dry_run:
            self.executor.cleanup()

    @property
    def exit_code(self):
        """Optional CLI hook; old runners without this hook are unchanged."""
        if self.controller is None or not self.controller.terminal:
            return None
        return {
            'succeeded': 0,
            'blocked': 2,
            'cancelled': 130
        }.get(self.controller.terminal['status'], 1)
