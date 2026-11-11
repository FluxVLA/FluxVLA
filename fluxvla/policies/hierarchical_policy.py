# Copyright 2026 Limx Dynamics
"""Planner/controller + VLA inference, independent of evaluation runners."""

import copy

from fluxvla.engines.utils.builder import build_from_cfg
from fluxvla.engines.utils.root import PLANNERS, POLICIES, TASK_MONITORS
from fluxvla.planning import (SkillCatalog, SubtaskController,
                              TrainingSubtaskCatalog,
                              TrainingSubtaskController)
from .vla_policy import PolicyOutput, VLAPolicy


@POLICIES.register_module()
class HierarchicalPolicy(VLAPolicy):
    """Plan before VLA tokenization, retaining subtasks until completion.

    A planner can be an API-backed config, a registered local VLM adapter,
    or an already initialized object with ``choose``/``reset``. Its model
    and tokenizer belong to that backend, never to ``input_processor``.
    Simulator facts arrive as observation['predicates']; no env object is
    passed into this policy. Expected task failures return a status so the
    caller can save its rollout rather than crash before writing artifacts.
    """

    def __init__(self,
                 skill_catalog=None,
                 planner=None,
                 monitor=None,
                 goal_conditions=None,
                 instruction_prefix='',
                 stable_observations=1,
                 max_rejections=1,
                 subtask_source=None,
                 replan_every_chunks=1,
                 max_episode_chunks=70,
                 max_control_replans=0,
                 max_episode_control_replans=12,
                 completion_requirements=None,
                 completion_guidance='',
                 completion_mode='environment',
                 planner_image_keys=(),
                 **kwargs):
        super().__init__(**kwargs)
        if not isinstance(instruction_prefix, str):
            raise ValueError('instruction_prefix must be a string')
        self.instruction_prefix = instruction_prefix
        if (skill_catalog is None) == (subtask_source is None):
            raise ValueError('Choose skill_catalog OR subtask_source')
        settings = dict(
            planner=(build_from_cfg(planner, PLANNERS) if isinstance(
                planner, dict) else planner),
            monitor=(build_from_cfg(monitor, TASK_MONITORS) if isinstance(
                monitor, dict) else monitor),
            goal_conditions=goal_conditions,
            stable_observations=stable_observations,
            max_rejections=max_rejections)
        if subtask_source is None:
            if completion_mode != 'environment':
                raise ValueError('Agent completion requires subtask_source')
            self.controller = SubtaskController(
                catalog=SkillCatalog(**skill_catalog), **settings)
        else:
            self.controller = TrainingSubtaskController(
                catalog=TrainingSubtaskCatalog(**subtask_source),
                replan_every_chunks=replan_every_chunks,
                max_episode_chunks=max_episode_chunks,
                max_control_replans=max_control_replans,
                max_episode_control_replans=max_episode_control_replans,
                completion_requirements=completion_requirements,
                completion_guidance=completion_guidance,
                completion_mode=completion_mode,
                planner_image_keys=planner_image_keys,
                **settings)
        self._pending_version = None

    def reset(self, goal, *, output_processor=None):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError('Hierarchical episodes require a nonempty goal')
        super().reset(goal, output_processor=output_processor)
        self.controller.reset(goal)
        self._pending_version = None

    def step(self, observation, observation_id, *, policy_seed=None):
        self._check_ready()
        # Both backends and the VLA process this same observation snapshot.
        observation = copy.deepcopy(observation)
        if getattr(self.controller, 'completion_mode', None) == 'agent':
            # Simulator truth must not enter planning or policy metadata.
            observation.pop('predicates', None)
        event = self.controller.step(observation, observation_id)
        active = self.controller.active
        metadata = dict(planning=event)
        if active is not None:
            metadata['subtask'] = dict(
                active.describe(), executed_chunks=self.controller.chunks)
            if isinstance(self.controller, TrainingSubtaskController):
                metadata['subtask']['completion_verified'] = False
            else:
                metadata['subtask'].update(
                    completion=copy.deepcopy(active.completion),
                    max_chunks=active.max_chunks)
        if 'predicates' in observation:
            metadata['predicates'] = copy.deepcopy(observation['predicates'])
        if event['status'] != 'ready':
            return PolicyOutput(status=event['status'], metadata=metadata)
        if event['decision']['decision'] == 'execute':
            reset = getattr(self.vla, 'reset_inference_history', None)
            if callable(reset):
                reset()
        result = super().step(
            observation,
            observation_id,
            policy_seed=policy_seed,
            instruction=self.instruction_prefix + event['instruction'])
        result.metadata = metadata
        self._pending_version = self.controller.version
        return result

    def acknowledge(self, executed_steps):
        super().acknowledge(executed_steps)
        if executed_steps:
            self.controller.record_dispatch(self._pending_version)
        self._pending_version = None
