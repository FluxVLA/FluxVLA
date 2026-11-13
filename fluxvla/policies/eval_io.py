# Copyright 2026 Limx Dynamics
"""Inference preprocessing and action conversion, outside evaluation loops."""

import numpy as np
import torch

from .vla_policy import ActionChunkProcessor


class EvaluationIO:

    def __init__(self, vla, dataset, denormalize_action, execute_horizon,
                 unnorm_key, mixed_precision_dtype, enable_mixed_precision):
        self.vla = vla
        self.dataset = dataset
        self.denormalize_action = denormalize_action
        self.execute_horizon = execute_horizon
        self.unnorm_key = unnorm_key
        self.mixed_precision_dtype = mixed_precision_dtype
        self.enable_mixed_precision = enable_mixed_precision
        self.goal = None

    @property
    def last_raw_state(self):
        return getattr(self.dataset, 'last_raw_state', None)

    @property
    def last_debug(self):
        return getattr(self.dataset, 'last_debug', {})

    def reset(self, goal, **context):
        self.goal = goal

    def before_predict(self, environment):
        pass

    def __call__(self, observation):
        return self.dataset(observation)

    def observe(self, observation, observation_id, **context):
        record = getattr(self.dataset, 'record_observation', None)
        if callable(record):
            record(observation, observation_id)


class LiberoInferenceIO(EvaluationIO):

    def __init__(self,
                 *args,
                 norm_stats_key,
                 controller_use_delta=None,
                 num_inference_steps=None,
                 inference_seed=None,
                 prefetch_observations=False,
                 record_history=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.task_suite_name = self.unnorm_key
        self.norm_stats_key = norm_stats_key
        self.controller_use_delta = controller_use_delta
        self.num_inference_steps = num_inference_steps
        self.inference_seed = inference_seed
        self.prefetch_observations = prefetch_observations
        self.record_history = record_history
        self.replay_image = None
        self.next_batch = None
        self.environment = None
        self.configured_environment = False
        self.normalized_actions = None

    def reset(self, goal, *, unnorm_key=None, **context):
        super().reset(goal)
        if unnorm_key is not None:
            self.unnorm_key = unnorm_key
        self.next_batch = self.replay_image = self.normalized_actions = None
        self.environment = None
        self.configured_environment = False

    def before_predict(self, environment):
        self.environment = environment
        if not self.configured_environment:
            configure = getattr(self.vla, 'configure_eval_env', None)
            if callable(configure):
                configure(environment, self.controller_use_delta)
            elif self.controller_use_delta is not None:
                raise RuntimeError('controller_use_delta requires the VLA to '
                                   'implement configure_eval_env().')
            self.configured_environment = True

    def _process(self, observation):
        processed = self.dataset(observation)
        if isinstance(processed, tuple):
            batch, self.replay_image = processed
        else:
            batch, self.replay_image = processed, None
        return batch

    def __call__(self, observation):
        batch = self.next_batch
        self.next_batch = None
        if batch is None:
            batch = self._process(observation)
        batch = dict(batch, unnorm_key=self.unnorm_key)
        kwargs = dict(batch)
        if self.num_inference_steps is not None:
            kwargs['num_inference_steps'] = self.num_inference_steps
        if self.inference_seed is not None:
            kwargs['seed'] = self.inference_seed
        build = getattr(self.vla, 'build_eval_predict_action_kwargs', None)
        if callable(build):
            kwargs.update(build(batch, env=self.environment))
        return kwargs

    def process_actions(self, actions, *, raw_state=None):
        normalized = actions[0] if actions.ndim == 3 else actions[0, None, :]
        self.normalized_actions = normalized.detach().float().cpu().numpy()

        def denormalize(data):
            return self.denormalize_action(
                dict(
                    action=data['action'],
                    task_suite_name=self.task_suite_name,
                    norm_stats_key=self.norm_stats_key))

        return ActionChunkProcessor(denormalize, self.execute_horizon,
                                    self.unnorm_key)(
                                        actions)

    def observe(self,
                observation,
                observation_id,
                *,
                action_index=None,
                terminal=False):
        if self.record_history:
            super().observe(observation, observation_id)
        if action_index is not None:
            feedback = getattr(self.vla, 'after_eval_env_step', None)
            if callable(feedback):
                feedback(self.normalized_actions[action_index])
            if self.prefetch_observations and not terminal:
                self.next_batch = self._process(
                    dict(
                        observation,
                        task_description=self.goal,
                        is_new_episode=False))
                return self.replay_image
        return None


class RobocasaInferenceIO(EvaluationIO):

    def __init__(self,
                 *args,
                 group_names=(),
                 group_stats=None,
                 group_denormalizers=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.group_names = group_names
        self.group_stats = group_stats
        self.group_denormalizers = group_denormalizers
        self.active_denormalizer = self.denormalize_action

    def reset(self, goal, *, task_index=0, **context):
        super().reset(goal)
        if self.group_stats is not None:
            group = self.group_names[task_index]
            stats = self.group_stats[group]
            self.dataset.set_active_stats_blob(stats[self.unnorm_key])
            self.vla.norm_stats = stats
            self.active_denormalizer = self.group_denormalizers[group]
        else:
            activate = getattr(self.dataset, 'set_active_stats_blob', None)
            if callable(activate):
                activate(None)
            self.active_denormalizer = self.denormalize_action

    def process_actions(self, actions, *, raw_state=None):
        return ActionChunkProcessor(self.active_denormalizer,
                                    self.execute_horizon, self.unnorm_key)(
                                        actions, raw_state=raw_state)


class RobotwinInferenceIO(EvaluationIO):

    def __init__(self, *args, device, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __call__(self, observation):
        batch = self.dataset(observation)
        if isinstance(batch, tuple):
            batch = batch[0]
        return {
            key:
            value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def process_actions(self, actions, *, raw_state=None):
        data = dict(action=actions.detach().float().cpu().numpy())
        if raw_state is not None:
            data['state'] = raw_state
        native = np.asarray(self.denormalize_action(data), dtype=np.float32)
        if native.ndim == 3 and native.shape[0] == 1:
            native = native[0]
        if (native.ndim != 2 or native.shape[1] != 14 or not len(native)
                or not np.isfinite(native).all()):
            raise ValueError('RoboTwin actions must be finite [T, 14]')
        if len(native) < self.execute_horizon:
            raise ValueError(f'FluxVLA returned {len(native)} actions, fewer '
                             f'than eval_chunk_size={self.execute_horizon}')
        return native[:self.execute_horizon], []
