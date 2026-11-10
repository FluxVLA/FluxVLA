# Copyright 2026 Limx Dynamics
"""Environment-independent preprocessing -> VLA -> action processing."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from fluxvla.engines.utils.root import POLICIES


@dataclass
class PolicyOutput:
    """A native action chunk or a non-execution status, never dummy actions."""

    status: str
    actions: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)

    def event(self):
        return dict(self.metadata, status=self.status)


class ActionChunkProcessor:
    """Convert normalized [1, T, D] VLA output into native [T, A] commands.

    Temporal normalization rows are relative to this prediction. Every
    state-relative target uses the SAME raw state as its input snapshot.
    The environment adapter remains responsible for mapping vector slices to
    its robot's action dictionary and executing those commands.
    """

    def __init__(self, denormalize_action, execute_horizon, unnorm_key):
        if type(execute_horizon) is not int or execute_horizon < 1:
            raise ValueError('execute_horizon must be a positive integer')
        self.denormalize_action = denormalize_action
        self.execute_horizon = execute_horizon
        self.unnorm_key = unnorm_key

    @staticmethod
    def validate_horizons(execute_horizon, model_horizon, stats_horizon):
        if execute_horizon > model_horizon:
            raise ValueError(
                f'eval_chunk_size={execute_horizon} exceeds model output '
                f'horizon={model_horizon}. Reduce the execution horizon or '
                'adapt the model; regenerating statistics cannot extend it.')
        if stats_horizon is not None and execute_horizon > stats_horizon:
            raise ValueError(
                f'eval_chunk_size={execute_horizon} exceeds action '
                f'statistics horizon={stats_horizon}. Regenerate temporal '
                'statistics before executing this chunk.')

    def __call__(self, actions, *, raw_state=None):
        if not isinstance(actions, torch.Tensor):
            raise TypeError('VLA must return a Tensor of normalized actions')
        if actions.ndim == 3 and actions.shape[0] == 1:
            model_horizon = int(actions.shape[1])
            chunk = actions[0, :self.execute_horizon, :]
        elif actions.ndim == 2 and actions.shape[0] == 1:
            model_horizon = 1
            chunk = actions[0, None, :]
        else:
            raise ValueError('Expected actions shaped [1, T, D] or [1, D], '
                             f'got {tuple(actions.shape)}')
        getter = getattr(self.denormalize_action, 'get_action_stats_horizon',
                         None)
        self.validate_horizons(self.execute_horizon, model_horizon,
                               getter() if callable(getter) else None)
        normalized = chunk.detach().float().cpu().numpy()
        if not normalized.size or not np.isfinite(normalized).all():
            raise ValueError('VLA produced empty or non-finite actions')
        diagnostics = [
            'Normalized action chunk: '
            f'min={normalized.min():.6g}, max={normalized.max():.6g}, '
            f'mean={normalized.mean():.6g}'
        ]
        state = None if raw_state is None else np.asarray(raw_state).copy()
        native = []
        for index, action in enumerate(normalized):
            inputs = dict(
                action=action.copy(),
                action_horizon_index=index,
                task_suite_name=self.unnorm_key)
            if state is not None:
                inputs['state'] = state.copy()
            native.append(self.denormalize_action(inputs))
        native = np.stack(native)
        if native.ndim != 2 or not native.size or not np.isfinite(
                native).all():
            raise ValueError('Action processing must return finite [T, A]')
        diagnostics.append(f'Denorm action: min={native[0].min():.6g}, '
                           f'max={native[0].max():.6g}')
        return native, diagnostics


@POLICIES.register_module()
class VLAPolicy:
    """Wrap already loaded components; never load models in a dataset.

    One instance owns one episode's runtime state. Model objects may be
    shared only when the caller provides appropriate inference isolation.
    ``acknowledge`` reports actual execution, not merely a prediction.
    """

    def __init__(self,
                 vla,
                 input_processor,
                 output_processor,
                 unnorm_key='',
                 mixed_precision_dtype=torch.bfloat16,
                 enable_mixed_precision=True):
        self.vla = vla
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.unnorm_key = unnorm_key
        self.mixed_precision_dtype = mixed_precision_dtype
        self.enable_mixed_precision = enable_mixed_precision
        self.goal = None
        self._pending_horizon = None

    def reset(self, goal, *, output_processor=None):
        if not isinstance(goal, str):
            raise ValueError('goal must be a string')
        self.goal = goal
        self._pending_horizon = None
        if output_processor is not None:
            self.output_processor = output_processor

    def _check_ready(self):
        if self.goal is None:
            raise RuntimeError('Reset the policy before starting an episode')
        if self._pending_horizon is not None:
            raise RuntimeError(
                'Acknowledge the previous chunk before replanning')

    def step(self,
             observation,
             observation_id,
             *,
             policy_seed=None,
             instruction=None):
        self._check_ready()
        data = dict(observation)
        # Environment feedback belongs to the controller, not VLA tensors.
        data.pop('predicates', None)
        data['task_description'] = (
            self.goal if instruction is None else instruction)
        data['is_new_episode'] = observation_id == 0
        processed = self.input_processor(data)
        batch = processed[0] if isinstance(processed, tuple) else processed
        if not isinstance(batch, Mapping):
            raise TypeError('Input processor must return a batch mapping')
        batch = dict(batch, unnorm_key=self.unnorm_key)
        debug = getattr(self.input_processor, 'last_debug', {})
        diagnostics = [
            'Prompt preview: ' +
            (debug.get('text') or debug.get('prompt') or '')[:240]
        ]
        if 'states' in batch:
            state = batch['states'].detach().float().cpu().numpy()
            diagnostics.append(
                f'State range: min={state.min():.6g}, max={state.max():.6g}')
        raw_state = getattr(self.input_processor, 'last_raw_state', None)
        # Preserve eval sampling: seed immediately before VLA inference, not
        # before a local planner that may itself consume random numbers.
        if policy_seed is not None:
            torch.manual_seed(int(policy_seed))
            torch.cuda.manual_seed_all(int(policy_seed))
        with torch.no_grad(), torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision):
            normalized = self.vla.predict_action(**batch)
        actions, action_diagnostics = self.output_processor(
            normalized, raw_state=raw_state)
        self._pending_horizon = len(actions)
        return PolicyOutput(
            status='ready',
            actions=actions,
            diagnostics=diagnostics + action_diagnostics)

    def acknowledge(self, executed_steps):
        if (self._pending_horizon is None or type(executed_steps) is not int
                or not 0 <= executed_steps <= self._pending_horizon):
            raise ValueError('Invalid execution acknowledgment')
        self._pending_horizon = None
