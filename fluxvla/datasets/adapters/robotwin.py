"""RoboTwin/Aloha transforms, matching RLinf's public PI0.5 recipe.

The Aloha constants and conversion order follow RLinf's Apache-2.0
``rlinf/models/embodiment/openpi/policies/aloha_policy.py``. No OpenPI runtime
is imported into the Flux/Transformers 5 environment.
"""

import numpy as np
import torch

from fluxvla.engines import build_transform_from_cfg
from fluxvla.engines.utils import PROCESSORS
from .libero import LiberoRLAdapter

JOINT_FLIP = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


def decode_state(state):
    """Convert runtime linear grippers to PI angular coordinates."""
    state = np.asarray(state) * JOINT_FLIP
    linear = state[..., [6, 13]] * (0.05800 - 0.01844) + 0.01844
    with np.errstate(divide='ignore', invalid='raise'):
        angular = np.arcsin(
            np.clip((0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear),
                    -1, 1))
    state[..., [6, 13]] = (angular - 0.5476) / (1.6296 - 0.5476)
    if not np.isfinite(state).all():
        raise ValueError('Non-finite Aloha state')
    return state


def encode_actions(actions):
    """Convert PI coordinates to public Aloha action commands."""
    actions = np.asarray(actions) * JOINT_FLIP
    actions[..., [6, 13]] = (actions[..., [6, 13]] + 0.5476 +
                             0.6213) / (1.4910 + 0.6213)
    return actions


@PROCESSORS.register_module()
class RoboTwinRLAdapter:
    """Bridge RoboTwin observations through a configured transform pipeline."""

    _numpy = staticmethod(LiberoRLAdapter._numpy)

    def __init__(self, *, norm_stats, transforms):
        if norm_stats is None:
            raise ValueError('RoboTwin requires checkpoint statistics')
        self.stats = norm_stats.get('norm_stats', norm_stats)
        # Action restoration requires quantiles; observation statistics follow
        # the configured normalization transform.
        for quantile in ('q01', 'q99'):
            array = np.asarray(self.stats['actions'][quantile])
            if array.shape != (14, ) or not np.isfinite(array).all():
                raise ValueError(f'Expected finite 14-D actions/{quantile}')
        upper = np.asarray(self.stats['actions']['q99'])
        lower = np.asarray(self.stats['actions']['q01'])
        if np.any(upper < lower):
            raise ValueError('Invalid quantile order for actions')
        self.transforms = [
            build_transform_from_cfg(transform) for transform in transforms
        ]

    def __call__(self, env_obs):
        state = self._numpy(env_obs['states'])
        main, wrist = (
            self._numpy(env_obs[key])
            for key in ('main_images', 'wrist_images'))
        prompts = env_obs['task_descriptions']
        if state.ndim != 2 or state.shape[1] != 14 or not np.isfinite(
                state).all():
            raise ValueError('Expected finite raw [B,14] Aloha state')
        batch = len(state)
        if (main.ndim != 4 or wrist.ndim != 5 or wrist.shape[1] != 2
                or main.shape[-1] != 3 or wrist.shape[-1] != 3
                or main.dtype != np.uint8 or wrist.dtype != np.uint8
                or len(main) != batch or len(wrist) != batch
                or len(prompts) != batch):
            raise ValueError(
                'Expected RGB uint8 main [B,H,W,3] and wrist [B,2,H,W,3]')
        rows = []
        for index in range(batch):
            images = [
                view.transpose(2, 0, 1)
                for view in (main[index], wrist[index, 0], wrist[index, 1])
            ]
            sample = dict(
                images=images,
                img_masks=np.ones(len(images), dtype=bool),
                states=state[index].copy(),
                stats=self.stats,
                task_description=prompts[index])
            for transform in self.transforms:
                sample = transform(sample)
            rows.append({
                key: torch.as_tensor(sample[key], dtype=dtype)
                for key, dtype in dict(
                    images=torch.float32,
                    img_masks=torch.bool,
                    lang_tokens=torch.int64,
                    lang_masks=torch.bool,
                    states=torch.float32).items()
            })
        return {
            key: torch.stack([row[key] for row in rows])
            for key in rows[0]
        }

    def env_actions(self,
                    model_actions,
                    action_chunk,
                    action_dim,
                    *,
                    env_obs=None):
        if action_dim != 14 or env_obs is None:
            raise ValueError(
                'RoboTwin action restoration requires 14 dimensions and '
                'the current observation')
        actions = self._numpy(model_actions[:, :action_chunk, :14]).copy()
        low, high = (
            np.asarray(self.stats['actions'][key]) for key in ('q01', 'q99'))
        actions = (actions + 1) / 2 * (high - low + 1e-6) + low
        state = decode_state(self._numpy(env_obs['states']))
        actions[..., DELTA_MASK] += state[:, None, DELTA_MASK]
        actions = encode_actions(actions)
        if not np.isfinite(actions).all():
            raise ValueError('Non-finite RoboTwin environment action')
        return torch.from_numpy(actions.astype(np.float32))
