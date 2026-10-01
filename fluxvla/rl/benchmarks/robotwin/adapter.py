"""RoboTwin/Aloha transforms, matching RLinf's public PI0.5 recipe.

The Aloha constants and conversion order follow RLinf's Apache-2.0
``rlinf/models/embodiment/openpi/policies/aloha_policy.py``. No OpenPI runtime
is imported into the Flux/Transformers 5 environment.
"""

import numpy as np
import torch

from fluxvla.rl.utils.tensors import as_numpy
from fluxvla.tokenizers.paligemma_tokenizer import PaligemmaTokenizer
from fluxvla.transforms.prompters import PreparePromptWithState
from fluxvla.transforms.transform_images import _resize_chw_with_pad_pil

JOINT_FLIP = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


def decode_state(state):
    """Map linear grippers to PI angles; not the inverse action transform."""
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
    """Map PI coordinates to Aloha commands using the public transform."""
    actions = np.asarray(actions) * JOINT_FLIP
    actions[..., [6, 13]] = (actions[..., [6, 13]] + 0.5476 +
                             0.6213) / (1.4910 + 0.6213)
    return actions


class RoboTwinObservationAdapter:
    """Three RGB views and 14-D state, without rotation or center cropping."""

    _numpy = staticmethod(as_numpy)

    def __init__(self,
                 *,
                 norm_stats,
                 tokenizer_path,
                 image_size=224,
                 max_token_len=200):
        if norm_stats is None or tokenizer_path is None:
            raise ValueError(
                'RoboTwin requires explicit checkpoint statistics '
                'and tokenizer')
        self.stats = norm_stats.get('norm_stats', norm_stats)
        for field in ('state', 'actions'):
            for quantile in ('q01', 'q99'):
                array = np.asarray(self.stats[field][quantile])
                if array.shape != (14, ) or not np.isfinite(array).all():
                    raise ValueError(
                        f'Expected finite 14-D {field}/{quantile}')
            high = np.asarray(self.stats[field]['q99'])
            if np.any(high < self.stats[field]['q01']):
                raise ValueError(f'Invalid quantile order for {field}')
        self.tokenizer = PaligemmaTokenizer(model_path=tokenizer_path)
        self.prompt = PreparePromptWithState(lowercase_task_description=False)
        self.image_size = int(image_size)
        self.max_token_len = int(max_token_len)

    def normalize_state(self, state):
        low, high = (
            np.asarray(self.stats['state'][key]) for key in ('q01', 'q99'))
        return (decode_state(state) - low) / (high - low + 1e-6) * 2 - 1

    def tokenize(self, prompt, normalized_state):
        text = self.prompt({
            'task_description': prompt,
            'states': normalized_state
        })['prompt']
        # Preserve the structured newline and trailing Action prefix. Flux's
        # ordinary text-only tokenizer intentionally cleans these characters.
        encoded = self.tokenizer._tokenizer.encode(text, add_bos=True)
        tokens = np.zeros(self.max_token_len, dtype=np.int64)
        mask = np.zeros(self.max_token_len, dtype=bool)
        count = min(len(encoded), self.max_token_len)
        tokens[:count], mask[:count] = encoded[:count], True
        return tokens, mask

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
        normalized = self.normalize_state(state)
        rows = []
        for index in range(batch):
            images = [
                _resize_chw_with_pad_pil(
                    view.transpose(2, 0, 1), self.image_size, self.image_size)
                for view in (main[index], wrist[index, 0], wrist[index, 1])
            ]
            tokens, mask = self.tokenize(prompts[index], normalized[index])
            padded = np.pad(normalized[index], (0, 18)).astype(np.float32)
            rows.append({
                'images':
                torch.from_numpy(
                    np.concatenate(images).astype(np.float32) / 255.0 * 2 - 1),
                'img_masks':
                torch.ones(3, dtype=torch.bool),
                'lang_tokens':
                torch.from_numpy(tokens),
                'lang_masks':
                torch.from_numpy(mask),
                'states':
                torch.from_numpy(padded)
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
                'RoboTwin action restoration requires 14 dimensions '
                'and current observation')
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


def build_adapter(flux_cfg, options, norm_stats):
    return RoboTwinObservationAdapter(
        norm_stats=norm_stats,
        tokenizer_path=options.tokenizer_path,
        image_size=options.get('image_size', 224),
        max_token_len=options.get('max_token_len', 200))
