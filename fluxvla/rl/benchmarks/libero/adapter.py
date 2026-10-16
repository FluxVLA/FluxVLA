"""Translate canonical RLinf LIBERO observations using Flux preprocessing."""

import copy

import numpy as np
import torch
from PIL import Image

from fluxvla.engines import build_transform_from_cfg
from fluxvla.rl.utils.tensors import as_numpy


class LiberoObservationAdapter:
    """Two already-rotated RGB cameras, raw 8D proprio, and task strings."""

    def __init__(self, flux_cfg, *, norm_stats=None, tokenizer_path=None):
        dataset = flux_cfg.train_dataloader.dataset
        self.stats = (
            norm_stats
            if norm_stats is not None else dataset.dataset_statistics)
        self.stats_key = dataset.statistic_name
        if self.stats_key not in self.stats:
            raise ValueError(
                f'Missing normalization statistics: {self.stats_key}')
        transforms = {
            item['type']: copy.deepcopy(dict(item))
            for item in flux_cfg.eval.dataset.transforms
        }
        image_cfg = transforms['TransformImage']
        if len(image_cfg['input_sizes']) != 2:
            raise ValueError(
                'The LIBERO bridge requires exactly two camera views')
        self.image_transform = build_transform_from_cfg(image_cfg)
        prompt_cfg = transforms['LiberoPromptFromInputs']
        if tokenizer_path is not None:
            prompt_cfg['tokenizer']['model_path'] = str(tokenizer_path)
        self.prompt_transform = build_transform_from_cfg(prompt_cfg)
        self.state_transform = build_transform_from_cfg(
            transforms['LiberoProprioFromInputs'])
        if self.state_transform.norm_type != 'mean_std':
            raise ValueError(
                'Flux LIBERO v1 requires mean_std proprio normalization')
        self.state_stats = self.stats[self.stats_key][
            self.state_transform.stat_key]
        self.state_dim = self.state_transform.state_dim
        self.action_transform = build_transform_from_cfg(
            dict(
                **copy.deepcopy(dict(flux_cfg.eval.denormalize_action)),
                norm_stats=self.stats))

    _numpy = staticmethod(as_numpy)

    def __call__(self, env_obs):
        states = self._numpy(env_obs['states'])
        main = self._numpy(env_obs['main_images'])
        wrist = self._numpy(env_obs['wrist_images'])
        prompts = env_obs['task_descriptions']
        batch_size = states.shape[0]
        if states.ndim != 2 or states.shape[1] != len(
                self.state_stats['mean']):
            raise ValueError('Expected unnormalized [B, 8] LIBERO proprio')
        if len(prompts) != batch_size or main.shape[
                0] != batch_size or wrist.shape[0] != batch_size:
            raise ValueError('Observation fields have different batch sizes')
        for images in (main, wrist):
            if images.ndim != 4 or images.shape[
                    -1] != 3 or images.dtype != np.uint8:
                raise ValueError(
                    'Expected already-rotated RGB uint8 images [B,H,W,3]')
        rows = []
        for index in range(batch_size):
            row = self.image_transform({
                'pixel_values':
                [Image.fromarray(main[index]),
                 Image.fromarray(wrist[index])]
            })
            row = self.prompt_transform(
                dict(row, task_description=prompts[index]))
            state = self.state_transform._normalize(states[index],
                                                    self.state_stats)
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:len(state)] = state
            rows.append({
                'images':
                row['pixel_values'],
                'img_masks':
                torch.ones(2, dtype=torch.bool),
                'lang_tokens':
                torch.as_tensor(row['lang_tokens'], dtype=torch.long),
                'lang_masks':
                torch.as_tensor(row['lang_masks'], dtype=torch.bool),
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
        """Denormalize and convert the gripper once per action."""
        raw = self._numpy(model_actions[:, :action_chunk, :action_dim])
        actions = [
            self.action_transform({
                'action': action.copy(),
                'norm_stats_key': self.stats_key
            }) for action in raw.reshape(-1, action_dim)
        ]
        return torch.from_numpy(
            np.asarray(actions, dtype=np.float32).reshape(raw.shape))


def build_adapter(flux_cfg, options, norm_stats):
    return LiberoObservationAdapter(
        flux_cfg,
        norm_stats=norm_stats,
        tokenizer_path=options.get('tokenizer_path'))
