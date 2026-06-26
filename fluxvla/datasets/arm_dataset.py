# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Union

import numpy as np

from fluxvla.datasets.sarm_dataset import SARMDataset
from fluxvla.engines import DATASETS


@DATASETS.register_module()
class ARMDataset(SARMDataset):
    """LeRobot dataset loader for ARM advantage reward training.

    Official implementation of https://arxiv.org/abs/2604.03037

    Compared with :class:`SARMDataset`, this dataset:

    * samples **causal** windows ``[t-k, ..., t]`` (history plus current only)
    * reads scalar ``progress`` from parquet rows (not stage/tau heads)
    * derives tri-state ``interval_targets`` in ``{-1, 0, +1}`` from progress
      deltas for the interval head
    * emits ``lerobot_video`` metadata for :class:`DecodeLeRobotVideoSequence`
      (CLIP features are computed online in
      :class:`~fluxvla.models.backbones.llms.arm.ARMBackbone`)
    """

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 video_keys: List[str],
                 transforms: List[Dict],
                 annotation_mode: str = 'single_stage',
                 n_obs_steps: int = 4,
                 frame_gap: int = 30,
                 max_rewind_steps: int = 0,
                 rewind_probability: float = 0.0,
                 state_key: str = 'observation.state',
                 training: bool = True,
                 interval_eps: float = 1e-3) -> None:
        """Initialize an ARM dataset.

        Args:
            data_root_path (Union[str, List[str]]): One or more LeRobot dataset
                roots containing ``meta/``, ``data/``, and ``videos/``.
            video_keys (List[str]): Camera keys decoded from episode videos.
            transforms (List[Dict]): Transform configs applied after sample
                assembly.
            annotation_mode (str): Kept for compatibility with SARM dataset
                construction; ARM training uses parquet ``progress`` directly.
            n_obs_steps (int): Number of history frames before the current
                frame.
            frame_gap (int): Frame stride between adjacent observations.
            max_rewind_steps (int): Maximum rewind frames for augmentation
                (typically ``0`` for ARM).
            rewind_probability (float): Probability of rewind augmentation
                during training.
            state_key (str): Parquet key used for robot state vectors.
            training (bool): Whether training-only sampling rules are enabled.
            interval_eps (float): Minimum absolute progress delta used to label
                an interval as Progressive or Regressive.
        """
        super().__init__(
            data_root_path=data_root_path,
            video_keys=video_keys,
            transforms=transforms,
            annotation_mode=annotation_mode,
            n_obs_steps=n_obs_steps,
            frame_gap=frame_gap,
            max_rewind_steps=max_rewind_steps,
            rewind_probability=rewind_probability,
            state_key=state_key,
            training=training,
        )
        self.interval_eps = interval_eps

    def _compute_causal_indices(self, frame_idx: int, ep_start: int,
                                ep_end: int) -> List[int]:
        """Build causal indices ``[t-k, ..., t]`` clamped in the episode."""
        indices: List[int] = []
        for step in range(self.n_obs_steps, -1, -1):
            target_idx = frame_idx - step * self.frame_gap
            clamped_idx = max(ep_start, min(ep_end - 1, target_idx))
            indices.append(clamped_idx)
        return indices

    def _compute_interval_targets_from_progress(
            self, progress_seq: np.ndarray) -> np.ndarray:
        """Map progress deltas to tri-state interval labels."""
        delta = progress_seq[1:] - progress_seq[:-1]
        labels = np.zeros_like(delta, dtype=np.int64)
        labels[delta > self.interval_eps] = 1
        labels[delta < -self.interval_eps] = -1
        return labels

    def __getitem__(self, index: int, dataset_statistics=None) -> Dict:
        """Build one transformed ARM sequence sample.

        Args:
            index (int): Global row index in the concatenated parquet dataset.
            dataset_statistics: Optional normalized statistics mapping supplied
                by dataset wrappers.

        Returns:
            Dict: Sample with images, states, ``progress``,
            ``interval_targets``, tokenized text, and metadata for
            :class:`~fluxvla.models.vlas.arm_reward_model.ARMRewardModel`.
        """
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        episode_index = int(data['episode_index'])
        episode_start, episode_end = self.episode_ranges[(dataset_idx,
                                                          episode_index)]
        obs_indices = self._compute_causal_indices(index, episode_start,
                                                   episode_end)
        sequence_indices = obs_indices
        valid_length = len(sequence_indices)

        sequence_rows = [self.dataset[seq_idx] for seq_idx in sequence_indices]
        timestamps = [float(row['timestamp']) for row in sequence_rows]
        states = np.stack([
            np.asarray(row[self.state_key], dtype=np.float32)
            for row in sequence_rows
        ],
                          axis=0)

        output = {
            'lerobot_video':
            self._make_lerobot_video_context(dataset_idx, episode_index,
                                             timestamps),
            'states':
            states,
            'stats':
            self._get_state_statistics(dataset_idx, dataset_statistics),
            'task_description':
            self._resolve_task_description(dataset_idx, data),
            'lengths':
            np.asarray(valid_length, dtype=np.int64),
            'episode_index':
            np.asarray(episode_index, dtype=np.int64),
            'current_index':
            np.asarray(index, dtype=np.int64),
        }

        if 'progress' not in sequence_rows[0]:
            raise ValueError(
                "ARMDataset requires dataset rows to contain 'progress'.")
        progress_seq = np.asarray(
            [float(row['progress']) for row in sequence_rows],
            dtype=np.float32,
        )
        center_idx = self.n_obs_steps
        output['progress'] = np.asarray(
            progress_seq[center_idx], dtype=np.float32)
        output['interval_targets'] = (
            self._compute_interval_targets_from_progress(progress_seq))

        for transform in self.transforms:
            output = transform(output)
        # Stats are only needed by NormalizeStatesAndActions.
        output.pop('stats', None)
        return output
