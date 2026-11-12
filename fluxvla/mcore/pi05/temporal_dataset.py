# Copyright 2025 VLASH team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""LeRobot datasets with temporal-delay augmentation for PI training.

Adapted from the Apache-2.0 VLASH dataset implementation. This module contains
only the behavior used by PMP and has no VLASH package dependency.
"""

import random
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PI0TemporalDataset(LeRobotDataset):
    """Apply a random action delay and use the previous action as state."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        max_delay_steps: int = 0,
    ):
        self.max_delay_steps = max_delay_steps
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        self._last_offset = 0

    def _get_query_indices(self, idx: int, ep_idx: int):
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        max_delta = self.delta_indices["action"][-1]
        max_offset = min(self.max_delay_steps, max(0, ep_end - 1 - idx - max_delta))
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
        self._last_offset = offset
        return self._get_query_indices_for_offset(idx, ep_idx, offset)

    def _get_query_indices_for_offset(self, idx: int, ep_idx: int, offset: int):
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {}
        padding = {}
        for key, delta_indices in self.delta_indices.items():
            query_indices[key] = [
                max(ep_start, min(ep_end - 1, idx + delta + offset))
                for delta in delta_indices
            ]
            padding[f"{key}_is_pad"] = torch.tensor(
                [
                    idx + delta + offset < ep_start or idx + delta + offset >= ep_end
                    for delta in delta_indices
                ],
                dtype=torch.bool,
            )
        return query_indices, padding

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        offset = self._last_offset
        if offset <= 0:
            return item
        ep_idx = int(item["episode_index"].item())
        ep = self.meta.episodes[ep_idx]
        prev_idx = max(ep["dataset_from_index"], min(ep["dataset_to_index"] - 1, idx + offset - 1))
        state = item["observation.state"]
        previous_action = self.hf_dataset[prev_idx]["action"]
        if state.ndim != 1 or previous_action.ndim != 1 or state.shape != previous_action.shape:
            raise ValueError("Temporal delay requires equal one-dimensional state and action shapes")
        item["observation.state"] = previous_action
        return item


class PI0SharedObservationDataset(PI0TemporalDataset):
    """Return every valid delay while sharing image and language observations."""

    def __getitem__(self, idx: int) -> dict:
        ep_idx = int(self.hf_dataset[idx]["episode_index"].item())
        ep = self.meta.episodes[ep_idx]
        max_delta = self.delta_indices["action"][-1]
        max_offset = min(
            self.max_delay_steps,
            max(0, ep["dataset_to_index"] - 1 - idx - max_delta),
        )
        self._last_offset = 0
        base_item = super(PI0TemporalDataset, self).__getitem__(idx)
        result = {
            key: value
            for key, value in base_item.items()
            if key.startswith("observation.images.") or key in ("task", "episode_index")
        }
        states = []
        actions = []
        action_padding = []
        for offset in range(max_offset + 1):
            if offset == 0:
                state = base_item["observation.state"]
            else:
                previous_action = self.hf_dataset[idx + offset - 1]["action"]
                if base_item["observation.state"].shape != previous_action.shape:
                    raise ValueError("Shared delay requires equal state and action shapes")
                state = previous_action
            indices, padding = self._get_query_indices_for_offset(idx, ep_idx, offset)
            states.append(state)
            actions.append(torch.stack([self.hf_dataset[i]["action"] for i in indices["action"]]))
            action_padding.append(padding["action_is_pad"])

        result["observation.state"] = torch.stack(states)
        result["action"] = torch.stack(actions)
        result["action_is_pad"] = torch.stack(action_padding)
        result["num_offsets"] = max_offset + 1
        return result


def shared_observation_collate_fn(batch: list[dict]) -> dict:
    """Pad variable offset counts and collate shared observations."""
    max_offsets = max(item["num_offsets"] for item in batch)
    per_offset_keys = {"observation.state", "action", "action_is_pad"}
    shared_keys = [key for key in batch[0] if key not in per_offset_keys and key != "num_offsets"]
    result = default_collate([{key: item[key] for key in shared_keys} for item in batch])
    batch_size = len(batch)
    device = batch[0]["observation.state"].device
    state_shape = batch[0]["observation.state"].shape[1:]
    action_shape = batch[0]["action"].shape[1:]
    states = torch.zeros(batch_size, max_offsets, *state_shape, device=device)
    actions = torch.zeros(batch_size, max_offsets, *action_shape, device=device)
    action_is_pad = torch.ones(
        batch_size, max_offsets, action_shape[0], dtype=torch.bool, device=device
    )
    offset_mask = torch.zeros(batch_size, max_offsets, dtype=torch.bool, device=device)
    for index, item in enumerate(batch):
        count = item["num_offsets"]
        states[index, :count] = item["observation.state"]
        actions[index, :count] = item["action"]
        action_is_pad[index, :count] = item["action_is_pad"]
        offset_mask[index, :count] = True
    result["observation.state"] = states
    result["action"] = actions
    result["action_is_pad"] = action_is_pad
    result["offset_mask"] = offset_mask
    result["max_offsets"] = max_offsets
    return result
