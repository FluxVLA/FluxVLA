"""LingBot-VLA-v2 data-contract transforms."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class LingBotRobotSchemaPack:
    """Pack named robot groups into the fixed LingBot 55D state/action space."""

    def __init__(self, feature_groups: Sequence[Mapping], state_key: str = "state", action_key: str = "actions"):
        self.feature_groups = [dict(group) for group in feature_groups]
        self.state_key = state_key
        self.action_key = action_key
        self._validate()

    def _validate(self):
        offsets = []
        for group in self.feature_groups:
            if "name" not in group or "state_key" not in group or "action_key" not in group:
                raise ValueError("Each LingBot schema group needs name/state_key/action_key.")
            dim = int(group["dim"])
            offset = int(group["offset"])
            if dim <= 0 or offset < 0 or offset + dim > 55:
                raise ValueError(f"Invalid 55D group range: offset={offset}, dim={dim}")
            offsets.extend(range(offset, offset + dim))
        if len(set(offsets)) != len(offsets):
            raise ValueError("LingBot 55D schema groups overlap.")

    def __call__(self, sample):
        state = torch.zeros(55, dtype=torch.float32)
        if not self.feature_groups:
            raise ValueError("LingBot schema must contain at least one feature group.")
        first_action = torch.as_tensor(sample[self.feature_groups[0]["action_key"]])
        if first_action.ndim != 2:
            raise ValueError("Action features must have shape [horizon, dim].")
        actions = torch.zeros((first_action.shape[0], 55), dtype=torch.float32)
        state_mask = torch.zeros(55, dtype=torch.bool)
        action_mask = torch.zeros(55, dtype=torch.bool)
        for group in self.feature_groups:
            start = int(group["offset"])
            dim = int(group["dim"])
            state_value = torch.as_tensor(sample[group["state_key"]], dtype=torch.float32).flatten()
            action_value = torch.as_tensor(sample[group["action_key"]], dtype=torch.float32)
            if state_value.numel() > dim or action_value.shape[-1] > dim:
                raise ValueError(f"Feature {group['name']} exceeds configured dimension {dim}.")
            state[start : start + state_value.numel()] = state_value
            actions[:, start : start + action_value.shape[-1]] = action_value
            state_mask[start : start + state_value.numel()] = True
            action_mask[start : start + action_value.shape[-1]] = True
        sample = dict(sample)
        sample["state"] = state
        sample["actions"] = actions
        sample["state_joint_mask"] = state_mask
        sample["joint_mask"] = action_mask.unsqueeze(0).expand(actions.shape[0], -1)
        if "action_is_pad" not in sample:
            sample["action_is_pad"] = torch.zeros(actions.shape[0], dtype=torch.bool)
        return sample


@TRANSFORMS.register_module()
class LingBotPad55D:
    """Validate/pad pre-packed state and actions to the native 55D contract."""

    def __init__(self, state_key: str = "state", action_key: str = "actions"):
        self.state_key = state_key
        self.action_key = action_key

    def __call__(self, sample):
        sample = dict(sample)
        state = torch.as_tensor(sample[self.state_key])
        actions = torch.as_tensor(sample[self.action_key])
        if state.shape[-1] > 55 or actions.shape[-1] > 55:
            raise ValueError("LingBot state/actions cannot exceed 55 dimensions.")
        sample["state"] = F.pad(state, (0, 55 - state.shape[-1])).float()
        sample["actions"] = F.pad(actions, (0, 55 - actions.shape[-1])).float()
        sample.setdefault("state_joint_mask", torch.arange(55) < state.shape[-1])
        sample.setdefault(
            "joint_mask",
            (torch.arange(55) < actions.shape[-1]).unsqueeze(0).expand(actions.shape[0], -1),
        )
        sample.setdefault("action_is_pad", torch.zeros(actions.shape[0], dtype=torch.bool))
        return sample


@TRANSFORMS.register_module()
class LingBotQwen3VLImageTransform:
    """Prepare per-view Qwen3-VL patches for the native LingBot contract.

    HuggingFace returns all image patches concatenated along the first
    dimension.  LingBot's vision bridge consumes one patch sequence per view,
    so this transform restores ``[num_images, patches, patch_width]`` while
    preserving the native ``image_grid_thw`` metadata.
    """

    def __init__(
        self,
        processor_path: str,
        image_key: str = "images",
        image_size: int | None = 224,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        self.image_key = image_key
        if image_size is not None:
            if min_pixels is not None or max_pixels is not None:
                raise ValueError("Set image_size or min_pixels/max_pixels, not both.")
            min_pixels = max_pixels = int(image_size) * int(image_size)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(self, sample):
        images = sample[self.image_key]
        processor_kwargs = {"return_tensors": "pt"}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        processed = self.processor.image_processor(images=images, **processor_kwargs)
        pixel_values = processed["pixel_values"]
        grid = torch.as_tensor(processed["image_grid_thw"], dtype=torch.long)
        # The processor emits one row per raw vision patch.  The native vision
        # tower applies ``spatial_merge_size`` internally, so splitting here
        # uses grid volume rather than the post-merge token count.
        patch_counts = grid.prod(dim=-1).tolist()
        if sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError(
                "Qwen3-VL patch/grid mismatch: "
                f"patches={pixel_values.shape[0]}, grid_counts={patch_counts}"
            )
        if len(set(patch_counts)) != 1:
            raise ValueError("LingBot currently requires equal patch counts per image.")
        sample = dict(sample)
        sample[self.image_key] = pixel_values.reshape(len(patch_counts), patch_counts[0], -1)
        sample["image_grid_thw"] = grid
        sample.setdefault("img_masks", torch.ones(len(patch_counts), dtype=torch.bool))
        sample["img_masks"] = torch.as_tensor(sample["img_masks"], dtype=torch.bool)
        return sample
