"""Online LeRobot input pipeline for LingBot-VLA-v2.

This module deliberately owns the complete raw-data to model-input contract.
It does not import the reference LingBot package and does not consume captured
model-ready batches.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from fluxvla.engines import DATASETS
from .flow_rng import sample_flow_by_index


_RAW_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
_ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
_EFFECTOR_INDICES = (6, 13)


def _as_uint8(image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.is_floating_point():
        if image.numel() and float(image.max()) <= 2.0:
            image = image.mul(255).round()
        image = image.clamp(0, 255).to(torch.uint8)
    return image


def _bounds_99(value: torch.Tensor, stats: Mapping[str, Any]) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32)
    low = torch.as_tensor(stats["q01"], dtype=torch.float32)
    high = torch.as_tensor(stats["q99"], dtype=torch.float32)
    if low.shape[-1] != value.shape[-1] or high.shape[-1] != value.shape[-1]:
        raise ValueError(
            "LingBot normalization dimension mismatch: "
            f"value={tuple(value.shape)}, q01={tuple(low.shape)}, q99={tuple(high.shape)}"
        )
    return (value - low) / (high - low + 1.0e-6) * 2.0 - 1.0


def _load_norm_stats(path: str | Path) -> Mapping[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LingBot normalization statistics not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("norm_stats", payload)
    required = {
        "observation.state.arm.position",
        "observation.state.effector.position",
        "action.arm.position",
        "action.effector.position",
    }
    missing = sorted(required.difference(stats))
    if missing:
        raise ValueError(f"LingBot normalization statistics are missing keys: {missing}")
    return stats


def _build_lerobot_dataset(
    data_path: str | Path,
    *,
    action_horizon: int,
    image_size: int,
    use_future_images: bool,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from torchvision.transforms.v2 import Resize

    root = Path(data_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LingBot LeRobot dataset not found: {root}")
    repo_id = root.name
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = float(metadata.fps)
    delta_timestamps = {"action": [index / fps for index in range(action_horizon)]}
    if use_future_images:
        future_offset = (action_horizon - 1) / fps
        delta_timestamps.update(
            {camera: [0.0, future_offset] for camera in _RAW_CAMERA_KEYS}
        )
    dataset_kwargs = {
        "root": root,
        "delta_timestamps": delta_timestamps,
        "image_transforms": Resize((image_size, image_size)),
    }
    # ``return_uint8`` was removed from newer LeRobot releases.  The online
    # adapter already normalizes image tensors through ``_as_uint8``, so pass
    # the option only when the installed API still exposes it.
    if "return_uint8" in inspect.signature(LeRobotDataset).parameters:
        dataset_kwargs["return_uint8"] = True
    return LeRobotDataset(repo_id, **dataset_kwargs)


@DATASETS.register_module()
class LingBotOnlineDataset(Dataset):
    """Convert raw RoboTwin LeRobot samples into LingBot model tensors."""

    def __init__(
        self,
        data_path: str,
        norm_stats_path: str,
        tokenizer_path: str,
        *,
        action_horizon: int = 50,
        image_size: int = 256,
        tokenizer_max_length: int = 72,
        use_future_images: bool = True,
        return_teacher_images: bool = False,
        model_dtype: torch.dtype = torch.bfloat16,
        dataset=None,
        processor=None,
        flow_rng_seed: int | None = None,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("LingBot action horizon must be positive")
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.use_future_images = bool(use_future_images)
        self.return_teacher_images = bool(return_teacher_images)
        if model_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported LingBot model input dtype: {model_dtype}")
        self.model_dtype = model_dtype
        self.flow_rng_seed = None if flow_rng_seed is None else int(flow_rng_seed)
        self.norm_stats = _load_norm_stats(norm_stats_path)
        self.dataset = dataset or _build_lerobot_dataset(
            data_path,
            action_horizon=self.action_horizon,
            image_size=self.image_size,
            use_future_images=self.use_future_images,
        )
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                tokenizer_path,
                padding_side="right",
                trust_remote_code=True,
                local_files_only=True,
            )
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _split_robot_features(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.as_tensor(value, dtype=torch.float32)
        if value.shape[-1] != 14:
            raise ValueError(f"RoboTwin state/action must be 14D, got {tuple(value.shape)}")
        return value[..., list(_ARM_INDICES)], value[..., list(_EFFECTOR_INDICES)]

    def _pack_robot(self, raw: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        raw_state = torch.as_tensor(raw["observation.state"], dtype=torch.float32)
        raw_actions = torch.as_tensor(raw["action"], dtype=torch.float32)
        if raw_actions.ndim != 2 or raw_actions.shape[0] != self.action_horizon:
            raise ValueError(
                "LingBot action chunk mismatch: "
                f"expected [{self.action_horizon}, 14], got {tuple(raw_actions.shape)}"
            )
        state_arm, state_effector = self._split_robot_features(raw_state)
        action_arm, action_effector = self._split_robot_features(raw_actions)
        state_arm = _bounds_99(
            state_arm, self.norm_stats["observation.state.arm.position"]
        )
        state_effector = _bounds_99(
            state_effector, self.norm_stats["observation.state.effector.position"]
        )
        action_arm = _bounds_99(action_arm, self.norm_stats["action.arm.position"])
        action_effector = _bounds_99(
            action_effector, self.norm_stats["action.effector.position"]
        )

        state = torch.zeros(55, dtype=torch.float32)
        actions = torch.zeros((self.action_horizon, 55), dtype=torch.float32)
        state[:12] = state_arm
        state[28:30] = state_effector
        actions[:, :12] = action_arm
        actions[:, 28:30] = action_effector
        state_joint_mask = torch.zeros(55, dtype=torch.bool)
        action_joint_mask = torch.zeros(55, dtype=torch.bool)
        state_joint_mask[:12] = True
        state_joint_mask[28:30] = True
        action_joint_mask[:12] = True
        action_joint_mask[28:30] = True
        joint_mask = action_joint_mask.unsqueeze(0).expand(self.action_horizon, -1).clone()
        action_is_pad = torch.as_tensor(
            raw.get("action_is_pad", torch.zeros(self.action_horizon)),
            dtype=torch.bool,
        )
        return {
            "state": state,
            "actions": actions,
            "state_joint_mask": state_joint_mask,
            "action_joint_mask": action_joint_mask,
            "joint_mask": joint_mask,
            "action_is_pad": action_is_pad,
        }

    def _process_images(self, raw: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        current_images = []
        future_images = []
        for camera in _RAW_CAMERA_KEYS:
            frames = torch.as_tensor(raw[camera])
            if frames.ndim == 3:
                current = frames
                future = frames
            elif frames.ndim == 4:
                current = frames[0]
                future = frames[-1]
            else:
                raise ValueError(f"Unexpected LingBot camera tensor {camera}: {tuple(frames.shape)}")
            current_images.append(_as_uint8(current))
            future_images.append(_as_uint8(future))

        patches = []
        grids = []
        for image in current_images:
            processed = self.processor.image_processor(image)
            pixel_values = torch.as_tensor(processed["pixel_values"])
            grid = torch.as_tensor(processed["image_grid_thw"], dtype=torch.long).reshape(-1, 3)
            if grid.shape[0] != 1 or pixel_values.shape[0] != int(grid.prod()):
                raise ValueError(
                    "LingBot Qwen3-VL image patch/grid mismatch: "
                    f"patches={tuple(pixel_values.shape)}, grid={grid.tolist()}"
                )
            patches.append(pixel_values)
            grids.append(grid[0])
        if len({tuple(value.shape) for value in patches}) != 1:
            raise ValueError("LingBot cameras must produce equal Qwen3-VL patch shapes")
        result = {
            "images": torch.stack(patches),
            "image_grid_thw": torch.stack(grids),
            "img_masks": torch.ones(len(patches), dtype=torch.bool),
        }
        if self.return_teacher_images:
            result["pil_images"] = torch.stack(current_images)
            result["future_pil_images"] = torch.stack(future_images)
        return result

    def _tokenize(self, task: str) -> dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(task)}],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            [prompt],
            padding="max_length",
            padding_side="right",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "lang_tokens": tokenized["input_ids"][0].long(),
            "lang_masks": tokenized["attention_mask"][0].bool(),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raw = self.dataset[index]
        sample = self._pack_robot(raw)
        sample.update(self._process_images(raw))
        sample.update(self._tokenize(raw["task"]))
        for key in ("state", "actions", "images"):
            sample[key] = sample[key].to(dtype=self.model_dtype)
        sample["sample_index"] = torch.tensor(index, dtype=torch.long)
        if self.flow_rng_seed is not None:
            noise, time = sample_flow_by_index(
                sample["sample_index"].reshape(1),
                (1, *sample["actions"].shape),
                seed=self.flow_rng_seed,
                dtype=sample["actions"].dtype,
            )
            sample["noise"] = noise[0]
            sample["time"] = time[0]
        return sample


__all__ = ["LingBotOnlineDataset"]
